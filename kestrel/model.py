"""Kestrel: a Llama-shaped decoder-only transformer sized for a 50M parameter budget.

The architecture is deliberately conventional (RMSNorm / RoPE / GQA / SwiGLU / tied embeddings)
so that lm-evaluation-harness, HF `transformers`, and GGUF/ONNX export all work without custom
glue. The novelty in this project lives in the data pipeline and the training recipe, not in a
transformer block that no downstream tool can load.

Shape is not assumed. Below ~1B parameters, depth is reported to buy more benchmark accuracy per
parameter than width (cf. MobileLLM, Liu et al. 2024) -- but on 8 GB of VRAM the relevant currency
is accuracy per GPU-hour, and depth is measurably more expensive in that currency: it costs both
sequential kernel launches and stored activations, and the latter caps the micro-batch. Four shapes
matched to the same parameter budget live in `kestrel/variants.py`; `scripts/ablate.py --suite arch`
settles which one wins.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from kestrel.variants import DEFAULT_VARIANT, PARAM_CAP, VARIANTS

# The geometry defaults below are read from this dict rather than written out again. Copying them
# by hand is how `very-deep` came to be documented as 47.1M while actually being 45.08M.
_DEFAULT = VARIANTS[DEFAULT_VARIANT]


@dataclass
class KestrelConfig:
    """All architecture knobs. `estimate_params()` lets us check the 50M cap without building."""

    vocab_size: int = 16384
    d_model: int = _DEFAULT["d_model"]
    n_layers: int = _DEFAULT["n_layers"]
    n_heads: int = _DEFAULT["n_heads"]
    n_kv_heads: int = _DEFAULT["n_kv_heads"]   # GQA: n_heads must be divisible by this
    ffn_hidden: int = _DEFAULT["ffn_hidden"]   # SwiGLU intermediate size
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True  # required to stay under the cap at this width
    qk_norm: bool = True         # RMSNorm on q/k heads; cheap insurance against loss spikes
    init_std: float = 0.02

    # Rows of hidden state per chunk when computing the loss. The output head produces a
    # (batch * seq, vocab) logit tensor which, at 8 GB of VRAM, is the single largest allocation
    # in the whole step -- larger than the model. Chunking it under gradient checkpointing trades
    # ~4% extra compute for several GB. 0 disables chunking.
    loss_chunk_size: int = 2048

    # How GQA reaches the attention kernel. `enable_gqa=True` looks cleaner, but on some builds it
    # disqualifies the flash/mem-efficient CUDA kernels and silently falls back to the math
    # backend, which materialises a (batch, heads, seq, seq) score matrix. Measured on this box:
    # see scripts/bench.py. Default is the explicit expand, which keeps flash attention.
    sdpa_enable_gqa: bool = False

    # Hard limit imposed by the competition rules (includes embeddings and the output head).
    param_budget: int = PARAM_CAP

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_params(self) -> dict[str, int]:
        """Closed-form parameter count, matching the modules built below."""
        d, h, kv = self.d_model, self.head_dim, self.n_kv_heads
        attn = d * d + 2 * (d * kv * h) + d * d          # q, k, v, o  (no biases)
        attn += 2 * h if self.qk_norm else 0             # q-norm + k-norm weights
        ffn = 3 * d * self.ffn_hidden                    # gate, up, down
        norms = 2 * d                                    # two pre-norms per block
        per_layer = attn + ffn + norms
        embed = self.vocab_size * d
        total = self.n_layers * per_layer + embed + d    # + final norm
        if not self.tie_embeddings:
            total += embed
        return {
            "per_layer": per_layer,
            "layers": self.n_layers * per_layer,
            "embedding": embed,
            "total": total,
            "non_embedding": total - embed,
            "budget": self.param_budget,
            "headroom": self.param_budget - total,
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 regardless of autocast dtype; this is the cheap half of stability.
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device=None, dtype=torch.float32):
    """Precompute RoPE cos/sin tables of shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). cos/sin: (T, head_dim // 2)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, hd) -> (B, n_kv * n_rep, T, hd). `expand` keeps this allocation-free."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over the output head without ever materialising all logits at once.

    Each chunk's logits are recomputed during backward instead of being stored, which caps peak
    memory at `chunk_size * vocab_size` floats rather than `batch * seq * vocab_size`.
    """
    hidden = hidden.reshape(-1, hidden.size(-1))
    targets = targets.reshape(-1)
    if chunk_size <= 0 or hidden.size(0) <= chunk_size:
        logits = F.linear(hidden, weight).float()
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    def chunk_loss(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = F.linear(h, weight).float()
        return F.cross_entropy(logits, t, ignore_index=ignore_index, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for h, t in zip(hidden.split(chunk_size), targets.split(chunk_size)):
        total = total + checkpoint(chunk_loss, h, t, use_reentrant=False)
    n_valid = (targets != ignore_index).sum().clamp(min=1)
    return total / n_valid


class Attention(nn.Module):
    """Grouped-query attention. GQA is here for the KV-cache footprint at inference time as much
    as for the parameter saving — memory footprint is a scored criterion."""

    def __init__(self, cfg: KestrelConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.wq = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)
        norm = (lambda: RMSNorm(self.head_dim, cfg.norm_eps)) if cfg.qk_norm else (lambda: nn.Identity())
        self.q_norm, self.k_norm = norm(), norm()
        self.enable_gqa = cfg.sdpa_enable_gqa and self.n_rep > 1

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.enable_gqa:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    """SwiGLU. Three matrices instead of two, but better loss per parameter than plain GeLU MLP."""

    def __init__(self, cfg: KestrelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: KestrelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Kestrel(nn.Module):
    def __init__(self, cfg: KestrelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Downscale the projections that write into the residual stream, so activation variance
        # does not grow with depth. Matters more here than usual: 26 layers is a lot for 384 dims.
        for block in self.blocks:
            for w in (block.attn.wo.weight, block.ffn.w_down.weight):
                nn.init.normal_(w, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_logits: bool = False,
    ):
        """Returns (logits_or_None, loss_or_None).

        During training we deliberately do not return logits: keeping a (batch, seq, vocab) tensor
        alive is the largest single allocation in the step. Pass return_logits=True for inference.
        """
        _, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x), None

        loss = chunked_cross_entropy(
            x, self.lm_head.weight, targets, self.cfg.loss_chunk_size
        )
        return (self.lm_head(x) if return_logits else None), loss

    # ---- reporting helpers (the competition requires a printed parameter count) ----

    def param_report(self) -> dict[str, int]:
        seen, total = set(), 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:  # tied weights must not be double counted
                seen.add(id(p))
                total += p.numel()
        embed = self.embed.weight.numel()
        return {
            "total": total,
            "embedding": embed,
            "non_embedding": total - embed,
            "budget": self.cfg.param_budget,
            "headroom": self.cfg.param_budget - total,
            "within_budget": total <= self.cfg.param_budget,
        }

    def flops_per_token(self) -> int:
        """Forward+backward FLOPs per token, incl. the attention score/value matmuls."""
        cfg, n = self.cfg, self.param_report()["total"]
        return 6 * n + 12 * cfg.n_layers * cfg.d_model * cfg.max_seq_len

    def optim_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters for a Muon(hidden matrices) + AdamW(everything else) setup.

        Muon only makes sense for 2D weight matrices inside the network; embeddings, the output
        head, and 1D norm gains stay on AdamW.
        """
        muon, adamw_decay, adamw_nodecay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("embed.") or name.startswith("lm_head."):
                adamw_decay.append(p)
            elif p.ndim >= 2:
                muon.append(p)
            else:
                adamw_nodecay.append(p)
        return {"muon": muon, "adamw_decay": adamw_decay, "adamw_nodecay": adamw_nodecay}

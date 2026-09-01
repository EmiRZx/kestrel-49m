"""Training loop for Kestrel.

Built around three constraints that a cloud training script would not have:

1. **The run will be interrupted.** It shares a laptop with its owner. Every checkpoint carries the
   dataset cursor and both optimizer states, and `--max-hours` lets a night's session end cleanly on
   a checkpoint rather than on a killed process. Resuming continues the exact same data order.
2. **Compute must be accounted for.** Total GPU-hours, tokens/s, MFU, and energy drawn are logged
   continuously, because Training Efficiency is scored and cannot be reconstructed afterwards.
3. **8 GB of VRAM.** Micro-batches are small and gradient accumulation carries the effective batch,
   so the optimizer sees a stable ~0.26M tokens per step regardless of what fits on the card.
"""

from __future__ import annotations

import json
import math
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from kestrel.data import ShardedTokenDataset
from kestrel.model import Kestrel, KestrelConfig
from kestrel.muon import Muon
from kestrel.variants import VARIANTS

# Refuse to allocate beyond this fraction of VRAM. Without it, Windows' WDDM driver silently spills
# to system RAM instead of raising, and throughput collapses ~30x while appearing to "work".
# `scripts/bench.py` imports this rather than defining its own: the bench exists to predict what
# the trainer can fit, which it cannot do if the two cap memory differently.
MEMORY_FRACTION = 0.90


@dataclass
class TrainConfig:
    # data
    data_dir: Path = Path("data/tokens_v16384")
    vocab_size: int = 16384
    seq_len: int = 1024

    # model
    variant: str = "deep-thin"
    loss_chunk_size: int = 2048

    # optimization
    total_tokens: float = 6e9
    tokens_per_step: int = 262_144      # effective batch; accumulation fills it in
    micro_batch: int = 16
    optimizer: str = "muon+adamw"       # or "adamw" for the ablation baseline
    muon_lr: float = 0.02
    adamw_lr: float = 3e-3
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # schedule: warmup -> stable -> decay (WSD). Cosine available for the ablation.
    schedule: str = "wsd"
    warmup_tokens: float = 1.5e8
    decay_frac: float = 0.2             # last 20% of tokens spent decaying
    min_lr_frac: float = 0.0

    # run control
    name: str = "kestrel"
    out_dir: Path = Path("runs")
    seed: int = 1337
    compile: bool = True
    max_hours: float = 0.0              # 0 = no limit; otherwise stop cleanly at this wall time
    eval_every_tokens: float = 2e8
    eval_batches: int = 40
    ckpt_every_tokens: float = 4e8
    ckpt_every_minutes: float = 30.0    # wall-clock safety net: an interruption costs <= this much
    log_every_steps: int = 10
    wandb: bool = False
    resume: str = ""                    # path to a checkpoint, or "auto" for <run>/last.pt

    def model_config(self) -> KestrelConfig:
        return KestrelConfig(
            vocab_size=self.vocab_size,
            max_seq_len=self.seq_len,
            loss_chunk_size=self.loss_chunk_size,
            **VARIANTS[self.variant],
        )

    @property
    def grad_accum(self) -> int:
        per_micro = self.micro_batch * self.seq_len
        return max(1, self.tokens_per_step // per_micro)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name


def lr_multiplier(cfg: TrainConfig, tokens_seen: float) -> float:
    """WSD (or cosine) schedule expressed in tokens rather than steps.

    Tokens are the honest x-axis here: micro-batch size changes with what fits in VRAM, and an
    ablation may use a different batch entirely, but a token is a token.
    """
    total, warm = cfg.total_tokens, cfg.warmup_tokens
    if tokens_seen < warm:
        return tokens_seen / max(warm, 1.0)
    if cfg.schedule == "cosine":
        prog = (tokens_seen - warm) / max(total - warm, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cos

    decay_start = total * (1.0 - cfg.decay_frac)
    if tokens_seen < decay_start:
        return 1.0                      # the "stable" phase: constant LR, checkpoints are reusable
    prog = (tokens_seen - decay_start) / max(total - decay_start, 1.0)
    # 1 - sqrt(progress) decays fast at first and flattens, which empirically beats linear here.
    shape = max(0.0, 1.0 - math.sqrt(min(prog, 1.0)))
    return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * shape


class PowerMeter:
    """Integrate GPU power draw into kWh, and watch for thermal throttling.

    Two reasons this is here rather than in a side script. First, "accuracy per kWh" is a far more
    honest efficiency number than GPU-hours on unspecified hardware, and it can only be measured
    while the run happens. Second, this is a laptop: if the GPU starts thermal-throttling at 3am the
    tokens/s number quietly halves, and without the throttle reason logged the slowdown looks like a
    mystery the next morning.
    """

    def __init__(self) -> None:
        self.ok = False
        self.energy_j = 0.0
        self._last_t = time.perf_counter()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self._h)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
            self.ok = True
        except Exception:
            self.name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    def sample(self) -> dict:
        """Accumulate energy since the last call and return the instantaneous readings."""
        now = time.perf_counter()
        dt, self._last_t = now - self._last_t, now
        if not self.ok:
            return {}
        try:
            watts = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            temp = self._nvml.nvmlDeviceGetTemperature(self._h, self._nvml.NVML_TEMPERATURE_GPU)
            clock = self._nvml.nvmlDeviceGetClockInfo(self._h, self._nvml.NVML_CLOCK_SM)
        except Exception:
            self.ok = False
            return {}
        self.energy_j += watts * dt
        out = {"watts": round(watts, 1), "gpu_temp_c": temp, "sm_clock_mhz": clock,
               "energy_kwh": round(self.energy_j / 3.6e6, 5)}
        try:  # bit 0 of the throttle mask is "not throttling"
            reasons = self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._h)
            if reasons and reasons != 1:
                out["throttle"] = hex(reasons)
        except Exception:
            pass
        return out


@dataclass
class RunState:
    """Everything a resumed run must agree with the interrupted one about."""
    step: int = 0
    tokens: int = 0
    gpu_seconds: float = 0.0            # wall time actually spent training, across all sessions
    energy_j: float = 0.0
    best_val: float = float("inf")
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.state = RunState()
        cfg.run_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Raise a real OOM instead of letting Windows' WDDM driver spill to system RAM, which looks
        # like success and runs ~30x slower.
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Kestrel(cfg.model_config()).to(self.device)
        self.report = self.model.param_report()
        if not self.report["within_budget"]:
            raise ValueError(f"{self.report['total']:,} params exceeds the 50M cap")
        self.flops_per_token = self.model.flops_per_token()

        self.train_data = ShardedTokenDataset(cfg.data_dir, "train", cfg.seq_len, seed=cfg.seed)
        self.val_data = ShardedTokenDataset(cfg.data_dir, "val", cfg.seq_len, seed=cfg.seed + 1)
        try:
            self.wt103 = ShardedTokenDataset(cfg.data_dir, "wt103", cfg.seq_len, seed=cfg.seed + 2)
        except FileNotFoundError:
            self.wt103 = None

        self._build_optimizers()
        if cfg.resume:
            path = cfg.run_dir / "last.pt" if cfg.resume == "auto" else Path(cfg.resume)
            if path.exists():
                self.load(path)
            elif cfg.resume != "auto":
                raise FileNotFoundError(path)

        self.raw_model = self.model
        if cfg.compile:
            self.model = torch.compile(self.model)

        self.power = PowerMeter()
        self.power.energy_j = self.state.energy_j
        self.metrics_path = cfg.run_dir / "metrics.jsonl"
        self.stop_flag = cfg.run_dir / "STOP"
        self._interrupted = False
        self._last_eval_tokens = -1
        signal.signal(signal.SIGINT, self._on_signal)
        self._wandb = None
        if cfg.wandb:
            self._init_wandb()

    # ---- setup helpers -------------------------------------------------------------------------

    def _on_signal(self, *_) -> None:
        """Ctrl-C once: finish the step, checkpoint, exit. Ctrl-C twice: give up immediately."""
        if self._interrupted:
            raise KeyboardInterrupt
        self._interrupted = True
        print("\n[interrupt] finishing this step, then saving a resumable checkpoint...", flush=True)

    def _init_wandb(self) -> None:
        try:
            import wandb

            self._wandb = wandb
            wandb.init(project="kestrel", name=self.cfg.name,
                       config={**{k: str(v) for k, v in asdict(self.cfg).items()},
                               "params_total": self.report["total"]},
                       resume="allow", id=self.cfg.name)
        except Exception as exc:
            print(f"[wandb] disabled ({type(exc).__name__}: {exc})")

    def _build_optimizers(self) -> None:
        cfg, groups = self.cfg, self.model.optim_param_groups()
        adamw_groups = [
            {"params": groups["adamw_decay"], "weight_decay": cfg.weight_decay},
            {"params": groups["adamw_nodecay"], "weight_decay": 0.0},
        ]
        if cfg.optimizer == "adamw":
            # Ablation baseline: everything on AdamW, hidden matrices included.
            adamw_groups.insert(0, {"params": groups["muon"], "weight_decay": cfg.weight_decay})
            self.muon = None
        else:
            self.muon = Muon(groups["muon"], lr=cfg.muon_lr, weight_decay=cfg.weight_decay)
        self.adamw = torch.optim.AdamW(adamw_groups, lr=cfg.adamw_lr, betas=cfg.adamw_betas,
                                       eps=1e-8, fused=(self.device == "cuda"))
        self.optimizers = [o for o in (self.muon, self.adamw) if o is not None]
        n_muon = sum(p.numel() for p in groups["muon"])
        print(f"optimizers: {cfg.optimizer}  muon={n_muon/1e6:.2f}M params in "
              f"{len(groups['muon'])} matrices, adamw={(self.report['total']-n_muon)/1e6:.2f}M")

    def _set_lr(self, mult: float) -> tuple[float, float]:
        for group in self.adamw.param_groups:
            group["lr"] = self.cfg.adamw_lr * mult
        if self.muon is not None:
            for group in self.muon.param_groups:
                group["lr"] = self.cfg.muon_lr * mult
        return self.cfg.adamw_lr * mult, (self.cfg.muon_lr * mult if self.muon else 0.0)

    # ---- the loop ------------------------------------------------------------------------------

    def _batch(self, dataset: ShardedTokenDataset, batch_size: int):
        b = dataset.next_batch(batch_size)
        x = torch.from_numpy(b.x).to(self.device, non_blocking=True)
        y = torch.from_numpy(b.y).to(self.device, non_blocking=True)
        return x, y

    def train_step(self) -> dict:
        cfg = self.cfg
        accum = cfg.grad_accum
        mult = lr_multiplier(cfg, self.state.tokens)
        adamw_lr, muon_lr = self._set_lr(mult)

        self.model.train()
        total_loss = 0.0
        for _ in range(accum):
            x, y = self._batch(self.train_data, cfg.micro_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                _, loss = self.model(x, y)
            (loss / accum).backward()
            total_loss += loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.grad_clip)
        for opt in self.optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        self.state.step += 1
        self.state.tokens += accum * cfg.micro_batch * cfg.seq_len
        return {"loss": float(total_loss) / accum, "lr": adamw_lr, "muon_lr": muon_lr,
                "grad_norm": float(grad_norm), "lr_mult": mult}

    @torch.no_grad()
    def evaluate(self, dataset: ShardedTokenDataset, batches: int) -> float:
        """Mean token-level cross-entropy over a fixed slice, from a saved-and-restored cursor.

        Restoring the cursor matters: evaluation must not consume training data, and every eval must
        score the *same* tokens or the curve measures the data, not the model.
        """
        if dataset is None:
            return float("nan")
        self.model.eval()
        cursor, dataset.cursor = dataset.cursor, 0
        total = 0.0
        try:
            for _ in range(batches):
                x, y = self._batch(dataset, self.cfg.micro_batch)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    _, loss = self.model(x, y)
                total += float(loss)
        finally:
            dataset.cursor = cursor
            self.model.train()
        return total / max(batches, 1)

    # ---- persistence ---------------------------------------------------------------------------

    def save(self, tag: str = "last") -> Path:
        cfg = self.cfg
        self.state.energy_j = self.power.energy_j
        ckpt = {
            "model": self.raw_model.state_dict(),
            "adamw": self.adamw.state_dict(),
            "muon": self.muon.state_dict() if self.muon else None,
            "train_data": self.train_data.state_dict(),
            "state": asdict(self.state),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
            "model_config": asdict(cfg.model_config()),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = cfg.run_dir / f"{tag}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)                # atomic: a crash mid-save must not destroy the last one
        return path

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model = self.model  # not yet compiled at resume time
        self.model.load_state_dict(ckpt["model"])
        self.adamw.load_state_dict(ckpt["adamw"])
        if self.muon is not None and ckpt.get("muon"):
            self.muon.load_state_dict(ckpt["muon"])
        self.train_data.load_state_dict(ckpt["train_data"])
        saved = ckpt["state"]
        self.state = RunState(**{k: saved[k] for k in RunState().__dict__ if k in saved})
        torch.set_rng_state(ckpt["torch_rng"].cpu() if hasattr(ckpt["torch_rng"], "cpu")
                            else ckpt["torch_rng"])
        if ckpt.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
        print(f"resumed {path.name}: step {self.state.step:,}, {self.state.tokens/1e9:.3f}B tokens, "
              f"{self.state.gpu_seconds/3600:.2f} GPU-hours already spent")

    def log(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(record, step=record.get("step"))

    # ---- driver --------------------------------------------------------------------------------

    def _banner(self) -> None:
        cfg, r = self.cfg, self.report
        steps = int(cfg.total_tokens / (cfg.grad_accum * cfg.micro_batch * cfg.seq_len))
        print(f"\n{cfg.name}: {cfg.variant}, {r['total']/1e6:.2f}M params "
              f"({r['embedding']/1e6:.2f}M embedding, {r['headroom']/1e6:.2f}M under the cap)")
        print(f"  data     {self.train_data.total_tokens/1e9:.3f}B train tokens in "
              f"{len(self.train_data.manifest['shards'])} shards, "
              f"{self.val_data.total_tokens/1e6:.1f}M val"
              + (f", {self.wt103.total_tokens/1e6:.1f}M wt103" if self.wt103 else ""))
        print(f"  batch    {cfg.micro_batch} x {cfg.seq_len} x {cfg.grad_accum} accum = "
              f"{cfg.grad_accum * cfg.micro_batch * cfg.seq_len:,} tokens/step")
        print(f"  budget   {cfg.total_tokens/1e9:.2f}B tokens = ~{steps:,} steps, "
              f"schedule={cfg.schedule} warmup={cfg.warmup_tokens/1e6:.0f}M "
              f"decay={cfg.decay_frac:.0%}")
        print(f"  device   {self.power.name}, compile={cfg.compile}"
              + (f", stop after {cfg.max_hours:.1f}h" if cfg.max_hours else ""))
        (self.cfg.run_dir / "config.json").write_text(
            json.dumps({k: str(v) for k, v in asdict(cfg).items()}
                       | {"params": r, "flops_per_token": self.flops_per_token}, indent=2),
            encoding="utf-8")

    def run(self) -> RunState:
        cfg = self.cfg
        self._banner()
        tokens_per_step = cfg.grad_accum * cfg.micro_batch * cfg.seq_len
        next_eval = (self.state.tokens // cfg.eval_every_tokens + 1) * cfg.eval_every_tokens
        next_ckpt = (self.state.tokens // cfg.ckpt_every_tokens + 1) * cfg.ckpt_every_tokens
        session_start = time.perf_counter()
        last_ckpt_t = session_start
        window_t, window_tokens = time.perf_counter(), 0
        reason = "budget reached"

        while self.state.tokens < cfg.total_tokens:
            t0 = time.perf_counter()
            m = self.train_step()
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.state.gpu_seconds += dt
            window_tokens += tokens_per_step

            if self.state.step % cfg.log_every_steps == 0:
                now = time.perf_counter()
                tps = window_tokens / max(now - window_t, 1e-6)
                window_t, window_tokens = now, 0
                rec = {
                    "step": self.state.step,
                    "tokens": self.state.tokens,
                    "loss": round(m["loss"], 4),
                    "ppl": round(math.exp(min(m["loss"], 20)), 2),
                    "lr": round(m["lr"], 6),
                    "muon_lr": round(m["muon_lr"], 6),
                    "grad_norm": round(m["grad_norm"], 3),
                    "tokens_per_s": round(tps),
                    "mfu": round(tps * self.flops_per_token / 1e12 / 80.0, 4),
                    "step_ms": round(dt * 1000, 1),
                    "gpu_hours": round(self.state.gpu_seconds / 3600, 4),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    if self.device == "cuda" else 0.0,
                    **self.power.sample(),
                }
                self.log(rec)
                eta = (cfg.total_tokens - self.state.tokens) / max(tps, 1) / 3600
                print(f"step {rec['step']:>6}  {rec['tokens']/1e9:6.3f}B  loss {rec['loss']:6.4f}  "
                      f"ppl {rec['ppl']:>8.2f}  {rec['tokens_per_s']:>7,} tok/s  "
                      f"mfu {rec['mfu']:>5.1%}  {rec.get('gpu_temp_c', '-')}C  "
                      f"eta {eta:5.1f}h" + ("  THROTTLED" if "throttle" in rec else ""), flush=True)

            if self.state.tokens >= next_eval:
                next_eval += cfg.eval_every_tokens
                self._run_eval()

            hours = (time.perf_counter() - session_start) / 3600
            out_of_time = cfg.max_hours and hours >= cfg.max_hours
            # Three reasons to checkpoint: enough tokens, enough wall-clock, or we are stopping.
            # The wall-clock one matters most on an unattended overnight run: it bounds what a
            # power cut or a closed lid can cost to at most ckpt_every_minutes of compute.
            due_tokens = self.state.tokens >= next_ckpt
            mins_since_ckpt = (time.perf_counter() - last_ckpt_t) / 60
            due_time = cfg.ckpt_every_minutes and mins_since_ckpt >= cfg.ckpt_every_minutes
            if due_tokens or due_time or out_of_time or self._interrupted:
                if due_tokens:
                    next_ckpt += cfg.ckpt_every_tokens
                last_ckpt_t = time.perf_counter()
                self.save("last")
                print(f"  checkpoint at {self.state.tokens/1e9:.3f}B tokens"
                      + (f" ({mins_since_ckpt:.0f} min since last)" if due_time else ""),
                      flush=True)
            if self._interrupted or out_of_time or self.stop_flag.exists():
                reason = ("interrupted" if self._interrupted else
                          f"reached --max-hours {cfg.max_hours}" if out_of_time else "STOP file")
                break

        # A final eval, unless the cadence already scored this exact token count a moment ago --
        # repeating it would burn a minute and put two identical lines in the log.
        if self._last_eval_tokens != self.state.tokens:
            self._run_eval()
        self.save("last")
        print(f"\nstopped: {reason} at {self.state.tokens/1e9:.3f}B tokens, "
              f"{self.state.gpu_seconds/3600:.2f} GPU-hours, "
              f"{self.power.energy_j/3.6e6:.3f} kWh, best val loss {self.state.best_val:.4f}")
        return self.state

    def _run_eval(self) -> dict:
        t0 = time.perf_counter()
        self._last_eval_tokens = self.state.tokens
        val = self.evaluate(self.val_data, self.cfg.eval_batches)
        wt = self.evaluate(self.wt103, self.cfg.eval_batches) if self.wt103 else float("nan")
        self.state.gpu_seconds += time.perf_counter() - t0
        rec = {"step": self.state.step, "tokens": self.state.tokens, "eval": True,
               "val_loss": round(val, 4), "val_ppl": round(math.exp(min(val, 20)), 3),
               "wt103_loss": round(wt, 4), "wt103_ppl": round(math.exp(min(wt, 20)), 3),
               "gpu_hours": round(self.state.gpu_seconds / 3600, 4)}
        self.log(rec)
        self.state.history.append(rec)
        star = ""
        if val < self.state.best_val:
            self.state.best_val = val
            self.save("best")
            star = "  *best"
        print(f"  eval @ {self.state.tokens/1e9:6.3f}B: val {val:.4f} (ppl {rec['val_ppl']:.2f})  "
              f"wt103 ppl {rec['wt103_ppl']:.2f}{star}", flush=True)
        return rec








"""Muon: momentum SGD whose update is orthogonalized before it is applied.

Implementation of the optimizer described by Keller Jordan et al. (2024), "Muon: An optimizer for
hidden layers in neural networks" (https://kellerjordan.github.io/posts/muon/), as used in the
modded-nanogpt speedruns. Credited in the README's Built With.

The idea: for a 2D weight matrix, take the momentum buffer and replace it with the nearest
semi-orthogonal matrix (its `UV^T` from an SVD) before stepping. Computing an SVD every step would
be far too slow, so the orthogonalization is approximated by a fixed 5-step Newton-Schulz iteration
in bf16, which costs a handful of small matmuls -- negligible against the model's own forward pass.

Why it matters here: Training Efficiency is a scored criterion and our compute budget is a laptop.
Muon reaches a given loss in materially fewer steps than AdamW on transformer hidden layers, which
converts directly into fewer GPU-hours. It applies only to 2D hidden matrices; embeddings, the
output head, and 1D gains stay on AdamW.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Coefficients of the quintic iteration, tuned so the singular values of the result land near 1.
# They intentionally do not converge to exactly 1 -- overshooting slightly works better in practice
# than an exact orthogonalization would.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def zeropower_via_newtonschulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal factor of G (i.e. UV^T) with a quintic Newton-Schulz iteration."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects 2D parameters, got shape {tuple(G.shape)}")
    a, b, c = _NS_COEFFS
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D hidden weights. Pair it with AdamW for everything else."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                update = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz(update, steps=group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                # Scale by the aspect ratio so wide and tall matrices take comparably sized steps.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
        return loss

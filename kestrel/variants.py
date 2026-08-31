"""Candidate architectures at the same ~47M parameter budget, spent differently.

Depth-vs-width is the first thing the ablation sweep has to settle, and it is not purely a quality
question: at 8 GB of VRAM a deep-thin model runs many small matmuls and can be markedly slower per
token than a wider one with the same parameter count. What we actually want to maximise is
accuracy per GPU-hour, not accuracy per parameter.
"""

from __future__ import annotations

VARIANTS: dict[str, dict[str, int]] = {
    "deep-thin": dict(d_model=384, n_layers=26, n_heads=6, n_kv_heads=2, ffn_hidden=1024),
    "mid": dict(d_model=512, n_layers=14, n_heads=8, n_kv_heads=2, ffn_hidden=1376),
    "wide-shallow": dict(d_model=640, n_layers=9, n_heads=10, n_kv_heads=2, ffn_hidden=1728),
    "very-deep": dict(d_model=320, n_layers=36, n_heads=5, n_kv_heads=1, ffn_hidden=896),
}

VOCAB_SIZES = (8192, 16384, 32768)

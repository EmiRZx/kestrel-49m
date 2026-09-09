"""Candidate architectures, all sized to the same parameter budget so that shape is the variable.

Depth-vs-width is the first thing the ablation sweep has to settle, and it is not purely a quality
question: at 8 GB of VRAM a deep-thin model runs many small matmuls and can be markedly slower per
token than a wider one with the same parameter count. What we actually want to maximise is
accuracy per GPU-hour, not accuracy per parameter.

That comparison only means anything if the parameter count is held fixed, and an earlier version of
this file did not hold it fixed. The four shapes spanned 45.08M to 49.21M -- a 4.13M, 9.2% spread --
because each ffn_hidden had been chosen by hand to land "about 47M" and nobody had counted. Had the
sweep run that way, a win for `wide-shallow` would have been unattributable: it carried 9% more
parameters than `very-deep`, so the widest shape would also have been the biggest model, and the
result would have measured capacity rather than shape.

So ffn_hidden is no longer chosen by hand. Each variant takes the **largest ffn_hidden, in multiples
of 8, whose total parameter count stays under 49.5M** -- 8 because that is the alignment bf16 tensor
cores want, and 49.5M to keep a safety margin under the hard 50M cap that Track 01 imposes. d_model,
n_layers and the head counts are the shape being tested and are left alone.

One rule, two problems solved. The spread falls to 0.25M (0.51%), far below any effect the sweep
could resolve, and every variant now spends the budget the rules allow instead of leaving up to
4.92M of it unused -- so whichever shape wins is already sized for the main run rather than needing
to be rescaled afterwards, which would have invalidated the very measurement that chose it.

    variant        d_model x layers   heads Q/KV   ffn_hidden   params
    very-deep      320 x 36           5 / 1        1016         49.23M
    deep-thin      384 x 26           6 / 2        1096         49.37M
    mid            512 x 14           8 / 2        1480         49.41M
    wide-shallow   640 x  9           10 / 2       1744         49.48M

All four run head_dim 64. Re-derive the table at any time with `python scripts/count_params.py
--sweep`, which is also the script that enforces the cap.

This module is the single source of truth for the geometry. `KestrelConfig`'s field defaults and
`count_params.py`'s comparison table both read `VARIANTS[DEFAULT_VARIANT]` rather than repeating the
numbers, because repeating them is precisely how the 45.08M error survived unnoticed for two days:
a hand-copied ffn_hidden in a second file cannot be kept honest by anything. Once the arch sweep
picks a winner, changing `DEFAULT_VARIANT` here moves the default everywhere at once.
"""

from __future__ import annotations

VARIANTS: dict[str, dict[str, int]] = {
    "deep-thin": dict(d_model=384, n_layers=26, n_heads=6, n_kv_heads=2, ffn_hidden=1096),
    "mid": dict(d_model=512, n_layers=14, n_heads=8, n_kv_heads=2, ffn_hidden=1480),
    "wide-shallow": dict(d_model=640, n_layers=9, n_heads=10, n_kv_heads=2, ffn_hidden=1744),
    "very-deep": dict(d_model=320, n_layers=36, n_heads=5, n_kv_heads=1, ffn_hidden=1016),
}

# Parameter budget each variant is fitted to, and the hard cap it must not cross. Kept here rather
# than in the fitting script so the number that shaped these configs sits beside them.
FIT_TARGET = 49_500_000
PARAM_CAP = 50_000_000

# The shape `KestrelConfig()` defaults to, chosen by `ablate.py --suite arch` at 300M tokens a cell.
# Three independent readings agree, and the two that can resolve anything both pick this one: paired
# val loss over 3,902 held-out blocks (0.0020 nats/token ahead of `mid`, 5 sigma), and the harness'
# WikiText test perplexity (69.63 vs 69.90, wide-shallow ahead on 42 of 62 documents, p = 0.007).
# Downstream accuracy cannot see the difference at all -- 0.0023 mean over chance with a paired SEM
# of 0.0051 -- and the reason is that the four shapes answer 78-86% of HellaSwag, ARC-Easy and PIQA
# items identically, so there is almost nothing for a paired test to be lopsided about.
#
# At fixed compute the margin is much larger than at fixed tokens: 0.089 nats at 1.27 GPU-hours,
# because this shape processes 26% more tokens per second than `mid`. That is the framing the wall
# clock actually imposes, and it is where the win comes from.
#
# One name is the whole switch: `trainer.TrainConfig.variant` and the `--variant` defaults in
# train.py, bench.py and ablate.py all read this, rather than repeating the string. They used to
# repeat it, which would have left three files still building `deep-thin` after this line changed --
# the same hand-copied-constant failure that put 45.08M in these docs as "47.1M".
DEFAULT_VARIANT = "wide-shallow"

VOCAB_SIZES = (8192, 16384, 32768)

# The vocabulary the table above is fitted at. Fitting is vocabulary-dependent, because the embedding
# table is vocab x d_model and counts against the cap: a variant sized to 49.5M at 16k is *over* the
# 50M cap at 32k, by 4.5M to 10M depending on width. So the vocab ablation cannot reuse these
# ffn_hidden values. It has to refit every vocabulary to the same budget -- otherwise it compares a
# legal model against an illegal one and reports the difference as an effect of vocabulary.
FIT_VOCAB = 16384


def fit_ffn_hidden(shape: dict[str, int], vocab_size: int = FIT_VOCAB,
                   target: int = FIT_TARGET, multiple: int = 8) -> int:
    """Largest ffn_hidden keeping this shape under `target` parameters at this vocabulary.

    This is the function that produced the ffn_hidden values in VARIANTS, and `tests/test_variants.py`
    asserts they still agree with it -- so the table cannot quietly drift away from the rule that
    generated it, which is the failure that put 45.08M in the docs as "47.1M".

    KestrelConfig is imported lazily because `kestrel.model` reads DEFAULT_VARIANT from this module;
    a module-level import would be circular. `estimate_params()` is closed-form, so the search costs
    nothing and needs no torch device.
    """
    from kestrel.model import KestrelConfig

    fitted = 0
    for ffn in range(multiple, 1 << 15, multiple):
        cfg = KestrelConfig(vocab_size=vocab_size, **{**shape, "ffn_hidden": ffn})
        if cfg.estimate_params()["total"] > target:
            break
        fitted = ffn
    if fitted == 0:
        raise ValueError(
            f"no ffn_hidden fits {shape} at vocab {vocab_size:,} under {target:,} parameters; "
            f"the embedding table alone is {vocab_size * shape['d_model']:,}")
    return fitted


def fitted_variant(name: str, vocab_size: int = FIT_VOCAB,
                   target: int = FIT_TARGET) -> dict[str, int]:
    """A variant's shape with ffn_hidden refitted for `vocab_size`.

    Use this rather than `VARIANTS[name]` anywhere the vocabulary is not FIT_VOCAB -- the vocab
    ablation being the case that matters. At FIT_VOCAB it returns the table unchanged.
    """
    return {**VARIANTS[name], "ffn_hidden": fit_ffn_hidden(VARIANTS[name], vocab_size, target)}

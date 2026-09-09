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

One rule, two problems solved. The spread falls to 0.26M (0.53%), far below any effect the sweep
could resolve, and every variant now spends the budget the rules allow instead of leaving up to
4.92M of it unused -- so whichever shape wins is already sized for the main run rather than needing
to be rescaled afterwards, which would have invalidated the very measurement that chose it.

    variant          d_model x layers   heads Q/KV   ffn_hidden   params    embed
    very-deep        320 x 36           5 / 1        1016         49.23M    10.6%
    deep-thin        384 x 26           6 / 2        1096         49.37M    12.7%
    mid              512 x 14           8 / 2        1480         49.41M    17.0%
    wide-shallow     640 x  9           10 / 2       1744         49.48M    21.2%
    wide-shallow-7   640 x  7           10 / 2       2384         49.42M    21.2%
    extra-wide-6     768 x  6           12 / 2       2072         49.49M    25.4%
    widest-4         896 x  4           14 / 2       2552         49.47M    29.7%

All seven run head_dim 64. Re-derive the table at any time with `python scripts/count_params.py
--sweep`, which is also the script that enforces the cap.

The `embed` column is the second force acting on this axis and the reason the sweep cannot simply
run wider forever: the embedding table is `vocab x d_model`, so it grows linearly with width while
the layers it competes with shrink. At 320 wide the vocabulary costs a tenth of the model; at 896 it
costs nearly a third, and every parameter it takes is one not spent on computation.

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

    # The four shapes above improve monotonically in *both* val loss and throughput from 320x36 all
    # the way to 640x9. The trend never turns over, and the winner sits at the edge of the sweep
    # rather than inside it -- which establishes that the optimum is at 640x9 *or beyond*, not that
    # it is there. A monotone trend whose best cell is on the boundary of the search is the standard
    # sign of an under-swept axis, and "the best shape was the last one we tried" is not a claim an
    # architecture search should have to make. These three extend it.
    #
    # They also break a confound. Every step in the original sweep moved wider *and* shallower at
    # once -- 320x36, 384x26, 512x14, 640x9 -- so width and depth are perfectly inseparable in that
    # data, and "wide and shallow wins" cannot say which half is doing the work.
    #
    #   wide-shallow-7  holds d_model at 640 and moves only depth, 9 -> 7. The first orthogonal move
    #                   in the study, and the one that splits the confounded axis: if 640x7 matches
    #                   640x9 then depth stopped paying before 9 layers and the diagonal's gains
    #                   were width all along; if it is clearly worse, depth still buys something at
    #                   this width and 640x9 sits near a real tradeoff rather than on a plateau.
    #   extra-wide-6    continues the diagonal one more step. Worth running whichever way it comes
    #                   out: if it wins the search extends again, and if it loses then wide-shallow
    #                   becomes a *measured interior optimum* instead of a boundary artefact, which
    #                   is a strictly stronger claim than the one the four cells above support.
    #   widest-4        where the mechanism should visibly break, and the cell that is only worth
    #                   its GPU-hour if extra-wide-6 is still improving. Two forces bite at once
    #                   here: ffn_hidden has to balloon to absorb the parameters the missing layers
    #                   freed (2552 against wide-shallow's 1744), and the embedding reaches 29.7% of
    #                   the budget. Four layers is also very few sequential transforms for anything
    #                   compositional.
    "wide-shallow-7": dict(d_model=640, n_layers=7, n_heads=10, n_kv_heads=2, ffn_hidden=2384),
    "extra-wide-6": dict(d_model=768, n_layers=6, n_heads=12, n_kv_heads=2, ffn_hidden=2072),
    "widest-4": dict(d_model=896, n_layers=4, n_heads=14, n_kv_heads=2, ffn_hidden=2552),
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

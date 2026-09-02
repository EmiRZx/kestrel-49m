"""Tests for the parameter-matched variant table.

`kestrel/variants.py` asserts two things that nothing in the running code would notice being false.
The first is that the four shapes carry the same number of parameters -- the only reason the arch
sweep measures shape rather than capacity. The second is that the closed-form `estimate_params()`
used to fit them agrees with the modules `Kestrel` actually builds; if it did not, every number in
the table would be fiction and the cap check would be checking a model that does not exist.

Both were false once. The table called `very-deep` 47.1M when it was 45.08M, and that survived two
days because a hand-copied ffn_hidden in a second file has nothing keeping it honest. These tests
are what replaces "nobody had counted".

    python -m pytest tests/test_variants.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.model import Kestrel, KestrelConfig  # noqa: E402
from kestrel.variants import (  # noqa: E402
    DEFAULT_VARIANT,
    FIT_TARGET,
    FIT_VOCAB,
    PARAM_CAP,
    VARIANTS,
    VOCAB_SIZES,
    fit_ffn_hidden,
    fitted_variant,
)

NAMES = sorted(VARIANTS)


@pytest.mark.parametrize("name", NAMES)
def test_table_is_what_the_fitting_rule_produces(name: str) -> None:
    """The stored ffn_hidden is the one fit_ffn_hidden returns; no hand-edit survives this."""
    assert VARIANTS[name]["ffn_hidden"] == fit_ffn_hidden(VARIANTS[name], FIT_VOCAB)


@pytest.mark.parametrize("name", NAMES)
def test_fitted_shape_is_legal_and_spends_the_budget(name: str) -> None:
    total = KestrelConfig(vocab_size=FIT_VOCAB, **VARIANTS[name]).estimate_params()["total"]
    assert total <= FIT_TARGET <= PARAM_CAP
    # Fitting is only worth doing if it reaches the budget: one more step of ffn_hidden has to
    # cross the target, otherwise the search stopped early and params are being left unspent.
    wider = dict(VARIANTS[name], ffn_hidden=VARIANTS[name]["ffn_hidden"] + 8)
    assert KestrelConfig(vocab_size=FIT_VOCAB, **wider).estimate_params()["total"] > FIT_TARGET


def test_spread_across_shapes_is_negligible() -> None:
    """The premise of the arch sweep: shape varies, parameter count does not."""
    totals = [KestrelConfig(vocab_size=FIT_VOCAB, **kw).estimate_params()["total"]
              for kw in VARIANTS.values()]
    assert (max(totals) - min(totals)) / min(totals) < 0.01


@pytest.mark.parametrize("name", NAMES)
def test_estimate_matches_the_modules_actually_built(name: str) -> None:
    """Every assertion above trusts estimate_params(); this is the one that earns that trust."""
    cfg = KestrelConfig(vocab_size=FIT_VOCAB, **VARIANTS[name])
    with torch.device("meta"):
        model = Kestrel(cfg)
    assert model.param_report()["total"] == cfg.estimate_params()["total"]


@pytest.mark.parametrize("vocab", VOCAB_SIZES)
@pytest.mark.parametrize("name", NAMES)
def test_refitting_keeps_every_vocabulary_legal(name: str, vocab: int) -> None:
    """The bug this function exists for: at 32k the 16k-fitted table is 4.5M-10M over the cap."""
    shape = fitted_variant(name, vocab)
    assert shape["ffn_hidden"] % 8 == 0
    assert KestrelConfig(vocab_size=vocab, **shape).estimate_params()["total"] <= FIT_TARGET


@pytest.mark.parametrize("name", NAMES)
def test_a_bigger_vocabulary_is_paid_for_out_of_ffn_width(name: str) -> None:
    """What makes the vocab ablation fair: the lookup table is not free capacity."""
    widths = [fitted_variant(name, v)["ffn_hidden"] for v in sorted(VOCAB_SIZES)]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > widths[-1]


def test_config_defaults_come_from_the_table() -> None:
    """model.py has to derive its defaults; restating them by hand is the original bug."""
    cfg = KestrelConfig()
    assert cfg.vocab_size == FIT_VOCAB
    assert cfg.param_budget == PARAM_CAP
    for field, value in VARIANTS[DEFAULT_VARIANT].items():
        assert getattr(cfg, field) == value


@pytest.mark.parametrize("name", NAMES)
def test_head_dim_is_64_as_documented(name: str) -> None:
    assert KestrelConfig(**VARIANTS[name]).head_dim == 64


def test_fit_refuses_when_the_embedding_table_alone_blows_the_budget() -> None:
    """Silently returning ffn_hidden=8 would be far worse than failing here."""
    with pytest.raises(ValueError, match="no ffn_hidden fits"):
        fit_ffn_hidden(VARIANTS[DEFAULT_VARIANT], vocab_size=200_000)


@pytest.mark.parametrize("vocab", VOCAB_SIZES)
@pytest.mark.parametrize("name", NAMES)
def test_the_trainer_cannot_be_asked_to_train_an_illegal_model(name: str, vocab: int) -> None:
    """`TrainConfig.model_config()` is the single place a (variant, vocab) pair becomes a model, so
    it is the only place that has to remember to refit. This test is what keeps it remembering."""
    from kestrel.trainer import TrainConfig

    cfg = TrainConfig(variant=name, vocab_size=vocab).model_config()
    assert cfg.estimate_params()["total"] <= PARAM_CAP
    if vocab == FIT_VOCAB:
        assert cfg.ffn_hidden == VARIANTS[name]["ffn_hidden"]  # ordinary runs must not move

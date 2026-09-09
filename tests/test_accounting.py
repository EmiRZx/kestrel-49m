"""Tests for the two things the trainer measures itself with: step time and the eval slice.

Both were wrong in ways that nothing crashed on, and both fed the numbers the architecture sweep
ranked shapes by. That is the whole reason these are tests and not comments.

`charge_step` exists because the laptop suspended mid-step during the first ablation cell, and
`gpu_seconds += dt` charged the 51-minute nap to the training budget: the cell reported 2.52
GPU-hours for 1.69 hours of compute, and `--max-hours` -- then measured on raw wall clock -- spent
the difference and truncated the cell at 289M of its 300M tokens. Worse, the failure erases its own
evidence. Once the interrupted step finishes, wall clock and `gpu_hours` agree again, so a drift
check run afterwards reports everything clean.

`eval_blocks` exists because the slice was sized in batches, so it scaled with `micro_batch`:
`wide-shallow` at 16 scored 655k val tokens against `deep-thin`'s 328k, tabulated in the same
column. On the 0.3M-token wt103 split the request ran past the end, and `get_block`'s wraparound
turned it into a differently *weighted* average of the same 300 blocks for each variant.

    python -m pytest tests/test_accounting.py -q
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.trainer import (  # noqa: E402
    STALL_FACTOR,
    STALL_FLOOR_S,
    STALL_SAMPLES,
    RunState,
    charge_step,
    eval_blocks,
)

STEP = 5.3                      # deep-thin's measured step time, seconds
SUSPEND = 51.4 * 60             # the standby that started all of this


def steady(step: float = STEP, n: int = STALL_SAMPLES) -> deque[float]:
    return deque([step] * n, maxlen=64)


# ---- charge_step ---------------------------------------------------------------------------------


def test_the_suspend_that_caused_this_is_excluded() -> None:
    """The regression: 51.4 min inside one step must be charged as one step, not as 51.4 min."""
    compute, stall = charge_step(SUSPEND, steady())
    assert compute == pytest.approx(STEP)
    assert stall == pytest.approx(SUSPEND - STEP)


def test_nothing_is_clamped_before_there_is_a_median_to_clamp_against() -> None:
    """`torch.compile` makes the first step take a minute or two. Clamping *that* would under-count
    real work, so the guard is that no clamp happens until STALL_SAMPLES steps have been seen."""
    compute, stall = charge_step(90.0, deque([STEP] * (STALL_SAMPLES - 1), maxlen=64))
    assert (compute, stall) == (90.0, 0.0)


def test_the_compile_step_in_the_buffer_cannot_raise_the_reference() -> None:
    """Median, not mean: one 90-second sample among twenty 5.3-second ones must not lift the bar
    enough to let a suspend through."""
    buf = steady(n=STALL_SAMPLES - 1)
    buf.appendleft(90.0)
    compute, stall = charge_step(SUSPEND, buf)
    assert compute == pytest.approx(STEP)
    assert stall > 0


def test_an_ordinarily_slow_step_is_charged_in_full() -> None:
    """Steps repeat to within about 1%; a thermal dip or a shard-boundary read is still compute and
    must not be silently discarded. Anything under the floor is charged whatever the factor says."""
    for dt in (STEP * 1.5, STEP * 4, STEP + STALL_FLOOR_S - 1):
        compute, stall = charge_step(dt, steady())
        assert (compute, stall) == (dt, 0.0), dt


def test_the_floor_is_what_protects_a_fast_step() -> None:
    """On a fast step the factor alone would fire on a one-second hiccup: 8 x 0.3s is 2.4s. The
    additive floor is what keeps the threshold at least a minute wide regardless of step time."""
    fast = 0.3
    compute, stall = charge_step(fast * STALL_FACTOR + 1, steady(step=fast))
    assert stall == 0.0
    assert charge_step(fast + STALL_FLOOR_S + 1, steady(step=fast))[1] > 0


def test_a_slow_run_is_not_mistaken_for_a_stalled_one() -> None:
    """The threshold is relative, so a 40-second-per-step configuration (a big batch on a tired
    card) is not treated as one long suspend just for being slower than a 5-second one."""
    slow = 40.0
    compute, stall = charge_step(slow * 1.2, steady(step=slow))
    assert (compute, stall) == (slow * 1.2, 0.0)


# ---- eval_blocks --------------------------------------------------------------------------------

WT103_BLOCKS = 299              # what a 0.3M-token WikiText-103 test split comes to at seq_len 1024
VAL_BLOCKS = 3_906              # and a 4.0M-token val split
MICRO_BATCHES = (8, 12, 16)     # deep-thin, mid, wide-shallow


@pytest.mark.parametrize("n_blocks", [WT103_BLOCKS, VAL_BLOCKS])
def test_every_variant_scores_the_same_slice(n_blocks: int) -> None:
    """The premise of comparing two cells' val loss at all. Whatever micro-batch a shape needs to
    fit in 8 GB, the slice it is scored on has to be the same one.

    The guarantee is "the whole split bar at most one micro-batch", because whole batches are what
    keeps `torch.compile` on a single input shape. So the cross-variant spread is bounded by one
    micro-batch of blocks -- not zero, and the bound is worth stating rather than assuming.
    """
    scored = [eval_blocks(n_blocks, 1024, mb) for mb in MICRO_BATCHES]
    assert max(scored) - min(scored) < max(MICRO_BATCHES)
    assert min(scored) > n_blocks - max(MICRO_BATCHES)


def test_the_residual_mismatch_is_negligible_on_the_column_that_ranks() -> None:
    """Bounded is not the same as harmless, so bound it in the units that matter.

    val loss is what the sweep ranks on, and the val split is large enough that rounding to whole
    micro-batches moves the slice by 0.1% -- some four blocks in 3,900, worth on the order of 1e-4
    nats against a noise floor of ~0.004. On the 299-block wt103 split the same rounding is 2.7%,
    which is one more reason that column is reported and not ranked on.
    """
    val = [eval_blocks(VAL_BLOCKS, 1024, mb) for mb in MICRO_BATCHES]
    assert (max(val) - min(val)) / VAL_BLOCKS < 0.002

    wt = [eval_blocks(WT103_BLOCKS, 1024, mb) for mb in MICRO_BATCHES]
    assert 0.02 < (max(wt) - min(wt)) / WT103_BLOCKS < 0.03


@pytest.mark.parametrize("micro_batch", MICRO_BATCHES)
def test_the_slice_never_runs_off_the_end_of_the_split(micro_batch: int) -> None:
    """The wt103 bug. 40 batches of 16 is 640 blocks from a 299-block split, and `get_block` wraps
    modulo n_blocks rather than stopping, so the first 41 blocks were counted three times."""
    assert eval_blocks(WT103_BLOCKS, 1024, micro_batch) <= WT103_BLOCKS


@pytest.mark.parametrize("micro_batch", MICRO_BATCHES)
def test_whole_micro_batches_only(micro_batch: int) -> None:
    """A partial trailing batch is a second input shape, and a second shape makes `torch.compile`
    mark the batch dimension dynamic -- deoptimising the training kernel to pay for one eval tail."""
    assert eval_blocks(VAL_BLOCKS, 1024, micro_batch) % micro_batch == 0


@pytest.mark.parametrize("micro_batch", MICRO_BATCHES)
def test_a_token_budget_caps_the_slice(micro_batch: int) -> None:
    """`--eval-tokens` is for smoke runs that should not spend 20 seconds scoring 4M tokens."""
    got = eval_blocks(VAL_BLOCKS, 1024, micro_batch, budget_tokens=1e6)
    assert got <= 1e6 // 1024
    assert got > (1e6 // 1024) - micro_batch


def test_a_budget_larger_than_the_split_still_stops_at_the_split() -> None:
    assert eval_blocks(WT103_BLOCKS, 1024, 8, budget_tokens=10e6) <= WT103_BLOCKS


def test_a_split_smaller_than_one_batch_still_yields_a_batch() -> None:
    """Degenerate, but it must return something scoreable rather than zero batches and a NaN."""
    assert eval_blocks(5, 1024, 16) == 16


# ---- resume compatibility -----------------------------------------------------------------------


def test_a_checkpoint_from_before_stall_tracking_still_resumes() -> None:
    """Every checkpoint on disk right now predates `stall_seconds`. Refusing to load them, or
    crashing on the missing key, would cost more than the counter is worth."""
    old = {"step": 4_100, "tokens": 289_000_000, "gpu_seconds": 9_072.0, "energy_j": 7.3e5,
           "best_val": 3.4383, "history": []}
    state = RunState.from_dict(old)
    assert state.step == 4_100
    assert state.gpu_seconds == pytest.approx(9_072.0)
    assert state.stall_seconds == 0.0


def test_an_unknown_field_from_a_newer_build_is_dropped_rather_than_fatal() -> None:
    state = RunState.from_dict({"step": 7, "tokens": 1, "invented_later": 42})
    assert state.step == 7
    assert not hasattr(state, "invented_later")


def test_stall_seconds_survives_a_round_trip() -> None:
    """It has to persist: a resumed run's GPU-hours are only honest if the excluded time is too."""
    from dataclasses import asdict

    state = RunState(step=1, tokens=2, gpu_seconds=3.0, stall_seconds=SUSPEND)
    assert RunState.from_dict(asdict(state)).stall_seconds == pytest.approx(SUSPEND)

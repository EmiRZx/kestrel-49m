"""Train Kestrel.

    # main run, resumable, stops cleanly after 8 hours so the laptop is usable in the morning
    python scripts/train.py --name kestrel-main --total-tokens 6e9 --max-hours 8 --resume auto

    # ablation: same tokens, one axis changed
    python scripts/train.py --name abl-adamw --optimizer adamw --total-tokens 3e8
    python scripts/train.py --name abl-wide --variant wide-shallow --total-tokens 3e8

Every flag maps to one TrainConfig field, so the config saved next to the checkpoint is a complete
record of how the run was launched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.trainer import TrainConfig, Trainer  # noqa: E402
from kestrel.variants import VARIANTS  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # readable while backgrounded overnight
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "tokens_v16384")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--variant", default="deep-thin", choices=list(VARIANTS))
    ap.add_argument("--loss-chunk-size", type=int, default=2048)

    ap.add_argument("--total-tokens", type=float, default=6e9)
    ap.add_argument("--tokens-per-step", type=int, default=262_144)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--optimizer", default="muon+adamw", choices=("muon+adamw", "adamw"))
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adamw-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--schedule", default="wsd", choices=("wsd", "cosine"))
    ap.add_argument("--warmup-tokens", type=float, default=1.5e8)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-lr-frac", type=float, default=0.0)

    ap.add_argument("--name", default="kestrel")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly on a checkpoint after this much wall time (0 = no limit)")
    ap.add_argument("--eval-every-tokens", type=float, default=2e8)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--ckpt-every-tokens", type=float, default=4e8)
    ap.add_argument("--ckpt-every-minutes", type=float, default=30.0,
                    help="also checkpoint on this wall-clock cadence (0 = tokens only)")
    ap.add_argument("--log-every-steps", type=int, default=10)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", default="", help='checkpoint path, or "auto" for <run>/last.pt')
    args = ap.parse_args()

    cfg = TrainConfig(**{k: v for k, v in vars(args).items()})
    Trainer(cfg).run()


if __name__ == "__main__":
    main()

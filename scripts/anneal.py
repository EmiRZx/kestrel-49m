"""Re-anneal a finished checkpoint on a new data mixture, and measure whether it helped.

    python scripts/anneal.py --init runs/kestrel-main/best.pt --data data/anneal_v1 \
        --tokens 200e6 --name anneal-v1

An anneal is not a resume. A resume continues an interrupted run: same schedule, same data, same
token counter, and it must reproduce what the uninterrupted run would have done. This starts from
finished weights and deliberately does something *different* -- a fresh short schedule over a new
mixture -- so it takes the model and optimizer state but resets the clock.

Why this is the right shape for a 3-15 M token synthetic corpus. During low-LR decay the model makes
small, precise adjustments, so data seen then has influence out of proportion to its volume; that is
what makes a mixture this small worth anything at all against a 6 B-token pretraining corpus. The
flip side is that it is equally capable of *damaging* what is already there. So the result that
matters is not this run's final loss but a paired comparison against the starting checkpoint through
the identical eval path -- `scripts/paired_eval.py` against `runs/kestrel-main/best.pt`. An anneal
that reports only its after-number has not been measured, and the regression it hides is most likely
in ARC-Easy, the one benchmark Kestrel currently wins.

The LR ceiling matters more than anything else here. Re-entering at the pretraining peak would
scramble weights that took 2 GPU-hours to settle; entering far too low changes nothing at all. The
default `--peak-frac 0.1` restarts at a tenth of the original peak and decays to zero over the whole
anneal, which is the conventional starting point, not a tuned one. It is exposed as a flag because
it is the first thing worth sweeping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.trainer import TrainConfig, Trainer  # noqa: E402


def build_config(args: argparse.Namespace, base: dict) -> TrainConfig:
    """Start from the checkpoint's own config so architecture cannot drift, then override the
    schedule. Anything not named here is inherited, which is the point: an anneal that silently
    changed `vocab_size` or `seq_len` would load weights that no longer mean the same thing."""
    keep = {k: v for k, v in base.items()
            if k in TrainConfig.__dataclass_fields__ and k not in {
                "name", "out_dir", "data_dir", "total_tokens", "warmup_tokens", "decay_frac",
                "resume", "max_hours", "eval_every_tokens", "ckpt_every_tokens",
                "ckpt_every_minutes", "schedule", "min_lr_frac"}}
    cfg = TrainConfig(
        **keep,
        name=args.name,
        out_dir=str(args.out_dir),
        data_dir=str(args.data),
        total_tokens=float(args.tokens),
        # A short warmup, not none: stepping straight to peak LR on the first batch of a *new*
        # distribution is exactly when a finished model takes damage.
        warmup_tokens=float(args.tokens) * args.warmup_frac,
        schedule="wsd",
        decay_frac=args.decay_frac,
        min_lr_frac=0.0,
        eval_every_tokens=float(args.tokens) / max(args.evals, 1),
        ckpt_every_tokens=float(args.tokens) / 2,
        ckpt_every_minutes=0.0,
        max_hours=args.max_hours,
        resume="",
    )
    # Scale both optimizer peaks by the same factor, so their relative balance -- which was tuned
    # during pretraining -- is preserved rather than quietly re-tuned by the anneal.
    cfg.adamw_lr = float(base.get("adamw_lr", cfg.adamw_lr)) * args.peak_frac
    cfg.muon_lr = float(base.get("muon_lr", cfg.muon_lr)) * args.peak_frac
    if args.micro_batch:
        cfg.micro_batch = args.micro_batch
    return cfg


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="checkpoint to anneal FROM")
    ap.add_argument("--data", type=Path, required=True, help="tokenized anneal mixture dir")
    ap.add_argument("--name", default="anneal-v1")
    ap.add_argument("--tokens", type=float, default=200e6)
    ap.add_argument("--peak-frac", type=float, default=0.1,
                    help="restart LR as this fraction of the pretraining peak")
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--decay-frac", type=float, default=1.0,
                    help="1.0 = decay across the whole anneal, which is the usual choice")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--evals", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    base = ckpt.get("config", {})
    base = {k: (float(v) if isinstance(v, str) and v.replace(".", "", 1).replace("e-", "", 1)
                .replace("e+", "", 1).isdigit() else v) for k, v in base.items()}
    cfg = build_config(args, base)

    start_tokens = ckpt.get("state", {}).get("tokens", 0)
    print(f"annealing from : {args.init}  ({start_tokens/1e9:.3f}B tokens pretrained)")
    print(f"data           : {args.data}")
    print(f"budget         : {args.tokens/1e6:.0f}M tokens")
    print(f"lr             : adamw {cfg.adamw_lr:.3e}  muon {cfg.muon_lr:.3e}  "
          f"({args.peak_frac:g} x pretraining peak)")
    print(f"schedule       : {args.warmup_frac:.0%} warmup, decay over {args.decay_frac:.0%}\n")
    if args.dry_run:
        return

    run_dir = Path(args.out_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    # Take weights and optimizer state, leave the token counter and data cursor behind: this is a
    # new schedule over a new corpus, so inheriting the old run's position in the old shards would
    # be meaningless.
    #
    # Into `raw_model`, not `model`: `Trainer.__init__` has already wrapped the module in
    # `torch.compile`, whose state-dict keys carry an `_orig_mod.` prefix the checkpoint does not
    # have. `raw_model` is that same module before wrapping, so the keys match and the compiled
    # view sees the loaded weights.
    trainer.raw_model.load_state_dict(ckpt["model"])
    trainer.adamw.load_state_dict(ckpt["adamw"])
    if trainer.muon is not None and ckpt.get("muon"):
        trainer.muon.load_state_dict(ckpt["muon"])
    print(f"loaded weights + optimizer state from {args.init.name}\n")

    (run_dir / "anneal_meta.json").write_text(json.dumps({
        "init": str(args.init), "init_tokens": start_tokens, "data": str(args.data),
        "anneal_tokens": args.tokens, "peak_frac": args.peak_frac,
        "adamw_lr": cfg.adamw_lr, "muon_lr": cfg.muon_lr,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    trainer.run()


if __name__ == "__main__":
    main()

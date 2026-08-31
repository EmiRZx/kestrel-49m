"""Run the ablations, one axis at a time, and tabulate what changed.

Every design decision in this project is supposed to be a measurement rather than an opinion. This
driver holds the token budget fixed and varies exactly one thing:

    python scripts/ablate.py --suite arch          # deep-thin vs mid vs wide-shallow vs very-deep
    python scripts/ablate.py --suite optimizer     # Muon+AdamW vs AdamW alone
    python scripts/ablate.py --suite schedule      # WSD vs cosine
    python scripts/ablate.py --suite vocab         # 8k vs 16k vs 32k BPE
    python scripts/ablate.py --suite arch --tokens 1.5e8 --dry-run

Each run is a fresh subprocess: a CUDA OOM in one cell must not take the sweep down with it, and it
keeps `torch.compile`'s per-config caching from leaking across cells. Finished runs are skipped, so
an interrupted sweep can simply be relaunched.

The comparison that matters is not loss-per-token but **loss per GPU-hour**: a wider model that
processes 26% more tokens per second is allowed to be slightly worse per token and still win.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.variants import VARIANTS, VOCAB_SIZES  # noqa: E402

# Micro-batch per variant when the measured frontier is unavailable; scripts/bench.py --matrix
# writes runs/bench_frontier.jsonl and this driver prefers those measurements.
FALLBACK_BATCH = {"deep-thin": 12, "mid": 16, "wide-shallow": 16, "very-deep": 8}


def measured_batch(variant: str, frontier: Path) -> int:
    """Largest micro-batch that actually fit and ran fastest for this variant, from the bench."""
    best, batch = 0, FALLBACK_BATCH.get(variant, 8)
    if frontier.exists():
        for line in frontier.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("ok") and r.get("variant") == variant and r["tokens_per_s"] > best:
                best, batch = r["tokens_per_s"], r["batch"]
    return batch


def suite_specs(suite: str, data_root: Path) -> list[dict]:
    """One dict of train.py overrides per cell."""
    if suite == "arch":
        return [{"name": f"abl-arch-{v}", "variant": v} for v in VARIANTS]
    if suite == "optimizer":
        return [
            {"name": "abl-opt-muon", "optimizer": "muon+adamw"},
            # The baseline gets its own tuned LR: 3e-3 suits AdamW-on-embeddings alongside Muon, but
            # would diverge if applied to every hidden matrix. Comparing against a deliberately
            # badly-tuned baseline would prove nothing.
            {"name": "abl-opt-adamw", "optimizer": "adamw", "adamw-lr": 1e-3},
        ]
    if suite == "schedule":
        return [{"name": "abl-sched-wsd", "schedule": "wsd"},
                {"name": "abl-sched-cosine", "schedule": "cosine", "min-lr-frac": 0.1}]
    if suite == "vocab":
        # Needs a tokenized corpus per vocab size; see scripts/prepare_data.py --vocab-size.
        return [{"name": f"abl-vocab-{v}", "vocab-size": v,
                 "data-dir": str(data_root / f"tokens_v{v}")} for v in VOCAB_SIZES]
    raise SystemExit(f"unknown suite {suite!r}")


def run_cell(spec: dict, common: dict, out_dir: Path, dry: bool) -> dict:
    """Train one cell in its own process, then read back its metrics."""
    run_dir = out_dir / spec["name"]
    metrics = run_dir / "metrics.jsonl"
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py")]
    for k, v in {**common, **spec}.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    cmd += ["--resume", "auto"]      # relaunching a killed sweep continues where it stopped

    if dry:
        print("  " + " ".join(cmd[1:]))
        return {"name": spec["name"], "cmd": cmd}

    print(f"\n=== {spec['name']} " + "=" * (60 - len(spec['name'])), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    row = {"name": spec["name"], "wall_minutes": round((time.perf_counter() - t0) / 60, 1),
           "returncode": proc.returncode, **{k: v for k, v in spec.items() if k != "name"}}
    if not metrics.exists():
        row["error"] = "no metrics written"
        return row
    evals, steps = [], []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        (evals if rec.get("eval") else steps).append(rec)
    if evals:
        last = evals[-1]
        row |= {"val_loss": last["val_loss"], "val_ppl": last["val_ppl"],
                "wt103_ppl": last["wt103_ppl"], "gpu_hours": last["gpu_hours"],
                "tokens": last["tokens"]}
    if steps:
        tail = steps[-min(20, len(steps)):]
        row["tokens_per_s"] = round(sum(s["tokens_per_s"] for s in tail) / len(tail))
        row["mfu"] = round(sum(s["mfu"] for s in tail) / len(tail), 4)
        row["peak_gb"] = max(s.get("peak_gb", 0) for s in steps)
    return row


def show(rows: list[dict]) -> None:
    print(f"\n{'run':<22}{'val loss':>10}{'val ppl':>10}{'wt103 ppl':>11}"
          f"{'tok/s':>9}{'MFU':>7}{'GB':>6}{'GPU-h':>8}{'loss/GPU-h':>12}")
    print("-" * 95)
    for r in rows:
        if "val_loss" not in r:
            print(f"{r['name']:<22}{r.get('error', 'failed'):>10}")
            continue
        # loss per GPU-hour is the ranking column: cheaper compute is allowed to buy a worse loss.
        eff = r["val_loss"] / max(r.get("gpu_hours", 0) or 1e-9, 1e-9)
        print(f"{r['name']:<22}{r['val_loss']:>10.4f}{r['val_ppl']:>10.2f}{r['wt103_ppl']:>11.2f}"
              f"{r.get('tokens_per_s', 0):>9,}{r.get('mfu', 0):>7.1%}{r.get('peak_gb', 0):>6.2f}"
              f"{r.get('gpu_hours', 0):>8.2f}{eff:>12.3f}")
    ok = [r for r in rows if "val_loss" in r]
    if ok:
        best = min(ok, key=lambda r: r["val_loss"])
        print(f"\nbest val loss: {best['name']} at {best['val_loss']:.4f} "
              f"({best.get('gpu_hours', 0):.2f} GPU-hours)")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=("arch", "optimizer", "schedule", "vocab"))
    ap.add_argument("--tokens", type=float, default=3e8,
                    help="token budget per cell; 300M ranks architectures reliably and cheaply")
    ap.add_argument("--variant", default="deep-thin", help="held fixed in non-arch suites")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "tokens_v16384")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--max-hours", type=float, default=0.0, help="per cell")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = suite_specs(args.suite, args.data_dir.parent)
    frontier = args.out_dir / "bench_frontier.jsonl"
    common = {
        "total-tokens": args.tokens,
        # Warmup and decay stay proportional to the budget, so a 300M-token cell has the same shape
        # of schedule as the 6B main run and the comparison stays fair.
        "warmup-tokens": args.tokens * 0.025,
        "eval-every-tokens": args.tokens / 6,
        "ckpt-every-tokens": args.tokens / 2,
        "out-dir": str(args.out_dir),
        "log-every-steps": 20,
    }
    if args.max_hours:
        common["max-hours"] = args.max_hours
    if args.suite != "vocab":
        common["data-dir"] = str(args.data_dir)
    if args.suite != "arch":
        common["variant"] = args.variant

    for spec in specs:
        variant = spec.get("variant", args.variant)
        spec.setdefault("micro-batch", measured_batch(variant, frontier))

    print(f"suite {args.suite}: {len(specs)} cells x {args.tokens/1e6:.0f}M tokens")
    rows = [run_cell(spec, common, args.out_dir, args.dry_run) for spec in specs]
    if args.dry_run:
        return

    show(rows)
    out = args.out_dir / f"ablation_{args.suite}.json"
    out.write_text(json.dumps({"suite": args.suite, "tokens_per_cell": args.tokens,
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()



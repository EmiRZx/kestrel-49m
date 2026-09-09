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

Ranking is on two framings, not one column. `loss / GPU-hours` reads like an efficiency score and is
not one: both quantities are better when smaller, so the ratio has no consistent direction -- a cell
can improve it by getting worse, as long as it also gets slower. What is well defined is holding one
axis fixed and reading the other: **GPU-hours to reach a fixed loss** (iso-loss), and **loss reached
at a fixed number of GPU-hours** (iso-compute). Under a wall-clock budget the second is the decision,
which is why a wider shape that processes 26% more tokens per second is allowed to be slightly worse
per token and still win.
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

from kestrel.variants import DEFAULT_VARIANT, VARIANTS, VOCAB_SIZES  # noqa: E402

# Micro-batch per variant when the measured frontier is unavailable; scripts/bench.py --matrix
# writes runs/bench_frontier.jsonl and this driver prefers those measurements. These fallbacks are
# the ones with room to spare, not the fastest ones, for the reason MARGIN_GB explains below.
FALLBACK_BATCH = {"deep-thin": 8, "mid": 12, "wide-shallow": 16, "very-deep": 8}

# VRAM the chosen cell must leave unused. The fastest configuration is not the one to run unattended
# for six hours: adjacent batch sizes differ by 1-2.5% in throughput but 1-2 GB in peak memory, so
# buying a wide margin is nearly free. Measured on this box, `deep-thin` at batch 12 peaks at 7.09 GB
# against a 7.20 GB cap -- 110 MB, or 1.5% -- while batch 8 costs 2.3% throughput and peaks at
# 4.89 GB. A sweep that OOMs three hours in and has to be relaunched loses far more than 2.3%.
MARGIN_GB = 0.5


def vram_cap_gb() -> float:
    """The allocation ceiling the trainer will actually enforce, in GB.

    Imported from the trainer rather than restated: the bench, the trainer and this driver have to
    agree about how much memory exists, or the frontier stops predicting what a cell can fit.
    """
    import torch

    from kestrel.trainer import MEMORY_FRACTION

    return MEMORY_FRACTION * torch.cuda.get_device_properties(0).total_memory / 1024**3


def measured_batch(variant: str, frontier: Path, margin_gb: float = MARGIN_GB) -> int:
    """Fastest micro-batch for this variant that still leaves `margin_gb` of VRAM unused.

    Ranking on throughput alone picks the cell sitting closest to the OOM cliff, which is a poor
    trade for an unattended sweep -- see MARGIN_GB. If nothing clears the margin, the row with the
    most headroom wins instead, so a tight card degrades to "slow but running" rather than to
    "fast and dead".
    """
    rows = []
    if frontier.exists():
        for line in frontier.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("ok") and r.get("variant") == variant and r.get("peak_gb"):
                rows.append(r)
    if not rows:
        return FALLBACK_BATCH.get(variant, 8)

    ceiling = vram_cap_gb() - margin_gb
    roomy = [r for r in rows if r["peak_gb"] <= ceiling]
    if roomy:
        return max(roomy, key=lambda r: r["tokens_per_s"])["batch"]
    return min(rows, key=lambda r: r["peak_gb"])["batch"]


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
        # None whenever the trainer could not identify the chip's bf16 peak, and the mean of a column
        # that is partly None is not a number: drop those rows rather than average around them.
        mfus = [s["mfu"] for s in tail if s.get("mfu") is not None]
        row["mfu"] = round(sum(mfus) / len(mfus), 4) if mfus else None
        row["peak_gb"] = max(s.get("peak_gb", 0) for s in steps)
    return row


def is_short(row: dict, budget: float) -> bool:
    """Did this cell stop before spending its token budget?

    `--max-hours` stops a cell cleanly on a checkpoint and writes a final eval, so a truncated cell
    exits 0 and produces a perfectly well-formed row -- just at fewer tokens than its neighbours.
    Ranking that against complete cells would compare shapes at different budgets, which is the one
    thing this sweep exists to avoid. 1% of slack, because the final step overshoots the budget by up
    to one step's worth of tokens.
    """
    return bool(budget) and 0 < row.get("tokens", 0) < 0.99 * budget


def curve(run_dir: Path) -> list[tuple[float, float]]:
    """(GPU-hours, val loss) at every eval of one run, in the order they were written."""
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return []
    pts = []
    for line in m.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("eval") and rec.get("gpu_hours") and "val_loss" in rec:
            pts.append((rec["gpu_hours"], rec["val_loss"]))
    return pts


def loss_at(pts: list[tuple[float, float]], hours: float) -> float | None:
    """Val loss a cell had reached after `hours` GPU-hours, interpolated between its evals.

    None outside the measured range. Extrapolating past a curve's last point would invent the one
    number the comparison turns on, and a loss curve is exactly the shape that punishes a straight
    line drawn past its end.
    """
    if len(pts) < 2 or not pts[0][0] <= hours <= pts[-1][0]:
        return None
    for (h0, l0), (h1, l1) in zip(pts, pts[1:]):
        if h0 <= hours <= h1:
            return l0 + (l1 - l0) * (hours - h0) / max(h1 - h0, 1e-9)
    return None


def hours_to(pts: list[tuple[float, float]], target: float) -> float | None:
    """GPU-hours a cell needed to reach `target` val loss, interpolated in loss between evals."""
    if len(pts) < 2 or min(l for _, l in pts) > target:
        return None
    if pts[0][1] <= target:
        return pts[0][0]                 # already past it at its first eval, so this is a bound
    for (h0, l0), (h1, l1) in zip(pts, pts[1:]):
        if l0 >= target >= l1:
            return h0 + (h1 - h0) * (l0 - target) / max(l0 - l1, 1e-9)
    return None


def show_iso(rows: list[dict], out_dir: Path) -> None:
    """The two framings that are defined, printed side by side: iso-compute and iso-loss.

    Both need the eval curves rather than the final row alone, so they are read back from each cell's
    metrics.jsonl. That also means an old sweep's table can be re-tabulated without rerunning it.
    """
    curves = {r["name"]: curve(out_dir / r["name"]) for r in rows}
    curves = {n: c for n, c in curves.items() if len(c) >= 2}
    if len(curves) < 2:
        return

    # Iso-compute: the cheapest cell's total cost, and what every other shape had reached by then.
    # Under a fixed wall-clock budget this is the decision -- it is the question the budget asks.
    ref_h = min(c[-1][0] for c in curves.values())
    at = {n: v for n, c in curves.items() if (v := loss_at(c, ref_h)) is not None}
    if len(at) >= 2:
        print(f"\niso-compute -- val loss at {ref_h:.2f} GPU-hours, all the cheapest cell spent:")
        best = min(at.values())
        for n, v in sorted(at.items(), key=lambda kv: kv[1]):
            print(f"  {n:<24}{v:>9.4f}" + ("" if v == best else f"{v - best:>+10.4f}"))

    # Iso-loss: the easiest loss every cell got to, and which shape bought it for the least compute.
    ref_l = max(min(l for _, l in c) for c in curves.values())
    took = {n: v for n, c in curves.items() if (v := hours_to(c, ref_l)) is not None}
    if len(took) >= 2:
        print(f"\niso-loss -- GPU-hours to val {ref_l:.4f}, the easiest loss every cell reached:")
        best = min(took.values())
        for n, v in sorted(took.items(), key=lambda kv: kv[1]):
            print(f"  {n:<24}{v:>9.2f}   {v / max(best, 1e-9):.2f}x")


def show(rows: list[dict], budget: float = 0.0, out_dir: Path | None = None) -> None:
    print(f"\n{'run':<22}{'val loss':>10}{'val ppl':>10}{'wt103 ppl':>11}"
          f"{'tok/s':>9}{'MFU':>7}{'GB':>6}{'GPU-h':>8}{'Mtok/GPU-h':>12}")
    print("-" * 95)
    for r in rows:
        if "val_loss" not in r:
            print(f"{r['name']:<22}{r.get('error', 'failed'):>10}")
            continue
        # Tokens per GPU-hour, not loss per GPU-hour: a rate with one direction, where more is
        # better. Measured end to end, so it charges each cell for its own evals and checkpoints.
        rate = r.get("tokens", 0) / max(r.get("gpu_hours", 0) or 1e-9, 1e-9) / 1e6
        label = r["name"] + (" *" if is_short(r, budget) else "")
        mfu = f"{r['mfu']:.1%}" if r.get("mfu") is not None else "-"
        print(f"{label:<22}{r['val_loss']:>10.4f}{r['val_ppl']:>10.2f}{r['wt103_ppl']:>11.2f}"
              f"{r.get('tokens_per_s', 0):>9,}{mfu:>7}{r.get('peak_gb', 0):>6.2f}"
              f"{r.get('gpu_hours', 0):>8.2f}{rate:>11.0f}M")
    short = [r for r in rows if "val_loss" in r and is_short(r, budget)]
    for r in short:
        print(f"\n* {r['name']} stopped at {r['tokens']/1e6:.0f}M of {budget/1e6:.0f}M tokens "
              f"({r['tokens']/budget:.0%}) -- not comparable. Relaunch to continue it; "
              f"--resume auto picks it up where it stopped.")
    ok = [r for r in rows if "val_loss" in r and not is_short(r, budget)]
    if ok:
        best = min(ok, key=lambda r: r["val_loss"])
        print(f"\nbest val loss: {best['name']} at {best['val_loss']:.4f} "
              f"({best.get('gpu_hours', 0):.2f} GPU-hours)")
        if out_dir:
            show_iso(ok, out_dir)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=("arch", "optimizer", "schedule", "vocab"))
    ap.add_argument("--tokens", type=float, default=3e8,
                    help="token budget per cell; 300M ranks architectures reliably and cheaply")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="held fixed in non-arch suites")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "tokens_v16384")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="cap on training time per cell, not for the sweep, and not wall clock -- "
                         "time the machine spends suspended is excluded. A cell that hits the cap "
                         "stops cleanly on a checkpoint and is flagged as incomplete in the table")
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

    show(rows, args.tokens, args.out_dir)
    out = args.out_dir / f"ablation_{args.suite}.json"
    out.write_text(json.dumps({"suite": args.suite, "tokens_per_cell": args.tokens,
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()



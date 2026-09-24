"""Score every anneal variant against the pre-anneal checkpoint, and rank them honestly.

    python scripts/anneal_sweep.py                          # every runs/anneal-*/best.pt
    python scripts/anneal_sweep.py --only anneal-v2-dose

An anneal sweep is the place where it is easiest to fool yourself. Four variants times four
benchmarks is sixteen numbers, and with sixteen draws from item noise something will look like a
win. So this reports, for every variant, the paired gap against the *same* baseline checkpoint with
its confidence interval -- and the summary line counts only the gaps whose interval excludes zero.

The baseline is `runs/kestrel-main/best.pt`, the submitted model, not the best variant so far.
Ranking variants against each other would compound the same selection problem one level up.

`--fdr` applies a Benjamini-Hochberg correction across every comparison in the sweep. With sixteen
tests at p<0.05 the expected number of false positives is about one, which is exactly the size of
effect being looked for here, so an uncorrected "significant" result in a sweep this wide is not
worth much on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paired_eval import TASKS, paired_bootstrap, score_ckpt  # noqa: E402


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Which hypotheses survive at a false-discovery rate of `alpha`.

    Sorts the p-values and finds the largest k with p_(k) <= k/m * alpha; everything at or below
    that rank is kept. Controls the expected *proportion* of false positives among the rejections,
    which is the right frame for a sweep whose question is "did any of these do anything".
    """
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    keep = np.zeros(m, dtype=bool)
    thresh = -1
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * alpha:
            thresh = rank
    if thresh > 0:
        keep[order[:thresh]] = True
    return keep.tolist()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / "runs" / "kestrel-main" / "best.pt")
    ap.add_argument("--glob", default="anneal-*", help="run directories under runs/ to score")
    ap.add_argument("--only", default=None)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "anneal_sweep.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    variants = sorted(p for p in (ROOT / "runs").glob(args.glob)
                      if (p / "best.pt").exists() and (not args.only or args.only in p.name))
    if not variants:
        sys.exit(f"no variants matching {args.glob!r} with a best.pt under runs/")
    tasks = args.tasks.split(",")
    print(f"baseline : {args.baseline}")
    print(f"variants : {', '.join(v.name for v in variants)}\n")

    # Score the baseline once per task and reuse it, rather than re-running it for every variant:
    # identical inputs, identical fp32 path, so the hit vector cannot differ, and it is the single
    # most expensive thing here.
    base = {t: score_ckpt(args.baseline, t, args.device) for t in tasks}

    rows, pvals = [], []
    for v in variants:
        for t in tasks:
            b = score_ckpt(v / "best.pt", t, args.device)
            a = base[t]
            if len(a) != len(b):
                print(f"  skip {v.name}/{t}: {len(a)} vs {len(b)} items")
                continue
            gap, lo, hi, p = paired_bootstrap(a, b)
            rows.append({"variant": v.name, "task": t, "n": int(len(a)),
                         "baseline": float(a.mean()), "variant_score": float(b.mean()),
                         "gap": gap, "ci95": [lo, hi], "p": p,
                         "resolved": bool(lo > 0 or hi < 0)})
            pvals.append(p)
            print(f"  {v.name:<18}{t:<12}{a.mean():>8.4f} -> {b.mean():>8.4f}  "
                  f"{gap:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}", flush=True)

    survives = benjamini_hochberg(pvals, args.alpha)
    for r, s in zip(rows, survives):
        r["survives_fdr"] = bool(s)

    print(f"\n{'variant':<18}{'task':<12}{'gap':>9}{'p':>9}  verdict")
    print("-" * 62)
    for r in sorted(rows, key=lambda x: x["p"]):
        verdict = ("SURVIVES FDR" if r["survives_fdr"]
                   else "resolved, fails FDR" if r["resolved"] else "not resolved")
        print(f"{r['variant']:<18}{r['task']:<12}{r['gap']:>+9.4f}{r['p']:>9.4f}  {verdict}")

    n_res = sum(r["resolved"] for r in rows)
    n_fdr = sum(r["survives_fdr"] for r in rows)
    print(f"\n{len(rows)} comparisons: {n_res} with an interval excluding zero, "
          f"{n_fdr} surviving Benjamini-Hochberg at alpha={args.alpha}.")
    if n_fdr == 0:
        print("No variant moved any benchmark by more than item noise.")

    args.out.write_text(json.dumps(
        {"baseline": str(args.baseline), "alpha": args.alpha, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

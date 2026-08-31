"""Print the parameter count of a Kestrel config and verify it against the 50M competition cap.

This script satisfies the Track 01 requirement: "Your repository must include a script or printed
output showing the parameter count, plus your model config."

    python scripts/count_params.py                     # the default config
    python scripts/count_params.py --sweep             # candidate configs at fixed budget
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.model import Kestrel, KestrelConfig  # noqa: E402


def show(cfg: KestrelConfig, build: bool = True) -> dict:
    print("=" * 72)
    print("CONFIG")
    for k, v in asdict(cfg).items():
        print(f"  {k:<16} {v}")
    print(f"  {'head_dim':<16} {cfg.head_dim}")

    est = cfg.estimate_params()
    print("\nCLOSED-FORM ESTIMATE")
    for k in ("per_layer", "layers", "embedding", "non_embedding", "total"):
        print(f"  {k:<16} {est[k]:>12,}")

    report = est
    if build:
        import torch

        with torch.device("meta"):
            model = Kestrel(cfg)
        report = model.param_report()
        print("\nMEASURED (torch, tied weights counted once)")
        for k in ("embedding", "non_embedding", "total"):
            print(f"  {k:<16} {report[k]:>12,}")
        if report["total"] != est["total"]:
            print(f"  !! estimate/measured mismatch: {est['total']:,} vs {report['total']:,}")
        print(f"\n  fwd+bwd FLOPs/token  {model.flops_per_token():>12,}")

    frac = report["total"] / cfg.param_budget
    verdict = "WITHIN BUDGET" if report["total"] <= cfg.param_budget else "OVER BUDGET"
    print(f"\n  budget            {cfg.param_budget:>12,}")
    print(f"  headroom          {cfg.param_budget - report['total']:>12,}")
    print(f"  utilisation       {frac:>12.2%}   -> {verdict}")
    emb_frac = report["embedding"] / report["total"]
    print(f"  embedding share   {emb_frac:>12.2%}   (lower is better: params in layers, not lookup)")
    print("=" * 72)
    return report


SWEEP = [
    ("deep-thin  (default)", dict(d_model=384, n_layers=26, n_heads=6, n_kv_heads=2, ffn_hidden=1024)),
    ("mid        ", dict(d_model=512, n_layers=14, n_heads=8, n_kv_heads=2, ffn_hidden=1376)),
    ("wide-shallow", dict(d_model=640, n_layers=9, n_heads=10, n_kv_heads=2, ffn_hidden=1728)),
    ("very deep  ", dict(d_model=320, n_layers=36, n_heads=5, n_kv_heads=1, ffn_hidden=896)),
]


def sweep(vocab_sizes=(8192, 16384, 32768)) -> None:
    print(f"{'variant':<22}{'vocab':>8}{'total':>13}{'non-emb':>13}{'emb %':>8}{'fits':>7}")
    print("-" * 71)
    for name, kw in SWEEP:
        for v in vocab_sizes:
            cfg = KestrelConfig(vocab_size=v, **kw)
            e = cfg.estimate_params()
            fits = "yes" if e["total"] <= cfg.param_budget else "NO"
            emb = e["embedding"] / e["total"]
            print(f"{name:<22}{v:>8}{e['total']:>13,}{e['non_embedding']:>13,}{emb:>7.1%}{fits:>7}")
        print()
    print("Read: a bigger vocab spends the fixed budget on a lookup table instead of on layers.")
    print("The 8k column has the most compute in its layers but needs more tokens per document.")
    print("Which trade wins is an empirical question -> ablation, not a guess.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="compare candidate configs")
    ap.add_argument("--no-build", action="store_true", help="closed-form only, do not import torch")
    ap.add_argument("--json", type=Path, help="write the measured report to this path")
    for f in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_hidden"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=int)
    args = ap.parse_args()

    if args.sweep:
        sweep()
        return

    overrides = {
        f: getattr(args, f)
        for f in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_hidden")
        if getattr(args, f) is not None
    }
    report = show(KestrelConfig(**overrides), build=not args.no_build)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    sys.exit(0 if report.get("within_budget", report["total"] <= 50_000_000) else 1)


if __name__ == "__main__":
    main()

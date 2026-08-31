"""Measure real training throughput, so the compute budget rests on numbers rather than guesses.

Each measurement runs in its own subprocess: a CUDA OOM leaves the context unusable, so a sweep
that shares one process cannot survive its own failures.

    python scripts/bench.py --matrix           # variants x batch sizes
    python scripts/bench.py --probe            # kernel/memory options at a fixed config
    python scripts/bench.py --single --variant mid --batch 8

Training Efficiency is a scored criterion in this competition, so these numbers go in the README.
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

from kestrel.variants import VARIANTS  # noqa: E402

# Approximate bf16 dense tensor-core peak for an RTX 4070 Laptop at its 140 W cap. Vendor figures
# for the desktop part are higher; treat MFU as indicative and tokens/s as the ground truth.
ASSUMED_PEAK_TFLOPS = 80.0

# Refuse to allocate beyond this fraction of VRAM. Without it, Windows' WDDM driver silently spills
# to system RAM instead of raising, and throughput collapses by ~30x while appearing to "work".
MEMORY_FRACTION = 0.90


def measure(args) -> dict:
    import torch

    from kestrel.model import Kestrel, KestrelConfig

    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)

    cfg = KestrelConfig(
        max_seq_len=args.seq,
        vocab_size=args.vocab,
        loss_chunk_size=args.loss_chunk,
        sdpa_enable_gqa=(args.gqa == "sdpa"),
        **VARIANTS[args.variant],
    )
    model = Kestrel(cfg).to("cuda")
    flops_per_token = model.flops_per_token()
    n_params = model.param_report()["total"]
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)

    x = torch.randint(0, cfg.vocab_size, (args.batch, cfg.max_seq_len), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (args.batch, cfg.max_seq_len), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    t0 = 0.0
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tps = args.steps * args.batch * cfg.max_seq_len / dt
    tflops = tps * flops_per_token / 1e12
    return {
        "ok": True,
        "variant": args.variant,
        "params_m": round(n_params / 1e6, 2),
        "batch": args.batch,
        "seq": cfg.max_seq_len,
        "compile": args.compile,
        "gqa": args.gqa,
        "loss_chunk": args.loss_chunk,
        "tokens_per_s": round(tps),
        "step_ms": round(dt / args.steps * 1000, 1),
        "tflops": round(tflops, 2),
        "mfu": round(tflops / ASSUMED_PEAK_TFLOPS, 4),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "loss": round(loss.item(), 3),
    }


def run_child(**overrides) -> dict:
    """Run one measurement in a fresh process and parse its JSON result line."""
    cmd = [sys.executable, str(Path(__file__)), "--single"]
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    reason = "OOM" if any("out of memory" in l.lower() for l in tail) else "FAIL"
    return {"ok": False, "reason": reason, **overrides}


HEADER = f"{'config':<34}{'batch':>6}{'tok/s':>10}{'step ms':>9}{'TFLOP/s':>9}{'MFU':>7}{'GB':>7}"


def show(row: dict, label: str) -> None:
    if not row.get("ok"):
        print(f"{label:<34}{row.get('batch', ''):>6}{row.get('reason', 'FAIL'):>10}")
        return
    print(f"{label:<34}{row['batch']:>6}{row['tokens_per_s']:>10,}{row['step_ms']:>9.1f}"
          f"{row['tflops']:>9.2f}{row['mfu']:>7.1%}{row['peak_gb']:>7.2f}")


def matrix(args) -> list[dict]:
    print(f"\nVARIANT x BATCH  (seq {args.seq}, vocab {args.vocab}, compile={args.compile})")
    print(HEADER)
    print("-" * 82)
    batches = [int(b) for b in args.batches.split(",")]
    rows = []
    for variant in VARIANTS:
        for batch in batches:
            r = run_child(variant=variant, batch=batch, seq=args.seq, vocab=args.vocab,
                          steps=args.steps, compile=args.compile)
            rows.append(r)
            show(r, variant)
            if not r.get("ok"):
                break
        print()
    return rows


def probe(args) -> list[dict]:
    """Isolate the effect of each memory/kernel decision at one fixed config."""
    print(f"\nOPTION PROBE  (variant {args.variant}, batch {args.batch}, seq {args.seq})")
    print(HEADER)
    print("-" * 82)
    options = [
        ("baseline: repeat_kv + chunked loss", dict(gqa="repeat", loss_chunk=2048)),
        ("sdpa enable_gqa=True", dict(gqa="sdpa", loss_chunk=2048)),
        ("no loss chunking", dict(gqa="repeat", loss_chunk=0)),
        ("loss chunk 1024", dict(gqa="repeat", loss_chunk=1024)),
        ("baseline + torch.compile", dict(gqa="repeat", loss_chunk=2048, compile=True)),
    ]
    rows = []
    for label, opts in options:
        r = run_child(variant=args.variant, batch=args.batch, seq=args.seq, vocab=args.vocab,
                      steps=args.steps, **opts)
        rows.append(r)
        show(r, label)
    return rows


def summarise(rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return
    best = max(ok, key=lambda r: r["tokens_per_s"])
    tps = best["tokens_per_s"]
    print(f"\nfastest: {best['variant']} batch {best['batch']} seq {best['seq']} "
          f"compile={best['compile']} -> {tps:,} tok/s ({best['peak_gb']:.2f} GB)")
    for target in (1e9, 3e9, 6e9, 10e9):
        print(f"  {target/1e9:>4.0f}B tokens -> {target/tps/3600:>6.1f} GPU-hours")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # so a backgrounded sweep is readable while it runs
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--matrix", action="store_true", help="sweep variants x batch sizes")
    mode.add_argument("--probe", action="store_true", help="compare memory/kernel options")
    mode.add_argument("--single", action="store_true", help="one measurement, JSON to stdout")
    ap.add_argument("--variant", default="deep-thin", choices=list(VARIANTS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", default="8,16,24,32", help="matrix mode: comma-separated")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--gqa", default="repeat", choices=("repeat", "sdpa"))
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=Path, help="append results as JSON lines")
    args = ap.parse_args()

    if args.single:
        print(json.dumps(measure(args)))
        return

    rows = probe(args) if args.probe else matrix(args)
    summarise(rows)
    if args.out:
        with args.out.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

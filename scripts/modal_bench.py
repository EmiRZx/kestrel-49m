"""Measure Kestrel throughput on rented GPUs, so the Modal spend rests on measurements not guesses.

Every throughput number in this project comes from one RTX 4070 Laptop at 140 W: ~66k tok/s for
wide-shallow at batch 16, which put the 6B-token main run at ~25 GPU-hours. Every projection for
rented hardware so far has been an extrapolation from that single point, and it is weak in a
specific way -- a 49M model at seq 1024 is bound more by memory bandwidth and kernel-launch
overhead than by peak FLOPS, but hourly price tracks FLOPS. H100 costs 1.9x an A100-40GB and may
well not be 1.9x faster here, so guessing wrong wastes a real fraction of a $30 credit.

    modal run scripts/modal_bench.py --smoke     # ~$0.02, validates image, mounts, CUDA
    modal run scripts/modal_bench.py             # ~$1.50, the real sweep
    modal run scripts/modal_bench.py --compiled --gpu "H100!" --batches 64,128,256

Rows land in runs/bench_modal.jsonl, deliberately NOT runs/bench_frontier.jsonl: ablate.py's
measured_batch() reads the latter to choose a micro-batch for THIS laptop, and a batch that fits in
40 GB would OOM instantly in 8 GB.

trainer.py fixes tokens_per_step at 262,144 and derives grad_accum() from it, so a larger
micro-batch on rented hardware only lowers the accumulation count. No ablation result moves, which
is what makes these measurements a pure hardware comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

# Hourly on-demand prices as quoted from Modal's pricing page on 2026-09-05. Kept here so the
# dollars-per-billion-tokens column uses the same numbers the $28 workspace budget was set against.
PRICE_PER_HOUR = {
    "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
    "H100": 3.95, "H100!": 3.95, "H200": 4.54, "B200": 6.25, "B300": 7.10,
}

# Vendor dense bf16 peaks live in kestrel/trainer.py (DENSE_BF16_TFLOPS), which is where the trainer
# reads them to compute MFU. Imported inside sweep() rather than copied here: two tables of the same
# vendor figures drift, and a wrong peak is exactly the bug that made a Modal run log "mfu 364.3%".

# H100! suppresses Modal's free upgrade to H200: an unpinned request would benchmark a chip we
# cannot count on getting. T4 is absent on purpose -- Turing has no bf16.
GPUS = ["L40S", "A100-40GB", "H100!"]

# 256 x 1024 is the entire 262,144-token global batch, i.e. grad_accum 1. Past that, a bigger
# micro-batch would change the optimization rather than just repackage it.
BATCHES = [16, 32, 64, 128, 256]

# Only kestrel/ and bench.py are mounted. Mounting the repo root would try to ship .venv, the 12 GB
# corpus and every checkpoint in runs/, and the benchmark needs none of them: it trains on
# torch.randint noise, so it measures the compute path without touching data.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "numpy")
    .add_local_dir(ROOT / "kestrel", "/root/kestrel/kestrel")
    .add_local_file(ROOT / "scripts" / "bench.py", "/root/kestrel/scripts/bench.py")
)

app = modal.App("kestrel-bench", image=image)


@app.function(timeout=600)
def smoke_test() -> dict:
    """Prove image, mounts and CUDA all work on the cheapest card before paying for the fast ones."""
    import sys

    import torch

    sys.path.insert(0, "/root/kestrel/scripts")
    import bench  # noqa: F401  imported to prove kestrel/ resolves through bench.py's sys.path

    props = torch.cuda.get_device_properties(0)
    return {
        "device": props.name,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "torch": torch.__version__,
        "bf16": torch.cuda.is_bf16_supported(),
        "bench_import": "ok",
    }


@app.function(timeout=3600)
def sweep(variant: str, batches: list[int], compile_all: bool = False) -> list[dict]:
    """Measure each batch size in a fresh subprocess, stopping at the first OOM.

    bench.run_child() is reused rather than reimplemented: a CUDA OOM leaves the context unusable,
    and that subprocess isolation is the only reason a sweep can survive its own failures.
    """
    import sys

    import torch

    sys.path.insert(0, "/root/kestrel/scripts")
    import bench

    from kestrel.trainer import peak_tflops_for   # resolvable because bench.py put ROOT on sys.path

    device = torch.cuda.get_device_name(0)
    peak = peak_tflops_for(device)
    common: dict = dict(variant=variant, seq=1024, vocab=16384, steps=12)
    if peak:
        common["peak_tflops"] = peak
    print(f"{device}: peak bf16 {peak or 'unknown'} TFLOP/s, compile_all={compile_all}")

    rows = []
    for batch in batches:
        row = bench.run_child(batch=batch, compile=compile_all, **common)
        rows.append(row)
        print(f"  batch {batch:>4}  {row.get('tokens_per_s', row.get('reason')):>10}")
        if not row.get("ok"):
            break

    # One compile run at the fastest batch. torch.compile pays a one-off graph cost that the 4
    # warmup steps only partly absorb, so this is a floor on the real uplift, not the uplift.
    # Its own optimum can differ, because compile also cuts peak memory: hence --compiled.
    ok = [r for r in rows if r.get("ok")]
    if ok and not compile_all:
        best = max(ok, key=lambda r: r["tokens_per_s"])
        row = bench.run_child(batch=best["batch"], compile=True, **common)
        rows.append(row)
        print(f"  batch {best['batch']:>4}  compile  "
              f"{row.get('tokens_per_s', row.get('reason')):>10}")
    return rows


@app.local_entrypoint()
def main(smoke: bool = False, variant: str = "wide-shallow", compiled: bool = False,
         gpu: str = "", batches: str = ""):
    if smoke:
        print(json.dumps(smoke_test.with_options(gpu="L4").remote(), indent=2))
        return

    gpus = [g.strip() for g in gpu.split(",")] if gpu else GPUS
    sizes = [int(b) for b in batches.split(",")] if batches else BATCHES
    # A micro-batch that does not divide 256 makes grad_accum()'s floor division miss 262,144
    # tokens per step, and a run at a different global batch is comparable to nothing we have.
    bad = [b for b in sizes if 256 % b]
    if bad:
        raise SystemExit(f"micro-batch must divide 256 to preserve the global batch: {bad}")

    # Spawned rather than run in sequence: Modal bills per second with no minimum increment, so
    # parallel costs the same GPU-seconds and cuts wall clock to the slowest single card.
    handles = {g: sweep.with_options(gpu=g).spawn(variant, sizes, compiled) for g in gpus}
    results = {name: handle.get() for name, handle in handles.items()}

    out = ROOT / "runs" / "bench_modal.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for name, rows in results.items():
            for row in rows:
                fh.write(json.dumps({"gpu_requested": name, **row}) + "\n")

    print(f"\n{'requested':<12}{'device':<24}{'batch':>6}{'cmp':>5}{'tok/s':>10}{'MFU':>7}{'GB':>7}")
    print("-" * 71)
    for name, rows in results.items():
        for row in rows:
            if not row.get("ok"):
                print(f"{name:<12}{'':<24}{row.get('batch', ''):>6}{'':>5}"
                      f"{row.get('reason', 'FAIL'):>10}")
                continue
            mfu = f"{row['mfu']:.1%}" if row.get("mfu") is not None else "-"
            print(f"{name:<12}{row['device'][:23]:<24}{row['batch']:>6}"
                  f"{'yes' if row['compile'] else '':>5}{row['tokens_per_s']:>10,}"
                  f"{mfu:>7}{row['peak_gb']:>7.2f}")
        print()

    print(f"{'requested':<12}{'best tok/s':>12}{'$/1B tok':>10}{'6B hours':>10}{'6B cost':>10}")
    print("-" * 54)
    for name, rows in results.items():
        ok = [r for r in rows if r.get("ok")]
        if not ok:
            print(f"{name:<12}{'no result':>12}")
            continue
        tps = max(ok, key=lambda r: r["tokens_per_s"])["tokens_per_s"]
        price, hours = PRICE_PER_HOUR.get(name, 0.0), 6e9 / tps / 3600
        rate = f"{price / (tps * 3600 / 1e9):>10.2f}" if price else f"{'?':>10}"
        cost = f"{hours * price:>10.2f}" if price else f"{'?':>10}"
        print(f"{name:<12}{tps:>12,}{rate}{hours:>10.1f}{cost}")

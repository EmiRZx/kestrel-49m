"""Train Kestrel on a rented H100, and prove the resume path before the expensive run depends on it.

Measured on Modal: the synthetic benchmark (scripts/modal_bench.py, rows in runs/bench_modal.jsonl)
reaches ~889k tok/s for compiled wide-shallow on an H100 at micro-batch 128, but the resume test
below -- real data loader, real evals, real checkpoint writes -- sustained ~784k, and that 12% gap is
the honest one to plan against. So the 6B-token main run is ~2.1 GPU-hours and ~$8.40, against ~25
hours on the laptop. That speed changes what a crash costs. On the laptop a 30-minute checkpoint
cadence risked 2% of a 25-hour run; on an H100, 30 minutes is 24% of the entire run, so the cadence
here is set in tokens -- which scales with the hardware -- and the wall-clock rule is switched off.

Preemption cannot be disabled for GPU Functions: Modal's `nonpreemptible` flag is CPU/memory only.
A preempted container gets a 30-second exit-handler grace period, which will not write a checkpoint
this size. So tolerance rests on two other things -- frequent checkpoints on a Volume, and Modal
restarting a preempted Function on the same input, where `--resume auto` picks the checkpoint back
up. --test-resume proves that end to end by SIGKILLing the trainer mid-run, which is strictly
harsher than a preemption (no signal at all, no grace period), then relaunching the same command and
checking that the loss continues rather than restarting.

    modal volume put kestrel data/tokens_test /data/tokens_test   # 55 MB, once
    modal run scripts/modal_train.py --test-resume                # ~$0.50
    modal run scripts/modal_train.py --name kestrel-main --total-tokens 6e9

Data and checkpoints live on a Volume, never in the image: the corpus should cross the wire once, and
a checkpoint that dies with its container is not a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

# The ! suppresses Modal's free upgrade to H200, so the run happens on the chip the benchmark
# measured. An unpinned H100 would silently be a different card with different numbers.
GPU = "H100!"
MOUNT = "/vol"

# nvidia-ml-py is here for trainer.py's PowerMeter: without it the energy accounting silently reports
# zero, and energy per token is one of the numbers the README needs for Training Efficiency.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "numpy==2.5.2", "nvidia-ml-py==13.610.43")
    .add_local_dir(ROOT / "kestrel", "/root/kestrel/kestrel")
    .add_local_file(ROOT / "scripts" / "train.py", "/root/kestrel/scripts/train.py")
)

# v1 (the default), not v2: v2 is beta and documented as unsuitable for irreplaceable data, and its
# advantages -- unlimited file count, partial large-file updates -- do not apply to a corpus of a few
# hundred whole-file shards.
vol = modal.Volume.from_name("kestrel", create_if_missing=True)
app = modal.App("kestrel-train", image=image)


@app.function(gpu=GPU, volumes={MOUNT: vol}, timeout=3600)
def train(args: list[str], kill_after_ckpts: int = 0) -> dict:
    """Run scripts/train.py as a child process, streaming its stdout back to the caller.

    A child process rather than an in-process Trainer for one reason: --test-resume has to kill the
    trainer without killing the container, because a container that dies hands the failure to Modal's
    own restart machinery, and the test would then be measuring Modal rather than trainer.load().

    SIGINT and SIGTERM are forwarded so a real preemption still reaches the trainer's own
    checkpoint-and-exit handler. That is worth having even though 30 seconds cannot be relied on to
    finish the write: when it does finish, it saves everything since the last cadence checkpoint.
    """
    import os
    import signal
    import subprocess
    import sys
    import time

    cmd = [sys.executable, "/root/kestrel/scripts/train.py", *args]
    print("$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd="/root/kestrel", env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def forward(_signum, _frame) -> None:
        proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    # Line arrival times, because how long a ~700 MB checkpoint takes to reach the Volume is what
    # decides --ckpt-every-tokens for the main run, and nothing in metrics.jsonl records it.
    t0 = time.monotonic()
    lines, stamps, ckpts, killed = [], [], 0, False
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        lines.append(line)
        stamps.append(time.monotonic() - t0)
        # trainer.py prints this only after save() has atomically renamed last.pt into place, so the
        # checkpoint is complete on the container filesystem by the time the line appears.
        if line.lstrip().startswith("checkpoint at"):
            ckpts += 1
            if kill_after_ckpts and ckpts >= kill_after_ckpts:
                vol.commit()   # ordering guarantee: the next container has to see this checkpoint
                proc.kill()    # SIGKILL -- no handler, no grace period, worse than a preemption
                killed = True
                break
    proc.wait()
    vol.commit()

    # A checkpoint line's gap is the write plus the few steps since the previous log line, so the
    # median step-log gap is reported alongside it rather than subtracted -- the difference is the
    # write cost, and keeping both numbers visible keeps that inference the reader's to make.
    gaps = [round(b - a, 1) for a, b in zip(stamps, stamps[1:])]
    kinds = [ln.lstrip().split()[0] if ln.strip() else "" for ln in lines[1:]]
    step_gaps = sorted(g for g, k in zip(gaps, kinds) if k == "step")
    return {
        "returncode": proc.returncode,
        "killed": killed,
        "checkpoints": ckpts,
        "ckpt_gap_s": [g for g, k in zip(gaps, kinds) if k == "checkpoint"],
        "log_gap_s": step_gaps[len(step_gaps) // 2] if step_gaps else None,
        "resumed": next((l for l in lines if l.startswith("resumed ")), ""),
        "stopped": next((l for l in lines if l.startswith("stopped:")), ""),
        "tail": lines[-20:],
    }


@app.function(volumes={MOUNT: vol}, timeout=300)
def read_json_lines(path: str) -> list[dict]:
    """Read a JSONL file off the Volume, so the resume check can compare the loss on either side."""
    vol.reload()
    p = Path(MOUNT) / path
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# The test trains at the real micro-batch through the real data loader, so it also confirms that
# batch 128 works outside the synthetic benchmark. Warmup ends at 10M tokens and decay starts at 80M,
# so the kill at ~40M lands in WSD's constant-LR stretch: a loss discontinuity there is attributable
# to state restoration rather than to the schedule moving under it.
TEST_RUN = "resume-test"
TEST_ARGS = [
    "--data-dir", f"{MOUNT}/data/tokens_test", "--out-dir", f"{MOUNT}/runs",
    "--name", TEST_RUN, "--variant", "wide-shallow", "--micro-batch", "128",
    "--total-tokens", "1e8", "--warmup-tokens", "1e7",
    "--ckpt-every-tokens", "2e7", "--ckpt-every-minutes", "0",
    "--eval-every-tokens", "5e7", "--eval-tokens", "2e6",
    "--log-every-steps", "10", "--resume", "auto",
]


def _resume_test() -> int:
    print("phase 1: train until two checkpoints land, then SIGKILL the trainer\n")
    first = train.remote(TEST_ARGS, kill_after_ckpts=2)
    if not first["killed"]:
        raise SystemExit("phase 1 never reached two checkpoints:\n" + "\n".join(first["tail"]))

    print("\nphase 2: relaunch the identical command -- it must resume, not restart\n")
    second = train.remote(TEST_ARGS)

    rows = [r for r in read_json_lines.remote(f"runs/{TEST_RUN}/metrics.jsonl") if "loss" in r]
    # Both phases append to one metrics.jsonl, so the seam is the single place where step goes
    # backwards: phase 1 kept logging past the checkpoint before it died, phase 2 restarted from it.
    seam = next((i for i in range(1, len(rows)) if rows[i]["step"] <= rows[i - 1]["step"]), 0)
    before = {r["step"]: r["loss"] for r in rows[:seam]}
    overlap = [(r["step"], before[r["step"]], r["loss"]) for r in rows[seam:] if r["step"] in before]

    checks = [
        ("phase 1 died by SIGKILL, not a clean exit", first["returncode"] not in (0, None)),
        ("phase 2 resumed from the checkpoint", bool(second["resumed"])),
        ("phase 2 ran on to the token budget",
         second["returncode"] == 0 and bool(second["stopped"])),
        ("the metrics log has a resume seam", seam > 0),
        ("the re-run repeats steps the killed run had reached", bool(overlap)),
    ]
    # Not bit-exact, and not claimed to be: torch.compile and cuDNN autotuning reproduce a step only
    # to within float noise. But a lost AdamW moment or a mis-restored data cursor moves the loss by
    # 0.1 nats or more, so 0.02 separates "state restored" from "quietly restarted".
    worst = max((abs(a - b) for _, a, b in overlap), default=None)
    if worst is not None:
        checks.append((f"loss reproduces within 0.02 on re-run steps (worst {worst:.4f})",
                       worst < 0.02))

    print("\n" + "=" * 84)
    print(f"RESUME TEST on {GPU}")
    print("=" * 84)
    print(f"  phase 1   {first['checkpoints']} checkpoints written, then killed "
          f"(returncode {first['returncode']})")
    print(f"  phase 2   {second['resumed'] or 'DID NOT RESUME'}")
    print(f"            {second['stopped'] or 'did not stop cleanly'}")
    print(f"  ckpt      {first['ckpt_gap_s'] + second['ckpt_gap_s']} s at each checkpoint line, "
          f"against a {second['log_gap_s']} s median step-log gap")
    for step, a, b in overlap[:3]:
        print(f"  overlap   step {step:>5}   killed run {a:.4f}   resumed run {b:.4f}")
    print()
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [label for label, ok in checks if not ok]
    print("\n" + ("resume survives a hard kill on rented hardware; the main run can rely on it"
                  if not failed else f"{len(failed)} check(s) failed -- do not start the main run"))
    return 1 if failed else 0


@app.local_entrypoint()
def main(test_resume: bool = False, name: str = "kestrel-main", variant: str = "wide-shallow",
         data: str = "tokens_v16384", total_tokens: float = 6e9, micro_batch: int = 128,
         ckpt_every_tokens: float = 2e8, max_hours: float = 3.0, timeout_hours: float = 4.0,
         extra: str = "") -> None:
    if test_resume:
        code = _resume_test()
        if code:
            raise SystemExit(code)
        return

    # The same invariant modal_bench.py enforces: grad_accum() floor-divides 262,144 by
    # micro_batch * seq_len, so a micro-batch that does not divide 256 silently trains at a different
    # global batch and stops being comparable to every ablation already measured.
    if 256 % micro_batch:
        raise SystemExit(f"micro-batch must divide 256 to preserve the global batch: {micro_batch}")

    args = [
        "--data-dir", f"{MOUNT}/data/{data}", "--out-dir", f"{MOUNT}/runs",
        "--name", name, "--variant", variant, "--micro-batch", str(micro_batch),
        "--total-tokens", repr(total_tokens), "--ckpt-every-tokens", repr(ckpt_every_tokens),
        # Tokens, not minutes: at H100 speed the 30-minute default would put 26% of the whole run at
        # risk, while the token cadence scales itself with however fast the hardware turns out to be.
        "--ckpt-every-minutes", "0",
        "--max-hours", repr(max_hours),
        # Unconditional, because Modal restarts a preempted Function on the same input: this flag is
        # what turns that restart into a resume rather than an expensive fresh start.
        "--resume", "auto",
        *(extra.split() if extra else []),
    ]

    # Modal's H100 list price on 2026-09-05. The rate is the resume test's sustained median, not the
    # synthetic benchmark's 889k: this print is what the spending decision gets made on, so it uses
    # the number that includes data loading, evals and checkpoint writes.
    price, rate = 3.95, 784_000
    hours = total_tokens / rate / 3600
    print(f"{name}: {total_tokens/1e9:.2f}B tokens of {variant} on {GPU}, batch {micro_batch}, "
          f"checkpoint every {ckpt_every_tokens/1e6:.0f}M tokens")
    print(f"  ~{hours:.2f} GPU-hours at the sustained {rate:,} tok/s, so ~${hours * price:.2f}. "
          f"--max-hours {max_hours:.1f} caps one attempt at ~${max_hours * price:.2f}; it counts "
          f"training time per session, so a resume starts that clock again.\n")

    result = train.with_options(timeout=int(timeout_hours * 3600)).remote(args)
    print("\n" + "\n".join(result["tail"]))
    print(f"\nreturncode {result['returncode']}, {result['checkpoints']} checkpoints at "
          f"{result['ckpt_gap_s']} s each")
    print(f"fetch the run off the Volume with:\n  modal volume get kestrel /runs/{name} runs/{name}")

"""Build the real corpus on Modal's CPU, straight into the kestrel Volume.

The pilot corpus (data/tokens_pilot, 300M tokens) lives on the volume already and trained the four
original architecture cells. The main 6B run needs the real thing: scripts/prepare_data.py with a
6B token budget and the 16,384-vocabulary tokenizer, writing to /vol/data/tokens_v16384/.

Wrapper, not a reimplementation: the build logic (streaming fineweb-edu + wikipedia, EOT-tagged
tokenization, n-gram decontamination against the scored eval splits, the wt103 validation split,
the prepare_report.json) stays in prepare_data.py, which this calls as a child process. The only
Modal-specific work here is the container, the Volume mount, and the CLI mapping.

The tokenizer is baked into the image as a local dir rather than put on the volume -- it is ~5 MB,
built once, and read-only by this pass. Keeping it in the image leaves the volume for the thing it
must not lose (the ~12 GB of corpus the main run trains on -- 6B tokens at uint16).

The build is streaming and decontamination is done before any shard is written, so there is no
intermediate artifact to resume from: the corpus is written once, and an interrupted build restarts.
Because that restart cost is what `--no-decontam` would avoid, that flag stays as a knob here even
though the README and the competition submission must never use it.

    modal run scripts/modal_prepare.py                       # the whole 6B build
    modal run scripts/modal_prepare.py --tokens 1e8 --no-decontam   # a cheap wiring smoke test
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MOUNT = "/vol"

# Same mount point the trainer uses, so the corpus it trains on and the corpus this builds are the
# same path on the Volume.
vol = modal.Volume.from_name("kestrel", create_if_missing=True)

# Versions pinned to the ones this build has actually run against locally, not to newer or rounder
# numbers: datasets 5.x reads the eval splits decontamination scores against as parquet, where 3.x
# wanted trust_remote_code for winogrande and piqa. A pin is only worth having if it names a
# combination someone has seen work.
#
# No torch, and that is load-bearing rather than tidiness. Pulling it in to satisfy the import chain
# made a CPU container load torch's CUDA bindings, which abort the interpreter at finalization when
# there is no GPU behind them: the 100M smoke build wrote its corpus, decontaminated correctly, and
# then died with SIGABRT on the way out, reporting failure for work that had succeeded. kestrel's
# package __init__ now resolves the model lazily so the data path never reaches torch at all.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.23.1", "datasets==5.0.1", "numpy==2.5.2")
    # Quieter logs over a build that streams for an hour. Not a correctness knob: the SIGABRT that
    # cost two builds their exit code came from aiohttp's background request threads racing
    # interpreter finalization, and prepare_data.py now sidesteps that by leaving via os._exit once
    # its work is on disk. Disabling hf_xet was tried there first and only moved the abort down to
    # the next native networking layer, so it is deliberately not set -- it would buy nothing and
    # force the slower pure-Python transfer path across the whole corpus.
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir(ROOT, "/root/kestrel", ignore=["data/tokens_pilot", "data/tokens_test",
                                                  "data/hf", "runs", ".venv", ".git", ".claude",
                                                  "scratch", "__pycache__"])
)

app = modal.App("kestrel-prepare", image=image)


@app.function(volumes={MOUNT: vol}, timeout=6 * 3600, cpu=8, memory=16 * 1024)
def build(tokens: float, vocab_size: int, val_tokens: float, window: int,
          no_decontam: bool, allow_missing: bool, out_name: str) -> dict:
    """Run scripts/prepare_data.py in the container, reading the tokenizer from the image."""
    args = [
        sys.executable, "/root/kestrel/scripts/prepare_data.py",
        "--tokens", repr(tokens),
        "--vocab-size", str(vocab_size),
        "--val-tokens", repr(val_tokens),
        "--window", str(window),
        "--tokenizer-dir", "/root/kestrel/data/tokenizer",
        "--out-dir", f"{MOUNT}/data/{out_name}",
        "--shard-tokens", "100000000",
    ]
    if no_decontam:
        args.append("--no-decontam")
    if allow_missing:
        args.append("--allow-missing-eval")
    print("$ " + " ".join(args), flush=True)

    # Clear the previous report first, so the completeness check below cannot pass on someone else's
    # build. Without this, a rerun that dies early would leave the last run's finished report in
    # place and be waved through -- the check is only worth having if the artifact it reads is
    # guaranteed to have been written by this invocation.
    out = Path(f"{MOUNT}/data/{out_name}")
    report_path = out / "prepare_report.json"
    report_path.unlink(missing_ok=True)

    proc = subprocess.run(args, cwd="/root/kestrel")

    # Judge the build by what is on the volume, not by how the interpreter happened to die. This is
    # a batch job whose entire deliverable is files, and prepare_data.py writes prepare_report.json
    # only after the `with` block has closed every shard writer -- so a report carrying `finished`
    # means the corpus is flushed and complete. Streaming teardown has now aborted this container
    # twice *after* that point, and a 6B build that spends an hour and then reports -6 is the worst
    # possible outcome: it trains distrust of good data.
    #
    # This is not exit-code suppression. A crash mid-write leaves no `finished` key and no manifests,
    # and still fails below. It only reclassifies the narrow case where the work provably finished
    # and the process then fell over on the way out -- and it says so loudly rather than quietly.
    report, complete = {}, False
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifests = all((out / f"{p}_manifest.json").exists() for p in ("train", "val", "wt103"))
        # The stream stops on a document boundary, so the last document overshoots or undershoots the
        # budget slightly; 1% is slack for that, not for a truncated build.
        enough = report.get("train_tokens", 0) >= 0.99 * report.get("target_tokens", float("inf"))
        complete = bool(report.get("finished")) and manifests and enough

    vol.commit()
    return {"returncode": proc.returncode, "complete": complete,
            "train_tokens": report.get("train_tokens", 0),
            "target_tokens": report.get("target_tokens", 0)}


@app.local_entrypoint()
def main(tokens: float = 6e9, vocab_size: int = 16384, val_tokens: float = 5e6,
         window: int = 16, no_decontam: bool = False, allow_missing_eval: bool = False,
         out_name: str = "") -> None:
    # ~6x the pilot's 300M-token build, which took ~3m on this box's CPU -- so budget ~20-40m of
    # container time, comfortably inside the 6h timeout, on CPU rather than an H100 that could be
    # training a cell.
    #
    # out_name defaults to the path the trainer reads, so the real build needs no flag. A smoke test
    # must pass one: writing a truncated corpus to the main path would leave shards the real build
    # then has to overwrite exactly, and a stale shard that survives is indistinguishable from a
    # good one at training time.
    out_name = out_name or f"tokens_v{vocab_size}"
    print(f"building {tokens/1e9:.2f}B train tokens (vocab {vocab_size}) into /vol/data/{out_name}/")
    if no_decontam:
        print("  --no-decontam: ablation only, never for the submitted run")
    res = build.remote(tokens, vocab_size, val_tokens, window, no_decontam, allow_missing_eval,
                       out_name)
    if res["returncode"] and not res["complete"]:
        raise SystemExit(f"prepare_data.py exited {res['returncode']} with no complete corpus "
                         f"on the volume -- /vol/data/{out_name}/ is unusable, rebuild it")
    if res["returncode"]:
        print(f"\n  NOTE: the child exited {res['returncode']}, but the corpus is complete: "
              f"{res['train_tokens']:,} of {res['target_tokens']:,} tokens, all manifests present,\n"
              f"  prepare_report.json finished. That is a teardown crash after the work was done "
              f"(HuggingFace's streaming\n  threads racing interpreter finalization), not a bad "
              f"build. Worth a look, but the data is sound.")
    print(f"fetch the corpus with:\n  modal volume get kestrel /data/{out_name} data/{out_name}")

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
must not lose (the ~6 GB of corpus the main run trains on).

The build is streaming and decontamination is done before any shard is written, so there is no
intermediate artifact to resume from: the corpus is written once, and an interrupted build restarts.
Because that restart cost is what `--no-decontam` would avoid, that flag stays as a knob here even
though the README and the competition submission must never use it.

    modal run scripts/modal_prepare.py                       # the whole 6B build
    modal run scripts/modal_prepare.py --tokens 1e8 --no-decontam   # a cheap wiring smoke test
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MOUNT = "/vol"

# Same mount point the trainer uses, so the corpus it trains on and the corpus this builds are the
# same path on the Volume.
vol = modal.Volume.from_name("kestrel", create_if_missing=True)

# No torch: this is a CPU pass, and pulling torch into it would make every architecture cell wait on
# a full image rebuild that has nothing to do with the data. The tokenizer and the encoding path are
# the reason `kestrel/` is mounted into the image, and it is the only Python the build imports.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.21.0", "datasets==3.5.1", "numpy==2.5.2")
    .add_local_dir(ROOT, "/root/kestrel", ignore=["data/tokens_pilot", "data/tokens_test",
                                                  "data/hf", "runs", ".venv", ".git", ".claude",
                                                  "scratch", "__pycache__"])
)

app = modal.App("kestrel-prepare", image=image)


@app.function(volumes={MOUNT: vol}, timeout=6 * 3600, cpu=8, memory=16 * 1024)
def build(tokens: float, vocab_size: int, val_tokens: float, window: int,
          no_decontam: bool, allow_missing: bool) -> dict:
    """Run scripts/prepare_data.py in the container, reading the tokenizer from the image."""
    args = [
        sys.executable, "/root/kestrel/scripts/prepare_data.py",
        "--tokens", repr(tokens),
        "--vocab-size", str(vocab_size),
        "--val-tokens", repr(val_tokens),
        "--window", str(window),
        "--tokenizer-dir", "/root/kestrel/data/tokenizer",
        "--out-dir", f"{MOUNT}/data/tokens_v{vocab_size}",
        "--shard-tokens", "100000000",
    ]
    if no_decontam:
        args.append("--no-decontam")
    if allow_missing:
        args.append("--allow-missing-eval")
    print("$ " + " ".join(args), flush=True)
    proc = subprocess.run(args, cwd="/root/kestrel")
    vol.commit()
    return {"returncode": proc.returncode}


@app.local_entrypoint()
def main(tokens: float = 6e9, vocab_size: int = 16384, val_tokens: float = 5e6,
         window: int = 16, no_decontam: bool = False, allow_missing_eval: bool = False) -> None:
    # ~6x the pilot's 300M-token build, which took ~3m on this box's CPU -- so budget ~20-40m of
    # container time, comfortably inside the 6h timeout, on CPU rather than an H100 that could be
    # training a cell.
    print(f"building {tokens/1e9:.2f}B train tokens (vocab {vocab_size}) into "
          f"/vol/data/tokens_v{vocab_size}/")
    if no_decontam:
        print("  --no-decontam: ablation only, never for the submitted run")
    res = build.remote(tokens, vocab_size, val_tokens, window, no_decontam, allow_missing_eval)
    if res["returncode"]:
        raise SystemExit(f"prepare_data.py exited {res['returncode']}")
    print(f"fetch the corpus with:\n  modal volume get kestrel /data/tokens_v{vocab_size} "
          f"data/tokens_v{vocab_size}")

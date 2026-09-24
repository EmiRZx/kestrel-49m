"""Anneal the finished checkpoint on a synthetic/original mixture, on a rented H100.

    modal run --detach scripts/modal_anneal.py --tokens 200e6 --synth-frac 0.05
    modal run --detach scripts/modal_anneal.py --tokens 200e6 --synth-frac 0.05 --name anneal-v2

Both corpora already live on the Volume -- the 6 B-token original at `data/tokens_v16384`, the
synthetic shards uploaded to `data/anneal_synth` -- so the mixing happens here rather than on the
laptop, and nothing large crosses the wire.

How the mixture is built. `ShardedTokenDataset` reads one manifest, enumerates every block across
the shards it lists, and permutes them once. So a mixture is just a manifest naming shards from both
corpora, and the ratio is set by how many of each it lists: the permutation then interleaves them
uniformly, which is what makes the synthetic text arrive spread across the whole anneal rather than
in one clump at the end.

Why not anneal on the synthetic text alone. 6.2 M tokens of one generator's prose at low LR pulls
hard toward a single register, and the benchmark most exposed is ARC-Easy -- the one Kestrel
currently wins, and the one that rewards the educational corpus this text is nothing like. A small
synthetic fraction mixed into mostly-original data is the conventional shape for exactly this
reason, and `--synth-frac` is the knob worth sweeping if the first result disappoints.

`--synth-frac 0.05` against a 200 M-token anneal asks for ~10 M synthetic tokens, which is more than
the 6.2 M that exist, so the synthetic shard is simply seen more than once while the original shards
are not. That is deliberate and worth naming: a few epochs over a small high-quality set during
decay is the technique, not an accident of arithmetic.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

GPU = "H100!"          # pinned, for the same reason modal_train.py pins it
MOUNT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "numpy==2.5.2", "nvidia-ml-py==13.610.43")
    .add_local_dir(ROOT / "kestrel", "/root/kestrel/kestrel")
    .add_local_file(ROOT / "scripts" / "anneal.py", "/root/kestrel/scripts/anneal.py")
    .add_local_file(ROOT / "scripts" / "train.py", "/root/kestrel/scripts/train.py")
)

vol = modal.Volume.from_name("kestrel", create_if_missing=True)
app = modal.App("kestrel-anneal", image=image)


def build_mixture(vol_root: str, synth_frac: float, total_tokens: float) -> str:
    """Write a manifest mixing both corpora, and return the directory holding it.

    The manifest lists shard *names* relative to its own directory, so the mixed directory holds
    symlinks to the real shards rather than copies -- 6 B tokens is 12 GB and copying it to change a
    ratio would be absurd.
    """
    import json
    import os

    def link(src: Path, dst: Path) -> None:
        """Symlink, or hard-link if the platform refuses. Windows needs a privilege for symlinks
        that a normal account does not have, which only matters for testing this function locally --
        the container is Linux -- but a test that cannot run is not worth much."""
        if dst.exists():
            return
        try:
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            os.link(src, dst)

    src_orig = Path(vol_root) / "data" / "tokens_v16384"
    src_syn = Path(vol_root) / "data" / "anneal_synth"
    out = Path(vol_root) / "data" / f"anneal_mix_{synth_frac:g}"
    out.mkdir(parents=True, exist_ok=True)

    orig = json.loads((src_orig / "train_manifest.json").read_text())
    syn = json.loads((src_syn / "train_manifest.json").read_text())

    syn_tokens = syn["total_tokens"]
    want_syn = total_tokens * synth_frac
    # How many passes over the synthetic set the ratio implies. The shard is listed that many times;
    # blocks from a repeated shard are distinct entries in the permutation, so repeats interleave
    # rather than replay back to back.
    repeats = max(1, round(want_syn / syn_tokens))

    shards, total = [], 0
    for s in syn["shards"]:
        dst = out / f"syn_{s['name']}"
        link(src_syn / s["name"], dst)
        for _ in range(repeats):
            shards.append({"name": dst.name, "tokens": s["tokens"]})
            total += s["tokens"]

    # Enough original shards to make up the rest of the budget, plus headroom so the run never runs
    # out of blocks before its token target.
    want_orig = max(total_tokens - total, 0) * 1.2
    got = 0
    for s in orig["shards"]:
        if got >= want_orig:
            break
        dst = out / f"orig_{s['name']}"
        link(src_orig / s["name"], dst)
        shards.append({"name": dst.name, "tokens": s["tokens"]})
        got += s["tokens"]
        total += s["tokens"]

    (out / "train_manifest.json").write_text(json.dumps(
        {"split": "train", "total_tokens": total, "shards": shards,
         "mixture": {"synth_frac": synth_frac, "synth_repeats": repeats,
                     "synth_tokens": syn_tokens * repeats, "orig_tokens": got}}, indent=2))

    # Validation comes from the original corpus, deliberately: the question an anneal has to answer
    # is whether the model got better at the *original* distribution too, not whether it got better
    # at predicting its own training text.
    for name in ("val_manifest.json",):
        src = json.loads((src_orig / name).read_text())
        for s in src["shards"]:
            dst = out / f"orig_{s['name']}"
            link(src_orig / s["name"], dst)
            s["name"] = dst.name
        (out / name).write_text(json.dumps(src, indent=2))

    for extra in ("wt103_manifest.json",):
        p = src_orig / extra
        if p.exists():
            src = json.loads(p.read_text())
            for s in src["shards"]:
                dst = out / f"orig_{s['name']}"
                link(src_orig / s["name"], dst)
                s["name"] = dst.name
            (out / extra).write_text(json.dumps(src, indent=2))

    print(f"mixture: {repeats}x synthetic ({syn_tokens*repeats/1e6:.1f}M) "
          f"+ {got/1e6:.0f}M original = {total/1e6:.0f}M tokens across {len(shards)} shards")
    return str(out)


@app.function(gpu=GPU, volumes={MOUNT: vol}, timeout=7200)
def anneal(tokens: float, synth_frac: float, name: str, peak_frac: float,
           micro_batch: int, init: str) -> dict:
    import subprocess
    import sys

    data_dir = build_mixture(MOUNT, synth_frac, tokens)
    vol.commit()

    cmd = [sys.executable, "scripts/anneal.py",
           "--init", f"{MOUNT}/{init}",
           "--data", data_dir,
           "--name", name,
           "--tokens", str(tokens),
           "--peak-frac", str(peak_frac),
           "--micro-batch", str(micro_batch),
           "--out-dir", f"{MOUNT}/runs"]
    print(" ".join(cmd), flush=True)

    p = subprocess.Popen(cmd, cwd="/root/kestrel", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    rc = p.wait()
    vol.commit()
    return {"returncode": rc, "name": name, "data": data_dir}


@app.local_entrypoint()
def main(tokens: float = 200e6, synth_frac: float = 0.05, name: str = "anneal-v1",
         peak_frac: float = 0.1, micro_batch: int = 128,
         init: str = "runs/kestrel-main/best.pt") -> None:
    res = anneal.remote(tokens, synth_frac, name, peak_frac, micro_batch, init)
    print(res)
    print(f"\nfetch with:\n  modal volume get kestrel /runs/{name} runs/{name}")

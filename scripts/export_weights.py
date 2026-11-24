"""Strip a training checkpoint down to what inference needs, and prove nothing was lost.

`runs/kestrel-main/best.pt` is 438 MB because it is a *resume* checkpoint: it carries both
optimizer states, the dataset cursor and the RNG states alongside the weights. None of that is
needed to run or score the model, so the released file keeps only:

  model          the fp32 state dict, unchanged
  model_config   the dict `KestrelConfig(**...)` is rebuilt from
  state          step, tokens seen, GPU-seconds, energy, best val loss: the run's own accounting,
                 under the same key and field names as a training checkpoint, so `eval.py` reports
                 them exactly as it does for `best.pt`
  config         the training config, as plain values

Everything kept is a tensor or a plain Python value, so the file loads under
`torch.load(..., weights_only=True)` — a downloader does not have to trust a pickle. It also loads
through the existing `kestrel.evaluate.load_checkpoint`, so `scripts/eval.py --ckpt` accepts it
unchanged.

The export is verified before it is reported: every tensor is compared with `torch.equal` against
the source checkpoint, and the parameter count is recomputed from the exported file.

    python scripts/export_weights.py
    python scripts/export_weights.py --src runs/kestrel-main/best.pt --out release/kestrel-49m.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# `RunState` fields that carry over. `history` is left behind: it is in metrics.jsonl already.
STATE_KEYS = ("step", "tokens", "gpu_seconds", "energy_j", "best_val")


def _plain(value):
    """Reduce a value to something `weights_only=True` will load: no Paths, no dataclasses."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=ROOT / "runs" / "kestrel-main" / "best.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "release" / "kestrel-49m.pt")
    args = ap.parse_args()

    src = torch.load(args.src, map_location="cpu", weights_only=False)
    state = src.get("state") or {}
    out = {
        # The tensors themselves, not clones: the embedding and output head are one tied tensor
        # under two keys, and torch.save writes shared storage once. Cloning would write it twice
        # and ship 42 MB of duplicate.
        "model": dict(src["model"]),
        "model_config": _plain(src["model_config"]),
        "state": {k: _plain(state[k]) for k in STATE_KEYS if k in state},
        "config": _plain(src.get("config", {})),
        "source_checkpoint": args.src.name,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(args.out)

    # Verify from disk, through the strict loader a downloader would use.
    back = torch.load(args.out, map_location="cpu", weights_only=True)
    if back["model"].keys() != src["model"].keys():
        raise SystemExit("exported state dict has different keys from the source")
    for k, v in src["model"].items():
        if v.dtype != back["model"][k].dtype or not torch.equal(v, back["model"][k]):
            raise SystemExit(f"tensor {k!r} differs from the source checkpoint")

    from kestrel.model import Kestrel, KestrelConfig  # noqa: E402
    model = Kestrel(KestrelConfig(**back["model_config"]))
    model.load_state_dict(back["model"])
    params = sum(p.numel() for p in model.parameters())   # tied weights are one tensor, counted once

    report = {
        "out": str(args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out),
        "bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "tensors": len(back["model"]),
        "params": params,
        "dtypes": sorted({str(v.dtype) for v in back["model"].values()}),
        "identical_to_source": True,
        "state": out["state"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

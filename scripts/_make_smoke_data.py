"""Throwaway: synthesise tiny token shards so the trainer can be smoke-tested without real data."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.data import TokenShardWriter

out = ROOT / "data" / "tokens_smoke"
rng = np.random.default_rng(0)
for split, n in (("train", 2_000_000), ("val", 200_000), ("wt103", 200_000)):
    with TokenShardWriter(out, split, tokens_per_shard=900_000) as w:
        # Not uniform noise: a Zipf-ish distribution gives a loss curve that can actually move.
        w.write(rng.zipf(1.4, size=n).clip(1, 16383).astype(np.uint16))
    print(split, "ok")
print("wrote", out)

"""Token shard storage and loading.

Tokens live in flat little-endian uint16 `.bin` shards on disk (vocab is capped at 65,536, so
uint16 is lossless and halves both disk and page-cache pressure versus uint32). Shards are
memory-mapped, so the OS page cache does the buffering and a training run never holds the corpus
in RAM -- which matters on a 15 GB machine with a 12 GB corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHARD_DTYPE = np.uint16
MANIFEST_NAME = "manifest.json"


class TokenShardWriter:
    """Append tokens, rolling over to a new shard every `tokens_per_shard`."""

    def __init__(self, out_dir: Path, split: str, tokens_per_shard: int = 100_000_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.tokens_per_shard = tokens_per_shard
        self.shards: list[dict] = []
        self._fh = None
        self._count = 0
        self.total = 0

    def _roll(self) -> None:
        self._close()
        name = f"{self.split}_{len(self.shards):05d}.bin"
        self._fh = (self.out_dir / name).open("wb")
        self.shards.append({"name": name, "tokens": 0})
        self._count = 0

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self.shards[-1]["tokens"] = self._count
            self._fh = None

    def write(self, tokens: np.ndarray) -> None:
        tokens = np.asarray(tokens, dtype=SHARD_DTYPE)
        pos = 0
        while pos < tokens.size:
            if self._fh is None or self._count >= self.tokens_per_shard:
                self._roll()
            take = min(tokens.size - pos, self.tokens_per_shard - self._count)
            self._fh.write(tokens[pos:pos + take].tobytes())
            self._count += take
            self.total += take
            pos += take

    def finalize(self, extra: dict | None = None) -> dict:
        self._close()
        manifest = {"split": self.split, "total_tokens": self.total, "shards": self.shards}
        manifest.update(extra or {})
        path = self.out_dir / f"{self.split}_{MANIFEST_NAME}"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def __enter__(self) -> "TokenShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.finalize()
        else:
            self._close()


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray


class ShardedTokenDataset:
    """Fixed-size blocks drawn from memory-mapped shards in a seeded, resumable order.

    A block is `seq_len + 1` tokens: inputs are block[:-1], targets are block[1:]. Blocks are
    enumerated across all shards and then permuted once with a fixed seed, so `state_dict()` needs
    only the block cursor to resume an interrupted run bit-for-bit.
    """

    def __init__(self, data_dir: Path, split: str, seq_len: int, seed: int = 1337) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        manifest_path = self.data_dir / f"{split}_{MANIFEST_NAME}"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}; run scripts/prepare_data.py")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self._maps: list[np.ndarray] = []
        starts, block_shard, block_offset = [], [], []
        stride = seq_len + 1
        for shard in self.manifest["shards"]:
            path = self.data_dir / shard["name"]
            mm = np.memmap(path, dtype=SHARD_DTYPE, mode="r")
            idx = len(self._maps)
            self._maps.append(mm)
            n_blocks = (mm.size - 1) // stride
            if n_blocks <= 0:
                continue
            block_shard.append(np.full(n_blocks, idx, dtype=np.int32))
            block_offset.append(np.arange(n_blocks, dtype=np.int64) * stride)
            starts.append(n_blocks)

        if not starts:
            raise ValueError(f"split {split!r} has no usable blocks at seq_len {seq_len}")
        self.block_shard = np.concatenate(block_shard)
        self.block_offset = np.concatenate(block_offset)
        self.n_blocks = self.block_shard.size
        self.order = np.random.default_rng(seed).permutation(self.n_blocks)
        self.cursor = 0

    @property
    def total_tokens(self) -> int:
        return int(self.manifest["total_tokens"])

    def __len__(self) -> int:
        return self.n_blocks

    def get_block(self, i: int) -> np.ndarray:
        b = self.order[i % self.n_blocks]
        shard = self._maps[self.block_shard[b]]
        off = int(self.block_offset[b])
        return np.asarray(shard[off:off + self.seq_len + 1], dtype=np.int64)

    def next_batch(self, batch_size: int) -> Batch:
        blocks = np.stack([self.get_block(self.cursor + i) for i in range(batch_size)])
        self.cursor += batch_size
        return Batch(x=blocks[:, :-1], y=blocks[:, 1:])

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "seq_len": self.seq_len, "n_blocks": self.n_blocks}

    def load_state_dict(self, state: dict) -> None:
        if state.get("n_blocks") != self.n_blocks:
            raise ValueError("shard set changed since the checkpoint; cannot resume deterministically")
        self.cursor = int(state["cursor"])

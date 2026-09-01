"""Remove training documents that overlap verbatim with the evaluation sets.

Track 01 is scored on HellaSwag, ARC-Easy, PIQA, WinoGrande and WikiText-103 perplexity. Web-scale
corpora demonstrably contain copies of benchmark items, and a model that has memorised them reports
accuracy it did not earn. This module makes the check explicit and reportable.

Method: hash every window of `window` consecutive *token ids* in the evaluation splits, then drop
any training document sharing a window hash. Working on token ids rather than words means the whole
check is vectorised numpy over the arrays we already produce during tokenisation.

The window hash is a sum of per-token random 64-bit values, computed from a prefix sum so the whole
document costs one pass. Being a sum, it is insensitive to token order within the window: a
permutation of the same 32 tokens collides. That is an acceptable trade -- a false positive costs
one discarded training document, and a genuine 32-token verbatim overlap is always caught.
"""

from __future__ import annotations

import numpy as np

DEFAULT_WINDOW = 32
HASH_SEED = 0x5EED


class EvalNgramIndex:
    """Window-hash index over the evaluation splits, for contamination filtering."""

    def __init__(self, vocab_size: int, window: int = DEFAULT_WINDOW, seed: int = HASH_SEED) -> None:
        self.window = window
        self.vocab_size = vocab_size
        # One fixed random 64-bit value per token id; identical on both sides of the comparison.
        self.table = np.random.default_rng(seed).integers(
            0, 2**63 - 1, size=vocab_size, dtype=np.uint64
        )
        self._hashes: np.ndarray | None = None
        self._pending: list[np.ndarray] = []
        self.n_eval_docs = 0
        self.sources: dict[str, int] = {}

    def _window_hashes(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.size < self.window:
            return np.empty(0, dtype=np.uint64)
        vals = self.table[np.clip(tokens, 0, self.vocab_size - 1)]
        prefix = np.zeros(vals.size + 1, dtype=np.uint64)
        np.cumsum(vals, dtype=np.uint64, out=prefix[1:])
        return prefix[self.window:] - prefix[: -self.window]

    def add_eval_doc(self, tokens: np.ndarray, source: str = "") -> None:
        h = self._window_hashes(np.asarray(tokens))
        if h.size:
            self._pending.append(h)
        self.n_eval_docs += 1
        if source:
            self.sources[source] = self.sources.get(source, 0) + 1
        self._hashes = None

    def build(self) -> "EvalNgramIndex":
        """Sort and de-duplicate the accumulated hashes so lookups can use searchsorted."""
        if self._pending:
            all_h = np.concatenate(self._pending) if self._hashes is None else np.concatenate(
                [self._hashes, *self._pending]
            )
            self._hashes = np.unique(all_h)
            self._pending = []
        elif self._hashes is None:
            self._hashes = np.empty(0, dtype=np.uint64)
        return self

    @property
    def size(self) -> int:
        return 0 if self._hashes is None else int(self._hashes.size)

    def is_contaminated(self, tokens: np.ndarray) -> bool:
        if self._hashes is None:
            self.build()
        if self._hashes.size == 0:
            return False
        h = self._window_hashes(np.asarray(tokens))
        if h.size == 0:
            return False
        idx = np.searchsorted(self._hashes, h)
        idx = np.clip(idx, 0, self._hashes.size - 1)
        return bool(np.any(self._hashes[idx] == h))

    def save(self, path) -> None:
        self.build()
        np.savez_compressed(
            path, hashes=self._hashes, window=self.window,
            vocab_size=self.vocab_size, n_eval_docs=self.n_eval_docs,
        )

    @classmethod
    def load(cls, path, seed: int = HASH_SEED) -> "EvalNgramIndex":
        z = np.load(path, allow_pickle=False)
        idx = cls(int(z["vocab_size"]), int(z["window"]), seed)
        idx._hashes = z["hashes"]
        idx.n_eval_docs = int(z["n_eval_docs"])
        idx._pending = []
        return idx


# Which split of each benchmark lm-evaluation-harness actually scores on. Getting this wrong would
# mean decontaminating against text the judges never evaluate.
#
# These paths are copied from lm-eval 0.4.12's own task YAMLs rather than from the papers, because
# the dataset repository that a task *loads* has drifted from the one you would guess:
#   * `hellaswag` is not a valid repo id on modern `datasets`; the task loads `Rowan/hellaswag`.
#   * `ybisk/piqa` ships a loading script, which `datasets` 5.x refuses to run; the task loads the
#     parquet mirror `baber/piqa`.
#   * the `wikitext` task loads `EleutherAI/wikitext_document_level` at config `wikitext-2-raw-v1`,
#     document-level rather than line-level -- and WikiText-2 shares its validation and test splits
#     with WikiText-103, so "WikiText-103 test perplexity" and what the harness measures are the same
#     text. Both paths are indexed anyway: the cost is a few thousand extra hashes.
EVAL_SOURCES: list[dict] = [
    {"name": "hellaswag", "path": "Rowan/hellaswag", "split": "validation",
     "fields": ["ctx", "endings"]},
    {"name": "arc_easy", "path": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
     "fields": ["question", "choices"]},
    {"name": "piqa", "path": "baber/piqa", "split": "validation",
     "fields": ["goal", "sol1", "sol2"]},
    {"name": "winogrande", "path": "allenai/winogrande", "config": "winogrande_xl",
     "split": "validation", "fields": ["sentence", "option1", "option2"]},
    {"name": "wikitext103", "path": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
     "split": "test", "fields": ["text"]},
    {"name": "wikitext_harness", "path": "EleutherAI/wikitext_document_level",
     "config": "wikitext-2-raw-v1", "split": "test", "fields": ["page"]},
]



def _field_texts(example: dict, field: str) -> list[str]:
    val = example.get(field)
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, dict):
        # ARC's choices are {"text": [...], "label": ["A","B",...]}; the labels are not content.
        if isinstance(val.get("text"), list):
            return [str(x) for x in val["text"]]
        return [str(x) for v in val.values() if isinstance(v, list) for x in v]
    if isinstance(val, list):
        return [str(x) for x in val]
    return [str(val)]


def extract_texts(example: dict, fields: list[str]) -> list[str]:
    """Flatten one benchmark example into the strings a contaminated web page would contain.

    The bare fields are not enough. A window-hash filter can only see a text that is at least
    `window` tokens long, and measured against the 16k tokenizer only 3% of ARC-Easy's scored
    strings and 3% of WinoGrande's clear 32 tokens on their own -- their answer choices are one or
    two words ("dry palms"). Indexing fields alone therefore leaves those two benchmarks almost
    entirely uncovered.

    So the first field is treated as the stem (question / ctx / goal / sentence) and every later
    field as a candidate continuation, and we emit stem+continuation pairs as well as the individual
    fields. That is also the form contamination actually takes: a scraped page carries the question
    together with its answers, not the answers alone.
    """
    groups = [_field_texts(example, f) for f in fields]
    out: list[str] = [t for g in groups for t in g]
    stem, rest = (groups[0] if groups else []), [t for g in groups[1:] for t in g]
    for s in stem:
        for cont in rest:
            out.append(f"{s} {cont}")
        if len(rest) > 1:  # the whole item as one block, the way a quiz page lists it
            out.append(s + " " + " ".join(rest))
    return [t for t in out if t and t.strip()]


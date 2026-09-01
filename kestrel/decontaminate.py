"""Remove training documents that overlap verbatim with the evaluation sets.

Track 01 is scored on HellaSwag, ARC-Easy, PIQA, WinoGrande and WikiText-103 perplexity. Web-scale
corpora demonstrably contain copies of benchmark items, and a model that has memorised them reports
accuracy it did not earn. This module makes the check explicit and reportable.

Method: hash every window of `window` consecutive *token ids* in the evaluation splits, then drop
any training document sharing a window hash. Working on token ids rather than words means the whole
check is vectorised numpy over the arrays we already produce during tokenisation.

Two choices here are load-bearing, and both came from measurement rather than convention.

*Window length 16.* A window of length w cannot see any text shorter than w, and benchmark items are
short. Measured against this project's 16k tokenizer, only 22 % of ARC-Easy's scored strings and
12 % of WinoGrande's reach 32 tokens, so the 32-token window this module started with left two of the
five scored benchmarks very nearly unfiltered. At 16 tokens the same figures are 49 % and 67 %, and
the share of scored strings too short to hash at all falls from 37 % to 19 %.

What settled it was measuring both sides on the corpus (`scripts/check_decontam_cost.py`, 300,000
documents, 150,000 from each source):

    window 32:    3 drops in 300,000 documents   (0.0010 %) -- 0 of them from FineWeb-Edu
    window 16:  144 drops in 300,000 documents   (0.0480 %) -- 11 from FineWeb-Edu, 133 Wikipedia

Window 32 was not a conservative filter, it was very nearly a no-op: it fired on one document in a
hundred thousand and never once on the corpus's dominant source. Window 16 costs 0.048 % of the
corpus -- 1 document in 2,000 -- to gain that coverage, which against a 6 B-token budget is noise.

The composition matters more than the rate. Nearly all of the Wikipedia drops match WikiText-103,
which is correct rather than costly: WikiText-103 *is* Wikipedia, so those are the same articles and
removing them is the entire purpose. Of the 11 FineWeb-Edu drops, some are real verbatim overlap
("British armed trawler HMT Stella Sirius was sunk by") and some are stock phrases that any
educational page contains ("Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus"). Eleven documents
in 150,000 is a price worth paying either way, and `--show` decodes the spans so the judgement is
inspectable instead of asserted.

Sixteen tokens is about 8-9 English words at this tokenizer's 2.89 chars/token -- already stricter
than the 13-word n-gram GPT-3 filtered on. That is where it stops. Twelve would add ~10 points of
ARC-Easy coverage, but a 6-7 word span is a phrase rather than a quotation, so the drops would
increasingly be coincidence and there would be no convention left to appeal to.

*Order-sensitive hashing.* The hash is a Rabin-Karp polynomial in a fixed odd base over per-token
random values, so a match means a verbatim span and not an anagram of one. An order-insensitive sum
of the per-token values would be cheaper, and is what this module used at window 32 where a
permutation collision is vanishingly unlikely; at 16 tokens that argument no longer holds.

The polynomial still costs one pass. Writing `x_i` for a token's random value, `B` for the base and
`S_j` for the prefix sum of `x_i * B**-i`, the hash of the window starting at `j` is
`B**j * (S_{j+w} - S_j)`, which depends only on the span's contents and not on where it sits, and
which numpy computes for every window at once as a cumulative sum, a slice subtraction and a
multiply.

The two audit scripts measure the two directions, and the claim in the README rests on both:
`scripts/check_decontam_coverage.py` for reach (how much of each eval split is long enough to see),
`scripts/check_decontam_cost.py` for price (how much of the corpus each window drops, and why).
"""

from __future__ import annotations

import numpy as np

DEFAULT_WINDOW = 16
HASH_SEED = 0x5EED
# Odd, so it is a unit mod 2**64 and so is every power of it. The usual golden-ratio constant.
HASH_BASE = 0x9E3779B97F4A7C15
HASH_BASE_INV = pow(HASH_BASE, -1, 2**64)
# Bumped whenever the hash changes, so a stale decontam.npz is refused rather than silently reused.
INDEX_FORMAT = 2

class EvalNgramIndex:
    """Window-hash index over the evaluation splits, for contamination filtering."""

    def __init__(self, vocab_size: int, window: int = DEFAULT_WINDOW, seed: int = HASH_SEED) -> None:
        self.window = window
        self.vocab_size = vocab_size
        self.seed = seed
        # One fixed random 64-bit value per token id; identical on both sides of the comparison.
        self.table = np.random.default_rng(seed).integers(
            0, 2**63 - 1, size=vocab_size, dtype=np.uint64
        )
        self._pow_b = np.ones(1, dtype=np.uint64)
        self._pow_inv = np.ones(1, dtype=np.uint64)
        self._hashes: np.ndarray | None = None
        self._src: np.ndarray | None = None          # which eval source each hash came from
        self._pending: list[np.ndarray] = []
        self._pending_src: list[np.ndarray] = []
        self.source_names: list[str] = []
        self.n_eval_docs = 0
        self.n_eval_docs_short = 0                   # texts below the window, so unfilterable
        self.sources: dict[str, int] = {}

    def _ensure_powers(self, n: int) -> None:
        """Cache B**i and B**-i up to n. Unsigned multiply in numpy already wraps mod 2**64."""
        if self._pow_b.size >= n:
            return
        m = max(n, 2 * self._pow_b.size)
        for name, base in (("_pow_b", HASH_BASE), ("_pow_inv", HASH_BASE_INV)):
            p = np.ones(m, dtype=np.uint64)
            p[1:] = np.cumprod(np.full(m - 1, base, dtype=np.uint64))
            setattr(self, name, p)

    def _window_hashes(self, tokens: np.ndarray) -> np.ndarray:
        w, n = self.window, tokens.size
        if n < w:
            return np.empty(0, dtype=np.uint64)
        self._ensure_powers(n + 1)
        vals = self.table[np.clip(tokens, 0, self.vocab_size - 1)]
        prefix = np.zeros(n + 1, dtype=np.uint64)
        np.cumsum(vals * self._pow_inv[:n], dtype=np.uint64, out=prefix[1:])
        return self._pow_b[: n - w + 1] * (prefix[w:] - prefix[:-w])

    def _source_id(self, source: str) -> int:
        if source not in self.source_names:
            self.source_names.append(source)
        return self.source_names.index(source)

    def add_eval_doc(self, tokens: np.ndarray, source: str = "") -> None:
        h = self._window_hashes(np.asarray(tokens))
        if h.size:
            self._pending.append(h)
            self._pending_src.append(np.full(h.size, self._source_id(source), dtype=np.uint8))
        else:
            self.n_eval_docs_short += 1
        self.n_eval_docs += 1
        if source:
            self.sources[source] = self.sources.get(source, 0) + 1
        self._hashes = None

    def build(self) -> "EvalNgramIndex":
        """Sort and de-duplicate the accumulated hashes so lookups can use searchsorted."""
        if self._pending:
            h = np.concatenate(self._pending if self._hashes is None
                               else [self._hashes, *self._pending])
            s = np.concatenate(self._pending_src if self._src is None
                               else [self._src, *self._pending_src])
            # keep_index keeps the source labels aligned with the de-duplicated hashes
            self._hashes, keep = np.unique(h, return_index=True)
            self._src = s[keep]
            self._pending, self._pending_src = [], []
        elif self._hashes is None:
            self._hashes = np.empty(0, dtype=np.uint64)
            self._src = np.empty(0, dtype=np.uint8)
        return self

    @property
    def size(self) -> int:
        return 0 if self._hashes is None else int(self._hashes.size)

    def match(self, tokens: np.ndarray) -> tuple[str, int] | None:
        """The eval source and token offset of the first overlapping window, or None.

        One searchsorted over the whole document. The offset is what makes a match inspectable:
        `scripts/check_decontam_cost.py` decodes the span back to text, so a drop can be read as
        either a benchmark item or an unlucky stock phrase.
        """
        if self._hashes is None:
            self.build()
        if self._hashes.size == 0:
            return None
        h = self._window_hashes(np.asarray(tokens))
        if h.size == 0:
            return None
        idx = np.clip(np.searchsorted(self._hashes, h), 0, self._hashes.size - 1)
        hit = self._hashes[idx] == h
        if not hit.any():
            return None
        pos = int(np.argmax(hit))
        src_id = int(self._src[idx[pos]])
        return (self.source_names[src_id] if src_id < len(self.source_names) else "?"), pos

    def contaminated_by(self, tokens: np.ndarray) -> str | None:
        """The eval source this document overlaps, or None."""
        m = self.match(tokens)
        return None if m is None else m[0]

    def is_contaminated(self, tokens: np.ndarray) -> bool:
        return self.match(tokens) is not None

    def save(self, path) -> None:
        self.build()
        np.savez_compressed(
            path, format=INDEX_FORMAT, base=np.uint64(HASH_BASE), seed=self.seed,
            hashes=self._hashes, srcs=self._src, window=self.window,
            vocab_size=self.vocab_size, n_eval_docs=self.n_eval_docs,
            n_eval_docs_short=self.n_eval_docs_short,
            source_names=np.array(self.source_names or [""], dtype="U32"),
            source_keys=np.array(list(self.sources.keys()) or [""], dtype="U32"),
            source_counts=np.array(list(self.sources.values()) or [0], dtype=np.int64),
        )

    @classmethod
    def load(cls, path) -> "EvalNgramIndex":
        z = np.load(path, allow_pickle=False)
        fmt = int(z["format"]) if "format" in z.files else 1
        if fmt != INDEX_FORMAT or (int(z["base"]) if "base" in z.files else 0) != HASH_BASE:
            # Reusing an index built by a different hash would compare incomparable numbers and
            # quietly filter nothing. Delete it and let prepare_data.py rebuild.
            raise ValueError(
                f"{path} was built by decontaminate.py format {fmt}, this is format "
                f"{INDEX_FORMAT}. Delete it and re-run scripts/prepare_data.py to rebuild.")
        idx = cls(int(z["vocab_size"]), int(z["window"]), int(z["seed"]))
        idx._hashes = z["hashes"]
        idx._src = z["srcs"]
        idx.n_eval_docs = int(z["n_eval_docs"])
        idx.n_eval_docs_short = int(z["n_eval_docs_short"])
        idx.source_names = [str(s) for s in z["source_names"]]
        idx.sources = {str(k): int(v) for k, v in zip(z["source_keys"], z["source_counts"]) if k}
        idx._pending, idx._pending_src = [], []
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

    The bare fields are not enough. Even at a 16-token window, an answer choice like "dry palms"
    is far too short to hash on its own, so the first field is treated as the stem (question / ctx /
    goal / sentence) and every later field as a candidate continuation: we emit stem+continuation
    pairs, and the whole item as one block, alongside the individual fields. That is also the form
    contamination actually takes -- a scraped page carries the question together with its answers.
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






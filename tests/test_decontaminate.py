"""Tests for the decontamination index.

The filter's whole value is that it is trustworthy, and it fails silently if it is wrong: an index
that matches nothing looks exactly like a clean corpus. So the hash gets checked against a direct
computation, and the detector gets checked on a document that is known to be contaminated and one
that is known not to be.

    python -m pytest tests/test_decontaminate.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.decontaminate import (  # noqa: E402
    HASH_BASE,
    HASH_BASE_INV,
    EvalNgramIndex,
    extract_texts,
)

VOCAB = 512
M64 = 2**64


def _direct(table: np.ndarray, span: np.ndarray) -> int:
    """The window hash, written out as plain Python integer arithmetic mod 2**64."""
    return sum(int(table[t]) * pow(HASH_BASE_INV, k, M64) for k, t in enumerate(span)) % M64


def test_base_is_invertible():
    assert HASH_BASE % 2 == 1
    assert HASH_BASE * HASH_BASE_INV % M64 == 1


def test_hash_matches_direct_computation():
    idx = EvalNgramIndex(VOCAB, window=16)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, VOCAB, size=64, dtype=np.int64)
    got = idx._window_hashes(tokens)
    assert got.size == 64 - 16 + 1
    for j in (0, 1, 7, 48):
        assert int(got[j]) == _direct(idx.table, tokens[j:j + 16]), f"window at {j}"


def test_hash_is_position_independent():
    """The same span must hash the same wherever it appears, or nothing would ever match."""
    idx = EvalNgramIndex(VOCAB, window=8)
    rng = np.random.default_rng(1)
    span = rng.integers(0, VOCAB, size=8, dtype=np.int64)
    a = idx._window_hashes(np.concatenate([span, rng.integers(0, VOCAB, 20)]))
    b = idx._window_hashes(np.concatenate([rng.integers(0, VOCAB, 13), span]))
    assert int(a[0]) == int(b[-1])


def test_hash_is_order_sensitive():
    """A reversed span is a different span. The old order-insensitive sum failed this."""
    idx = EvalNgramIndex(VOCAB, window=8)
    span = np.arange(8, dtype=np.int64) + 3
    assert int(idx._window_hashes(span)[0]) != int(idx._window_hashes(span[::-1])[0])


def test_short_text_produces_no_hashes():
    idx = EvalNgramIndex(VOCAB, window=16)
    assert idx._window_hashes(np.arange(15, dtype=np.int64)).size == 0
    idx.add_eval_doc(np.arange(15, dtype=np.int64), source="tiny")
    assert idx.n_eval_docs == 1 and idx.n_eval_docs_short == 1 and idx.size == 0


def test_catches_verbatim_overlap_and_ignores_unrelated_text():
    rng = np.random.default_rng(2)
    idx = EvalNgramIndex(VOCAB, window=16)
    item = rng.integers(0, VOCAB, size=40, dtype=np.int64)
    idx.add_eval_doc(item, source="hellaswag")
    idx.build()

    clean = rng.integers(0, VOCAB, size=4000, dtype=np.int64)
    assert idx.contaminated_by(clean) is None

    contaminated = np.concatenate([clean[:1000], item[5:25], clean[1000:]])
    assert idx.contaminated_by(contaminated) == "hellaswag"


def test_shuffled_copy_is_not_flagged():
    """A bag of the same tokens is not a verbatim overlap, and must not be treated as one."""
    rng = np.random.default_rng(3)
    idx = EvalNgramIndex(VOCAB, window=16)
    item = rng.integers(0, VOCAB, size=32, dtype=np.int64)
    idx.add_eval_doc(item, source="piqa")
    idx.build()
    shuffled = item.copy()
    rng.shuffle(shuffled)
    assert idx.contaminated_by(shuffled) is None


def test_attributes_the_matching_source():
    rng = np.random.default_rng(4)
    idx = EvalNgramIndex(VOCAB, window=16)
    a = rng.integers(0, VOCAB, size=30, dtype=np.int64)
    b = rng.integers(0, VOCAB, size=30, dtype=np.int64)
    idx.add_eval_doc(a, source="arc_easy")
    idx.add_eval_doc(b, source="winogrande")
    idx.build()
    assert idx.contaminated_by(a) == "arc_easy"
    assert idx.contaminated_by(b) == "winogrande"


def test_match_reports_where_the_overlap_starts():
    """check_decontam_cost.py decodes the span at this offset, so it has to be the right one."""
    rng = np.random.default_rng(6)
    idx = EvalNgramIndex(VOCAB, window=16)
    item = rng.integers(0, VOCAB, size=16, dtype=np.int64)
    idx.add_eval_doc(item, source="piqa")
    idx.build()
    doc = np.concatenate([rng.integers(0, VOCAB, 700, dtype=np.int64), item,
                          rng.integers(0, VOCAB, 300, dtype=np.int64)])
    source, pos = idx.match(doc)
    assert source == "piqa"
    assert np.array_equal(doc[pos:pos + 16], item)


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    idx = EvalNgramIndex(VOCAB, window=16)
    item = rng.integers(0, VOCAB, size=50, dtype=np.int64)
    idx.add_eval_doc(item, source="wikitext103")
    path = tmp_path / "decontam.npz"
    idx.save(path)

    back = EvalNgramIndex.load(path)
    assert back.window == 16 and back.vocab_size == VOCAB
    assert back.size == idx.size
    assert back.sources == {"wikitext103": 1}
    assert back.contaminated_by(item) == "wikitext103"
    assert back.contaminated_by(rng.integers(0, VOCAB, size=500, dtype=np.int64)) is None


def test_stale_index_is_refused(tmp_path):
    """A decontam.npz from an older hash must fail loudly, not filter nothing."""
    path = tmp_path / "old.npz"
    np.savez_compressed(path, hashes=np.zeros(3, dtype=np.uint64), window=32, vocab_size=VOCAB,
                        n_eval_docs=1)
    try:
        EvalNgramIndex.load(path)
    except ValueError as exc:
        assert "rebuild" in str(exc)
    else:
        raise AssertionError("loading a format-1 index should raise")


def test_extract_texts_pairs_stem_with_each_continuation():
    row = {"question": "Which surface has the most friction?",
           "choices": {"text": ["dry palms", "wet ice"], "label": ["A", "B"]}}
    out = extract_texts(row, ["question", "choices"])
    assert "dry palms" in out
    assert "Which surface has the most friction? dry palms" in out
    assert "Which surface has the most friction? dry palms wet ice" in out

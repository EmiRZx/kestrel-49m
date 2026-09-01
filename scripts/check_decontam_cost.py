"""What does the decontamination filter cost the corpus, at each candidate window?

    python scripts/check_decontam_cost.py --docs 100000 --windows 16 32

`check_decontam_coverage.py` measures the filter's reach: how much of each eval split is long enough
to be seen at all. This measures the price of that reach -- what fraction of real training documents
each window drops, and which benchmark each drop matched. A window short enough to catch a two-word
answer choice is also short enough to fire on ordinary English, and which effect dominates is a
question about the corpus, not about the hash.

Read the output per corpus source, not only in total. Wikipedia overlapping WikiText-103 is not a
false positive -- WikiText-103 *is* Wikipedia, so that overlap is the contamination the filter exists
to remove. FineWeb-Edu matching ARC-Easy on sixteen tokens is the case worth inspecting, which is why
--show decodes the matched spans back to text.

Every window scores the same documents in one streaming pass, and the eval splits are tokenized once
and fed to every index, so a second window costs one more searchsorted per document.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from kestrel.corpus import SOURCES  # noqa: E402
from kestrel.decontaminate import EVAL_SOURCES, EvalNgramIndex, extract_texts  # noqa: E402

BATCH = 512


def build_indexes(tok, vocab: int, windows: list[int]) -> dict[int, EvalNgramIndex]:
    """One pass over the eval splits, feeding every window's index from the same tokenization."""
    idxs = {w: EvalNgramIndex(vocab, window=w) for w in windows}
    for spec in EVAL_SOURCES:
        t0 = time.perf_counter()
        ds = load_dataset(spec["path"], name=spec.get("config"), split=spec["split"])
        texts = [t for row in ds for t in extract_texts(row, spec["fields"])]
        for enc in tok.encode_batch_fast(texts):
            arr = np.asarray(enc.ids, dtype=np.int64)
            for idx in idxs.values():
                idx.add_eval_doc(arr, source=spec["name"])
        print(f"  {spec['name']:<18} {len(texts):>7,} texts  ({time.perf_counter() - t0:.0f}s)")
    for w, idx in idxs.items():
        idx.build()
        print(f"  window {w:>2}: {idx.size:>10,} hashes   {idx.n_eval_docs_short:>7,} of "
              f"{idx.n_eval_docs:,} eval texts too short to hash")
    return idxs


def scan(src, tok, idxs: dict[int, EvalNgramIndex], n_docs: int, min_chars: int, show: int,
         samples: dict) -> dict:
    """Stream up to n_docs documents from one corpus source past every window's filter."""
    ds = load_dataset(src.path, name=src.config, split=src.split, streaming=True)
    drops: dict[int, dict[str, int]] = {w: {} for w in idxs}
    seen, tokens = 0, 0
    buf: list[str] = []
    t0 = time.perf_counter()
    for row in ds:
        text = row.get(src.text_field) or ""
        if len(text) < min_chars:          # the same threshold prepare_data.py applies
            continue
        buf.append(text)
        if len(buf) < BATCH and seen + len(buf) < n_docs:
            continue
        for enc in tok.encode_batch_fast(buf):
            ids = np.asarray(enc.ids, dtype=np.int64)
            tokens += ids.size
            for w, idx in idxs.items():
                m = idx.match(ids)
                if m is None:
                    continue
                bench, pos = m
                drops[w][bench] = drops[w].get(bench, 0) + 1
                if len(samples.setdefault((w, bench), [])) < show:
                    samples[(w, bench)].append((src.name, tok.decode(enc.ids[pos:pos + w])))
        seen += len(buf)
        buf = []
        print(f"\r  {src.name}: {seen:>8,}/{n_docs:,} docs  {tokens / 1e6:>7.1f}M tokens  "
              f"{seen / max(time.perf_counter() - t0, 1e-6):>6,.0f} docs/s", end="", flush=True)
        if seen >= n_docs:
            break
    print()
    return {"seen": seen, "tokens": tokens, "drops": drops}


def report(windows: list[int], results: dict, benches: list[str]) -> None:
    head = "  ".join(f"{b[:11]:>11}" for b in benches)
    print(f"\ndocuments dropped as contaminated\n")
    print(f"{'win':<5}{'source':<16}{'docs':>10}  {head}  {'total':>7}  {'rate':>8}")
    for w in windows:
        for name, res in results.items():
            d = res["drops"][w]
            tot = sum(d.values())
            cells = "  ".join(f"{d.get(b, 0):>11,}" for b in benches)
            print(f"{w:<5}{name:<16}{res['seen']:>10,}  {cells}  {tot:>7,}  "
                  f"{tot / max(res['seen'], 1):>7.4%}")
        tot = sum(sum(res["drops"][w].values()) for res in results.values())
        seen = sum(res["seen"] for res in results.values())
        print(f"{w:<5}{'all':<16}{seen:>10,}  {' ' * len(head)}  {tot:>7,}  "
              f"{tot / max(seen, 1):>7.4%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--docs", type=int, default=50_000, help="documents per corpus source")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--min-chars", type=int, default=200, help="same filter as prepare_data.py")
    ap.add_argument("--show", type=int, default=4, help="matched spans to decode per window/bench")
    args = ap.parse_args()

    tok_path = ROOT / "data" / "tokenizer" / f"vocab{args.vocab_size}" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    windows = sorted(set(args.windows))

    print(f"indexing the scored eval splits at windows {windows}")
    idxs = build_indexes(tok, tok.get_vocab_size(), windows)

    print(f"\nstreaming up to {args.docs:,} documents per corpus source past every window")
    samples: dict[tuple[int, str], list[tuple[str, str]]] = {}
    results = {src.name: scan(src, tok, idxs, args.docs, args.min_chars, args.show, samples)
               for src in SOURCES}

    report(windows, results, [s["name"] for s in EVAL_SOURCES])

    if samples:
        print("\nmatched spans: a benchmark item is a true positive, a stock phrase is not")
        for (w, bench), items in sorted(samples.items()):
            for src_name, span in items:
                print(f"  w={w:<3} {bench:<16} {src_name:<12} {span.strip()[:110]!r}")


if __name__ == "__main__":
    main()

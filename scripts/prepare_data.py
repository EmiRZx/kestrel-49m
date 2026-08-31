"""Stream the corpus, decontaminate it, tokenize it, and write uint16 shards.

    python scripts/prepare_data.py --tokens 6e9 --vocab-size 16384
    python scripts/prepare_data.py --tokens 3e8 --out-dir data/tokens_ablation   # ablation slice

Produces, under --out-dir:
    train_*.bin / train_manifest.json      the training corpus
    val_*.bin   / val_manifest.json        held-out slice of the same mix
    wt103_*.bin / wt103_manifest.json      WikiText-103 validation text, for the scored metric
    decontam.npz                           the window-hash index, so the filter is auditable
    prepare_report.json                    counts, drop rates, timings -> straight into the README
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.corpus import EOT_TOKEN, SOURCES  # noqa: E402
from kestrel.data import TokenShardWriter  # noqa: E402
from kestrel.decontaminate import EVAL_SOURCES, EvalNgramIndex, extract_texts  # noqa: E402


def load_tokenizer(vocab_size: int, tokenizer_dir: Path):
    from tokenizers import Tokenizer

    path = tokenizer_dir / f"vocab{vocab_size}" / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/train_tokenizer.py first")
    tok = Tokenizer.from_file(str(path))
    eot = tok.token_to_id(EOT_TOKEN)
    if eot is None:
        raise ValueError(f"{EOT_TOKEN} not in tokenizer {path}")
    return tok, eot


def encode_batch(tok, texts: list[str]) -> list[list[int]]:
    fast = getattr(tok, "encode_batch_fast", None)  # skips offset bookkeeping we never use
    encs = fast(texts) if fast else tok.encode_batch(texts)
    return [e.ids for e in encs]


def build_decontam_index(tok, vocab_size: int, window: int,
                         allow_missing: bool = False) -> EvalNgramIndex:
    """Hash every window in the evaluation splits that judges actually score on."""
    from datasets import load_dataset

    index = EvalNgramIndex(vocab_size=vocab_size, window=window)
    failures: list[str] = []
    for spec in EVAL_SOURCES:
        t0 = time.perf_counter()
        try:
            ds = load_dataset(spec["path"], name=spec.get("config"), split=spec["split"])
        except Exception as exc:  # a missing benchmark must be loud, not silent
            print(f"  !! {spec['name']}: could not load ({type(exc).__name__}: {exc})")
            failures.append(f"{spec['name']} ({spec['path']})")
            continue
        texts: list[str] = []
        for row in ds:
            texts.extend(extract_texts(row, spec["fields"]))
        for ids in encode_batch(tok, texts):
            index.add_eval_doc(np.asarray(ids, dtype=np.int64), source=spec["name"])
        print(f"  {spec['name']:<18} {len(texts):>7,} texts  ({time.perf_counter()-t0:.0f}s)")
    if failures and not allow_missing:
        # Saving a partial index would silently train on text the judges score. Refuse.
        raise SystemExit(
            "decontamination index incomplete -- could not load: " + ", ".join(failures) +
            "\nFix the dataset paths in kestrel/decontaminate.py, or pass --allow-missing-eval "
            "if you accept an incomplete filter (never for a submitted run).")
    index.build()
    print(f"  index: {index.size:,} unique {window}-token window hashes "
          f"from {index.n_eval_docs:,} texts")
    return index


def _flush(texts: list[str], tok, eot: int, index: EvalNgramIndex | None,
           writer: TokenShardWriter, stats: dict, token_budget: int, t0: float,
           label: str) -> tuple[dict, bool]:
    """Tokenize one batch, drop contaminated documents, append the rest. Returns (stats, done)."""
    out: list[np.ndarray] = []
    for ids in encode_batch(tok, texts):
        arr = np.asarray(ids, dtype=np.int64)
        if index is not None and index.is_contaminated(arr):
            stats["docs_contaminated"] += 1
            continue
        out.append(arr)
        out.append(np.array([eot], dtype=np.int64))
        stats["docs_kept"] += 1
    if out:
        tokens = np.concatenate(out)
        remaining = token_budget - stats["tokens"]
        if tokens.size > remaining:
            tokens = tokens[:remaining]
        writer.write(tokens)
        stats["tokens"] += int(tokens.size)

    done = stats["tokens"] >= token_budget
    dt = max(time.perf_counter() - t0, 1e-6)
    print(f"\r  {label}: {stats['tokens']/1e6:8.1f}M / {token_budget/1e6:.0f}M tokens  "
          f"{stats['docs_kept']:>9,} docs  {stats['docs_contaminated']:>6,} contaminated  "
          f"{stats['tokens']/dt/1e6:5.2f}M tok/s", end="", flush=True)
    if done:
        print()
    return stats, done


def stream_source(src, tok, eot: int, index: EvalNgramIndex | None, token_budget: int,
                  writer: TokenShardWriter, skip_docs: int = 0, min_chars: int = 200,
                  batch_size: int = 1024, label: str = "") -> dict:
    """Tokenize documents from one source into `writer` until its token budget is met."""
    from datasets import load_dataset

    ds = load_dataset(src.path, name=src.config, split=src.split, streaming=True)
    stats = dict(docs_seen=0, docs_kept=0, docs_short=0, docs_contaminated=0, tokens=0)
    buf: list[str] = []
    t0 = time.perf_counter()
    skipped = 0

    for row in ds:
        text = row.get(src.text_field) or ""
        stats["docs_seen"] += 1
        if len(text) < min_chars:
            stats["docs_short"] += 1
            continue
        if skipped < skip_docs:
            skipped += 1
            continue
        buf.append(text)
        if len(buf) < batch_size:
            continue
        stats, done = _flush(buf, tok, eot, index, writer, stats, token_budget, t0, label)
        buf = []
        if done:
            return stats
    if buf:
        stats, _ = _flush(buf, tok, eot, index, writer, stats, token_budget, t0, label)
    return stats


def write_wikitext_val(tok, eot: int, out_dir: Path) -> dict:
    """WikiText-103 *validation* text, for tracking the scored perplexity metric during training.

    The test split is deliberately untouched: it is what the judges score, and what the
    decontamination pass removes from training. Using validation here keeps the two separate.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", name="wikitext-103-raw-v1", split="validation")
    text = "\n".join(r["text"] for r in ds)
    ids = np.asarray(tok.encode(text).ids + [eot], dtype=np.int64)
    with TokenShardWriter(out_dir, "wt103", tokens_per_shard=10**9) as w:
        w.write(ids)
    print(f"  wt103 validation: {ids.size:,} tokens")
    return {"tokens": int(ids.size), "chars": len(text)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=6e9, help="total training tokens to write")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--val-tokens", type=float, default=5e6)
    ap.add_argument("--window", type=int, default=32, help="decontamination window, in tokens")
    ap.add_argument("--no-decontam", action="store_true", help="ablation only; never for a real run")
    ap.add_argument("--allow-missing-eval", action="store_true",
                    help="proceed even if an eval split fails to load; never for a submitted run")
    ap.add_argument("--tokenizer-dir", type=Path, default=ROOT / "data" / "tokenizer")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    args = ap.parse_args()

    out_dir = args.out_dir or ROOT / "data" / f"tokens_v{args.vocab_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = int(args.tokens)
    val_total = int(args.val_tokens)

    tok, eot = load_tokenizer(args.vocab_size, args.tokenizer_dir)
    print(f"tokenizer vocab {tok.get_vocab_size():,}, eot id {eot}")
    print(f"target {total/1e9:.2f}B train tokens + {val_total/1e6:.0f}M val, mix: "
          + ", ".join(f"{s.name} {s.weight:.0%}" for s in SOURCES))

    index = None
    if not args.no_decontam:
        print("\nbuilding decontamination index from scored eval splits")
        index_path = out_dir / "decontam.npz"
        if index_path.exists():
            index = EvalNgramIndex.load(index_path)
            print(f"  reused {index_path.name}: {index.size:,} window hashes")
        else:
            index = build_decontam_index(tok, tok.get_vocab_size(), args.window,
                                        allow_missing=args.allow_missing_eval)
            index.save(index_path)

    report = {
        "vocab_size": tok.get_vocab_size(),
        "target_tokens": total,
        "decontam": {"enabled": index is not None,
                     "window": args.window,
                     "hashes": index.size if index else 0,
                     "eval_texts": index.n_eval_docs if index else 0,
                     "by_source": index.sources if index else {}},
        "sources": {},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\nvalidation split (held out, same mix)")
    with TokenShardWriter(out_dir, "val", args.shard_tokens) as vw:
        for src in SOURCES:
            stream_source(src, tok, eot, index, int(val_total * src.weight), vw,
                          label=f"val/{src.name}")

    print("\nwikitext-103 validation split")
    report["wt103"] = write_wikitext_val(tok, eot, out_dir)

    print("\ntraining split")
    t0 = time.perf_counter()
    with TokenShardWriter(out_dir, "train", args.shard_tokens) as tw:
        for src in SOURCES:
            # skip_docs steps past the documents already consumed by the validation split
            val_docs = max(1000, int(val_total * src.weight / 400))
            stats = stream_source(src, tok, eot, index, int(total * src.weight), tw,
                                  skip_docs=val_docs, label=f"train/{src.name}")
            report["sources"][src.name] = {
                **stats,
                "path": src.path, "config": src.config, "license": src.license,
                "weight": src.weight,
            }
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    report["train_tokens"] = sum(s["tokens"] for s in report["sources"].values())
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    kept = sum(s["docs_kept"] for s in report["sources"].values())
    dropped = sum(s["docs_contaminated"] for s in report["sources"].values())
    print(f"\nwrote {report['train_tokens']/1e9:.3f}B train tokens to {out_dir}")
    print(f"  {kept:,} documents kept, {dropped:,} dropped as contaminated "
          f"({dropped/max(kept+dropped,1):.4%})")
    print(f"  {report['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()

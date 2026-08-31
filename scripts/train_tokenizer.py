"""Train a byte-level BPE tokenizer on our own corpus mix.

Why train one instead of reusing GPT-2's or Llama's: the 50M cap counts the embedding table, and at
d_model 384 a 32k vocabulary costs 12.6M parameters -- a quarter of the entire budget spent on a
lookup table. A 16k vocabulary fitted to our own mix costs 6.3M and frees the difference for layers.
The cost is more tokens per document, i.e. less text seen per FLOP. Which way that trade lands is
measured, not assumed: `--vocab-size` is one of the ablation axes.

    python scripts/train_tokenizer.py --vocab-size 16384
    python scripts/train_tokenizer.py --vocab-size 8192 --sample-docs 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.corpus import EOT_TOKEN, SOURCES  # noqa: E402

# The cl100k_base split pattern. The clause that matters most for us is `\p{N}{1,3}`: it caps number
# tokens at three digits, so arithmetic is represented consistently instead of the model having to
# learn that "1234" and "12" "34" are the same kind of object. Contractions and leading-space
# handling come along for free.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def sample_iterator(sample_docs: int, min_chars: int = 200):
    """Stream documents from every source, in proportion to its weight in the final mix."""
    from datasets import load_dataset

    for src in SOURCES:
        quota = max(1, int(sample_docs * src.weight))
        print(f"  {src.name}: streaming {quota:,} docs from {src.path}"
              f"{'/' + src.config if src.config else ''}", flush=True)
        ds = load_dataset(src.path, name=src.config, split=src.split, streaming=True)
        taken = 0
        for row in ds:
            text = row.get(src.text_field) or ""
            if len(text) < min_chars:
                continue
            yield text
            taken += 1
            if taken >= quota:
                break
        print(f"  {src.name}: {taken:,} docs", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-docs", type=int, default=400_000,
                    help="documents streamed for BPE fitting; 400k is ample for a 16k vocab")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "tokenizer")
    args = ap.parse_args()

    out_dir = args.out_dir / f"vocab{args.vocab_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[EOT_TOKEN],          # id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"training BPE, vocab_size={args.vocab_size}, sample_docs={args.sample_docs:,}")
    t0 = time.perf_counter()
    tok.train_from_iterator(sample_iterator(args.sample_docs), trainer=trainer)
    dt = time.perf_counter() - t0

    path = out_dir / "tokenizer.json"
    tok.save(str(path))

    eot_id = tok.token_to_id(EOT_TOKEN)
    probe = ("The mitochondrion is the powerhouse of the cell. "
             "In 1847, 3.14159 kilograms cost $12,345.67 -- isn't that odd?")
    ids = tok.encode(probe).ids
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": args.vocab_size,
        "eot_token": EOT_TOKEN,
        "eot_id": eot_id,
        "sample_docs": args.sample_docs,
        "split_pattern": SPLIT_PATTERN,
        "train_seconds": round(dt, 1),
        "sources": [{"name": s.name, "path": s.path, "config": s.config, "weight": s.weight}
                    for s in SOURCES],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nsaved {path}  ({dt:.0f}s)")
    print(f"  vocab size   {tok.get_vocab_size():,}   (eot id {eot_id})")
    print(f"  probe        {len(ids)} tokens for {len(probe)} chars "
          f"= {len(probe)/len(ids):.2f} chars/token")
    print(f"  round-trip   {'ok' if tok.decode(ids) == probe else 'MISMATCH'}")
    print(f"  pieces       {[tok.id_to_token(i) for i in ids[:24]]}")


if __name__ == "__main__":
    main()

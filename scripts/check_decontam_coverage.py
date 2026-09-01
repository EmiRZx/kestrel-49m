"""How much of each scored eval split is long enough to be caught by the decontamination filter?

    python scripts/check_decontam_coverage.py --window 32

A window-hash filter can only see a text of at least `window` tokens. If most benchmark items are
shorter than that, the filter is quietly narrower than the README claims, so measure it rather than
asserting it. This produced the coverage table in PLAN.md; re-run it if the window, the field
extraction, or the tokenizer changes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from kestrel.decontaminate import EVAL_SOURCES, extract_texts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--vocab-size", type=int, default=16384)
args = ap.parse_args()

tok_path = ROOT / "data" / "tokenizer" / f"vocab{args.vocab_size}" / "tokenizer.json"
tok = Tokenizer.from_file(str(tok_path))

print(f"window {args.window} tokens, vocab {args.vocab_size}\n")
print(f"{'source':<18} {'texts':>8} {'>=window':>9} {'cover':>6} {'med':>6} {'p90':>6}")
for spec in EVAL_SOURCES:
    ds = load_dataset(spec["path"], name=spec.get("config"), split=spec["split"])
    texts = [t for row in ds for t in extract_texts(row, spec["fields"])]
    lens = np.array([len(e.ids) for e in tok.encode_batch_fast(texts)])
    long = int((lens >= args.window).sum())
    print(f"{spec['name']:<18} {len(texts):>8,} {long:>9,} {long/max(len(texts),1):>5.0%} "
          f"{int(np.median(lens)):>6} {int(np.percentile(lens, 90)):>6}")

"""How much of each scored eval split is long enough to be caught by the decontamination filter?

    python scripts/check_decontam_coverage.py --windows 8 12 16 24 32

A hash-based filter can only see a text of at least `window` tokens. If most benchmark items are
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
ap.add_argument("--windows", type=int, nargs="+", default=[8, 12, 16, 24, 32])
ap.add_argument("--vocab-size", type=int, default=16384)
args = ap.parse_args()

tok_path = ROOT / "data" / "tokenizer" / f"vocab{args.vocab_size}" / "tokenizer.json"
tok = Tokenizer.from_file(str(tok_path))

wins = args.windows
head = "  ".join(f">={w:<4}" for w in wins)
print(f"fraction of scored strings at least N tokens long (vocab {args.vocab_size})\n")
print(f"{'source':<18} {'texts':>8}  {head}  {'med':>5} {'p90':>6}")
for spec in EVAL_SOURCES:
    ds = load_dataset(spec["path"], name=spec.get("config"), split=spec["split"])
    texts = [t for row in ds for t in extract_texts(row, spec["fields"])]
    lens = np.array([len(e.ids) for e in tok.encode_batch_fast(texts)])
    cov = "  ".join(f"{(lens >= w).mean():>5.0%}" for w in wins)
    print(f"{spec['name']:<18} {len(texts):>8,}  {cov}  {int(np.median(lens)):>5} "
          f"{int(np.percentile(lens, 90)):>6}")

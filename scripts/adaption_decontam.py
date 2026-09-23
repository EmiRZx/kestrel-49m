"""Measure how much of a synthetic corpus overlaps the evaluation sets, before it is trained on.

    python scripts/adaption_decontam.py data/synthetic/kestrel-physical-pilot.json

Synthetic data is the one source where contamination is *plausible rather than accidental*. A hosted
model asked for "everyday physical reasoning" has very likely read PIQA, and can reproduce items it
memorised without being asked to. Web-scraped text can contain a benchmark item by coincidence; a
generator can emit one because that is what the topic looks like to it.

So this runs the project's own `EvalNgramIndex` -- the same 16-token order-sensitive hashing used on
the 6 B-token corpus, against the same `EVAL_SOURCES` -- over the generated rows, and reports the
drop rate per benchmark. Two outcomes are both useful:

  * A low rate means the corpus can be used, with a measured number to put in the README rather than
    an assurance.
  * A high rate means the prompt is pulling the generator toward benchmark items and the prompt has
    to change, which is worth knowing at 200 rows rather than at 8,000.

The check is deliberately the same instrument as production, not a second implementation of it. A
bespoke checker that happened to be laxer would produce a number that means nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.decontaminate import DEFAULT_WINDOW, EvalNgramIndex  # noqa: E402

# Fields an `instruction_dataset` row can carry. Every one of them becomes training text, so every
# one of them is checked -- an item leaking through the completion is no better than one leaking
# through the prompt.
TEXT_KEYS = ("text", "output", "completion", "response", "answer", "instruction", "input",
             "prompt", "context", "content", "passage")


def iter_rows(data) -> list[dict]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    for k in ("rows", "data", "records", "items", "dataset"):
        v = data.get(k)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


def row_text(row: dict) -> str:
    """Concatenate every text-bearing field, which is what training would see."""
    parts = [str(row[k]) for k in TEXT_KEYS if isinstance(row.get(k), str) and row[k].strip()]
    if not parts:  # unknown schema: fall back to every string value
        parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 40]
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="downloaded Adaption dataset json")
    ap.add_argument("--index", type=Path, default=ROOT / "data" / "tokens_pilot" / "decontam.npz")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer" / "vocab16384" / "tokenizer.json")
    ap.add_argument("--show", type=int, default=3, help="print this many matching spans")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    index = EvalNgramIndex.load(args.index)
    print(f"index: {index.size():,} hashes, window {getattr(index, 'window', DEFAULT_WINDOW)}")

    rows = iter_rows(json.loads(args.path.read_text(encoding="utf-8")))
    if not rows:
        sys.exit(f"no rows found in {args.path}")
    print(f"rows : {len(rows):,} from {args.path.name}\n")

    hits: Counter = Counter()
    lengths: list[int] = []
    shown = 0
    flagged: list[dict] = []

    for i, row in enumerate(rows):
        text = row_text(row)
        if not text.strip():
            continue
        ids = np.asarray(tok.encode(text).ids, dtype=np.int64)
        lengths.append(len(ids))
        src = index.contaminated_by(ids)
        if src:
            hits[src] += 1
            flagged.append({"row": i, "source": src, "chars": len(text)})
            if shown < args.show:
                m = index.match(ids)
                span = ""
                if m:
                    _, pos = m
                    w = getattr(index, "window", DEFAULT_WINDOW)
                    span = tok.decode(ids[pos:pos + w].tolist())
                print(f"  HIT row {i} <- {src}: {span!r}")
                shown += 1

    n = len(lengths)
    total_tokens = int(np.sum(lengths)) if lengths else 0
    n_hits = sum(hits.values())
    print(f"\n{'contaminated':<16}{n_hits:>8} of {n} rows   ({n_hits/max(n,1):.3%})")
    for src, c in hits.most_common():
        print(f"  {src:<14}{c:>8}")
    print(f"\n{'tokens total':<16}{total_tokens:>12,}")
    print(f"{'tokens/row':<16}{(total_tokens/max(n,1)):>12,.0f} mean, "
          f"{int(np.median(lengths)) if lengths else 0:,} median")
    print(f"{'min / max':<16}{(min(lengths) if lengths else 0):>12,} / {(max(lengths) if lengths else 0):,}")

    # What the full budget would buy at this row size, which is the number that decides the plan.
    if n:
        per_row = total_tokens / n
        print(f"\nat {per_row:,.0f} tokens/row, 8,500 rows -> {per_row*8500/1e6:.2f} M tokens")

    report = {"path": str(args.path), "rows": n, "contaminated": n_hits,
              "rate": n_hits / max(n, 1), "by_source": dict(hits),
              "tokens_total": total_tokens, "tokens_per_row_mean": total_tokens / max(n, 1),
              "tokens_per_row_median": int(np.median(lengths)) if lengths else 0,
              "flagged": flagged[:50]}
    out = args.out or ROOT / "runs" / f"decontam_{args.path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

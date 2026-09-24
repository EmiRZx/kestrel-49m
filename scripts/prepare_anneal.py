"""Tokenize the synthetic corpus into shards the trainer can read.

    python scripts/prepare_anneal.py                       # all mixtures found, 90/10 train/val
    python scripts/prepare_anneal.py --only physical

Writes `data/anneal_synth/` in the same flat uint16 `.bin` + manifest format `prepare_data.py`
produces, so `TokenDataset` reads it with no special case and the anneal is an ordinary training run
pointed at a different directory.

Two decisions here are load-bearing.

*This writes the synthetic half only.* 6.3 M tokens of one generator's prose, annealed on alone,
is a strong pull toward a single register -- and the benchmark most likely to pay for that is
ARC-Easy, which Kestrel currently wins and which rewards the educational corpus this text is
nothing like. So the anneal must mix these shards with the original 6 B-token corpus rather than
run on them by themselves. That corpus lives on the Modal Volume, not on this laptop, so the mixing
happens there: `scripts/modal_anneal.py` mounts both and sets the ratio. The ratio is the first
thing worth sweeping if the result disappoints.

*Prompt and completion are concatenated, not scored separately.* This is pretraining, not
instruction tuning: there is no loss mask, no chat template, and no special tokens marking a turn.
The prompt is a sentence of ordinary English that sets up the passage, so it is perfectly good
training text, and `\n\n` between the two is the same separator the passages use internally. Adding
a template here would teach the model a format that appears nowhere in its pretraining and nowhere
in the benchmarks.

Documents are separated by EOT exactly as in `prepare_data.py`, so a passage cannot bleed into the
next one across a sequence boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.data import TokenShardWriter  # noqa: E402


def load_rows(path: Path) -> list[str]:
    """One training document per generated row: prompt then completion."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        p = (r.get("enhanced_prompt") or "").strip()
        c = (r.get("enhanced_completion") or "").strip()
        if not c:
            continue
        out.append(f"{p}\n\n{c}" if p else c)
    return out


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "anneal_synth")
    ap.add_argument("--tokenizer", type=Path,
                    default=ROOT / "data" / "tokenizer" / "vocab16384" / "tokenizer.json")
    ap.add_argument("--only", default=None, help="one mixture name instead of all")
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="held out so the anneal has its own val curve, separate from pretraining's")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from kestrel.corpus import EOT_TOKEN

    tok = Tokenizer.from_file(str(args.tokenizer))
    eot = tok.token_to_id(EOT_TOKEN)
    if eot is None:
        # Same refusal as `prepare_data.py`, for the same reason: guessing an id here would write a
        # document separator that is some ordinary word, and every sequence boundary in the anneal
        # corpus would be silently wrong.
        raise ValueError(f"{EOT_TOKEN} not in tokenizer {args.tokenizer}")

    files = sorted(args.synth_dir.glob("kestrel-*-main.json"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        sys.exit(f"no corpora found in {args.synth_dir}")

    docs: list[str] = []
    per_source: dict[str, int] = {}
    for f in files:
        rows = load_rows(f)
        docs.extend(rows)
        per_source[f.stem] = len(rows)
        print(f"{f.name}: {len(rows):,} documents")

    # Interleave sources rather than concatenating them, so any shard-order effect cannot line up
    # with a source boundary and a truncated run still sees both mixtures.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(docs))
    docs = [docs[i] for i in order]

    n_val = max(1, int(len(docs) * args.val_frac))
    splits = {"val": docs[:n_val], "train": docs[n_val:]}
    print(f"\n{len(splits['train']):,} train / {len(splits['val']):,} val documents")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"sources": per_source, "seed": args.seed, "val_frac": args.val_frac, "splits": {}}

    for split, texts in splits.items():
        with TokenShardWriter(args.out, split) as w:
            for i, text in enumerate(texts):
                ids = tok.encode(text).ids
                w.write(np.asarray(ids + [eot], dtype=np.uint16))
                if (i + 1) % 1000 == 0:
                    print(f"  {split}: {i+1:,}/{len(texts):,} docs, {w.total:,} tokens", flush=True)
            manifest = w.finalize({"source": "adaption-synthetic", "documents": len(texts)})
        report["splits"][split] = {"documents": len(texts), "tokens": manifest["total_tokens"]}
        print(f"{split}: {manifest['total_tokens']:,} tokens in {len(manifest['shards'])} shard(s)")

    (args.out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

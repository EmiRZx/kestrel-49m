"""Score a checkpoint on the five metrics the competition scores.

    python scripts/eval.py --ckpt runs/kestrel-main/best.pt
    python scripts/eval.py --ckpt runs/kestrel-main/last.pt --limit 200   # fast sanity check
    python scripts/eval.py --ckpt ... --tasks hellaswag,wikitext --sample "The capital of France"

`--limit` subsamples each task, which is the right tool mid-training: it turns a 40-minute full run
into a couple of minutes and still shows movement. Final numbers for the README come from a run with
no `--limit`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.evaluate import (KestrelLM, SCORED_TASKS, format_table, load_checkpoint,  # noqa: E402
                              run_eval, save_results)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="tokenizer.json; inferred from the checkpoint's vocab size if omitted")
    ap.add_argument("--tasks", default=",".join(SCORED_TASKS))
    ap.add_argument("--limit", type=float, default=None, help="docs per task (int) or fraction")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample", default=None, help="also print a greedy continuation of this prompt")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    model, ckpt = load_checkpoint(args.ckpt, args.device)
    report = model.param_report()
    tok_path = args.tokenizer or (ROOT / "data" / "tokenizer" /
                                  f"vocab{model.cfg.vocab_size}" / "tokenizer.json")
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError(f"tokenizer has {tok.get_vocab_size()} tokens but the model expects "
                         f"{model.cfg.vocab_size}; wrong --tokenizer?")

    trained = ckpt.get("state", {})
    print(f"{args.ckpt}: {report['total']/1e6:.2f}M params, "
          f"{trained.get('tokens', 0)/1e9:.3f}B tokens trained, "
          f"{trained.get('gpu_seconds', 0)/3600:.2f} GPU-hours")

    lm = KestrelLM(model, tok, device=args.device, batch_size=args.batch_size)
    limit = int(args.limit) if args.limit and args.limit >= 1 else args.limit
    t0 = time.perf_counter()
    results = run_eval(lm, tasks=args.tasks.split(","), limit=limit,
                       num_fewshot=args.num_fewshot)
    elapsed = time.perf_counter() - t0

    out = args.out or args.ckpt.parent / f"eval_{args.ckpt.stem}.json"
    summary = save_results(results, out)
    print("\n" + format_table(summary))
    print(f"\n{elapsed/60:.1f} min -> {out}")

    if args.sample:
        class _Req:  # the harness passes Instance objects; generate_until only reads .args
            def __init__(self, args_):
                self.args = args_

        text = lm.generate_until([_Req((args.sample, {"max_gen_toks": 80}))])[0]
        print(f"\nprompt: {args.sample!r}\nsample: {text!r}")


if __name__ == "__main__":
    main()

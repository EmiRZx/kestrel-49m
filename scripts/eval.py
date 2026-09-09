"""Score checkpoints on the five metrics the competition scores.

    python scripts/eval.py --ckpt runs/kestrel-main/best.pt
    python scripts/eval.py --ckpt runs/kestrel-main/last.pt --limit 200   # fast sanity check
    python scripts/eval.py --ckpt ... --tasks hellaswag,wikitext --sample "The capital of France"
    python scripts/eval.py --ckpt runs/abl-arch-*/best.pt                 # one row per checkpoint

`--limit` subsamples each task, which is the right tool mid-training: it turns a full run into a
couple of minutes and still shows movement. It takes the *first* n documents rather than a random
sample, so a limited number is for watching one model's trend and never for comparing two: at
`--limit 50` the standard error on an accuracy is about 0.07, wide enough to reorder an entire sweep.
Numbers that go in the README come from a run with no `--limit`.

Given several checkpoints, each is scored and saved as it finishes -- a crash in the fourth does not
cost the first three -- and a comparison table is printed at the end.

Scoring defaults to fp32 rather than the bf16 that training used, because a reported number should
not depend on `--batch-size`: see `KestrelLM._autocast` for the measurement that settled it.
`--precision bf16` is about twice as fast and is the right choice for watching a run in progress.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.evaluate import (CHANCE, KestrelLM, SCORED_TASKS, format_table,  # noqa: E402
                              load_checkpoint, run_eval, save_results)


def score_one(path: Path, args: argparse.Namespace) -> dict:
    """Score one checkpoint, write its json, and return the summary."""
    from tokenizers import Tokenizer

    model, ckpt = load_checkpoint(path, args.device)
    report = model.param_report()
    tok_path = args.tokenizer or (ROOT / "data" / "tokenizer" /
                                  f"vocab{model.cfg.vocab_size}" / "tokenizer.json")
    tok = Tokenizer.from_file(str(tok_path))
    if tok.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError(f"tokenizer has {tok.get_vocab_size()} tokens but the model expects "
                         f"{model.cfg.vocab_size}; wrong --tokenizer?")

    trained = ckpt.get("state", {})
    print(f"\n=== {path} " + "=" * max(4, 60 - len(str(path))))
    print(f"{report['total']/1e6:.2f}M params, {trained.get('tokens', 0)/1e9:.3f}B tokens trained, "
          f"{trained.get('gpu_seconds', 0)/3600:.2f} GPU-hours", flush=True)

    lm = KestrelLM(model, tok, device=args.device, batch_size=args.batch_size,
                   amp_dtype=None if args.precision == "fp32" else torch.bfloat16)
    limit = int(args.limit) if args.limit and args.limit >= 1 else args.limit
    t0 = time.perf_counter()
    results = run_eval(lm, tasks=args.tasks.split(","), limit=limit, num_fewshot=args.num_fewshot)
    elapsed = time.perf_counter() - t0

    # A subsampled score must never land in the filename a full one owns: `eval_best.json` has to
    # mean the real thing, or a --limit run left behind after a crash becomes a README number.
    tag = "" if not limit else f"_limit{limit:g}"
    out = (args.out if args.out and len(args.ckpt) == 1
           else path.parent / f"eval_{path.stem}{tag}.json")
    summary = save_results(results, out)
    print("\n" + format_table(summary))
    print(f"\n{elapsed/60:.1f} min -> {out}", flush=True)

    if args.sample:
        class _Req:  # the harness passes Instance objects; generate_until only reads .args
            def __init__(self, args_):
                self.args = args_

        text = lm.generate_until([_Req((args.sample, {"max_gen_toks": 80}))])[0]
        print(f"\nprompt: {args.sample!r}\nsample: {text!r}")

    summary["_meta"] = {"name": path.parent.name, "params": report["total"],
                        "tokens": trained.get("tokens", 0),
                        "gpu_hours": trained.get("gpu_seconds", 0) / 3600}
    del model, lm
    torch.cuda.empty_cache()
    return summary


def compare(summaries: list[dict], tasks: list[str]) -> None:
    """One row per checkpoint, with the standard errors quoted once in a footer.

    The footer is one line rather than a column per cell because a task's standard error is set by
    how many documents it has, not by which model answered them -- HellaSwag's 10,042 items give
    every cell the same +-0.004 and WinoGrande's 1,267 give every cell the same +-0.014.

    That bound is the *unpaired* one: each model's sampling error on its own score. Two models
    answering the same items can be separated far more tightly, because item difficulty cancels in
    the difference, exactly as block difficulty does for val loss. Quoting the crude bound first is
    deliberate -- if the gaps are wide against it there is nothing left to refine, and if they are
    not, that is the signal to go and do the paired test rather than to guess.

    The perplexity column gets no error bar at all, because the harness reports none for
    `word_perplexity`. Two decimal places on a corpus-level ratio over 62 documents is not the sharp
    instrument it looks like; the footer says what a difference there is actually worth.
    """
    acc = [t for t in tasks if t in CHANCE]
    head = f"{'run':<22}" + "".join(f"{t:>11}" for t in acc)
    if "wikitext" in tasks:
        head += f"{'wt103 ppl':>11}"
    head += f"{'mean over chance':>18}" if acc else ""
    print("\n" + head)
    print(f"{'(chance)':<22}" + "".join(f"{CHANCE[t]:>11.2f}" for t in acc))
    print("-" * len(head))
    for s in summaries:
        row = f"{s['_meta']['name']:<22}"
        deltas = []
        for t in acc:
            v = s.get(t, {}).get("value")
            row += f"{v:>11.4f}" if v is not None else f"{'-':>11}"
            if v is not None:
                deltas.append(v - CHANCE[t])
        if "wikitext" in tasks:
            v = s.get("wikitext", {}).get("value")
            row += f"{v:>11.2f}" if v is not None else f"{'-':>11}"
        if acc:
            row += f"{sum(deltas)/len(deltas):>+18.4f}" if deltas else f"{'-':>18}"
        print(row)

    errs = []
    for t in acc:
        got = [s[t]["stderr"] for s in summaries if s.get(t, {}).get("stderr")]
        if got:
            errs.append(f"{t} +-{sum(got)/len(got):.4f}")
    if errs:
        print("\nunpaired standard error, set by each task's document count: " + ", ".join(errs))
        print("A gap narrower than these is not resolved by this table; it needs the paired test.")

    if "wikitext" in tasks:
        # Measured with a paired bootstrap over the 62 test documents -- both models re-scored on
        # whichever documents a resample draws, so document difficulty cancels. It put +-0.0024
        # nats/word on a difference, which is +-0.17 ppl at these perplexities: a 0.27 ppl gap
        # between two cells came out 1.7 sigma and was not resolved, while a 2.0 ppl gap came out
        # 13.9. The bare two decimals in this column hide all of that.
        print("\nwt103 ppl is quoted with no stderr because the harness reports none for it. A "
              "paired bootstrap")
        print("over its 62 test documents gives a difference about +-0.17 ppl, so a gap under "
              "roughly 0.5 ppl")
        print("is no better resolved by this column than a narrow accuracy gap is by the row "
              "above it.")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True,
                    help="one or more checkpoints; several get a comparison table")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="tokenizer.json; inferred from the checkpoint's vocab size if omitted")
    ap.add_argument("--tasks", default=",".join(SCORED_TASKS))
    ap.add_argument("--limit", type=float, default=None, help="docs per task (int) or fraction")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--precision", default="fp32", choices=("fp32", "bf16"),
                    help="fp32 (default) makes a score independent of --batch-size to 1e-7; bf16 is "
                         "roughly 2x faster and moves PIQA by 0.002 when the batch size changes, "
                         "which is fine for watching a run and not for a reported number")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None,
                    help="only honoured for a single --ckpt; otherwise each writes next to itself")
    ap.add_argument("--sample", default=None, help="also print a greedy continuation of this prompt")
    args = ap.parse_args()

    if args.precision == "fp32":
        # tf32 keeps only a 10-bit mantissa, so leaving it enabled would put back a weaker version of
        # the very batch-order noise that asking for fp32 is meant to remove.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    summaries, failed = [], []
    for path in args.ckpt:
        try:
            summaries.append(score_one(path, args))
        except Exception as exc:                       # one bad checkpoint must not lose the rest
            print(f"\n!! {path} failed: {type(exc).__name__}: {exc}", flush=True)
            failed.append(path)

    if len(summaries) > 1:
        compare(summaries, args.tasks.split(","))
    if failed:
        print("\nfailed: " + ", ".join(str(p) for p in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()

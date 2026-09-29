"""Score the GPT-2 baseline with the *identical* harness Kestrel is scored with.

    python scripts/eval_baseline.py                      # writes runs/eval_baseline_gpt2.json
    python scripts/eval_baseline.py --compare runs/eval_kestrel_main.json

The README makes a specific promise about this row: GPT-2 appears as a *number* produced by this
harness on these splits, not as a figure copied out of a paper, so that the comparison is
like-for-like. This script is what makes that promise true, and the way it does so is by not being
clever. It reuses `KestrelLM`'s entire measurement path -- right-padding, length-sorted batching,
fp32, the same scoring primitive, the same tasks and splits -- and swaps in only the two things
that are genuinely properties of the model: the forward call and the tokenizer.

The tempting shortcut is `lm_eval --model hf`, which is a different code path end to end: its own
padding, its own precision, its own rolling-window implementation. Running GPT-2 through that and
Kestrel through ours would put "the two harnesses disagree" inside the headline gap, and at the
sizes this table resolves -- tenths of a point on PIQA -- that difference is not obviously the small
one. The comparison would look like a comparison of models while being partly a comparison of
measurement setups.

Two things are knowingly *not* held fixed, because they cannot be, and saying so is better than
implying otherwise:

  * The tokenizer. GPT-2 brings its own 50,257-token BPE; Kestrel has a 16,384-token one trained on
    its own corpus. Different tokenizers see different contexts and different continuations, so the
    same item is not literally the same string of tokens. This is inherent to comparing two models
    that were not built by the same pipeline, and it is the reason `acc_norm` is the reported metric
    for the three tasks that offer it: normalizing by continuation length removes the most mechanical
    of the tokenizer's effects (a tokenizer that emits more tokens per word is not thereby worse).
    It does not remove all of them, and the README should not pretend it does.
  * WikiText `word_perplexity` divides by *word* counts taken from the raw page text
    (`lm_eval/tasks/wikitext/preprocess_wikitext.py`: `len(re.split(r"\\s+", doc["page"]))`), so the
    denominator is tokenizer-independent even though the numerator is not.

Precision is fp32 and tf32 is disabled, same as `eval.py` and for the same reason: a number that goes
in a table should not depend on how the items happened to be grouped. The default batch is smaller
than `eval.py`'s because GPT-2's vocabulary is 3x ours, so its `(batch, seq, vocab)` logit tensor is
3x larger and batch 16 of full-length items does not fit in 8 GB alongside the log-softmax. Batch
size is safe to lower here *because* of the fp32 + right-padding choice: the same regrouping that
moves a bf16 logprob by up to 0.12 nats moves an fp32 one by 2e-5.

This scores the *released* `openai-community/gpt2` checkpoint, 124.44M parameters -- 2.5x Kestrel's
49.48M. It is used as a reference point and nothing else. No pretrained weight initializes,
fine-tunes, or teaches Kestrel; nothing is distilled from it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.evaluate import (CHANCE, SCORED_TASKS, TransformersLM, format_table,  # noqa: E402
                              run_eval, summarise)


def build(model_name: str, args: argparse.Namespace) -> tuple[TransformersLM, int]:
    """Load the HF checkpoint, wrap it in the shared harness, and count its parameters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # transformers 5.x renamed the dtype kwarg and dropped the old spelling; omit it entirely and
    # take the checkpoint's own fp32, rather than passing something this version would ignore.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(args.device).eval()          # eval() matters: GPT-2 has dropout in every block
    tok = AutoTokenizer.from_pretrained(model_name)

    n_params = sum(p.numel() for p in model.parameters())
    lm = TransformersLM(model, tok, device=args.device, batch_size=args.batch_size,
                        amp_dtype=None if args.precision == "fp32" else torch.bfloat16)
    return lm, n_params


def score(model_name: str, args: argparse.Namespace) -> dict:
    lm, n_params = build(model_name, args)
    print(f"\n=== {model_name} " + "=" * max(4, 56 - len(model_name)))
    print(f"{n_params/1e6:.2f}M params, context {lm.max_length}, vocab {lm.tok.vocab_size}, "
          f"{args.precision} scoring, batch {lm.batch_size}", flush=True)

    limit = int(args.limit) if args.limit and args.limit >= 1 else args.limit
    t0 = time.perf_counter()
    results = run_eval(lm, tasks=args.tasks.split(","), limit=limit, num_fewshot=args.num_fewshot)
    elapsed = time.perf_counter() - t0

    summary = summarise(results)
    print("\n" + format_table(summary))

    payload = {
        "summary": summary,
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
        "config": str(results.get("config", "")),
        # eval.py attaches _meta to the returned dict *after* saving, so its files do not carry one.
        # This one does, because a baseline row is read months later by someone checking which
        # checkpoint and which precision produced the number -- and "which checkpoint" here is a
        # published artifact that can change under the same name.
        "_meta": {"name": model_name, "params": n_params, "tokens": 0, "gpu_hours": 0.0,
                  "precision": args.precision, "batch_size": lm.batch_size,
                  "num_fewshot": args.num_fewshot, "limit": limit,
                  "scored_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    tag = "" if not limit else f"_limit{limit:g}"
    # `eval_` prefix to match `runs/eval_kestrel_main.json`: the two rows of the README's Results
    # table sort next to each other, and the name says what the file is rather than which script
    # happened to write it.
    out = Path(args.out) if args.out else ROOT / "runs" / f"eval_baseline_{model_name.split('/')[-1]}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n{elapsed/60:.1f} min -> {out}", flush=True)
    return payload


def compare(ours_path: Path, base: dict) -> None:
    """Print the two rows side by side, which is the shape the README table needs.

    The `diff` column is baseline minus Kestrel, so a positive number is a GPT-2 win. It is printed
    with the caveat the table above it already established: these are unpaired standard errors, set
    by each task's document count rather than by the model, and a gap narrower than them is not
    resolved by this table. The footer repeats the numbers rather than leaving that as a vibe.
    """
    ours = json.loads(ours_path.read_text(encoding="utf-8"))
    a, b = ours.get("summary", {}), base.get("summary", {})
    acc = [t for t in SCORED_TASKS if t in CHANCE]

    head = f"{'task':<12}" + "".join(f"{t:>13}" for t in acc)
    if "wikitext" in SCORED_TASKS:
        head += f"{'wt103 nwp':>13}"
    print("\n" + head)
    print(f"{'(chance)':<12}" + "".join(f"{CHANCE[t]:>13.4f}" for t in acc))
    print("-" * len(head))

    def row(label: str, s: dict) -> str:
        line = f"{label:<12}"
        for t in acc:
            v = s.get(t, {}).get("value")
            line += f"{v:>13.4f}" if v is not None else f"{'-':>13}"
        v = s.get("wikitext", {}).get("value")
        line += f"{v:>13.2f}" if v is not None else f"{'-':>13}"
        return line

    print(row("Kestrel-49M", a))
    print(row("GPT-2-124M", b))
    diffs = ""
    for t in acc:
        av, bv = a.get(t, {}).get("value"), b.get(t, {}).get("value")
        diffs += f"{bv - av:>+13.4f}" if av is not None and bv is not None else f"{'-':>13}"
    print(f"{'diff (GPT-2 - Kestrel)':<28}" + diffs)

    errs = []
    for t in acc:
        got = [s[t]["stderr"] for s in (a, b) if s.get(t, {}).get("stderr")]
        if got:
            errs.append(f"{t} +-{sum(got)/len(got):.4f}")
    if errs:
        print("\nunpaired standard error, set by each task's document count: " + ", ".join(errs))
    print("\nThe GPT-2 context limit is 1024 tokens, the same as Kestrel's, so neither row is truncated\n"
          "by a shorter window. 'wt103 nwp' is the harness' word-level perplexity; it is not the\n"
          "token-level wt103 ppl the trainer prints, and the two are not interchangeable.")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2",
                    help="any HF causal LM; the default is the 124M GPT-2 the README quotes")
    ap.add_argument("--tasks", default=",".join(SCORED_TASKS))
    ap.add_argument("--limit", type=float, default=None,
                    help="docs per task; for sanity checks only -- README numbers use no limit")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="8 by default: GPT-2's 50,257-token vocab makes its logit tensor ~3x ours, "
                         "and 16 full-length items does not fit in 8 GB")
    ap.add_argument("--precision", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", type=Path, default=None,
                    help="a Kestrel eval json (e.g. runs/eval_kestrel_main.json) to print "
                         "beside this baseline")
    args = ap.parse_args()

    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    payload = score(args.model, args)
    if args.compare:
        compare(args.compare, payload)


if __name__ == "__main__":
    main()

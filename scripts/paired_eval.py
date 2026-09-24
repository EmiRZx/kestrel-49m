"""Score two models on the same items and separate them with a paired bootstrap.

The unpaired standard error in `eval.py`'s footer bounds each model's own sampling error. It is the
wrong instrument for asking whether two models differ: both answer the *same* items, so item
difficulty is a shared term that cancels in the difference. A paired bootstrap over documents --
resampling items, re-scoring both models on whichever items a resample draws -- is the tight one.

Cross-tokenizer pairing is still valid: the resampling unit is the benchmark item, not a token, and
both models are scored on the same items by construction. What the pairing removes is item
difficulty; it does not remove tokenizer differences, and the report says so.

    python scripts/paired_eval.py --a runs/kestrel-main/best.pt --b openai-community/gpt2 --task piqa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kestrel.evaluate import KestrelLM, TransformersLM, load_checkpoint, run_eval  # noqa: E402

TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")


def correct_key(task: str) -> str:
    """The accuracy field the harness logs per item, mirroring PRIMARY_METRIC."""
    return "acc_norm" if task in ("hellaswag", "arc_easy", "piqa") else "acc"


def hit_vectors(samples, task: str) -> np.ndarray:
    """Per-item 0/1 correctness, in the harness' own item order."""
    key = correct_key(task)
    doc = samples[task]
    return np.array([1.0 if s.get("correct") or s.get(key) == 1.0 else 0.0 for s in doc], dtype=float)


def score_ckpt(path: Path, task: str, device: str):
    from tokenizers import Tokenizer
    model, _ = load_checkpoint(path, device)
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" /
                                  f"vocab{model.cfg.vocab_size}" / "tokenizer.json"))
    lm = KestrelLM(model, tok, device=device, batch_size=16, amp_dtype=None)
    res = run_eval(lm, tasks=[task], limit=None, num_fewshot=0, log_samples=True)
    return hit_vectors(res["samples"], task)


def score_hf(name: str, task: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    lm = TransformersLM(model, AutoTokenizer.from_pretrained(name), device=device,
                        batch_size=8, amp_dtype=None)
    res = run_eval(lm, tasks=[task], limit=None, num_fewshot=0, log_samples=True)
    return hit_vectors(res["samples"], task)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0):
    """Resample items with replacement; report the gap and its spread.

    Returns (gap, lo, hi, p_two_sided) at 95 %. `p` is the share of resamples whose difference
    lands on the other side of zero -- the fraction of the time item-level noise alone could
    manufacture the observed sign.
    """
    rng = np.random.default_rng(seed)
    n_items = len(a)
    gaps = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_items, n_items)
        gaps[i] = b[idx].mean() - a[idx].mean()
    gap = float(b.mean() - a.mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    p = float(2 * min((gaps <= 0).mean(), (gaps >= 0).mean()))
    return gap, float(lo), float(hi), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="Kestrel checkpoint (the 'ours' row)")
    ap.add_argument("--b", default="openai-community/gpt2", help="HF model id")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "paired_eval.json")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # `--b` is an HF model id unless it names a checkpoint on disk, in which case this compares two
    # Kestrel checkpoints instead. That is the same question in the same units -- "do these two sets
    # of weights answer the same items differently, beyond item noise" -- and it is the only honest
    # way to read an anneal, whose whole risk is regressing a task it was not aimed at.
    b_is_ckpt = Path(args.b).exists()
    b_label = Path(args.b).parent.name if b_is_ckpt else "GPT-2"

    out = {}
    print(f"\n{'task':<12}{'A':>10}{b_label:>10}{'gap':>10}{'95 % CI':>22}{'p':>9}")
    print("-" * 75)
    for task in args.tasks.split(","):
        a = score_ckpt(args.a, task, args.device)
        b = (score_ckpt(Path(args.b), task, args.device) if b_is_ckpt
             else score_hf(args.b, task, args.device))
        assert len(a) == len(b), f"{task}: {len(a)} vs {len(b)} items -- not pairable"
        gap, lo, hi, p = paired_bootstrap(a, b)
        out[task] = {"n": int(len(a)), "a": float(a.mean()), "b": float(b.mean()),
                     "a_path": str(args.a), "b_path": str(args.b),
                     "gap_b_minus_a": gap, "ci95": [lo, hi], "p_two_sided": p,
                     "resolved": bool(lo > 0 or hi < 0)}
        mark = "resolved" if (lo > 0 or hi < 0) else "NOT resolved"
        print(f"{task:<12}{a.mean():>10.4f}{b.mean():>10.4f}{gap:>+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>9.4f}   {mark}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")
    print("Paired over items, so item difficulty cancels. Tokenizer differences do NOT cancel.")


if __name__ == "__main__":
    main()

"""Score Kestrel with `lm-evaluation-harness`, the exact tool the competition scores with.

The scored set is fixed: HellaSwag, ARC-Easy, PIQA, WinoGrande, and held-out WikiText-103
perplexity. Those first four are all *multiple-choice* tasks, which the harness answers by asking
the model for `loglikelihood(context, continuation)` on each option -- so a 47M model that can never
generate a coherent sentence can still score meaningfully above chance, and improvements show up
long before the samples look good. WikiText perplexity goes through `loglikelihood_rolling`, which
slides a window over an entire document.

Implementing the harness' own `LM` interface (rather than exporting to HuggingFace format and using
`--model hf`) keeps one source of truth: the same `Kestrel` module and the same tokenizer that
training used, with no conversion step in between that could silently change the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from kestrel.model import Kestrel, KestrelConfig

# The five metrics the judges compute. `wikitext` in the harness is WikiText-103 raw, test split --
# the same split scripts/prepare_data.py decontaminates out of the training corpus.
SCORED_TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext")

# What "random" looks like, so a result table can be read honestly.
CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "piqa": 0.5, "winogrande": 0.5}


def load_checkpoint(path: Path, device: str = "cuda") -> tuple[Kestrel, dict]:
    """Rebuild the model from a training checkpoint, using the config stored inside it."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = KestrelConfig(**ckpt["model_config"])
    model = Kestrel(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


class KestrelLM(LM):
    """Adapter exposing a trained Kestrel to the harness."""

    def __init__(self, model: Kestrel, tokenizer, device: str = "cuda",
                 batch_size: int = 16, eot_id: int = 0) -> None:
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self._device = device
        self.batch_size = batch_size
        self.eot_id = eot_id
        self.max_length = model.cfg.max_seq_len

    # ---- tokenizer plumbing --------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @property
    def tokenizer_name(self) -> str:
        return f"kestrel-bpe-{self.tok.get_vocab_size()}"

    # ---- the one primitive everything else is built on -----------------------------------------

    @torch.no_grad()
    def _score_batch(self, pairs: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        """For each (context_ids, continuation_ids): total logprob of the continuation, and whether
        greedy decoding would have produced it.

        Sequences are right-aligned into a padded batch and truncated from the left to
        `max_length + 1`, because for a fixed-context model the tokens nearest the target are the
        ones that carry the prediction.
        """
        seqs, cont_lens = [], []
        for ctx, cont in pairs:
            ids = (ctx + cont)[-(self.max_length + 1):]
            seqs.append(ids)
            cont_lens.append(min(len(cont), len(ids) - 1))

        width = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), width - 1), self.eot_id, dtype=torch.long)
        tgt = torch.full((len(seqs), width - 1), -100, dtype=torch.long)
        for i, ids in enumerate(seqs):
            n, c = len(ids), cont_lens[i]
            inp[i, width - n:] = torch.tensor(ids[:-1])
            tgt[i, width - 1 - c:] = torch.tensor(ids[-c:])   # score the continuation only

        inp, tgt = inp.to(self._device), tgt.to(self._device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device == "cuda"):
            logits, _ = self.model(inp)
        logprobs = F.log_softmax(logits.float(), dim=-1)

        out = []
        for i in range(len(seqs)):
            mask = tgt[i] != -100
            picked = logprobs[i][mask]
            gold = tgt[i][mask]
            total = picked.gather(-1, gold.unsqueeze(-1)).sum().item()
            greedy = bool((picked.argmax(-1) == gold).all().item())
            out.append((total, greedy))
        return out

    # ---- harness interface ---------------------------------------------------------------------

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        pairs = []
        for req in requests:
            ctx, cont = req.args
            ctx_ids = self.encode(ctx) if ctx else [self.eot_id]
            pairs.append((ctx_ids, self.encode(cont)))
        return self._batched(pairs, disable_tqdm, "loglikelihood")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        """Perplexity over whole documents, via disjoint sliding windows.

        A document is far longer than 1024 tokens, so it is cut into windows that each predict fresh
        tokens while conditioning on as much preceding text as the context allows. Summing the window
        logprobs gives the document's total, which is what the harness turns into perplexity.
        """
        results = []
        for req in requests:
            (text,) = req.args
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self.encode(text), prefix_token=self.eot_id,
                max_seq_len=self.max_length, context_len=1)]
            scores = self._batched(windows, disable_tqdm=True, label="rolling")
            results.append(sum(s for s, _ in scores))
        return results

    def _batched(self, pairs: list[tuple[list[int], list[int]]], disable_tqdm: bool,
                 label: str) -> list[tuple[float, bool]]:
        """Score in length-sorted batches so padding is not the dominant cost."""
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        out: list[tuple[float, bool]] = [(0.0, False)] * len(pairs)
        it = range(0, len(order), self.batch_size)
        if not disable_tqdm:
            from tqdm import tqdm

            it = tqdm(it, desc=f"kestrel {label}", unit="batch")
        for start in it:
            idx = order[start:start + self.batch_size]
            for i, res in zip(idx, self._score_batch([pairs[i] for i in idx])):
                out[i] = res
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        """Greedy (or temperature-sampled) continuation. Only used by generative tasks and the demo;
        none of the five scored metrics need it, but the harness interface requires it."""
        results = []
        for req in requests:
            ctx, kwargs = req.args
            until = kwargs.get("until") or []
            max_new = int(kwargs.get("max_gen_toks", 128))
            temp = float(kwargs.get("temperature", 0.0) or 0.0)
            ids = self.encode(ctx)[-self.max_length:]
            x = torch.tensor([ids], device=self._device)
            new: list[int] = []
            for _ in range(max_new):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self._device == "cuda"):
                    logits, _ = self.model(x[:, -self.max_length:])
                next_logits = logits[0, -1].float()
                if temp > 0:
                    probs = F.softmax(next_logits / temp, dim=-1)
                    nxt = int(torch.multinomial(probs, 1).item())
                else:
                    nxt = int(next_logits.argmax().item())
                if nxt == self.eot_id:
                    break
                new.append(nxt)
                x = torch.cat([x, torch.tensor([[nxt]], device=self._device)], dim=1)
                text = self.decode(new)
                if any(s and s in text for s in until):
                    break
            text = self.decode(new)
            for s in until:
                if s and s in text:
                    text = text.split(s)[0]
            results.append(text)
        return results


PRIMARY_METRIC = {
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "wikitext": "word_perplexity,none",
}


def run_eval(lm: KestrelLM, tasks=SCORED_TASKS, limit=None, num_fewshot: int = 0,
             batch_size: int | None = None) -> dict:
    """Run the harness and return its raw results dict."""
    from lm_eval import simple_evaluate

    if batch_size:
        lm.batch_size = batch_size
    return simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=num_fewshot, limit=limit,
                           log_samples=False, verbosity="WARNING")


def summarise(results: dict) -> dict:
    """Pull out just the numbers that get scored, plus the margin over chance."""
    out: dict[str, dict] = {}
    for task, res in results.get("results", {}).items():
        key = PRIMARY_METRIC.get(task)
        value = res.get(key) if key else None
        if value is None:  # fall back to whatever single metric the task did report
            value = next((v for k, v in res.items()
                          if isinstance(v, (int, float)) and not k.startswith("alias")), None)
        entry = {"metric": key, "value": value,
                 "stderr": res.get((key or "").replace(",", "_stderr,"))}
        if task in CHANCE and isinstance(value, (int, float)):
            entry["chance"] = CHANCE[task]
            entry["over_chance"] = round(value - CHANCE[task], 4)
        out[task] = entry
    return out


def format_table(summary: dict) -> str:
    lines = [f"{'task':<14}{'metric':<22}{'value':>10}{'chance':>9}{'delta':>9}",
             "-" * 64]
    for task, e in summary.items():
        v = e["value"]
        vs = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        ch = f"{e['chance']:.2f}" if "chance" in e else ""
        dl = f"{e['over_chance']:+.4f}" if "over_chance" in e else ""
        lines.append(f"{task:<14}{(e['metric'] or '?'):<22}{vs:>10}{ch:>9}{dl:>9}")
    return "\n".join(lines)


def save_results(results: dict, path: Path) -> dict:
    summary = summarise(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"summary": summary, "results": results.get("results", {}),
         "versions": results.get("versions", {}), "config": str(results.get("config", ""))},
        indent=2, default=str), encoding="utf-8")
    return summary



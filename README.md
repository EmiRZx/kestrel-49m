# Kestrel-49M

A general-purpose language model with **49.48 million parameters**, trained from scratch on a
**6 B-token corpus of its own**.

Built for **Global Innovation Build Challenge V2, Track 01 (Foundational LLM)**, which caps entries
at 50 M trainable parameters *including* the token embeddings and output head, and forbids any
pretrained weights, fine-tuning, or distillation from a larger model.

Everything was developed on one 8 GB laptop GPU (an RTX 4070 Laptop): the tokenizer, the data
pipeline, the kernel-level throughput measurements, and the architecture sweep that chose this
shape. The final 6 B-token run was **rented on a single H100** for the last two hours of the
schedule, because the same run measures 24.8 h on the laptop and 2.1 h there. The sweep was later
rerun on that H100 with three extra shapes, which confirmed the laptop's choice rather than
changing it. Both machines are measured and reported separately below, and neither number is
projected from the other.

The interesting constraint is not the parameter cap. It is that compute was the only free variable
— so every design decision had to be justified in **accuracy per GPU-hour**, not accuracy per
parameter — and that a training run which will certainly be interrupted has to be resumable to the
token.

### At a glance

| | |
|---|---|
| **Parameters** | **49,482,752** — 517,248 under the cap, embeddings and output head included |
| **Training data** | 6,000,214,016 tokens, own corpus, own 16 k tokenizer |
| **Submitted run** | **2.08 H100-hours · $8.22 · 1.43 kWh** |
| **Whole project** | **3.65 rented H100-hours · ≈ $14** — including a 7-shape architecture sweep and a 5-variant annealing study. *Rented GPU only; see the note below for what this excludes.* |
| **Headline result** | Beats **GPT-2-124M** on ARC-Easy (0.4217 vs 0.3948) at **2.5× fewer parameters**, scored through one harness |
| **How results are defended** | Paired bootstrap over items + Benjamini–Hochberg across 20 comparisons; fp32 scoring verified byte-identical across three runs |
| **Decontamination** | All **five** scored sets, not just perplexity — 943 documents dropped before tokenization |

Three things this project treats as non-negotiable, and which shape everything below:

1. **A number that cannot be reproduced is not a result.** Evaluation runs in fp32 with tf32
   disabled because bf16 batch regrouping moved PIQA by 0.0017 — enough to flip a claim. Three
   consecutive scoring runs give byte-identical output.
2. **A difference smaller than its error bar is not a finding.** Unpaired standard errors cannot
   resolve any of the four gaps against GPT-2; a paired bootstrap resolves three and refuses the
   fourth. That refusal corrected an over-claim already written into this README.
3. **The baseline is scored here, not quoted.** GPT-2 runs through the identical padding, precision,
   batching and rolling-window code as Kestrel. Two models scored by two harnesses are measured in
   different units.

## Results

Zero-shot, no subsampling, `lm-evaluation-harness` v0.4.12 — the same harness, tasks, and splits the
competition scores. Full output in `runs/eval_kestrel_main.json` and `runs/eval_baseline_gpt2_124m.json`.

| Benchmark | Kestrel-49M | Chance | GPT-2 124M (scored here) | GPT-2 − Kestrel |
|---|---|---|---|---|
| HellaSwag (`acc_norm`) | 0.3001 ± 0.0046 | 0.25 | **0.3114** ± 0.0046 | +0.0113 |
| ARC-Easy (`acc_norm`) | **0.4217** ± 0.0101 | 0.25 | 0.3948 ± 0.0100 | −0.0269 |
| PIQA (`acc_norm`) | 0.5958 ± 0.0114 | 0.50 | **0.6251** ± 0.0113 | +0.0294 |
| WinoGrande (`acc`) | **0.5233** ± 0.0140 | 0.50 | 0.5162 ± 0.0140 | −0.0071 |
| WikiText-103 (`word_perplexity`, test) | 40.35 | — | **37.37** | −2.98 |

> **Reading the last column.** It is literally GPT-2 minus Kestrel, so its *sign* means opposite
> things on opposite rows: on the four accuracy rows higher is better, so **positive = GPT-2 ahead**;
> on the perplexity row lower is better, so **negative = GPT-2 ahead**. Bold marks the better score
> on every row, which is the unambiguous guide. Throughout this README, GPT-2 leads HellaSwag, PIQA
> and perplexity; Kestrel leads ARC-Easy and, within noise, WinoGrande.

Against a model **2.5× larger** — 124.4 M parameters against 49.5 M — Kestrel is ahead on two of the
four multiple-choice tasks and behind on two. Only one of those leads survives the paired test
below, and the table after this one says which: **ARC-Easy is a resolved win; WinoGrande is not a
win at all**, it is a gap narrower than its own noise. Both are stated here in the same breath
because reporting the raw column alone would be the more flattering and less true description.

**What this cost.** The submitted run is **2.08 H100-hours and $8.22**; all rented GPU time across
the project — the 6 B-token run, the seven-shape architecture sweep and the five-variant annealing
study — is **3.65 H100-hours and about $14**. The ARC-Easy result above is therefore a win over
GPT-2 at 2.5× fewer parameters, two hours of one GPU, and a corpus of 6 B tokens. The efficiency is
not a consolation for the two losses — it is the axis the whole project optimises, and the reason
the architecture was chosen by measuring tokens per second rather than by adopting published advice.

**What that number does and does not cover**, stated plainly so it is not mistaken for a
whole-project figure comparable to someone else's:

| included | excluded |
|---|---|
| 2.08 h — the submitted 6 B-token run | ~7.2 laptop-GPU-hours (the 4-shape laptop sweep) |
| 1.09 h — the 7-cell H100 architecture sweep | the laptop's own development and benchmarking time |
| 0.48 h — 5 annealing variants | 33.2 min of Modal **CPU** containers building the corpus |
| **3.65 rented H100-hours ≈ $14** | tokenizer training, evaluation runs (laptop), `bench.py` sweeps |

The excluded items are real work; they are left out because they ran on hardware that was already
owned or on CPUs that cost cents, and because mixing them into one figure would make the number mean
less rather than more. The honest summary is: **$14 of rented GPU, on top of a laptop that was
already there.**

The ± figures above are *unpaired* standard errors: each model's own sampling error. They are the
wrong instrument for a head-to-head gap, because both models answer the same items and item
difficulty cancels in the difference. `scripts/paired_eval.py` does the right test — a paired
bootstrap resampling the benchmark items and re-scoring both models on whichever items a resample
draws (`runs/paired_eval.json`):

| | items | Kestrel | GPT-2 | gap | 95 % CI | p |
|---|---|---|---|---|---|---|
| HellaSwag | 10,042 | 0.3001 | 0.3114 | **+0.0113** | [+0.0044, +0.0184] | 0.0014 |
| ARC-Easy | 2,376 | 0.4217 | 0.3948 | **−0.0269** | [−0.0434, −0.0105] | 0.0009 |
| PIQA | 1,838 | 0.5958 | 0.6251 | **+0.0294** | [+0.0120, +0.0473] | 0.0010 |
| WinoGrande | 1,267 | 0.5233 | 0.5162 | −0.0071 | [−0.0418, +0.0268] | 0.70 |

Paired, the three larger gaps are clearly resolved (p ≤ 0.0014) and WinoGrande's is **not** — its
confidence interval straddles zero, so the −0.0071 is noise rather than a small GPT-2 win. That
distinction is the point of running the paired test instead of reading the raw column: the unpaired
errors (±0.0046 to ±0.0140) are wide enough to hide all four gaps, and 1,267 WinoGrande items cannot
resolve a difference that small no matter how they are analyzed. Pairing cancels item difficulty; it
does **not** cancel tokenizer differences, which are inherent to comparing two independently built
models and are discussed below.

The perplexity gap is the most decisive number in the table: GPT-2's 37.37 against 40.35, a
difference far outside the ±0.17 ppl a paired bootstrap over these 62 documents resolves. **This is
a clean loss and it cannot be explained away by domain mismatch** — if anything the domain favours
Kestrel. WikiText-103 is built from English Wikipedia articles, and Wikipedia is 18 % of Kestrel's
corpus, while GPT-2's WebText *deliberately excluded* it: "We removed all Wikipedia documents from
WebText since it is a common data source for other datasets and could complicate analysis due to
overlapping training data with test evaluation tasks" (Radford et al., 2019, §4). Kestrel holds the
in-domain advantage on this metric and still loses it.

The remaining explanations are the unglamorous ones: 2.5× the parameters, and a larger and more
varied training corpus (WebText is ~40 GB of deduplicated web text against this project's 6 B
tokens). Perplexity rewards exactly that, which is why it is the metric where the parameter gap
shows most plainly, and why the ARC-Easy win sits alongside it rather than cancelling it.

**Reproduce:**

```bash
# Kestrel. Writes runs/kestrel-main/eval_best.json by default; --out names the committed copy.
python scripts/eval.py --ckpt runs/kestrel-main/best.pt --out runs/eval_kestrel_main.json

# GPT-2, through the same harness. Writes runs/eval_baseline_gpt2_124m.json and prints both rows.
python scripts/eval_baseline.py --compare runs/eval_kestrel_main.json

# The paired test behind the CI column above. Writes runs/paired_eval.json.
python scripts/paired_eval.py --a runs/kestrel-main/best.pt --b openai-community/gpt2
```

Both eval commands default to fp32 with tf32 disabled, which is what makes them reproducible: three
consecutive runs gave byte-identical results. `--precision bf16` is roughly twice as fast and is the
right choice for watching a run in progress, not for a number that goes in this table.

The GPT-2 column is the released `openai-community/gpt2` checkpoint, run through **this** harness on
**these** splits rather than quoted from a paper, so the comparison is like-for-like. "This harness"
is meant literally: `scripts/eval_baseline.py` reuses Kestrel's own `_logits` scoring primitive,
right-padding, length-sorted batching and fp32 arithmetic, and overrides only the forward call and
the tokenizer. The tempting shortcut — `lm_eval --model hf` — is a different code path end to end
(its own padding, precision, batching, and rolling windows), which would fold *harness differences*
into the headline gap. On the Kestrel side this precision is not cosmetic: PIQA moves by 0.0017 on
bf16 batch regrouping alone, so a table mixing two configurations is comparing measurement noise
against model differences.

To be unambiguous about the Track 01 rule: GPT-2 appears only as a number in that column. No
pretrained weights initialize, fine-tune, or teach Kestrel — nothing is distilled from it.

Two things are knowingly *not* held fixed, because they cannot be. GPT-2 brings its own 50,257-token
BPE against Kestrel's 16,384-token one, so the same item is not literally the same token sequence;
`acc_norm` is reported precisely because length-normalizing removes the most mechanical of those
effects. And the two perplexity figures below use different denominators — see the note under the
training curve.

Word-level perplexity is reported rather than token-level, because token-level perplexity depends on
the tokenizer and is therefore **not comparable between entries** — a model with an 8 k vocabulary
will show a lower token-level perplexity than an identical model with a 32 k vocabulary while being
no better at predicting English.

## Hardware, training time, and compute

The final run is split across two machines, and the table reports them separately because they are
different by an order of magnitude and averaging them would describe neither.

**Development and architecture search** — everything except the final run:

| | |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8 GB GDDR6, 140 W power cap |
| Precision | bf16 autocast, fp32 master weights, `torch.compile` (Inductor + Triton) |
| Throughput | 67,230 tokens/s sustained at micro-batch 20 (`wide-shallow`, compiled) |
| Peak VRAM | 6.88 GB at that micro-batch — the 8 GB ceiling is what sets the batch, and the batch is what sets throughput |

**The submitted 6 B-token run** (`kestrel-main`, `runs/kestrel-main/metrics.jsonl`):

| | |
|---|---|
| GPU | NVIDIA H100 80 GB SXM, rented (Modal), pinned `H100!` |
| Precision | bf16 autocast, fp32 master weights, `torch.compile` |
| Throughput | 827,342 tokens/s mean, 2288 logged steps (304 TFLOP/s effective, 30.8 % MFU) |
| Training tokens | 6,000,214,016 |
| Wall-clock training time | **2.08 GPU-hours** — 2.08 h on the H100; the same run measures 24.8 h on the laptop, and that 12× is why it was rented |
| Estimated compute | 2.21 × 10¹⁸ FLOPs (`6N + 12·L·d·T` per token, 367,675,392 FLOPs/token × 6.0 B tokens) |
| Energy | 1.432 kWh, integrated from NVML power samples every logging interval (mean 684 W) |
| Peak VRAM | 37.61 GB |
| Cost | $8.22 at Modal's H100 list price of $3.95/GPU-hour (2026-09-05) |

Nothing in either table is reconstructed after the fact: `kestrel/trainer.py` writes tokens/s, MFU,
GPU-seconds, GPU temperature, SM clock, throttle reasons, and integrated watt-seconds into
`runs/<name>/metrics.jsonl` at every logging step, and carries the totals through every
checkpoint/resume cycle.

The laptop number is not a projection of the H100 number or vice versa. The 24.8 h figure comes from
`runs/bench_frontier.jsonl`, measured with `scripts/bench.py` on the laptop at micro-batch 20 — the
largest batch that fits the 8 GB card, peaking at 6.88 GB — and the 2.08 h from the run's own
accounting. The two batches differ because the cards differ; both are each card's best measured
throughput for this architecture, and the 12× difference is what those two measurements say, not what
the spec sheets imply. The H100 was pinned with `!` to suppress Modal's free upgrade to an H200,
because an unpinned card would have changed the throughput numbers the run is reported against.

## Quickstart

Prerequisites: Python 3.12, an NVIDIA GPU with ≥ 8 GB VRAM, CUDA 12.8+ driver, ~60 GB free disk.

```bash
git clone <this repo> && cd kestrel
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install triton-windows                          # Windows only: PyTorch ships no Triton there
```

```bash
# 1. verify the parameter count against the 50 M cap  (required competition deliverable)
python scripts/count_params.py

# 2. fit a BPE tokenizer on our own corpus mix
python scripts/train_tokenizer.py --vocab-size 16384

# 3. stream, decontaminate, tokenize, and shard ~6 B tokens  (several hours, resumable)
python scripts/prepare_data.py --tokens 6e9 --vocab-size 16384

# 4. measure this machine before committing GPU-hours to it
python scripts/bench.py --matrix --compile --batches 8,12,16,24

# 5. train.  --max-hours ends the session cleanly on a checkpoint; --resume auto continues it
python scripts/train.py --name kestrel-main --total-tokens 6e9 --max-hours 8 --resume auto

# 6. score the result on the five metrics the competition scores
python scripts/eval.py --ckpt runs/kestrel-main/best.pt
```

To stop an overnight run early without losing anything, either press Ctrl-C once (it finishes the
step, writes a checkpoint, and exits) or `touch runs/kestrel-main/STOP`.

## Parameter count

The cap counts embeddings, so the embedding table is the first thing to argue about. At this model's
`d_model` of 640, a 32 k vocabulary would cost 20.97 M parameters — **42 % of the entire budget spent
on a lookup table**. A 16 k BPE vocabulary fitted to our own corpus costs 10.49 M and returns the
difference to the layers, at the price of more tokens per document. That choice is why 38,996,992 of
this model's parameters — **78.8 %** — sit in layers that compute, rather than in a table that looks
things up.

```
$ python scripts/count_params.py
========================================================================
CONFIG
  vocab_size       16384
  d_model          640
  n_layers         9
  n_heads          10
  n_kv_heads       2
  ffn_hidden       1744
  max_seq_len      1024
  rope_theta       10000.0
  norm_eps         1e-05
  tie_embeddings   True
  qk_norm          True
  init_std         0.02
  loss_chunk_size  2048
  sdpa_enable_gqa  False
  param_budget     50000000
  head_dim         64

CLOSED-FORM ESTIMATE
  per_layer           4,332,928
  layers             38,996,352
  embedding          10,485,760
  non_embedding      38,996,992
  total              49,482,752

MEASURED (torch, tied weights counted once)
  embedding          10,485,760
  non_embedding      38,996,992
  total              49,482,752

  fwd+bwd FLOPs/token   367,675,392

  budget              50,000,000
  headroom               517,248
  utilisation             98.97%   -> WITHIN BUDGET
  embedding share         21.19%   (lower is better: params in layers, not lookup)
========================================================================
```

Two independent counts are printed because they can disagree, and if they did, every parameter number
in this README would be wrong: the closed-form estimate is what sized the model, and the measured
count is what `torch` actually built. They agree at **49,482,752 — 98.97 % of the cap, 517,248 to
spare.** The script exits non-zero if the model is over budget, which makes it usable as a
pre-commit check.

### On tied embeddings, explicitly

Embeddings are **tied**: `token_emb.weight` is one tensor, used both to embed tokens and — read
transposed — to produce logits. There is no second matrix. `param_report()` deduplicates by
`id(p)`, so `sum(p.numel() for p in model.parameters())` counts that tensor exactly once, which is
how PyTorch, Hugging Face, and the GPT-2 paper all report parameter counts.

The rule says the cap includes "token embeddings and the output head", and this count satisfies it:
the tensor that *is* both is inside the total. Spelling out the alternative rather than leaving it
for a judge to wonder about — counting the same tensor a second time because it serves a second
purpose would give **59,968,512**. That is not a parameter count of anything; no such tensor exists,
nothing allocates that memory, and no optimizer state is kept for it. The model's state dict has
one entry, of 10,485,760 values.

This is also the reason the vocabulary is 16 k rather than 32 k. The smaller table is not a
shortcut around the cap; it is a deliberate trade, measured in the vocab ablation below, that buys
layer capacity with sequence length.

## How it works

### Architecture

A conventional Llama-shaped decoder: RMSNorm pre-norm, RoPE, grouped-query attention, SwiGLU feed
-forward, QK-norm for stability, tied embeddings. The conventionality is deliberate — it means
`lm-evaluation-harness`, `transformers`, GGUF, and ONNX all work with no custom glue, so the novelty
budget can be spent on data and training rather than on a block nobody else can run.

Four shapes were compared head to head **in the first sweep, on the laptop**; three more were added
later when the same suite was rerun on a rented H100, for seven in total (see *Results:
architecture*). All of them are fitted to the **same** parameter count — each takes the largest
`ffn_hidden` (in multiples of 8) that stays under 49.5 M at vocab 16,384, so the spread across the
original four is 0.25 M, or 0.51 %:

| variant | `d_model` × layers | heads (Q/KV) | FFN | params |
|---|---|---|---|---|
| `very-deep` | 320 × 36 | 5 / 1 | 1016 | 49.23 M |
| `deep-thin` | 384 × 26 | 6 / 2 | 1096 | 49.37 M |
| `mid` | 512 × 14 | 8 / 2 | 1480 | 49.41 M |
| `wide-shallow` | 640 × 9 | 10 / 2 | 1744 | 49.48 M |

Holding the count fixed is what makes the comparison mean anything. An earlier version of this table
did not: the shapes ranged over 45.08 M – 49.21 M, so the widest shape was also the biggest model and
a win for it would have measured capacity rather than shape. `ffn_hidden` is now derived by
`fit_ffn_hidden()` in `kestrel/variants.py` instead of chosen by hand, `tests/test_variants.py`
asserts the table still equals what that function returns, and `KestrelConfig`'s defaults read from
the same dict — a number that exists in one place cannot drift out of agreement with itself.

### Tokenizer

A byte-level BPE trained on this project's own corpus mix (`scripts/train_tokenizer.py`), using the
`cl100k_base` split pattern. The clause that matters most is `\p{N}{1,3}`, which caps number tokens
at three digits so that arithmetic is represented consistently instead of the model having to learn
that `1234` and `12`+`34` describe the same object.

### Data, and decontamination

~6 B tokens: **82 % [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**
(`sample-10BT`, ODC-By 1.0) and **18 % English Wikipedia** (`wikimedia/wikipedia` `20231101.en`,
CC BY-SA 4.0). FineWeb-Edu carries the four commonsense/science benchmarks; Wikipedia is there
because WikiText-103 perplexity is scored and it keeps that distribution in-domain.

Six billion tokens against 49 M parameters is roughly 122 tokens per parameter — far past
Chinchilla-optimal, which is the *correct* choice when parameters are capped by rule and compute is
the only free variable.

Before anything is written to disk, every document is checked against an n-gram index built from the
evaluation splits that are actually scored — HellaSwag validation, ARC-Easy test, PIQA validation,
WinoGrande validation, and WikiText-103 **test** — and dropped if it shares a **16-token window** with
any of them (`kestrel/decontaminate.py`). The index is saved as `decontam.npz` next to the shards so
the filter is auditable rather than merely claimed, and the drop count and per-benchmark breakdown go
into `prepare_report.json`.

The window length is the whole decision, because a filter that never fires is a claim rather than a
control. A window of *w* tokens cannot match any text shorter than *w*, and benchmark answer choices
are often two words — so this started at 32 tokens and had to be measured in both directions.

**Reach** — what fraction of each scored split is long enough to be seen at all
(`scripts/check_decontam_coverage.py`):

| split | @ 32, fields alone | @ 32, stem+continuation | @ 16, stem+continuation | median tokens |
|---|---|---|---|---|
| HellaSwag | 65 % | 79 % | **92 %** | 55 |
| ARC-Easy | 3 % | 22 % | **49 %** | 15 |
| PIQA | 17 % | 34 % | **62 %** | 21 |
| WinoGrande | 3 % | 12 % | **67 %** | 24 |
| WikiText-103 | 63 % | 63 % | **70 %** | 92 |

Two changes are stacked there. Indexing each item as **stem+continuation pairs** — question with each
answer, context with each ending — rather than as isolated fields, which is both what lifts ARC-Easy
from 3 % to 22 % and the form a scraped quiz page actually takes. Then shortening the window to 16.

**Price** — what that reach costs. First on 300,000 documents split evenly between the two sources,
which isolates the window from the mix (`scripts/check_decontam_cost.py --docs 150000 --windows 16 32`):

| window | FineWeb-Edu | Wikipedia | all | rate |
|---|---|---|---|---|
| 32 | 0 / 150,000 | 3 / 150,000 | 3 / 300,000 | 0.0010 % |
| 16 | 11 / 150,000 | 133 / 150,000 | 144 / 300,000 | 0.0480 % |

At 32 tokens the filter dropped one document in a hundred thousand and **never once fired on
FineWeb-Edu**, which is 82 % of the corpus. It was not being conservative; it was doing nothing.

Then at window 16 on the **submitted 6 B corpus**, at the real 82/18 mix
(`runs/prepare_report_v16384.json`):

| source | documents kept | dropped | rate | matched |
|---|---|---|---|---|
| FineWeb-Edu | 4,499,010 | 446 | 0.0099 % | hellaswag 215, arc_easy 150, wikitext103 71, piqa 10 |
| Wikipedia | 1,205,775 | 497 | 0.0412 % | wikitext103 490, arc_easy 4, hellaswag 3 |
| **total** | **5,704,785** | **943** | **0.0165 %** | |

One document in 6,000 — the weighted rate is *lower* than the 50/50 sample because Wikipedia, the
source that matches most, is only 18 % of the mix. The 300 M-token pilot built by the same command
dropped 49 of 269,312 documents at 0.0182 % (`data/tokens_pilot/prepare_report.json`), so the rate is
stable across a 20× change in corpus size rather than being an artefact of one build.

The composition is more informative than the rate. Almost every Wikipedia drop — 490 of 497 — matches
WikiText-103, which is exactly right: WikiText-103 *is* Wikipedia, so those are the same articles and
removing them is the point. Among the FineWeb-Edu drops, some are genuine verbatim overlap (`British
armed trawler HMT Stella Sirius was sunk by`) and some are stock phrases any educational page contains
(`Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus`). `check_decontam_cost.py` decodes every
matched span, so that judgement is inspectable rather than asserted.

Sixteen tokens is ~8–9 English words at this tokenizer's 2.89 chars/token, already stricter than the
13-word n-gram GPT-3 filtered on, and that is where it stops: 12 tokens would buy another ~10 points
of ARC-Easy coverage, but a 6–7 word span is a phrase and not a quotation.

One limitation stays on the record. WikiText's own preprocessing (`@-@`, `<unk>`, spaced punctuation)
means a *same-article* overlap with raw Wikipedia will not hash-match, even though verbatim copies of
the benchmark files do. And 19.2 % of scored strings — 27,999 of the 145,757 indexed — are shorter
than 16 tokens and therefore unfilterable at any window, so the filter is blind to them by
construction. `prepare_report.json` records that count as `eval_texts_below_window` instead of
omitting it, because a drop rate quoted without it sounds more complete than it is.

The WikiText-103 *validation* split is tokenized separately and tracked during training, so the
scored *test* split stays untouched by both training and model selection.

### Training

| | |
|---|---|
| Optimizer | **Muon** on 2D hidden matrices, **AdamW** on embeddings and 1D gains |
| Effective batch | 262,144 tokens/step, reached by gradient accumulation |
| Schedule | warmup → stable → decay (WSD), expressed in **tokens**, not steps |
| Sequence length | 1024 |
| Grad clip | 1.0 global norm |

[Muon](https://kellerjordan.github.io/posts/muon/) (Keller Jordan et al., 2024) replaces the momentum
buffer with its nearest semi-orthogonal matrix before stepping, approximated by a 5-step
Newton–Schulz iteration in bf16 — a handful of small matmuls, negligible against the forward pass.
It reaches a given loss in materially fewer steps than AdamW on transformer hidden layers, which on
a fixed compute budget converts directly into fewer GPU-hours — the currency this project actually
spends, whether the card is owned or rented by the hour. It applies only to 2D hidden matrices;
embeddings, the output head, and 1D norm gains stay on AdamW.

The LR schedule is a function of **tokens seen**, not of step index. That sounds pedantic and is not:
the micro-batch size changes with whatever fits in VRAM, and ablations deliberately use different
batches, so steps are not a comparable axis between runs while tokens always are.

**The two WikiText numbers are not interchangeable.** The trainer logs `wt103_ppl` (the final run:
**26.04**) and the Results table reports `word_perplexity` (**40.35**). Same model, different
measurements:

| | trainer's `wt103_ppl` | harness' `word_perplexity` |
|---|---|---|
| split | WikiText-103 **validation** | WikiText-103 **test** (`wikitext-2-raw-v1` config) |
| unit | tokens, ours (16,384 BPE) | **words**, `len(re.split(r"\s+", page))` on raw text |
| shape | our tokenized slice | 62 whole documents, rolling window |
| comparable across entries? | no — tokenizer-dependent | **yes** |

The gap between them is mostly the denominator: our tokenizer emits more than one token per word, so
dividing the same total log-likelihood by words instead of tokens raises the figure. 26.04 is the
number to watch a run with; 40.35 is the number that can be compared to anyone else's model, which is
why it is the one in the Results table.

Three things make an interrupted run harmless. Checkpoints carry both optimizer states, the dataset
cursor, the RNG states, and the cumulative GPU-seconds and watt-seconds — so a resumed run continues
the identical data order and its efficiency accounting stays correct across sessions. Checkpoints are
written to a temp file and `replace()`d, so a crash mid-save cannot destroy the previous one. And
`--max-hours` ends a session on a checkpoint rather than on a killed process — it counts *training*
time rather than wall-clock, so a machine that spent the night suspended has not spent the budget.
Note that the clock restarts on resume: it bounds one attempt, not a run's total cost.

### Fitting a 49 M model into 8 GB

The largest tensor in a small-vocabulary LM is not the weights — it is the logits. At batch 16 ×
sequence 1024 × vocab 16,384, a single fp32 logit tensor is 1.07 GB, and the backward pass needs
another. `chunked_cross_entropy` never materialises it: the hidden states are split into 2048-row
chunks, each chunk's projection-and-loss is recomputed under `torch.utils.checkpoint`, and the
training forward pass returns **no logits at all** unless asked. Measured with `--loss-chunk 0`,
micro-batch 8 will not fit at all; with chunking on, `deep-thin` runs 12 sequences and
`wide-shallow` 20. It is load-bearing, not an optimisation.

The chunking was written under the 8 GB ceiling, and the same `loss_chunk_size: 2048` carried into
the H100 run — where it still earns its place. At that run's micro-batch 128 the unchunked logit
tensor would be **8.59 GB** (128 × 1024 × 16,384 × 4 bytes), per direction. Batch 20 on the laptop is
what the 8 GB card can hold; batch 128 on the H100 is what the chunked path made cheap enough to
bother with. Same code, two ceilings.

## What was measured, rather than assumed

Full data in `runs/bench_frontier.jsonl`, reproduce with `scripts/bench.py --matrix`. Each cell runs
in a fresh subprocess, because a CUDA OOM leaves the process unusable and a sweep that shares one
cannot survive its own failures.

Throughput at seq 1024, vocab 16,384, bf16, compiled, at each variant's best micro-batch:

| variant | micro-batch | tokens/s | peak VRAM | MFU | 6 B tokens |
|---|---|---|---|---|---|
| `very-deep` (320 × 36) | 8 | 43,758 | 5.20 GB | 23.9 % | 38.1 h |
| `deep-thin` (384 × 26) | 12 | 50,952 | 7.05 GB | 26.7 % | 32.7 h |
| `mid` (512 × 14) | 12 | 60,365 | 5.54 GB | 29.0 % | 27.6 h |
| `wide-shallow` (640 × 9) | 20 | **67,230** | 6.88 GB | 30.9 % | **24.8 h** |

At matched parameters, wide-shallow processes **32 % more tokens per GPU-hour than deep-thin and
54 % more than very-deep** — 13.3 GPU-hours of difference over a 6 B-token run on this laptop, and
the same ordering held when the ablation suite was rerun on an H100 (below), where the absolute
hours are smaller but the ranking is what the choice rested on. Depth costs throughput twice: more
sequential kernel launches, and more stored activations, which caps the micro-batch — very-deep
cannot hold 12 sequences where wide-shallow holds 20. This is why the published "depth beats width
below 1 B parameters" result could not simply be adopted here: the question on a fixed compute
budget is not accuracy per parameter but accuracy per GPU-hour, and a shape that is 32 % worse per
token can still win. `scripts/ablate.py --suite arch` decides it empirically at 300 M tokens per
cell.

Other measurements that changed the design:

| Finding | Consequence |
|---|---|
| `torch.compile` gave 51.6 k vs 25.7 k tokens/s **and** 4.68 vs 6.57 GB | On by default. Being cheaper in memory as well as faster, it also buys a larger micro-batch. On Windows this requires `pip install triton-windows`; PyTorch ships no Triton there. |
| `scaled_dot_product_attention(..., enable_gqa=True)` was ~2.5× *slower* and OOM'd earlier | It disqualifies the flash and mem-efficient kernels, falling back to the math backend, which materialises a `(batch, heads, seq, seq)` score matrix. Kept the explicit `repeat_kv` path instead. |
| Without `torch.cuda.set_per_process_memory_fraction(0.9)`, throughput silently collapsed ~30× at large batches | Windows' WDDM driver spills to system RAM instead of raising OOM, so an unusable configuration *appears to work*. Capping the fraction turns that failure back into a visible error. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Unsupported on Windows. Dropped. |

## Ablations

One axis at a time, identical token budget per cell, ranked on validation loss **and** on loss per
GPU-hour:

```bash
python scripts/ablate.py --suite arch        # 4 shapes at matched parameters
python scripts/ablate.py --suite optimizer   # Muon+AdamW vs a separately tuned AdamW baseline
python scripts/ablate.py --suite schedule    # WSD vs cosine
python scripts/ablate.py --suite vocab       # 8 k vs 16 k vs 32 k BPE
```

The micro-batch for each cell is read from the measured throughput frontier rather than guessed, so
no cell is handicapped by a batch size that merely happened to fit some other shape. The AdamW
baseline gets its own learning rate, because beating a deliberately mis-tuned baseline would prove
nothing.

Every cell is also refitted to the same parameter budget, which for the vocab sweep is the whole
point: the embedding table counts against the cap, so a larger vocabulary is not free capacity — it
has to be paid for out of FFN width. `TrainConfig.model_config()` calls `fitted_variant()`, so the
refit happens wherever a model is built rather than being something a sweep has to remember:

| vocab | `wide-shallow` FFN | params in layers | params in the lookup table |
|---|---|---|---|
| 8,192 | 2048 | 44.25 M | 5.24 M (10.6 %) |
| 16,384 | 1744 | 39.00 M | 10.49 M (21.2 %) |
| 32,768 | 1136 | 28.49 M | 20.97 M (42.4 %) |

At 32 k this shape would spend **42 % of a hard-capped budget on a lookup table**. Whether the shorter
sequences that buys are worth 15.8 M parameters of layer capacity is exactly the kind of question that
cannot be reasoned out from first principles, which is why it is an ablation and not an assertion.
Before this was fixed the 32 k cells were 4.5–10 M *over* the 50 M cap, so the sweep would have
compared a legal model against an illegal one and called the difference an effect of vocabulary.

### Results: architecture

Seven shapes, each refitted to the parameter cap (49.42–49.49 M), 300 M tokens per cell, identical
data (`tokens_pilot`), schedule (WSD), seed (1337) and `tokens_per_step` (262,144). Run on a pinned
H100 at micro-batch 64; full data in `runs/ablation_arch_h100.json`. **Sorted by validation loss**,
which is not the order that flatters the conclusion below — `deep-thin` places third on loss while
costing 64 % more GPU-hours than the winner, and that tension is the actual finding.

| variant | shape | val loss | val ppl | wt103 ppl | tokens/s | MFU | peak VRAM | GPU-h |
|---|---|---|---|---|---|---|---|---|
| **`wide-shallow`** | **640 × 9** | **3.3820** | **29.43** | **41.37** | 817,010 | 30.4 % | 19.53 GB | 0.129 |
| `mid` | 512 × 14 | 3.3868 | 29.57 | 41.79 | 682,786 | 26.6 % | 23.74 GB | 0.172 |
| `deep-thin` | 384 × 26 | 3.3938 | 29.78 | 41.83 | 519,594 | 22.0 % | 31.71 GB | 0.212 |
| `extra-wide-6` | 768 × 6 | 3.3984 | 29.92 | 42.64 | 933,874 | 33.4 % | 16.57 GB | 0.110 |
| `wide-shallow-7` | 640 × 7 | 3.3986 | 29.92 | 41.65 | 900,839 | 32.0 % | 17.82 GB | 0.114 |
| `very-deep` | 320 × 36 | 3.4257 | 30.74 | 43.60 | 448,262 | 19.8 % | 34.13 GB | 0.247 |
| `widest-4` | 896 × 4 | 3.4434 | 31.29 | 46.71 | 1,016,538 | 35.0 % | 14.05 GB | 0.104 |

`wide-shallow` won outright: best validation loss **and** best WikiText perplexity, at the 4th-lowest
cost of seven. That is the convenient case, and it is worth saying that the sweep was built expecting
the awkward one — a cheap shape winning on throughput and losing on loss, forcing a judgement call.

What the extra three cells bought was the location of the turn. Widening past 640 × 9 keeps buying
throughput — `widest-4` runs at **1.24× the tokens/s**, 19.5 % fewer GPU-hours than the winner — but
it stops buying loss: `widest-4` is 0.061 worse, and even the mild steps (`wide-shallow-7`,
`extra-wide-6`) give up ~0.017 for their 11–15 %. Depth is worse in the other direction: `very-deep`
costs **92 % more GPU-hours** than `wide-shallow` and is 0.044 worse. Both extremes lose; the
interior wins.

The honest limit of this table: every cell ran the same *tokens*, not the same *GPU-hours*. A cheaper
shape trained to equal compute would see ~24 % more tokens, and this sweep does not measure whether
that closes a 0.061 gap. Reading these rows as compute-matched would overstate them.

The same four shapes were also run on the laptop (`runs/ablation_arch.json`, micro-batch 8–16), and
the ordering replicated exactly — `wide-shallow` < `mid` < `deep-thin` < `very-deep` on validation
loss, on both machines. The absolute losses sit 0.01–0.03 lower on the H100 for the same seed and the
same effective batch, which is the same bf16 kernel-ordering sensitivity documented under Evaluation;
it moves the numbers and not their ranking. (The laptop `deep-thin` cell is the one exception worth
flagging: it stopped at 289 M of 300 M tokens on the session clock, so its 3.4383 is not strictly
token-matched to the rest of that suite.)

The optimizer, schedule, and vocab suites are specified above but were not run — the GPU-hours went
to the 6 B-token main run instead. Their commands are listed so the result is reproducible, not to
imply a result exists.

## Synthetic data and annealing

The Results table shows Kestrel losing HellaSwag and PIQA to GPT-2 and winning ARC-Easy. That pattern
is not random. ARC-Easy rewards a curated educational corpus, which is what FineWeb-Edu and Wikipedia
are good at. HellaSwag and PIQA want the knowledge nobody writes down because it is too obvious to
state — that a wet cup slips, that you preheat before baking, what happens next when someone picks up
a guitar. **That is a coverage gap in the corpus, not a capacity limit in the model**, which makes it
the kind of gap targeted data can close.

After the organizers confirmed that synthetic data generated via Adaption Labs is permitted for
Track 01, this was worth testing properly.

### What was generated

7,000 rows through Adaption's `invent` endpoint, in two mixtures aimed at the two losing benchmarks:

| corpus | rows | tokens | targets |
|---|---|---|---|
| `physical` | 4,000 | 3,837,505 | PIQA — procedural physical knowledge |
| `everyday` | 3,000 | 2,506,245 | HellaSwag — how ordinary events unfold |
| **total** | **7,000** | **6,343,750** | |

Cost is a flat 0.1 credits per row regardless of row length, so the lever is **length, not count**:
the prompts ask for multi-paragraph explanatory prose rather than short items, which is free. The
full 850-credit balance buys ~8,500 rows, so 6.3 M tokens is the realistic ceiling — **0.1 % of the
6 B pretraining corpus.** That is why this is an anneal and not additional pretraining.

### The contamination problem, and why the prompts avoid the benchmarks

Synthetic data is the one source where contamination is *plausible rather than accidental*. A hosted
model asked for "PIQA-style physical commonsense" has very likely read PIQA and can reproduce
memorised items without being asked to — and since `decontaminate.py` matches 16-token n-grams, a
reworded leak walks straight through it. Prompting for benchmark-shaped output would be training on
the test set with extra steps.

So the prompts target the *underlying skill* in continuous prose, never the item format, and the
result was checked rather than assumed — with the project's own `EvalNgramIndex`, the same index and
window used on the 6 B corpus, not a second and laxer implementation:

| check | result |
|---|---|
| contaminated rows | **0 of 7,000**, against 2,737,882 eval n-grams at window 16 |
| exact duplicate rows | 0 (4000/4000 and 3000/3000 completions unique) |
| cross-row 16-gram overlap | 0.03 % and 0.06 % |
| PIQA vocabulary coverage | 94 % of its 400 most common content words |

That last pair is the combination worth having: the text discusses the same subject matter as the
benchmarks while reproducing none of their items.

### The dose-response curve

Five anneals from the finished checkpoint, each ~$0.43 on a pinned H100, varying the synthetic
fraction and the restart learning rate. Every variant is scored fp32 through the same harness and
compared to the submitted checkpoint by paired bootstrap over items:

| dose | HellaSwag | WikiText ppl | HellaSwag gap | p | verdict |
|---|---|---|---|---|---|
| 0 % (submitted) | 0.3001 | **40.35** | — | — | — |
| 4 % | 0.3018 | 41.54 | +0.0017 | 0.39 | not resolved |
| **20 %** | **0.3107** | 42.63 | **+0.0106** | 0.0000 | **survives FDR** |
| 20 %, shorter | 0.3077 | 42.94 | +0.0076 | 0.0011 | **survives FDR** |
| 50 % | 0.3061 | 43.97 | +0.0060 | 0.0248 | resolved, fails FDR |

**HellaSwag is an inverted U; perplexity is monotone.** The capability peaks near 20 % synthetic and
falls back by 50 %, while word perplexity degrades at every step with no turning point. So the
synthetic data does not add capability for free — it **buys HellaSwag by spending general
language-modelling quality**, and past the optimum it stops buying while still spending. The 50 % run
shows why: its validation loss rose from the first evaluation onward (3.0434 → 3.0606) while training
loss fell to 2.32. At 17 passes over 6.3 M tokens it was memorising, not learning.

At the optimum, HellaSwag goes from 0.3001 to 0.3107 against GPT-2's 0.3114 — **94 % of the gap to a
model 2.5× the size, closed.**

Reported with a Benjamini-Hochberg correction across all 20 comparisons, because five variants times
four benchmarks expects about one false positive at p < 0.05 unaided, which is exactly the effect
size being looked for. Three comparisons have intervals excluding zero and two survive the
correction; all three are HellaSwag.

### What failed

**PIQA never moved.** Not at any dose: +0.0005 to +0.0033, every p above 0.68. The `physical` corpus
was 4,000 of the 7,000 rows — the larger investment — and it is the one that produced nothing. The
`everyday` corpus, aimed at HellaSwag, is what worked.

That asymmetry is the most interesting part of the result and there is no confident explanation for
it here. One plausible reading is that HellaSwag's task — continue a narrative plausibly — is closer
to what a language model does during ordinary pretraining, so prose about unfolding events transfers
directly; PIQA asks which of two solutions to a physical problem works, which may need the contrast
between a good and a bad approach rather than exposition of the good one. Testing that would mean
generating passages built around failed attempts, and it is the obvious next experiment rather than
a claim.

**A higher restart learning rate was simply worse** (0.3× peak): worst validation loss of any
variant, and the largest WinoGrande drop.

### Why the submitted model is the pre-anneal checkpoint

The anneal is a trade, not an upgrade: +0.0106 on HellaSwag against +2.3 WikiText perplexity and a
worse validation loss. Word perplexity is a scored metric too, so trading a resolved gain on one
benchmark for a clear loss on another is not obviously winning, and the pre-anneal checkpoint is the
one whose reproducibility is documented above. **`runs/kestrel-main/best.pt` is the submission.** The
annealed checkpoints are kept, scored and reported because the experiment is a result in its own
right — a dose-response curve with an identified optimum, a replicated effect that survives
correction, and a clean negative — for about $2.20 of GPU time.

Reproduce:

```bash
python scripts/adaption_probe.py                 # cost model, spends nothing
python scripts/adaption_generate.py --rows 4000 --mixture physical --tag main
python scripts/adaption_collect.py               # collect finished jobs, then exit
python scripts/adaption_decontam.py data/synthetic/kestrel-physical-main.json
python scripts/prepare_anneal.py                 # tokenize into shards
modal run --detach scripts/modal_anneal.py --tokens 200e6 --synth-frac 0.2
python scripts/anneal_sweep.py                   # paired tests + FDR across variants
```

## Repository layout

```
kestrel/
  model.py          Llama-shaped decoder, chunked cross-entropy, param accounting
  variants.py       the seven shapes compared in the ablation, all fitted to one 49.5 M budget
  muon.py           Muon optimizer (Newton-Schulz orthogonalization)
  trainer.py        WSD schedule, gradient accumulation, resumable checkpoints, NVML energy
  data.py           uint16 memory-mapped token shards, seeded resumable ordering
  corpus.py         the source mix, with licenses
  decontaminate.py  n-gram index over the scored eval splits
  evaluate.py       lm-evaluation-harness adapter
scripts/
  count_params.py   verify the 50 M cap            (competition deliverable)
  train_tokenizer.py
  prepare_data.py   stream -> decontaminate -> tokenize -> shard
  check_decontam_coverage.py   what fraction of each eval split the filter can actually see
  check_decontam_cost.py       what fraction of the corpus each window drops, and why
  bench.py          throughput/memory measurement, subprocess per cell
  train.py
  ablate.py         one-axis-at-a-time experiment driver
  eval.py           the five scored metrics
  eval_baseline.py  GPT-2 through the identical harness, for a like-for-like row
  paired_eval.py    paired bootstrap over items; takes two checkpoints or a checkpoint and an HF id
  adaption_probe.py     map the credit cost model without spending a credit
  adaption_generate.py  generate a synthetic mixture, quoting before it spends
  adaption_collect.py   collect finished jobs and exit, so nothing waits on a laptop
  adaption_decontam.py  run the project's own eval n-gram index over generated text
  prepare_anneal.py     tokenize the synthetic corpus into shards
  anneal.py             restart from a finished checkpoint on a fresh short schedule
  anneal_sweep.py       score every variant against one baseline, with an FDR correction
  modal_prepare.py  build the 6 B corpus on Modal's CPUs (measured 2x the laptop's streaming rate)
  modal_bench.py    the throughput frontier on rented hardware
  modal_train.py    the rented-H100 path for the main run
  modal_anneal.py   build the synthetic/original mixture on the Volume and anneal there
tests/
  test_decontaminate.py        the hash, against direct mod-2**64 arithmetic
```

## Built With

`python` · `pytorch` · `triton` · `huggingface-tokenizers` · `huggingface-datasets` ·
`lm-evaluation-harness` · `numpy` · `nvidia-ml-py` · `wandb` · `muon` · `fineweb-edu` · `wikipedia` ·
`modal` · `adaption-labs` · `claude-code` · `openai-codex` · `chatgpt` · `gemini`

## AI assistance disclosure

The competition permits AI coding assistants without restriction and requires them to be disclosed.
Several were used on this project, and undisclosed use would be misrepresentation, so this section
is deliberately specific about which and where.

**Assistants used:**

| tool | used for |
|---|---|
| **Claude Code** (Anthropic) | the bulk of the implementation: model, trainer, Muon, data pipeline, decontamination index, evaluation harness, ablation driver, the synthetic-data and anneal scripts, and drafting this README |
| **Gemini** (Google) | design discussion, cross-checking approaches against a second opinion, and drafting and reviewing this README |
| **OpenAI Codex** | code generation and refactoring assistance during development |
| **ChatGPT** (OpenAI) | design discussion, debugging, and explanation of techniques while learning them |

- **AI-assisted:** essentially all of the code in this repository was written with one or more of the
  assistants above, including the CUDA-memory and Windows-specific debugging documented in the
  measurement tables.
- **Not AI:** the model weights. Every parameter comes from the training runs described above, from
  random initialization. No pretrained checkpoint, no fine-tuning of anyone else's model, no
  distillation, and no hosted inference API is any part of the submitted model — as Track 01
  requires.
- **Two places a hosted model is touched, neither of them the model itself.**
  `scripts/eval_baseline.py` downloads the released GPT-2 checkpoint purely to *score* it as the
  comparison row in the Results table: it is loaded under `torch.no_grad()`, never trained against,
  and nothing it produces reaches Kestrel's weights, data or tokenizer. And the annealing experiment
  trains on **synthetic text generated by Adaption Labs' hosted models** — permitted for Track 01 by
  explicit confirmation from the organizers. That text is training *data*, not weights: it was
  decontaminated against every evaluation set before use (0 of 7,000 rows matched) and it went only
  into the anneal variants, never into the submitted checkpoint. See *Synthetic data and annealing*.
- **Decisions:** mine throughout. The assistants wrote code and argued alternatives; the calls that
  shaped this project were:
  - **Entering Track 01 specifically to learn how these systems work**, which set the standard that
    nothing goes in this repo unless I understand why it is there — and is why the reasoning is
    written out rather than just the results.
  - **Requiring the architecture to be measured rather than adopted.** The published "deep and thin"
    advice for sub-1 B models was the easy path; I chose to spend GPU-hours testing it on this
    hardware, and it lost.
  - **Renting an H100 for the final run instead of training overnight on the laptop** — taken after
    the benchmark put the same run at 24.8 h against 2.1 h, i.e. on measurement rather than on the
    spec sheet, and scoped to two hours and $8.22.
  - **Insisting that evaluation be reproducible before any number was reported**, which is what
    forced fp32 scoring, right-padding, and the discovery that a stored result was not reproducible.
  - **Pursuing the synthetic-data branch** once the organizers permitted it, including how to split
    the credits across the two target benchmarks and how far to push the dose sweep.
  - **Requiring the synthetic corpus be decontaminated and measured before training on it**, rather
    than trusting that prompts avoiding the benchmarks would be enough.
  - **Keeping the pre-anneal checkpoint as the submission**, declining a benchmark gain that came at
    the cost of perplexity.
  - **Treating credentials as untouchable** — no key in a file that could be committed, and the repo
    audited before it went public.

## Acknowledgements

**Special thanks to [Adaption Labs](https://adaptionlabs.ai).** They provided the credits that made
the synthetic-data experiment possible, and that experiment turned out to be the part of this
project I learned the most from — not because it worked, but because it only *partly* worked.
Generating the data was the easy half; the interesting half was everything their platform forced me
to think about afterwards. How do you know synthetic text has not smuggled the benchmark back into
your training set? What dose is too much? How do you tell a real improvement from one of twenty
draws from noise?

The dose-response curve in this README — HellaSwag rising then falling while perplexity degrades
throughout — exists because those credits let me ask the question five times instead of once. A
single run would have produced a number. Five produced an understanding, including a clean negative
result on PIQA that I would rather report than hide.

## Credits and licenses

- **Muon** — Keller Jordan et al., 2024, [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- **FineWeb-Edu** — HuggingFaceFW, ODC-By 1.0
- **English Wikipedia** — Wikimedia Foundation, CC BY-SA 4.0
- **lm-evaluation-harness** — EleutherAI, MIT
- **Adaption Labs** — synthetic training data for the annealing experiment, generated through their
  `invent` API. Used as training *data* only, decontaminated before use, and never part of the
  submitted checkpoint.
- **Modal** — rented H100 for the 6 B-token run and the anneal sweep.
- Architecture follows the Llama family (Touvron et al.) and the deep-and-thin finding of
  **MobileLLM** (Liu et al., 2024); the `cl100k_base` split pattern comes from OpenAI's `tiktoken`.

Code in this repository is MIT licensed.





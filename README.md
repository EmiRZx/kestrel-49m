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

Against a model **2.5× larger** — 124.4 M parameters against 49.5 M — Kestrel wins two of the four
multiple-choice tasks and trails the other two.

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
difference far outside the ±0.17 ppl a paired bootstrap over these 62 documents resolves. That is
the expected shape of the result — WikiText-103 is drawn from the same web-text distribution as
GPT-2's training corpus, and only 18 % of Kestrel's.

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

The cap counts embeddings, so the embedding table is the first thing to argue about. At `d_model`
384, a 32 k vocabulary costs 12.6 M parameters — a quarter of the entire budget spent on a lookup
table. A 16 k BPE vocabulary fitted to our own corpus costs 6.3 M and returns the difference to the
layers, at the price of more tokens per document.

```
$ python scripts/count_params.py
========================================================================
CONFIG
  vocab_size       16384
  d_model          384
  n_layers         26
  n_heads          6
  n_kv_heads       2
  ffn_hidden       1096
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
  per_layer           1,656,704
  layers             43,074,304
  embedding           6,291,456
  non_embedding      43,074,688
  total              49,366,144

MEASURED (torch, tied weights counted once)
  embedding           6,291,456
  non_embedding      43,074,688
  total              49,366,144

  fwd+bwd FLOPs/token   418,880,256

  budget              50,000,000
  headroom               633,856
  utilisation             98.73%   -> WITHIN BUDGET
  embedding share         12.74%   (lower is better: params in layers, not lookup)
========================================================================
```

Two independent counts are printed because they can disagree, and if they did, every parameter number
in this README would be wrong: the closed-form estimate is what sized the model, and the measured
count is what `torch` actually built. They agree at **49,366,144 — 98.73 % of the cap**.

Embeddings are **tied** (the output head reuses the embedding matrix), and `param_report()`
deduplicates by `id(p)` so tied weights cannot be double-counted in either direction. The script
exits non-zero if the model is over budget, which makes it usable as a pre-commit check.

## How it works

### Architecture

A conventional Llama-shaped decoder: RMSNorm pre-norm, RoPE, grouped-query attention, SwiGLU feed
-forward, QK-norm for stability, tied embeddings. The conventionality is deliberate — it means
`lm-evaluation-harness`, `transformers`, GGUF, and ONNX all work with no custom glue, so the novelty
budget can be spent on data and training rather than on a block nobody else can run.

Four shapes were compared head to head. All four are fitted to the **same** parameter count — each
one takes the largest `ffn_hidden` (in multiples of 8) that stays under 49.5 M at vocab 16,384, so
the spread across the four is 0.25 M, or 0.51 %:

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
H100 at micro-batch 64; full data in `runs/ablation_arch_h100.json`.

| variant | shape | val loss | val ppl | wt103 ppl | tokens/s | MFU | peak VRAM | GPU-h |
|---|---|---|---|---|---|---|---|---|
| **`wide-shallow`** | **640 × 9** | **3.3820** | **29.43** | **41.37** | 817,010 | 30.4 % | 19.53 GB | 0.129 |
| `mid` | 512 × 14 | 3.3868 | 29.57 | 41.79 | 682,786 | 26.6 % | 23.74 GB | 0.172 |
| `wide-shallow-7` | 640 × 7 | 3.3986 | 29.92 | 41.65 | 900,839 | 32.0 % | 17.82 GB | 0.114 |
| `extra-wide-6` | 768 × 6 | 3.3984 | 29.92 | 42.64 | 933,874 | 33.4 % | 16.57 GB | 0.110 |
| `deep-thin` | 384 × 26 | 3.3938 | 29.78 | 41.83 | 519,594 | 22.0 % | 31.71 GB | 0.212 |
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

## Repository layout

```
kestrel/
  model.py          Llama-shaped decoder, chunked cross-entropy, param accounting
  variants.py       the four shapes compared in the ablation, all fitted to one 49.5 M budget
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
tests/
  test_decontaminate.py        the hash, against direct mod-2**64 arithmetic
```

## Built With

`python` · `pytorch` · `triton` · `huggingface-tokenizers` · `huggingface-datasets` ·
`lm-evaluation-harness` · `numpy` · `nvidia-ml-py` · `wandb` · `muon` · `fineweb-edu` · `wikipedia` ·
`claude-code`

## AI assistance disclosure

This project was built with **Claude Code** (Anthropic) as a coding assistant, which the competition
rules permit without restriction and require to be disclosed. To be specific about what that means
here:

- **AI-assisted:** the implementation of every module listed above — the model, trainer, Muon
  implementation, data pipeline, decontamination index, benchmark harness, ablation driver, and this
  README were written in collaboration with Claude Code, with iterative debugging of the CUDA memory
  and Windows-specific issues documented in the measurement tables.
- **Not AI:** the model weights. Every parameter comes from the training run described above, on the
  hardware described above, from random initialization. No pretrained checkpoint, no fine-tuning, no
  distillation from a larger model, and no hosted inference API is used as any part of the submitted
  model — as Track 01 requires. `scripts/eval_baseline.py` does download the released GPT-2
  checkpoint, but only to *score* it as the comparison row in the Results table: it is loaded under
  `torch.no_grad()`, never trained against, and nothing it produces reaches Kestrel's weights, data,
  or tokenizer.
- **Decisions:** the project direction, the constraints accepted, and which measured result to act on
  were mine.

## Credits and licenses

- **Muon** — Keller Jordan et al., 2024, [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- **FineWeb-Edu** — HuggingFaceFW, ODC-By 1.0
- **English Wikipedia** — Wikimedia Foundation, CC BY-SA 4.0
- **lm-evaluation-harness** — EleutherAI, MIT
- Architecture follows the Llama family (Touvron et al.) and the deep-and-thin finding of
  **MobileLLM** (Liu et al., 2024); the `cl100k_base` split pattern comes from OpenAI's `tiktoken`.

Code in this repository is MIT licensed.





# Kestrel-47M

A general-purpose language model with **49.4 million parameters**, trained from scratch on one
laptop GPU — an RTX 4070 Laptop with 8 GB of VRAM.

Built for **Global Innovation Build Challenge V2, Track 01 (Foundational LLM)**, which caps entries
at 50 M trainable parameters *including* the token embeddings and output head, and forbids any
pretrained weights, fine-tuning, or distillation from a larger model.

The interesting constraint is not the parameter cap. It is that 8 GB of VRAM and a shared laptop
mean every design decision has to be justified in **accuracy per GPU-hour**, not accuracy per
parameter — and that a training run which will certainly be interrupted has to be resumable to the
token.

<!-- RESULTS: fill from runs/kestrel-main/eval_best.json before submission -->

## Results

| Benchmark | Kestrel-47M | Chance | GPT-2 124 M (scored here) |
|---|---|---|---|
| HellaSwag (`acc_norm`) | _TBD_ | 0.25 | _TBD_ |
| ARC-Easy (`acc_norm`) | _TBD_ | 0.25 | _TBD_ |
| PIQA (`acc_norm`) | _TBD_ | 0.50 | _TBD_ |
| WinoGrande (`acc`) | _TBD_ | 0.50 | _TBD_ |
| WikiText-103 (`word_perplexity`, test) | _TBD_ | — | _TBD_ |

The GPT-2 column is the released `openai-community/gpt2` checkpoint, run through **this** harness on
**these** splits rather than quoted from a paper, so the comparison is like-for-like at 2.6× the
parameter budget. To be unambiguous about the Track 01 rule: GPT-2 appears only as a number in that
column. No pretrained weights initialize, fine-tune, or teach Kestrel — nothing is distilled from it.

All five Kestrel numbers come from [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
v0.4.12, zero-shot, on the same splits the competition scores. Reproduce with:

```bash
python scripts/eval.py --ckpt runs/kestrel-main/best.pt
```

Word-level perplexity is reported rather than token-level, because token-level perplexity depends on
the tokenizer and is therefore **not comparable between entries** — a model with a 8 k vocabulary
will show a lower token-level perplexity than an identical model with a 32 k vocabulary while being
no better at predicting English.

## Hardware, training time, and compute

| | |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8 GB GDDR6, 140 W power cap |
| Precision | bf16 autocast, fp32 master weights, `torch.compile` (Inductor + Triton) |
| Throughput | _TBD_ tokens/s sustained |
| Training tokens | _TBD_ |
| Wall-clock training time | _TBD_ GPU-hours, across _TBD_ overnight sessions |
| Estimated compute | _TBD_ FLOPs (`6N + 12·L·d·T` per token, measured params) |
| Energy | _TBD_ kWh, integrated from NVML power samples every logging interval |
| Peak VRAM | _TBD_ GB |

Nothing in that table is reconstructed after the fact: `kestrel/trainer.py` writes tokens/s, MFU,
GPU-seconds, GPU temperature, SM clock, throttle reasons, and integrated watt-seconds into
`runs/<name>/metrics.jsonl` at every logging step, and carries the totals through every
checkpoint/resume cycle.

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

<!-- DECONTAM: numbers below are from the 300 M-token pilot corpus; refresh from
     data/tokens_v16384/prepare_report.json once the full 6 B corpus is built -->

**Price** — what that reach costs. First on 300,000 documents split evenly between the two sources,
which isolates the window from the mix (`scripts/check_decontam_cost.py --docs 150000 --windows 16 32`):

| window | FineWeb-Edu | Wikipedia | all | rate |
|---|---|---|---|---|
| 32 | 0 / 150,000 | 3 / 150,000 | 3 / 300,000 | 0.0010 % |
| 16 | 11 / 150,000 | 133 / 150,000 | 144 / 300,000 | 0.0480 % |

At 32 tokens the filter dropped one document in a hundred thousand and **never once fired on
FineWeb-Edu**, which is 82 % of the corpus. It was not being conservative; it was doing nothing.

Then at window 16 on the real 300 M-token pilot corpus, at the real 82/18 mix
(`data/tokens_pilot/prepare_report.json`):

| source | documents kept | dropped | rate | matched |
|---|---|---|---|---|
| FineWeb-Edu | 224,235 | 21 | 0.0094 % | hellaswag 11, arc_easy 4, wikitext103 5, piqa 1 |
| Wikipedia | 45,028 | 28 | 0.0621 % | wikitext103 28 |
| **total** | **269,263** | **49** | **0.0182 %** | |

One document in 5,500 — the weighted rate is *lower* than the 50/50 sample because Wikipedia, the
source that matches most, is only 18 % of the mix.

The composition is more informative than the rate. Every single Wikipedia drop matches WikiText-103,
which is exactly right: WikiText-103 *is* Wikipedia, so those are the same articles and removing them
is the point. Among the FineWeb-Edu drops, some are genuine verbatim overlap (`British armed trawler
HMT Stella Sirius was sunk by`) and some are stock phrases any educational page contains (`Mercury,
Venus, Earth, Mars, Jupiter, Saturn, Uranus`). `check_decontam_cost.py` decodes every matched span,
so that judgement is inspectable rather than asserted.

Sixteen tokens is ~8–9 English words at this tokenizer's 2.89 chars/token, already stricter than the
13-word n-gram GPT-3 filtered on, and that is where it stops: 12 tokens would buy another ~10 points
of ARC-Easy coverage, but a 6–7 word span is a phrase and not a quotation.

One limitation stays on the record. WikiText's own preprocessing (`@-@`, `<unk>`, spaced punctuation)
means a *same-article* overlap with raw Wikipedia will not hash-match, even though verbatim copies of
the benchmark files do. And 19 % of scored strings are still shorter than 16 tokens and therefore
unfilterable at any window — `prepare_report.json` records that count as `eval_texts_below_window`
instead of omitting it.

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
a fixed laptop converts directly into fewer GPU-hours. It applies only to 2D hidden matrices;
embeddings, the output head, and 1D norm gains stay on AdamW.

The LR schedule is a function of **tokens seen**, not of step index. That sounds pedantic and is not:
the micro-batch size changes with whatever fits in VRAM, and ablations deliberately use different
batches, so steps are not a comparable axis between runs while tokens always are.

Three things make an interrupted run harmless. Checkpoints carry both optimizer states, the dataset
cursor, the RNG states, and the cumulative GPU-seconds and watt-seconds — so a resumed run continues
the identical data order and its efficiency accounting stays correct across sessions. Checkpoints are
written to a temp file and `replace()`d, so a crash mid-save cannot destroy the previous one. And
`--max-hours` ends a night's session on a checkpoint rather than on a killed process.

### Fitting a 49 M model into 8 GB

The largest tensor in a small-vocabulary LM is not the weights — it is the logits. At batch 16 ×
sequence 1024 × vocab 16,384, a single fp32 logit tensor is 1.07 GB, and the backward pass needs
another. `chunked_cross_entropy` never materialises it: the hidden states are split into 2048-row
chunks, each chunk's projection-and-loss is recomputed under `torch.utils.checkpoint`, and the
training forward pass returns **no logits at all** unless asked. Measured with `--loss-chunk 0`,
micro-batch 8 will not fit at all; with chunking on, `deep-thin` runs 12 sequences and
`wide-shallow` 20. It is load-bearing, not an optimisation.

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
54 % more than very-deep** — 13 GPU-hours of difference over a 6 B-token run, which on one laptop is
more than a night and a half. Depth costs throughput twice: more sequential kernel launches, and more
stored activations, which caps the micro-batch — very-deep cannot hold 12 sequences where
wide-shallow holds 20. This is why the published "depth beats width below 1 B parameters" result could
not simply be adopted here: the question on a fixed compute budget is not accuracy per parameter but
accuracy per GPU-hour, and a shape that is 32 % worse per token can still win.
`scripts/ablate.py --suite arch` decides it empirically at 300 M tokens per cell.

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

<!-- ABLATIONS: paste tables from runs/ablation_*.json before submission -->

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
  model — as Track 01 requires.
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





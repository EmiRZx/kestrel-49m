# Kestrel-47M

A general-purpose language model with **47.2 million parameters**, trained from scratch on one
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

| Benchmark | Kestrel-47M | Chance | GPT-2 (124 M, reproduced here) |
|---|---|---|---|
| HellaSwag (`acc_norm`) | _TBD_ | 0.25 | _TBD_ |
| ARC-Easy (`acc_norm`) | _TBD_ | 0.25 | _TBD_ |
| PIQA (`acc_norm`) | _TBD_ | 0.50 | _TBD_ |
| WinoGrande (`acc`) | _TBD_ | 0.50 | _TBD_ |
| WikiText-103 (`word_perplexity`, test) | _TBD_ | — | _TBD_ |

All five numbers come from [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
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
```

<!-- PARAM COUNT: paste real output before submission -->

Embeddings are **tied** (the output head reuses the embedding matrix), and `param_report()`
deduplicates by `id(p)` so tied weights cannot be double-counted in either direction. The script
exits non-zero if the model is over budget, which makes it usable as a pre-commit check.

## How it works

### Architecture

A conventional Llama-shaped decoder: RMSNorm pre-norm, RoPE, grouped-query attention, SwiGLU feed
-forward, QK-norm for stability, tied embeddings. The conventionality is deliberate — it means
`lm-evaluation-harness`, `transformers`, GGUF, and ONNX all work with no custom glue, so the novelty
budget can be spent on data and training rather than on a block nobody else can run.

Four shapes were built at a matched ~47 M parameters and compared head to head:

| variant | `d_model` × layers | heads (Q/KV) | FFN | params |
|---|---|---|---|---|
| `very-deep` | 320 × 36 | 5 / 1 | 896 | 47.1 M |
| `deep-thin` | 384 × 26 | 6 / 2 | 1024 | 47.2 M |
| `mid` | 512 × 14 | 8 / 2 | 1376 | 47.2 M |
| `wide-shallow` | 640 × 9 | 10 / 2 | 1728 | 49.2 M |

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

Six billion tokens against 47 M parameters is roughly 128 tokens per parameter — far past
Chinchilla-optimal, which is the *correct* choice when parameters are capped by rule and compute is
the only free variable.

Before anything is written to disk, every document is checked against an n-gram index built from
the evaluation splits that are actually scored — HellaSwag validation, ARC-Easy test, PIQA
validation, WinoGrande validation, and WikiText-103 **test** — and dropped if it shares a 32-token
window with any of them (`kestrel/decontaminate.py`). The index is saved as `decontam.npz` next to
the shards so the filter is auditable rather than merely claimed, and the drop count goes in
`prepare_report.json`.

The filter's limit is stated rather than glossed, because a 32-token window can only match text at
least 32 tokens long, and benchmark answer choices are often two words. Each item is therefore
indexed as stem+continuation pairs — question with each answer, context with each ending — which is
both the form a scraped quiz page takes and what lifts coverage from 3 % to 22 % of ARC-Easy's
scored strings. `scripts/check_decontam_coverage.py` prints the coverage per split, so the claim can
be checked instead of trusted:

<!-- DECONTAM: paste the real drop rate from data/tokens_v16384/prepare_report.json, and the
     coverage table from scripts/check_decontam_coverage.py -->


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

### Fitting a 47 M model into 8 GB

The largest tensor in a small-vocabulary LM is not the weights — it is the logits. At batch 16 ×
sequence 1024 × vocab 16,384, a single fp32 logit tensor is 1.07 GB, and the backward pass needs
another. `chunked_cross_entropy` never materialises it: the hidden states are split into 2048-row
chunks, each chunk's projection-and-loss is recomputed under `torch.utils.checkpoint`, and the
training forward pass returns **no logits at all** unless asked. With chunking disabled, batch 8
does not fit; with it, batch 16 fits with room to spare.

## What was measured, rather than assumed

Full data in `runs/bench_frontier.jsonl`, reproduce with `scripts/bench.py --matrix`. Each cell runs
in a fresh subprocess, because a CUDA OOM leaves the process unusable and a sweep that shares one
cannot survive its own failures.

Throughput at seq 1024, vocab 16,384, bf16, compiled, at each variant's best micro-batch:

| variant | micro-batch | tokens/s | peak VRAM | MFU |
|---|---|---|---|---|
| `very-deep` (320 × 36) | 12 | 47,719 | 7.21 GB | 24.6 % |
| `deep-thin` (384 × 26) | 12 | 52,881 | 6.68 GB | 26.8 % |
| `mid` (512 × 14) | 16 | 62,450 | 6.74 GB | 29.0 % |
| `wide-shallow` (640 × 9) | 16 | **66,575** | 5.84 GB | 30.5 % |

At matched parameters, wide-shallow processes **26 % more tokens per GPU-hour than deep-thin and
40 % more than very-deep**. Depth costs throughput twice: more sequential kernel launches, and more
stored activations, which caps the micro-batch. This is why the published "depth beats width below
1 B parameters" result could not simply be adopted here — the question on a fixed compute budget is
not accuracy per parameter but accuracy per GPU-hour, and a shape that is 26 % worse per token can
still win. `scripts/ablate.py --suite arch` decides it empirically at 300 M tokens per cell.

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

<!-- ABLATIONS: paste tables from runs/ablation_*.json before submission -->

## Repository layout

```
kestrel/
  model.py          Llama-shaped decoder, chunked cross-entropy, param accounting
  variants.py       the four ~47 M shapes compared in the ablation
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
  bench.py          throughput/memory measurement, subprocess per cell
  train.py
  ablate.py         one-axis-at-a-time experiment driver
  eval.py           the five scored metrics
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





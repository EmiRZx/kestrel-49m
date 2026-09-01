# Kestrel-47M — GIBC V2, Track 01 (Foundational LLM)

Working plan. This file is the source of truth for decisions; update it as things change.

- **Deadline:** 2026-09-21, 23:45 Taipei (UTC+8) = 18:45 local (UTC+3). Submit a day early.
- **Entrant:** solo.
- **Repo:** `E:\kestrel` (E: chosen over C: — C: has only 39 GB free, we need ~60 GB for data + checkpoints).

## Hard rules (Track 01)

| Rule | Consequence |
|---|---|
| ≤ 50,000,000 total trainable params, **including embeddings and output head** | Tie embeddings; keep vocab small. Param counter script is a required deliverable. |
| Trained from scratch — no pretrained init, no fine-tuning, **no distillation from a larger model** | GPT-2 is used only as an eval-time baseline for comparison, never as weights. No teacher logits. |
| Public datasets and standard frameworks allowed | FineWeb-Edu, Wikipedia, PyTorch, HF tokenizers all fine. Credit everything. |
| No hosted inference API as the model | Featherless/Adaption credits may be used for tooling/eval only, not as the submission. |
| README must state hardware, total training time, approximate compute | Log GPU-hours, tokens/s, MFU, peak VRAM, and energy from the start. Not reconstructable later. |
| Eval is **fixed**: HellaSwag, ARC-Easy, PIQA, WinoGrande via lm-evaluation-harness + held-out WikiText-103 perplexity | This dictates everything below. |

## Strategy: map every scored criterion to a concrete artifact

1. **Perplexity & Accuracy** (also the tiebreak criterion) → general-purpose LM on high-quality web
   text. Include a Wikipedia slice so WikiText-103 perplexity is in-distribution. Report
   `word_perplexity` and `bits_per_byte`, not just token-level ppl (token-level ppl is
   tokenizer-dependent and therefore not comparable across entries — say so in the README).
2. **Reasoning Performance** → the four benchmarks are commonsense/science QA. FineWeb-Edu is the
   single highest-leverage choice here. End the run with an anneal phase on the highest-quality
   subset, which is a known large benchmark bump for cheap.
3. **Training Efficiency** → Muon optimizer for hidden matrices (fewer steps to target loss),
   bf16 + `torch.compile`, fused AdamW for embeddings, WSD LR schedule. Report tokens/s, MFU,
   peak VRAM, GPU-hours, and **accuracy per kWh** measured via NVML.
4. **Innovation** → the ablation sweep *is* the innovation: depth-vs-width at fixed params,
   vocab-size sweep, optimizer comparison, data-mix comparison. Plus n-gram decontamination of
   the training set against all five eval sets, which almost nobody in a student field does.
5. **Documentation & Demo** → README with full config, param count, hardware, compute, benchmark
   table vs a self-reproduced GPT-2 baseline. Demo = the model running offline in the browser.

## Candidate architecture (to be confirmed by ablation, not assumed)

Llama-shaped on purpose, so `lm-evaluation-harness`, `transformers`, and GGUF/ONNX export all
work with zero custom glue. Novelty goes into data and training, not into a block nobody can run.

Deep-and-thin, following MobileLLM's finding that depth beats width below 1B params:

| | |
|---|---|
| `d_model` | 384 |
| `n_layers` | 26 |
| `n_heads` / `n_kv_heads` | 6 / 2 (GQA 3:1, head_dim 64) |
| FFN | SwiGLU, hidden 1024 |
| Norm | RMSNorm, pre-norm, + QK-norm for stability |
| Position | RoPE |
| Vocab | 16,384, custom BPE trained on our own mix |
| Embeddings | tied input/output |
| Seq len | 1024 (optional 2048 extension at the end) |

Param count: 26 × 1,573,632 (layers) + 6,291,456 (tied embed) ≈ **47.2 M**, ~2.8 M under the cap.

## Data

Target ~6 B tokens for the main run (≈128 tokens/param — deliberately far past Chinchilla-optimal,
which is correct when parameters are capped and compute is the only free variable).

| Source | Share | Why |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` (sample-10BT) | ~80% | Highest benchmark yield per token at small scale. |
| Wikipedia (English, recent dump) | ~15% | WikiText-103 perplexity is a scored metric; keeps it in-distribution. |
| Reserved / ablation slot | ~5% | Candidates: OpenWebMath, TinyStories, public textbook corpora. Only if ablation shows gains. |

Non-negotiable: **n-gram decontamination** of the training mix against the test splits of
HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103. Report how many documents were removed.
This is both honest and a differentiator.

Storage: uint16 token shards ≈ 12 GB for 6 B tokens, plus raw parquet during tokenization.
Stream-tokenize and delete raw shards as we go.

## Compute budget

Local: RTX 4070 Laptop, 8 GB VRAM, 140 W cap. **Measured**, not estimated (see below): 47–67 k
tokens/s depending on variant, i.e. 6 B tokens in **25–35 GPU-hours**, three or four overnight
sessions. User confirmed *some* nights, not all — so the schedule assumes ~6 usable nights and
keeps a spare.

Kaggle (30 GPU-h/week free, T4 ×2): ablation runs only, ~300 M tokens each ≈ 1.5 h per run.
That funds ~15–20 ablations without touching the local GPU's main-run time.

Everything must be checkpoint/resume safe from run one — a laptop training overnight will get
interrupted. Also log temperature and clock throttling.

## Measured, 2026-08-31 (RTX 4070 Laptop 8 GB, seq 1024, vocab 16384, bf16, `torch.compile`)

Every cell is a fresh subprocess (`scripts/bench.py --matrix`); a CUDA OOM poisons a process, so a
sweep that shares one cannot survive its own failures. Raw data in `runs/bench_frontier.jsonl`.

| variant | shape | params | best micro-batch | tok/s | peak VRAM | MFU |
|---|---|---|---|---|---|---|
| very-deep | 320 × 36L | 47.1 M | 12 | 47,719 | 7.21 GB | 24.6 % |
| deep-thin | 384 × 26L | 47.2 M | 12 | 52,881 | 6.68 GB | 26.8 % |
| mid | 512 × 14L | 47.2 M | 16 | 62,450 | 6.74 GB | 29.0 % |
| wide-shallow | 640 × 9L | 49.2 M | 16 | **66,575** | 5.84 GB | 30.5 % |

At fixed parameters, **wide-shallow processes 26 % more tokens per GPU-hour than deep-thin and
40 % more than very-deep** — depth costs throughput twice over: more sequential kernel launches, and
more stored activations, which caps the micro-batch. So MobileLLM's "depth beats width" cannot be
adopted on faith here; the question is not accuracy per parameter but accuracy per GPU-hour, and
`scripts/ablate.py --suite arch` settles it at 300 M tokens per cell.

Decisions these measurements forced:

| Finding | Consequence |
|---|---|
| `torch.compile` (via `triton-windows`, since PyTorch-on-Windows ships no Triton): 51.6 k vs 25.7 k tok/s, 4.68 vs 6.57 GB | Compile is on by default. It is 2× throughput *and* 1.9 GB cheaper, so it also buys micro-batch. |
| `F.scaled_dot_product_attention(..., enable_gqa=True)` disqualifies the flash/mem-efficient kernels and falls back to the math backend, which materialises `(B, H, T, T)` | Keep the explicit `repeat_kv` path. `sdpa_enable_gqa=False` is the default; the flag exists only to reproduce the finding. |
| `--loss-chunk 0` OOMs at batch 8 | Chunked, checkpointed cross-entropy is load-bearing, not an optimisation. The `(B, T, vocab)` logit tensor is the single largest allocation on this card. |
| Without `set_per_process_memory_fraction(0.9)`, Windows' WDDM driver silently spills to system RAM: throughput collapses ~30× while *appearing to work* | Cap it, so a real OOM is raised. Applies to the trainer too. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is unsupported on Windows | Dropped. |

## Deliverables (all six required on Devpost)


- [ ] Project description on Devpost
- [ ] Public repo with README: setup, prerequisites, usage, param count, hardware, training time,
      compute, benchmark table, AI-assistance disclosure (required by the rules — Claude Code will
      be listed in Built With and the README)
- [ ] Demo video, 2–5 min, English audio or subtitles, unlisted OK, shows the thing running
- [ ] Built With: every library, dataset, tool
- [ ] Team info: real full name, Devpost account
- [ ] ≥3 screenshots: loss curves, benchmark table, browser demo

## Timeline

| Days | Work |
|---|---|
| Aug 31 – Sep 2 | Env, repo, tokenizer, data pipeline, model, trainer, param counter, eval wiring, smoke run |
| Sep 3 – Sep 7 | Ablations on Kaggle (vocab size, depth vs width, optimizer, data mix). Lock the config. |
| Sep 8 – Sep 13 | Main run on the 4070, resumable, across nights. Build browser demo in parallel. |
| Sep 14 – Sep 16 | Anneal phase on high-quality data, context extension, final eval + GPT-2 baseline reproduction |
| Sep 17 – Sep 19 | README, screenshots, demo video, Devpost writeup |
| Sep 20 | Submit. One day of slack before the Sep 21 deadline. |

## Status (updated 2026-09-01)

Built and smoke-tested:

| Module | State |
|---|---|
| `kestrel/model.py` | done — 47.2 M at deep-thin, chunked CE, GQA, QK-norm, `param_report`, `flops_per_token`, `optim_param_groups` |
| `kestrel/variants.py`, `corpus.py`, `data.py`, `muon.py` | done |
| `kestrel/decontaminate.py` | done — dataset paths taken from lm-eval 0.4.12's own task YAMLs; `extract_texts` emits stem+continuation pairs, see the coverage table below |
| `kestrel/trainer.py`, `scripts/train.py` | done — WSD schedule in *tokens*, grad accumulation to 0.26 M tok/step, atomic checkpoints, resume verified (step 13 → 14 with best-val and kWh carried across sessions), NVML power/temp/throttle logging, `--max-hours`, `STOP` file, Ctrl-C → checkpoint, and a 30-minute wall-clock checkpoint cadence so an unattended night can lose at most half an hour |
| `kestrel/evaluate.py`, `scripts/eval.py` | written — `lm_eval.api.model.LM` adapter (0.4.12), the same module and tokenizer training used, so no export step can move the numbers. **Still never executed**; needs a checkpoint + tokenizer pair. |
| `scripts/bench.py`, `count_params.py` | done, results above |
| `scripts/ablate.py` | done — one axis per suite, subprocess per cell, micro-batch read from the measured frontier, ranks on **loss per GPU-hour** |
| tokenizer @ 16 k | done — 16,384 vocab in 180 s from 400 k streamed docs, eot id 0, 2.89 chars/token on a number-heavy probe |
| `scripts/prepare_data.py` | done — 300 M-token pilot corpus built in 194 s (~1.55 M tok/s); the 6 B run is the same command with `--tokens 6e9` and should take ~65 min |
| first real training | 4 M tokens on the pilot corpus, uncompiled: loss 8.38 → 6.25, WikiText-103 val ppl 1755 → 1021. The pipeline end-to-end works on real data. |

### Decontamination coverage, measured 2026-09-01

A window-hash filter can only see a text at least `window` tokens long. Measured against the 16 k
tokenizer, indexing the scored fields *individually* left two benchmarks almost uncovered, because
their answer choices are one or two words:

| split | cover @ 32, fields alone | cover @ 32, with stem+continuation | median tokens |
|---|---|---|---|
| hellaswag | 65 % | **79 %** | 55 |
| arc_easy | 3 % | **22 %** | 15 |
| piqa | 17 % | **34 %** | 21 |
| winogrande | 3 % | **12 %** | 24 |
| wikitext103 | 63 % | 63 % | 92 |
| wikitext_harness | 100 % | 100 % | 3338 |

Pairing each stem with each continuation is now done, and is the right form regardless: a scraped
page carries the question together with its answers. What it does not fix is that ARC-Easy and
WinoGrande items are simply shorter than 32 tokens (medians 15 and 24) — **open decision**, below.

The pilot run dropped 2 documents out of 269,310 as contaminated (0.0007 %). That number is
believable rather than reassuring: with coverage as above it mostly reflects long-text overlap.
Also note WikiText's own preprocessing (`@-@`, `<unk>`, spaced punctuation) means a *same-article*
overlap with raw Wikipedia will not hash-match; verbatim copies of the benchmark files will.

Open questions, in the order they get answered:

1. **Decontamination window.** Drop to 16 tokens (covers most of ARC and WinoGrande, at some rate of
   innocent training documents discarded), keep 32, or run two windows and record both drop rates?
   Cheap to decide — it only costs an index rebuild and a 3-minute pilot re-run.
2. **Architecture.** wide-shallow is 26 % faster per token than deep-thin; is it worse *per token*
   by more than 26 %? → `ablate.py --suite arch`, 300 M tokens per cell, on the pilot corpus.
3. **Vocab size.** 8 k frees 3.1 M params for layers but costs more tokens per document.
   Needs a tokenized corpus per size, so it runs after the main data prep.
4. **Optimizer.** Muon+AdamW vs a fairly-tuned AdamW baseline at equal tokens *and* equal wall time.
5. **Data mix.** Whether the reserved 5 % slot earns its place.



## Environment facts

- Python 3.12.9 venv at `E:\kestrel\.venv` (system default is 3.14, too new for the ML stack).
- torch from the cu130 index; driver is NVIDIA 610.88 / CUDA UMD 13.3.
- Git 2.52, Node 25.4 (for the browser demo). WSL2 Ubuntu available as a `torch.compile` fallback
  if Triton on Windows misbehaves.


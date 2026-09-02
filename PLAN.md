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
5. **Documentation & Demo** → README with full config, param count, hardware, compute, and a
   benchmark table beside GPT-2-124M **scored through our own harness on the same splits** (the
   released checkpoint as a reference number only — never as initialization, which the track forbids).
   Demo = the model running offline in the browser.

## Candidate architecture (to be confirmed by ablation, not assumed)

Llama-shaped on purpose, so `lm-evaluation-harness`, `transformers`, and GGUF/ONNX export all
work with zero custom glue. Novelty goes into data and training, not into a block nobody can run.

Deep-and-thin, following MobileLLM's finding that depth beats width below 1B params:

| | |
|---|---|
| `d_model` | 384 |
| `n_layers` | 26 |
| `n_heads` / `n_kv_heads` | 6 / 2 (GQA 3:1, head_dim 64) |
| FFN | SwiGLU, hidden 1096 |
| Norm | RMSNorm, pre-norm, + QK-norm for stability |
| Position | RoPE |
| Vocab | 16,384, custom BPE trained on our own mix |
| Embeddings | tied input/output |
| Seq len | 1024 (optional 2048 extension at the end) |

Param count: 26 × 1,656,704 (layers) + 6,291,456 (tied embed) + 384 (final norm) = **49,366,144**,
0.63 M under the cap. `ffn_hidden` is not hand-chosen — `fit_ffn_hidden()` in `kestrel/variants.py`
returns the largest multiple of 8 that stays under 49.5 M, and all four candidate shapes are fitted
the same way so the arch ablation varies shape alone (spread 0.25 M, 0.51 %).

## Data

Target ~6 B tokens for the main run (≈122 tokens/param — deliberately far past Chinchilla-optimal,
which is correct when parameters are capped and compute is the only free variable).

| Source | Share | Why |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` (sample-10BT) | 82 % | Highest benchmark yield per token at small scale. |
| Wikipedia (English, `20231101.en`) | 18 % | WikiText-103 perplexity is a scored metric; keeps it in-distribution. |
| Reserved / ablation slot | 0 % for now | Candidates: OpenWebMath, TinyStories, public textbook corpora. Enters the mix only if `ablate.py --suite data` shows a gain; until then its share stays with FineWeb-Edu. |

`kestrel/corpus.py` holds the authoritative weights — the tokenizer trainer and the data pipeline
both read it, so they cannot disagree about what the corpus is. Keep this table in step with it.

Non-negotiable: **n-gram decontamination** of the training mix against the test splits of
HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103. Report how many documents were removed.
This is both honest and a differentiator.

Storage: uint16 token shards ≈ 12 GB for 6 B tokens, plus raw parquet during tokenization.
Stream-tokenize and delete raw shards as we go.

## Compute budget

Local: RTX 4070 Laptop, 8 GB VRAM, 140 W cap. **Measured**, not estimated (see below): 44–67 k
tokens/s depending on variant, i.e. 6 B tokens in **25–38 GPU-hours**, three to five overnight
sessions. User confirmed *some* nights, not all — so the schedule assumes ~6 usable nights and
keeps a spare.

Kaggle (30 GPU-h/week free, T4 ×2): ablation runs only, ~300 M tokens each ≈ 1.5 h per run.
That funds ~15–20 ablations without touching the local GPU's main-run time.

Everything must be checkpoint/resume safe from run one — a laptop training overnight will get
interrupted. Also log temperature and clock throttling.

## Measured, 2026-09-02 (RTX 4070 Laptop 8 GB, seq 1024, vocab 16384, bf16, `torch.compile`)

Every cell is a fresh subprocess (`scripts/bench.py --matrix`); a CUDA OOM poisons a process, so a
sweep that shares one cannot survive its own failures. Raw data in `runs/bench_frontier.jsonl`
(the pre-refit measurements are kept in `runs/bench_frontier_prefit.jsonl`).

| variant | shape | params | best micro-batch | tok/s | peak VRAM | MFU | 6 B tokens |
|---|---|---|---|---|---|---|---|
| very-deep | 320 × 36L | 49.23 M | 8 | 43,758 | 5.20 GB | 23.9 % | 38.1 h |
| deep-thin | 384 × 26L | 49.37 M | 12 | 50,952 | 7.05 GB | 26.7 % | 32.7 h |
| mid | 512 × 14L | 49.41 M | 12 | 60,365 | 5.54 GB | 29.0 % | 27.6 h |
| wide-shallow | 640 × 9L | 49.48 M | 20 | **67,230** | 6.88 GB | 30.9 % | **24.8 h** |

At matched parameters — and this table is the first version where they really are matched, within
0.51 % — **wide-shallow processes 32 % more tokens per GPU-hour than deep-thin and 54 % more than
very-deep**. Depth costs throughput twice over: more sequential kernel launches, and more stored
activations, which caps the micro-batch (very-deep cannot hold 12 sequences; wide-shallow holds 20).
Matching the parameter counts *widened* the gap rather than closing it, because the shapes that were
previously undersized had to spend their new parameters on FFN width, which is exactly what the
micro-batch was paying for. So MobileLLM's "depth beats width" cannot be adopted on faith here; the
question is not accuracy per parameter but accuracy per GPU-hour, and the 13 GPU-hours between the
fastest and slowest shape is over a night and a half of this project's compute.
`scripts/ablate.py --suite arch` settles it at 300 M tokens per cell.

Decisions these measurements forced:

| Finding | Consequence |
|---|---|
| `torch.compile` (via `triton-windows`, since PyTorch-on-Windows ships no Triton): 51.6 k vs 25.7 k tok/s, 4.68 vs 6.57 GB | Compile is on by default. It is 2× throughput *and* 1.9 GB cheaper, so it also buys micro-batch. |
| `F.scaled_dot_product_attention(..., enable_gqa=True)` disqualifies the flash/mem-efficient kernels and falls back to the math backend, which materialises `(B, H, T, T)` | Keep the explicit `repeat_kv` path. `sdpa_enable_gqa=False` is the default; the flag exists only to reproduce the finding. |
| `--loss-chunk 0` OOMs at batch 8 | Chunked, checkpointed cross-entropy is load-bearing, not an optimisation. The `(B, T, vocab)` logit tensor is the single largest allocation on this card. |
| Without `set_per_process_memory_fraction(MEMORY_FRACTION)` (0.90, defined once in `kestrel/trainer.py` and imported by the bench so the two cannot drift), Windows' WDDM driver silently spills to system RAM: throughput collapses ~30× while *appearing to work* | Cap it, so a real OOM is raised. Applies to the trainer too. |
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

## Status (updated 2026-09-02)

Built and smoke-tested:

| Module | State |
|---|---|
| `kestrel/model.py` | done — 49.37 M at deep-thin, chunked CE, GQA, QK-norm, `param_report`, `flops_per_token`, `optim_param_groups`; geometry defaults derive from `variants.py` rather than restating it |
| `kestrel/variants.py` | done — four shapes fitted to one budget by `fit_ffn_hidden()`, per-vocabulary refitting via `fitted_variant()`, single source of truth for `KestrelConfig` and `count_params.py`; 35 tests in `tests/test_variants.py`, including the closed-form estimate against the modules actually built |
| `corpus.py`, `data.py`, `muon.py` | done |
| `kestrel/decontaminate.py` | done — dataset paths taken from lm-eval 0.4.12's own task YAMLs; order-sensitive Rabin–Karp window hash at **window 16**, per-benchmark attribution of every drop, stale-index refusal via `INDEX_FORMAT`; 12 tests in `tests/test_decontaminate.py`, including the hash against direct mod-2⁶⁴ arithmetic |
| `kestrel/trainer.py`, `scripts/train.py` | done — WSD schedule in *tokens*, grad accumulation to 0.26 M tok/step, atomic checkpoints, resume verified (step 13 → 14 with best-val and kWh carried across sessions), NVML power/temp/throttle logging, `--max-hours`, `STOP` file, Ctrl-C → checkpoint, and a 30-minute wall-clock checkpoint cadence so an unattended night can lose at most half an hour |
| `kestrel/evaluate.py`, `scripts/eval.py` | done — `lm_eval.api.model.LM` adapter (0.4.12), the same module and tokenizer training used, so no export step can move the numbers. **Executed 2026-09-02** on a 4 M-token checkpoint at `--limit 50`: all five scored tasks ran, 3.3 min, JSON written beside the checkpoint. The accuracies at n=50 are noise (SE ≈ 0.07) — the run tested the plumbing, and a full unlimited pass is 10–20 min. |
| `scripts/bench.py`, `count_params.py` | done, results above; `count_params.py --sweep` and the frontier both read `variants.py` |
| `scripts/ablate.py` | done — one axis per suite, subprocess per cell, micro-batch read from the measured frontier, ranks on **loss per GPU-hour** |
| tokenizer @ 16 k | done — 16,384 vocab in 180 s from 400 k streamed docs, eot id 0, 2.89 chars/token on a number-heavy probe |
| `scripts/prepare_data.py` | done — 300 M-token pilot corpus rebuilt at window 16 in 191 s (~1.6 M tok/s), 49 of 269,312 docs dropped as contaminated; the 6 B run is the same command with `--tokens 6e9` and should take ~65 min |
| first real training | 4 M tokens on the pilot corpus, uncompiled: loss 8.38 → 6.25, WikiText-103 val ppl 1755 → 1021. The pipeline end-to-end works on real data. |
| trainer vs bench | compiled smoke run sustained 52,187–52,352 tok/s against a bench prediction of 52,881, MFU 26.5 % vs 26.8 %, and exercised eval + checkpoint + the 403 MB optimizer state. The frontier is a trustworthy source for `ablate.py`'s micro-batches. |

### Decontamination window, decided 2026-09-01 — **16 tokens**

A window-hash filter can only see a text at least `window` tokens long. Both directions were
measured before choosing, because this is the project's main rigour claim and a filter that never
fires is indistinguishable from a clean corpus.

**Reach** (`scripts/check_decontam_coverage.py`, 16 k tokenizer) — fraction of each split's scored
strings long enough to be hashed at all:

| split | @ 32, fields alone | @ 32, stem+continuation | @ 16, stem+continuation | median tokens |
|---|---|---|---|---|
| hellaswag | 65 % | 79 % | **92 %** | 55 |
| arc_easy | 3 % | 22 % | **49 %** | 15 |
| piqa | 17 % | 34 % | **62 %** | 21 |
| winogrande | 3 % | 12 % | **67 %** | 24 |
| wikitext103 | 63 % | 63 % | **70 %** | 92 |
| wikitext_harness | 100 % | 100 % | 100 % | 3338 |

Two stacked fixes: indexing stem+continuation pairs instead of isolated fields (which is also the
form a scraped quiz page takes), then halving the window. Unfilterable scored strings: 37 % → 19 %.

**Price** (`scripts/check_decontam_cost.py --docs 150000 --windows 16 32`, 300 k documents split
evenly between the sources, so the window is isolated from the mix):

| window | fineweb-edu | wikipedia | all | rate |
|---|---|---|---|---|
| 32 | 0 / 150,000 | 3 / 150,000 | 3 / 300,000 | 0.0010 % |
| 16 | 11 / 150,000 | 133 / 150,000 | 144 / 300,000 | 0.0480 % |

The decisive number is the zero. At 32 tokens the filter never fired on FineWeb-Edu at all — 82 % of
the corpus, 150,000 documents, not one match. It was not conservative, it was inert.

At window 16 on the rebuilt 300 M-token pilot, at the real 82/18 mix: **49 of 269,312 documents
dropped, 0.0182 %** — one in 5,500, lower than the 50/50 sample because Wikipedia is the source that
matches most and is only 18 % of the mix. FineWeb-Edu 21 (hellaswag 11, arc_easy 4, wikitext103 5,
piqa 1); Wikipedia 28, all wikitext103.

Composition beats rate here: every Wikipedia drop matches WikiText-103, which is *correct* —
WikiText-103 is Wikipedia, so those are the same articles. Among the FineWeb-Edu drops, some are real
overlap ("British armed trawler HMT Stella Sirius was sunk by") and some are stock phrases
("Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus"). `--show` decodes every matched span.

Stopping at 16 rather than 12: 16 tokens ≈ 8–9 English words at 2.89 chars/token, already stricter
than GPT-3's 13-word n-gram. Twelve would add ~10 points of ARC coverage but a 6–7 word span is a
phrase, not a quotation, and there would be no convention left to appeal to. Order-sensitive hashing
became a prerequisite at 16 — the previous order-insensitive sum would have matched anagrams, which
is tolerable at 32 and not at 16.

Two limitations stay on the record rather than being glossed: WikiText's own preprocessing (`@-@`,
`<unk>`, spaced punctuation) means a *same-article* overlap with raw Wikipedia does not hash-match
(verbatim copies of the benchmark files do), and 19 % of scored strings are shorter than 16 tokens
and therefore unfilterable at any window — logged as `eval_texts_below_window`.

Open questions, in the order they get answered:

1. **Architecture.** wide-shallow is 32 % faster per token than deep-thin and 54 % faster than
   very-deep; is it worse *per token* by more than that? → `ablate.py --suite arch`, 300 M tokens per
   cell, on the pilot corpus. This gates the 25–38 GPU-hour main run, so it is the next thing decided.
2. **Vocab size.** 8 k frees 3.1 M params for layers but costs more tokens per document. Each cell has
   to refit `ffn_hidden` for its vocabulary (`fitted_variant()`) or the 32 k cell would be over the
   50 M cap and the comparison would be against an illegal model. Needs a tokenized corpus per size,
   so it runs after the main data prep.
3. **Optimizer.** Muon+AdamW vs a fairly-tuned AdamW baseline at equal tokens *and* equal wall time.
4. **Data mix.** Whether the reserved slot earns its place.
5. **Context extension.** Whether to extend 1024 → 2048 after the main run. Defer; it only matters if
   there is GPU time left after the anneal.



## Environment facts

- Python 3.12.9 venv at `E:\kestrel\.venv` (system default is 3.14, too new for the ML stack).
- torch from the cu130 index; driver is NVIDIA 610.88 / CUDA UMD 13.3.
- Git 2.52, Node 25.4 (for the browser demo). WSL2 Ubuntu available as a `torch.compile` fallback
  if Triton on Windows misbehaves.


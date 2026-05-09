# TurboQuant KV Memory Bench

I built this repo to measure one thing TurboQuant delivers reliably in a HuggingFace/Kaggle workflow: KV storage compression gain (`estimated_kv_storage_gain_x`). I track token disagreement, KL drift, and latency ratio vs baseline as guardrails.

Scope is memory compression in Python-first HuggingFace workflows. Backend or kernel integration would be needed for absolute decode speedups. That's not what this repo claims.

## At A Glance

- measurable KV memory win: `2.27x` to `3.30x` estimated storage gain on the latest Kaggle 3B run,
- best balanced config: `tq_k8_v4_rw128` (`2.27x` gain with low disagreement),
- practical runtime profile: compressed runs remain close to baseline latency (about `1.05x` to `1.10x`).

## Core Goal

The goal is finding TurboQuant cache settings that squeeze the most KV storage gain without destroying generation quality.

Success criteria here:

- strong KV storage gain (`>2x`) on long-context prompts,
- quality drift that stays manageable (tracked by disagreement, KL, NLL delta),
- latency overhead that stays close to baseline.

## Scope And Plan

What this repo covers on the HuggingFace eager path:

- optimizing cache update overhead,
- improving KV storage gain vs quality tradeoff selection,
- keeping compressed-run latency close to baseline,
- providing reproducible memory-gain measurements.

What this repo does not cover:

- universal decode speedups over fp16 in eager Python attention,
- backend/kernel-level acceleration (vLLM, Triton, custom CUDA).

If you want backend acceleration, fork the settings and telemetry from here into a separate integration project.

![Latest Kaggle 3B snapshot](results/qwen25_3b_tradeoff.png)

The chart is generated from [`results/qwen25_3b_snapshot.csv`](results/qwen25_3b_snapshot.csv) by running:

```bash
python scripts/plot_results.py
```

The CSV was extracted from the latest executed outputs already present in the Kaggle-first notebook.

## What We Achieved

- Built a working TurboQuant-style `DynamicCache` wrapper with asymmetric K/V bit allocation and fixed or dynamic residual windows.
- Added an end-to-end benchmark that compares baseline and compressed-cache runs on the exact same prompt sets.
- Added a memory-first reporting layer: `primary_metric=estimated_kv_storage_gain_x`, quality guardrails, and latency-vs-baseline ratio.
- Measured more than just output text: KV compression gain, KL drift, teacher-forced NLL delta, token disagreement, and latency.
- Added notebook checks that support two useful observations:
  - asymmetric K/V quantization is motivated by observed key/value norm asymmetry,
  - the long-context retrieval sanity check is directionally useful, though the latest stored probe is qualitative rather than a decisive failure-threshold test.

## Latest Result Snapshot (Kaggle 3B)

These numbers come from the latest executed analysis already stored in the Kaggle notebook using `Qwen/Qwen2.5-3B-Instruct`, `20` harmful prompts, `20` harmless prompts, and `filler_repetitions=32`.

| Run | KV storage gain (x) | Token disagreement | KL to baseline | Avg latency | Latency vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_fp16_cache` | `1.00x` | `0.0000` | `0.000` | `3.48s` | `1.00x` |
| `tq_k8_v4_rw128` | `2.27x` | `0.0328` | `1.279` | `3.82s` | `1.10x` |
| `tq_k6_v4_rw128` | `2.27x` | `0.0828` | `2.862` | `3.69s` | `1.06x` |
| `tq_k4_v2_rw128_prot2` | `3.30x` | `0.1344` | `3.459` | `3.66s` | `1.05x` |

What this means in practice:

- `2.27x` to `3.30x` KV storage gain is a clear, measurable win.
- `tq_k8_v4_rw128` is the best quality-preserving operating point in the current setup.
- Latency stays close to baseline (about `1.05x` to `1.10x`), so the memory win is actually usable.

## Tangible Improvement Added

The repo now includes a concrete TurboQuant cache-path optimization that makes the Hugging Face eager path noticeably more practical.

- Added approximation-materialized compressed updates so compressed tokens still perturb decode while avoiding full-history re-decompression each step.
- Added chunked compression updates (`compression_chunk_size`, default `16`) to avoid paying compression/decompression cost for every single-token overflow.
- Added explicit telemetry for `estimated_compression_ratio` and `pending_uncompressed_tokens` so we can track the tradeoff between update cost and compression strictness.

Measured impact:

- End-to-end 3B benchmark latency for compressed runs dropped from about `24-25s` on the older path to about `3.66-3.82s` on the current path. KL and token disagreement stayed non-zero.
- A synthetic cache-update microbenchmark (same tensor shapes, prefill `1024`, decode `256`) cut update time from `0.340s` at chunk size `1` to `0.163s` at chunk size `32`.

Reproduce the microbenchmark locally:

```bash
PYTHONPATH=src python scripts/bench_cache_update.py \
  --prefill-tokens 1024 \
  --decode-steps 256 \
  --chunk-sizes 1 4 8 16 32
```

The notebook also includes two supporting checks:

- K/V norm asymmetry: a stored run shows key norms are substantially larger than value norms. That supports allocating more bits to keys than values.
- Needle retrieval sanity check: the notebook compares baseline, moderate compression, and aggressive compression on a long-context retrieval task. In the latest stored probe, all configurations still recovered the needle. So this check is more about robustness inspection than a hard failure claim.

## Methodology

The benchmark isolates cache compression effects rather than general model variation.

```mermaid
flowchart LR
    A[Load harmful and harmless prompt sets] --> B[Build long-context prompts with repeated filler]
    B --> C[Run baseline model with normal cache]
    B --> D[Run same model with TurboQuant-style cache]
    C --> E[Collect baseline logprobs and target continuations]
    D --> F[Collect compressed-cache generations and cache stats]
    E --> G[Compare KL, NLL delta, token disagreement]
    F --> G
    D --> H[Measure latency and optional safety telemetry]
```

### Core evaluation path

- Harmful prompts come from `mlabonne/harmful_behaviors`.
- Harmless prompts come from `mlabonne/harmless_alpaca`.
- Each run uses the same model and prompt slices for baseline and compressed-cache comparisons.
- Harmless prompts are used to measure distributional drift against baseline.
- Harmful prompts can be used as optional safety telemetry under compression.
- Long-context filler can be repeated to push more tokens into the cache and make compression effects easier to observe.

### Metrics that matter here

- `refusal_rate` (optional): how often the model declines harmful prompts using a lightweight refusal-marker heuristic.
- `avg_kl_to_baseline`: continuation-level KL divergence from baseline log-probabilities.
- `avg_nll_delta_to_baseline`: how much worse the compressed run scores the baseline continuation tokens.
- `avg_token_disagreement_to_baseline`: token mismatch rate versus baseline greedy continuations.
- `avg_latency_sec`: generation time per prompt.
- `cache_stats`: fraction of tokens that were actually compressed and related cache settings.

### Why this framing

The evaluation framing is inspired by public Heretic methodology.

- Heretic repository: https://github.com/p-e-w/heretic

## Repository Map

- `src/tqhk/quantization.py`: rotation, Lloyd-Max, and MSE-driven KV compression helpers.
- `src/tqhk/cache.py`: TurboQuant-style cache wrapper with asymmetric K/V bits and residual-window logic.
- `src/tqhk/prompting.py`: dataset loading and long-context prompt construction.
- `src/tqhk/evaluation.py`: KL, NLL, disagreement, generation utilities, and optional refusal telemetry.
- `src/tqhk/benchmark.py`: orchestration for baseline vs compressed ablations and result export.
- `scripts/run_benchmark.py`: CLI entrypoint with a small set of useful cache presets.
- `scripts/plot_results.py`: reads `results/qwen25_3b_snapshot.csv` and regenerates `results/qwen25_3b_tradeoff.png`.
- `notebooks/turboquant_heretic_eval.ipynb`: Kaggle-first notebook with the main 3B result plus supplemental validations.

## Rerun The Benchmark

I tried to keep this easy to rerun on Kaggle or any CUDA machine without a massive setup section.

```bash
pip install -e .
python scripts/run_benchmark.py \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --harmful-split "test[:100]" \
  --harmless-split "test[:100]" \
  --batch-size 4 \
  --max-new-tokens 64 \
  --filler-repetitions 0
```

Artifacts are written to:

- `results/benchmark_results.csv`
- `results/benchmark_results.json`

For the full analysis path, use the notebook in `notebooks/`. It is the canonical place for the stored 3B result snapshot and the supplementary validation checks.

## Recommended Runtime

- Python `>=3.10`
- CUDA GPU
- Kaggle T4 is the intended low-friction target
- The notebook is meant for Kaggle, not a local laptop or CPU-only setup

If you're on Kaggle, the notebook already has clone, install, and reload cells. You won't need to reconstruct the environment manually.

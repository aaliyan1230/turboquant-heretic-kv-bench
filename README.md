# TurboQuant x Refusal Eval Bench

This repo asks a narrow question: if you compress the KV cache with a TurboQuant-style scheme, what changes first, safety behavior or model fidelity?

The current answer is: on the latest stored 3B notebook run, refusal rate stayed flat while distributional drift rose as compression became more aggressive. In other words, this benchmark already surfaces a measurable compression-vs-fidelity tradeoff, but it does not show an obvious refusal-rate effect in the sampled run.

## What We Achieved

- Built a working TurboQuant-style `DynamicCache` wrapper with asymmetric K/V bit allocation and fixed or dynamic residual windows.
- Added an end-to-end benchmark that compares baseline and compressed-cache runs on the exact same harmful and harmless prompts.
- Measured more than just output text: refusal rate, KL drift, teacher-forced NLL delta, token disagreement, latency, and cache compression stats.
- Added notebook checks that support two useful claims:
  - asymmetric K/V quantization is motivated by observed key/value norm asymmetry,
  - moderate cache compression can preserve long-context retrieval better than aggressive settings.

## Latest Result Snapshot

These numbers come from the latest executed analysis already stored in the Kaggle notebook using `Qwen/Qwen2.5-3B-Instruct`, `20` harmful prompts, `20` harmless prompts, and `filler_repetitions=32`.

| Run | Refusal rate | KL to baseline | Token disagreement | Avg latency |
| --- | ---: | ---: | ---: | ---: |
| `baseline_fp16_cache` | `0.25` | `0.000` | `0.0000` | `3.97s` |
| `tq_k8_v4_rw128` | `0.25` | `1.275` | `0.0328` | `25.22s` |
| `tq_k6_v4_rw128` | `0.25` | `2.862` | `0.0828` | `25.19s` |
| `tq_k4_v2_rw32` | `0.25` | `3.886` | `0.2047` | `25.17s` |

Practical interpretation:

- The benchmark is sensitive enough to detect fidelity drift.
- Moderate compression is measurably closer to baseline than aggressive compression.
- In this implementation, latency gets worse rather than better because the cache path is still Python-heavy.
- The meaningful signal today is fidelity degradation under compression, not a shift in refusal rate.

The notebook also includes two supporting checks:

- K/V norm asymmetry: a stored run reports that key norms are substantially larger than value norms, which supports allocating more bits to keys than values.
- Needle retrieval sanity check: the notebook compares baseline, moderate compression, and aggressive compression on a long-context retrieval task to verify that weaker settings fail first.

## Methodology

The benchmark is designed to isolate cache compression effects rather than general model variation.

```mermaid
flowchart LR
    A[Load harmful and harmless prompt sets] --> B[Build long-context prompts with repeated filler]
    B --> C[Run baseline model with normal cache]
    B --> D[Run same model with TurboQuant-style cache]
    C --> E[Collect baseline logprobs and target continuations]
    D --> F[Collect compressed-cache generations and cache stats]
    E --> G[Compare KL, NLL delta, token disagreement]
    F --> G
    D --> H[Measure harmful-prompt refusals and latency]
```

### Core evaluation path

- Harmful prompts come from `mlabonne/harmful_behaviors`.
- Harmless prompts come from `mlabonne/harmless_alpaca`.
- Each run uses the same model and prompt slices for baseline and compressed-cache comparisons.
- Harmless prompts are used to measure distributional drift against baseline.
- Harmful prompts are used to measure refusal counts under compression.
- Long-context filler can be repeated to push more tokens into the cache and make compression effects easier to observe.

### Metrics that matter here

- `refusal_rate`: how often the model declines harmful prompts using a lightweight refusal-marker heuristic.
- `avg_kl_to_baseline`: continuation-level KL divergence from baseline log-probabilities.
- `avg_nll_delta_to_baseline`: how much worse the compressed run scores the baseline continuation tokens.
- `avg_token_disagreement_to_baseline`: token mismatch rate versus baseline greedy continuations.
- `avg_latency_sec`: generation time per prompt.
- `cache_stats`: fraction of tokens that were actually compressed and related cache settings.

### Why this framing

The evaluation framing is inspired by public Heretic methodology, but this repo does not copy Heretic source code and does not implement Heretic optimization loops.

- Heretic repository: https://github.com/p-e-w/heretic

## Repository Map

- `src/tqhk/quantization.py`: rotation, Lloyd-Max, and MSE-driven KV compression helpers.
- `src/tqhk/cache.py`: TurboQuant-style cache wrapper with asymmetric K/V bits and residual-window logic.
- `src/tqhk/prompting.py`: dataset loading and long-context prompt construction.
- `src/tqhk/evaluation.py`: refusal detection, KL, NLL, disagreement, and generation utilities.
- `src/tqhk/benchmark.py`: orchestration for baseline vs compressed ablations and result export.
- `scripts/run_benchmark.py`: CLI entrypoint with a small set of useful cache presets.
- `notebooks/turboquant_heretic_eval.ipynb`: Kaggle-first notebook with the main 3B result plus supplemental validations.

## Rerun The Benchmark

This repo is intended to be easy to rerun on Kaggle or any CUDA machine without a large setup section.

```bash
pip install -e .
python scripts/run_benchmark.py \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
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

If you are running on Kaggle, the notebook already contains clone, install, and reload cells so you do not have to reconstruct the environment manually.

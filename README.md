# TurboQuant x Refusal Eval Bench

Small benchmark project to test whether TurboQuant-style KV cache compression changes refusal behavior and first-token distribution drift on harmful/harmless prompt sets.

This is intended for rapid experimentation on Kaggle GPUs (T4) and local setups.

## What this project does

- Runs baseline and TurboQuant-cache ablations on the same prompts.
- Measures:
  - refusal rate on harmful prompts,
  - KL divergence (first-token logprobs) on harmless prompts vs baseline,
  - average generation latency,
  - cache-side token compression stats.
- Supports fixed or dynamic residual windows.

## Methodology note

The refusal/KL evaluation framing is inspired by public methodology from the Heretic project:

- Repository: https://github.com/p-e-w/heretic

This repo does not copy Heretic source code and does not run Heretic optimization loops.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run benchmark:

```bash
python scripts/run_benchmark.py \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --harmful-split "test[:100]" \
  --harmless-split "test[:100]" \
  --batch-size 4 \
  --max-new-tokens 64 \
  --filler-repetitions 0
```

Outputs:

- `results/benchmark_results.csv`
- `results/benchmark_results.json`

## Project layout

- `src/tqhk/quantization.py` - rotation + Lloyd-Max + MSE compressor
- `src/tqhk/cache.py` - TurboQuant cache wrapper with fixed/dynamic residual window
- `src/tqhk/prompting.py` - prompt loading and long-context wrapping
- `src/tqhk/evaluation.py` - refusal + KL + generation helpers
- `src/tqhk/benchmark.py` - end-to-end runner
- `scripts/run_benchmark.py` - CLI entrypoint
- `notebooks/` - Kaggle notebook scaffold and analysis notebook(s)

## Suggested Kaggle runtime

- GPU: 2xT4 (or 1xT4)
- Python >= 3.10
- Install this repo via:

```bash
!git clone <your-fork-url>
%cd turboquant-heretic-kv-bench
!pip install -e .
```

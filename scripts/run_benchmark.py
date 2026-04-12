#!/usr/bin/env python3
"""CLI entrypoint for TurboQuant KV memory benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tqhk.benchmark import BenchmarkConfig, RunConfig, run_benchmark
from tqhk.cache import CacheConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TurboQuant benchmark (memory gain + quality guardrails)"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--harmful-split", default="test[:100]")
    parser.add_argument("--harmless-split", default="test[:100]")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--filler-repetitions", type=int, default=0)
    parser.add_argument("--output-csv", default="results/benchmark_results.csv")
    parser.add_argument("--output-json", default="results/benchmark_results.json")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = BenchmarkConfig(
        model_name=args.model,
        device=args.device,
        load_in_4bit=not args.no_4bit,
        harmless_split=args.harmless_split,
        harmful_split=args.harmful_split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        filler_repetitions=args.filler_repetitions,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )

    runs = [
        RunConfig(
            name="baseline_fp16_cache",
            use_turboquant_cache=False,
            cache_config=CacheConfig(),
        ),
        RunConfig(
            name="tq_k8_v4_rw128",
            use_turboquant_cache=True,
            cache_config=CacheConfig(
                key_bits=8,
                value_bits=4,
                residual_mode="fixed",
                residual_window=128,
            ),
        ),
        RunConfig(
            name="tq_k6_v4_rw128",
            use_turboquant_cache=True,
            cache_config=CacheConfig(
                key_bits=6,
                value_bits=4,
                residual_mode="fixed",
                residual_window=128,
            ),
        ),
        RunConfig(
            name="tq_k4_v2_rw128_prot2",
            use_turboquant_cache=True,
            cache_config=CacheConfig(
                key_bits=4,
                value_bits=2,
                residual_mode="fixed",
                residual_window=128,
                protected_layers=2,
                protected_bits=8,
            ),
        ),
    ]

    rows = run_benchmark(cfg=cfg, run_configs=runs)
    print("Completed runs:")
    for row in rows:
        primary = row.get("primary_metric_value", 0.0)
        disagree = row.get("quality_guardrail_token_disagreement", 0.0)
        latency_ratio = row.get("latency_vs_baseline_ratio", 0.0)
        print(
            f"- {row['run_name']}: refusals={row['refusals']}/{row['total']}, "
            f"refusal_rate={row['refusal_rate']:.3f}, kl={row['avg_kl_to_baseline']:.4f}, "
            f"kv_gain={primary:.2f}x, disagreement={disagree:.4f}, "
            f"latency={row['avg_latency_sec']:.3f}s ({latency_ratio:.2f}x baseline)"
        )


if __name__ == "__main__":
    main()

"""End-to-end benchmark runner for TurboQuant KV memory experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .cache import CacheConfig, TurboQuantDynamicCache
from .evaluation import (
    compute_token_disagreement,
    compute_teacher_forced_nll,
    EvalResult,
    compute_kl_to_baseline,
    generate_target_token_ids,
    generate_responses,
    get_first_token_logprobs,
    is_refusal,
)
from .prompting import build_long_context_prompt, load_prompt_sets


@dataclass
class RunConfig:
    name: str
    use_turboquant_cache: bool
    cache_config: CacheConfig


@dataclass
class BenchmarkConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"
    load_in_4bit: bool = True
    harmless_split: str = "test[:100]"
    harmful_split: str = "test[:100]"
    batch_size: int = 4
    kl_max_new_tokens: int = 8
    nll_target_new_tokens: int = 32
    max_new_tokens: int = 64
    filler_repetitions: int = 0
    output_csv: str = "results/benchmark_results.csv"
    output_json: str = "results/benchmark_results.json"


def _prompt_token_stats(tokenizer, prompts: list[str]) -> dict[str, float]:
    lengths: list[int] = []
    for i in range(0, len(prompts), 32):
        batch = prompts[i : i + 32]
        encoded = tokenizer(
            batch,
            padding=False,
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    if not lengths:
        return {"avg": 0.0, "max": 0.0, "min": 0.0}

    return {
        "avg": sum(lengths) / len(lengths),
        "max": float(max(lengths)),
        "min": float(min(lengths)),
    }


def _cuda_memory_stats(device: str) -> dict[str, float]:
    if device != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


def _make_cache_factory(run: RunConfig, n_layers: int):
    if not run.use_turboquant_cache:
        return None

    def factory() -> TurboQuantDynamicCache:
        return TurboQuantDynamicCache(config=run.cache_config, n_layers=n_layers)

    return factory


def _prepare_prompts(cfg: BenchmarkConfig) -> tuple[list[str], list[str]]:
    prompt_set = load_prompt_sets(
        harmless_split=cfg.harmless_split,
        harmful_split=cfg.harmful_split,
    )
    harmless_prompts = [
        build_long_context_prompt(p, filler_repetitions=cfg.filler_repetitions)
        for p in prompt_set.harmless
    ]
    harmful_prompts = [
        build_long_context_prompt(p, filler_repetitions=cfg.filler_repetitions)
        for p in prompt_set.harmful
    ]
    return harmless_prompts, harmful_prompts


def _load_model_and_tokenizer(cfg: BenchmarkConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    dtype = torch.float16
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if cfg.device == "cuda" else None,
        attn_implementation="eager",
    )
    model.eval()
    if cfg.device != "cuda":
        model.to(cfg.device)
    return model, tokenizer


def run_benchmark(cfg: BenchmarkConfig, run_configs: list[RunConfig]) -> list[dict]:
    harmless_prompts, harmful_prompts = _prepare_prompts(cfg)
    model, tokenizer = _load_model_and_tokenizer(cfg)
    n_layers = model.config.num_hidden_layers
    harmless_prompt_stats = _prompt_token_stats(tokenizer, harmless_prompts)
    harmful_prompt_stats = _prompt_token_stats(tokenizer, harmful_prompts)

    baseline_run = next((x for x in run_configs if not x.use_turboquant_cache), None)
    if baseline_run is None:
        raise ValueError(
            "At least one baseline run (without TurboQuant cache) is required."
        )

    baseline_logprobs = get_first_token_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=harmless_prompts,
        batch_size=cfg.batch_size,
        device=cfg.device,
        max_new_tokens=cfg.kl_max_new_tokens,
        cache_factory=None,
    )

    baseline_target_ids = generate_target_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompts=harmless_prompts,
        batch_size=cfg.batch_size,
        device=cfg.device,
        max_new_tokens=cfg.nll_target_new_tokens,
        cache_factory=None,
    )
    baseline_nll = compute_teacher_forced_nll(
        model=model,
        tokenizer=tokenizer,
        prompts=harmless_prompts,
        target_token_ids=baseline_target_ids,
        device=cfg.device,
        cache_factory=None,
    )

    rows: list[dict] = []

    for run in run_configs:
        if cfg.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        cache_factory = _make_cache_factory(run, n_layers=n_layers)

        logprobs = get_first_token_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompts=harmless_prompts,
            batch_size=cfg.batch_size,
            device=cfg.device,
            max_new_tokens=cfg.kl_max_new_tokens,
            cache_factory=cache_factory,
        )
        kl = compute_kl_to_baseline(logprobs, baseline_logprobs)
        run_nll = compute_teacher_forced_nll(
            model=model,
            tokenizer=tokenizer,
            prompts=harmless_prompts,
            target_token_ids=baseline_target_ids,
            device=cfg.device,
            cache_factory=cache_factory,
        )
        nll_delta = run_nll - baseline_nll

        run_target_ids = generate_target_token_ids(
            model=model,
            tokenizer=tokenizer,
            prompts=harmless_prompts,
            batch_size=cfg.batch_size,
            device=cfg.device,
            max_new_tokens=cfg.nll_target_new_tokens,
            cache_factory=cache_factory,
        )
        token_disagreement = compute_token_disagreement(
            run_target_ids,
            baseline_target_ids,
        )

        harmful_responses, avg_latency, cache_stats = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=harmful_prompts,
            batch_size=cfg.batch_size,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            cache_factory=cache_factory,
        )
        refusal_count = sum(is_refusal(r) for r in harmful_responses)
        memory_stats = _cuda_memory_stats(cfg.device)

        result = EvalResult(
            refusal_rate=refusal_count / max(len(harmful_responses), 1),
            refusals=refusal_count,
            total=len(harmful_responses),
            avg_kl_to_baseline=kl,
            avg_nll_delta_to_baseline=nll_delta,
            avg_token_disagreement_to_baseline=token_disagreement,
            avg_latency_sec=avg_latency,
            cache_stats=cache_stats,
        )

        row = {
            "run_name": run.name,
            "model_name": cfg.model_name,
            "filler_repetitions": cfg.filler_repetitions,
            "harmless_prompt_avg_tokens": harmless_prompt_stats["avg"],
            "harmless_prompt_max_tokens": harmless_prompt_stats["max"],
            "harmful_prompt_avg_tokens": harmful_prompt_stats["avg"],
            "harmful_prompt_max_tokens": harmful_prompt_stats["max"],
            **asdict(result),
            **memory_stats,
            "cache_config": asdict(run.cache_config),
        }
        rows.append(row)

    baseline_row = next(
        (row for row, run in zip(rows, run_configs) if not run.use_turboquant_cache),
        None,
    )
    baseline_latency = baseline_row.get("avg_latency_sec", 0.0) if baseline_row else 0.0

    for row, run in zip(rows, run_configs):
        cache_stats = row.get("cache_stats") or {}
        estimated_ratio = cache_stats.get("estimated_compression_ratio")
        if run.use_turboquant_cache:
            primary_value = (
                float(estimated_ratio)
                if isinstance(estimated_ratio, (float, int)) and estimated_ratio > 0
                else 0.0
            )
        else:
            primary_value = 1.0

        row["primary_metric"] = "estimated_kv_storage_gain_x"
        row["primary_metric_value"] = primary_value
        row["quality_guardrail_token_disagreement"] = row[
            "avg_token_disagreement_to_baseline"
        ]
        row["passes_quality_guardrail_0p05"] = (
            row["avg_token_disagreement_to_baseline"] <= 0.05
        )
        row["latency_vs_baseline_ratio"] = (
            row["avg_latency_sec"] / baseline_latency if baseline_latency > 0 else 0.0
        )
        row["latency_overhead_pct_vs_baseline"] = (
            100.0 * (row["latency_vs_baseline_ratio"] - 1.0)
        )

    _write_results(cfg, rows)
    return rows


def _write_results(cfg: BenchmarkConfig, rows: list[dict]) -> None:
    csv_path = Path(cfg.output_csv)
    json_path = Path(cfg.output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for row in rows:
        cache_stats = row.get("cache_stats", {})
        cache_config = row.get("cache_config", {})
        flat = {
            k: v for k, v in row.items() if k not in {"cache_stats", "cache_config"}
        }
        for key, value in cache_stats.items():
            flat[f"cache_stats.{key}"] = value
        for key, value in cache_config.items():
            flat[f"cache_config.{key}"] = value
        flat_rows.append(flat)

    if flat_rows:
        fieldnames = sorted({k for row in flat_rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

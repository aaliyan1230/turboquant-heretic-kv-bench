"""End-to-end benchmark runner for TurboQuant x refusal evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .cache import CacheConfig, TurboQuantDynamicCache
from .evaluation import (
    EvalResult,
    compute_kl_to_baseline,
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
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    load_in_4bit: bool = True
    harmless_split: str = "test[:100]"
    harmful_split: str = "test[:100]"
    batch_size: int = 4
    max_new_tokens: int = 64
    filler_repetitions: int = 0
    output_csv: str = "results/benchmark_results.csv"
    output_json: str = "results/benchmark_results.json"


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
        cache_factory=None,
    )

    rows: list[dict] = []

    for run in run_configs:
        cache_factory = _make_cache_factory(run, n_layers=n_layers)

        logprobs = get_first_token_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompts=harmless_prompts,
            batch_size=cfg.batch_size,
            device=cfg.device,
            cache_factory=cache_factory,
        )
        kl = compute_kl_to_baseline(logprobs, baseline_logprobs)

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

        result = EvalResult(
            refusal_rate=refusal_count / max(len(harmful_responses), 1),
            refusals=refusal_count,
            total=len(harmful_responses),
            avg_kl_to_baseline=kl,
            avg_latency_sec=avg_latency,
            cache_stats=cache_stats,
        )

        row = {
            "run_name": run.name,
            "model_name": cfg.model_name,
            "filler_repetitions": cfg.filler_repetitions,
            **asdict(result),
            "cache_config": asdict(run.cache_config),
        }
        rows.append(row)

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

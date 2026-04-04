"""Prompt loading and long-context prompt builders."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


FILLER_TEXT = (
    "The committee reviewed operational updates for ongoing quarterly planning, "
    "including staffing allocations, infrastructure requests, delivery milestones, "
    "and retrospective notes from cross-functional teams. "
    "Each section contains procedural details and references to internal tracking codes."
)


@dataclass
class PromptSet:
    harmless: list[str]
    harmful: list[str]


def load_prompt_sets(
    harmless_split: str = "test[:100]",
    harmful_split: str = "test[:100]",
    harmless_dataset: str = "mlabonne/harmless_alpaca",
    harmful_dataset: str = "mlabonne/harmful_behaviors",
    text_column: str = "text",
) -> PromptSet:
    harmless_ds = load_dataset(harmless_dataset, split=harmless_split)
    harmful_ds = load_dataset(harmful_dataset, split=harmful_split)

    harmless = [str(x).strip() for x in harmless_ds[text_column] if str(x).strip()]
    harmful = [str(x).strip() for x in harmful_ds[text_column] if str(x).strip()]
    return PromptSet(harmless=harmless, harmful=harmful)


def build_long_context_prompt(
    user_prompt: str,
    filler_repetitions: int = 0,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    if filler_repetitions <= 0:
        long_context_block = ""
    else:
        long_context_block = "\n\n".join([FILLER_TEXT] * filler_repetitions)

    if long_context_block:
        user_content = (
            f"Context document:\n\n{long_context_block}\n\nUser request:\n{user_prompt}"
        )
    else:
        user_content = user_prompt

    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

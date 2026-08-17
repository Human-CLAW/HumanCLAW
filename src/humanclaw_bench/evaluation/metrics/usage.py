"""Provider-neutral token accounting for the paper's cost columns.

Adapters expose the provider's structured usage object through ``last_usage``.
If an endpoint does not report usage, the result is explicitly labelled as an
estimate instead of silently presenting estimated values as exact tokens.
Reasoning tokens are excluded from the visible-output column used in the paper.
"""

from __future__ import annotations

import math
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    """Return a value as a mapping, or an empty mapping for incompatible inputs."""

    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _number(mapping: dict[str, Any], *keys: str) -> int | None:
    """Extract the first finite numeric token-usage field from a mapping."""

    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return None


def normalize_usage(value: Any) -> dict[str, Any] | None:
    """Normalize OpenAI, Gemini-queue, or vLLM token fields."""

    usage = _mapping(value)
    if not usage:
        return None
    prompt = _number(usage, "input_tokens", "prompt_tokens", "prompt_token_count")
    # Providers disagree about whether ``completion_tokens`` includes hidden
    # reasoning.  Prefer an explicitly visible/candidate count when present;
    # only OpenAI-style completion totals need reasoning subtracted below.
    visible_completion = _number(
        usage,
        "output_tokens",
        "visible_output_tokens",
        "candidates_token_count",
    )
    completion_total = _number(usage, "completion_tokens")
    reasoning = _number(
        usage,
        "reasoning_tokens",
        "num_reasoning_tokens",
        "thoughts_token_count",
        "thinking_tokens",
    )
    details = _mapping(
        usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    )
    if reasoning is None:
        reasoning = _number(details, "reasoning_tokens")

    if prompt is None or (visible_completion is None and completion_total is None):
        return None
    reasoning = int(reasoning or 0)

    if visible_completion is not None:
        visible = int(visible_completion)
        inferred_total = int(prompt) + visible + reasoning
    else:
        # OpenAI's completion_tokens includes hidden reasoning.  A
        # FilesystemQueueModel normalizes Gemini's candidate-only
        # completion_tokens to output_tokens before reaching this function.
        assert completion_total is not None
        visible = max(0, int(completion_total) - reasoning)
        inferred_total = int(prompt) + int(completion_total)
    return {
        "input_tokens": int(prompt),
        "visible_output_tokens": int(visible),
        "reasoning_tokens": reasoning,
        "total_tokens": int(
            _number(usage, "total_tokens", "total_token_count") or inferred_total
        ),
    }


def _estimated_tokens(text: str) -> int:
    # Character/4 is deliberately simple and stable.  The source label makes
    # clear that this fallback is not provider billing usage.
    """Estimate token count only when the provider supplied no usage metadata."""

    return int(math.ceil(len(str(text or "")) / 4.0))


class UsageTracker:
    """Accumulate one compact cost record from planner/verifier calls."""

    def __init__(self) -> None:
        """Initialize per-call and per-episode model-usage accumulators."""

        self.calls = 0
        self.exact_calls = 0
        self.input_tokens = 0
        self.visible_output_tokens = 0
        self.reasoning_tokens = 0

    def record(self, stage_output: Any) -> None:
        """Add planner/verifier usage for one agent decision without double counting."""

        self.calls += 1
        normalized = normalize_usage(getattr(stage_output, "usage", None))
        if normalized is not None:
            self.exact_calls += 1
            self.input_tokens += int(normalized["input_tokens"])
            self.visible_output_tokens += int(normalized["visible_output_tokens"])
            self.reasoning_tokens += int(normalized["reasoning_tokens"])
            return
        self.input_tokens += _estimated_tokens(getattr(stage_output, "prompt", ""))
        self.visible_output_tokens += _estimated_tokens(
            getattr(stage_output, "raw_output", "")
        )

    def summary(self, decision_steps: int) -> dict[str, Any]:
        """Return paper-table token totals and per-step averages."""

        denominator = max(1, int(decision_steps))
        # Zero calls is not "exact": it means planning failed before a
        # provider response was recorded.  Label it explicitly so a broken
        # episode cannot masquerade as a zero-token exact measurement.
        if self.calls == 0:
            source = "unavailable_no_calls"
        elif self.calls == self.exact_calls:
            source = "provider_exact"
        elif self.exact_calls == 0:
            source = "estimated_chars_div_4"
        else:
            source = "mixed_exact_and_estimated"
        return {
            "decision_steps": int(decision_steps),
            "vlm_calls": int(self.calls),
            "token_source": source,
            "input_tokens": int(self.input_tokens),
            "visible_output_tokens": int(self.visible_output_tokens),
            "reasoning_tokens": int(self.reasoning_tokens),
            "input_tokens_per_step": self.input_tokens / denominator,
            "visible_output_tokens_per_step": (
                self.visible_output_tokens / denominator
            ),
        }


__all__ = ["UsageTracker", "normalize_usage"]

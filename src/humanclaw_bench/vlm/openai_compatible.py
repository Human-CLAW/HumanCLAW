"""OpenAI-compatible adapter for vLLM and direct provider endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class OpenAICompatibleModel:
    """Call OpenAI-compatible multimodal chat APIs behind the common VLM contract."""

    def __init__(
        self,
        *,
        model: str,
        output_dir: str | Path,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        """Create the API client and pin model, decoding, and response-format settings."""

        from openai import OpenAI

        key = api_key or os.environ.get(api_key_env)
        if not key and base_url:
            key = "EMPTY"
        if not key:
            raise RuntimeError(f"Missing API credential in {api_key_env}")
        self.model_name = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.extra_body = dict(extra_body or {})
        self.response_format = response_format
        self.client = (
            OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)
        )
        del output_dir
        self.current_episode_step = -1
        self.last_usage: dict[str, Any] = {}

    def respond(self, messages: list[dict[str, Any]]) -> str:
        """Send one chat completion request and retain normalized token usage."""

        self.last_usage = {}
        kwargs: dict[str, Any] = {}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.response_format:
            kwargs["response_format"] = self.response_format
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            if isinstance(usage, dict):
                self.last_usage = dict(usage)
        text = response.choices[0].message.content or ""
        return text

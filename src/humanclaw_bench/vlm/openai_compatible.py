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
        client_type: str = "openai",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        azure_endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        azure_endpoint: str | None = None,
        api_version_env: str = "AZURE_OPENAI_API_VERSION",
        api_version: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        send_temperature: bool = True,
        max_tokens_parameter: str = "max_tokens",
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        """Create the API client and pin model, decoding, and response-format settings."""

        client_type = str(client_type)
        if client_type not in {"openai", "azure_openai"}:
            raise ValueError(f"Unsupported OpenAI client type: {client_type!r}")
        max_tokens_parameter = str(max_tokens_parameter)
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "max_tokens_parameter must be 'max_tokens' or "
                "'max_completion_tokens'"
            )
        key = api_key or os.environ.get(api_key_env)
        if not key and base_url and client_type == "openai":
            key = "EMPTY"
        if not key:
            raise RuntimeError(f"Missing API credential in {api_key_env}")
        self.model_name = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.send_temperature = bool(send_temperature)
        self.max_tokens_parameter = max_tokens_parameter
        self.reasoning_effort = reasoning_effort
        self.extra_body = dict(extra_body or {})
        self.response_format = response_format
        if client_type == "azure_openai":
            from openai import AzureOpenAI

            endpoint = azure_endpoint or os.environ.get(azure_endpoint_env)
            version = api_version or os.environ.get(api_version_env)
            if not endpoint:
                raise RuntimeError(
                    f"Missing Azure OpenAI endpoint in {azure_endpoint_env}"
                )
            if not version:
                raise RuntimeError(
                    f"Missing Azure OpenAI API version in {api_version_env}"
                )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_version=version,
                api_key=key,
            )
        else:
            from openai import OpenAI

            self.client = (
                OpenAI(base_url=base_url, api_key=key)
                if base_url
                else OpenAI(api_key=key)
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
        kwargs[self.max_tokens_parameter] = self.max_tokens
        if self.send_temperature:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
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

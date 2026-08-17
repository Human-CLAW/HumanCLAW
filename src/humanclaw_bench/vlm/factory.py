"""Create a VLM adapter from a complete model config object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import VLM
from .filesystem_queue import FilesystemQueueModel
from .openai_compatible import OpenAICompatibleModel


def build_model(config: dict[str, Any], output_dir: str | Path) -> VLM:
    """Construct the configured direct-API or filesystem-queue VLM adapter."""

    required = ("backend", "model", "max_tokens", "temperature", "response_format")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "Model config is incomplete; missing fields: " + ", ".join(missing)
        )
    if int(config["max_tokens"]) <= 0:
        raise ValueError("model max_tokens must be positive")
    kind = str(config["backend"])
    model = str(config["model"])
    model_env = config.get("model_env")
    if model_env:
        model = os.environ.get(str(model_env), model)
    common = {
        "model": model,
        "output_dir": output_dir,
        "max_tokens": int(config["max_tokens"]),
        "temperature": float(config["temperature"]),
    }
    if kind in {"openai_compatible", "azure_openai"}:
        return OpenAICompatibleModel(
            **common,
            client_type=("azure_openai" if kind == "azure_openai" else "openai"),
            base_url=config.get("base_url"),
            api_key_env=str(
                config.get(
                    "api_key_env",
                    "AZURE_OPENAI_API_KEY"
                    if kind == "azure_openai"
                    else "OPENAI_API_KEY",
                )
            ),
            api_key=config.get("api_key"),
            azure_endpoint_env=str(
                config.get("azure_endpoint_env", "AZURE_OPENAI_ENDPOINT")
            ),
            azure_endpoint=config.get("azure_endpoint"),
            api_version_env=str(
                config.get("api_version_env", "AZURE_OPENAI_API_VERSION")
            ),
            api_version=config.get("api_version"),
            send_temperature=bool(config.get("send_temperature", True)),
            max_tokens_parameter=str(
                config.get("max_tokens_parameter", "max_tokens")
            ),
            reasoning_effort=config.get("reasoning_effort"),
            extra_body=config.get("extra_body"),
            response_format=config["response_format"],
        )
    if kind == "filesystem_queue":
        if not config.get("queue_dir"):
            raise ValueError("filesystem_queue requires queue_dir")
        return FilesystemQueueModel(
            **common,
            queue_dir=config["queue_dir"],
            reasoning_budget_tokens=config.get("reasoning_budget_tokens"),
            response_format=config["response_format"],
            timeout_s=float(config.get("timeout_s", 900)),
            poll_interval_s=float(config.get("poll_interval_s", 1)),
        )
    raise ValueError(f"Unknown VLM backend: {kind}")

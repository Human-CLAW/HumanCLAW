"""Provider-neutral VLM contract."""

from __future__ import annotations

from typing import Any, Protocol


class VLM(Protocol):
    """Define the provider-neutral multimodal request and usage-accounting contract."""

    last_usage: dict[str, Any]

    def respond(self, messages: list[dict[str, Any]]) -> str:
        """Send one multimodal conversation and return the provider's text."""

        ...


__all__ = ["VLM"]

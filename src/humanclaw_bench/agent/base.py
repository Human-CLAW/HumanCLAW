"""Base ego-agent interface for HumanClawBench."""

from __future__ import annotations

from typing import Any, Protocol


class EgoAgent(Protocol):
    """Define the reset/act interface implemented by benchmark ego agents."""

    def reset(self, task: Any) -> None:
        """Reset per-episode state."""

    def act(self, ego_rgb: Any, history: list[dict], env_feedback: list[dict]) -> Any:
        """Return the next planner decision."""

"""Plan-and-skill ego-agent wrapper for HumanClawBench Find/Nav/Interact."""

from __future__ import annotations

from typing import Any

from humanclaw_bench.agent.base import EgoAgent
from humanclaw_bench.agent.planner import (
    HumanClawBenchPSVPlanSkillPlanner,
)
from humanclaw_bench.vlm.base import VLM


class PSVEgoAgent(EgoAgent):
    """Versioned planner exposed through the HumanClawBench ego-agent API."""

    def __init__(
        self,
        model: VLM,
        *,
        prompt_version: str = "v4",
        verifier_version: str = "v3",
        max_history: int = 10,
        plan_horizon_steps: int = 6,
    ) -> None:
        """Create the versioned planner used by the evaluation."""

        self.prompt_version = prompt_version
        self.verifier_version = verifier_version
        self._planner = HumanClawBenchPSVPlanSkillPlanner(
            model=model,
            prompt_version=prompt_version,
            verifier_version=verifier_version,
            max_history=max_history,
            plan_horizon_steps=plan_horizon_steps,
        )

    def reset(self, task: Any) -> None:
        """Clear planner history and bind the agent to a new episode task."""

        self._planner.reset(task)

    def act(self, ego_rgb: Any, history: list[dict], env_feedback: list[dict]) -> Any:
        """Choose one skill from the current ego image and retain the planner result."""

        return self._planner.act(ego_rgb, history, env_feedback)


__all__ = ["PSVEgoAgent"]

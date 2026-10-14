"""Explicit single-stage baseline using the common atomic action interface."""

from __future__ import annotations

from typing import Any

from humanclaw_bench.agent.planner import HumanClawBenchPSVPlanSkillPlanner
from humanclaw_bench.agent.prompts.v4 import ACTION_SPACE, SYSTEM_PROMPT
from humanclaw_bench.agent.types import PlannerResult
from humanclaw_bench.experiments.recorder import event


class BareAPIAgent:
    """Current image and task only; parsing/retry mechanics are explicit plumbing.

    Uses the same action converter as PSV but no planner history, mid-level
    goal, verifier, or retry-exhaustion movement fallback. Action parameters
    follow the shared converter's defaults and clamps, recorded in the trace.
    Exhausted calls end the run as an
    infrastructure failure, never as a successful Stop.
    """

    def __init__(self, model: Any) -> None:
        """Reuse the existing model transport and atomic action conversion."""
        self._adapter = HumanClawBenchPSVPlanSkillPlanner(model)

    def reset(self, task: Any) -> None:
        """Bind the next episode without retaining prior episode history."""
        self._adapter.reset(task)

    def act(self, ego_rgb: Any, history: list[dict], env_feedback: list[dict]) -> PlannerResult:
        """Choose one action from the current observation and task instruction."""
        del env_feedback
        adapter = self._adapter
        if adapter.task is None:
            raise RuntimeError("Call reset(task) before act().")
        if hasattr(adapter.model, "current_episode_step"):
            adapter.model.current_episode_step = len(history)
        prompt = (
            SYSTEM_PROMPT + "\n\n" + ACTION_SPACE + "\n\n"
            + "Task: " + adapter.task.instruction + "\n"
            + 'Use the current image to choose one action. Return JSON with '
            + '"visible_state" (brief current-view description), "action_id" '
            + '(integer 0 through 7), and "action_name" (an exact action name).'
        )
        attempts = adapter._call_with_retries(ego_rgb, prompt, stage="bare_api")
        result = attempts[-1]
        if result.error:
            raise RuntimeError("Bare API request failed after retries: " + result.error)
        action_id = result.raw.get("action_id")
        if isinstance(action_id, bool) or not isinstance(action_id, int) or action_id not in range(8):
            event("tool.rejected", kind="tool", name="invalid_action",
                  data={"response": result.raw, "reason": "invalid action_id"})
            raise ValueError("Bare API response must contain action_id 0 through 7")
        action = adapter._chooser_action(result.raw)
        event("action.normalized", name="bare_api",
              data={"response": result.raw, "action": action.to_json(),
                    "policy": "shared PSV action parameter defaults and clamps"})
        event("tool.requested", kind="tool", name=action.skill,
              data={"action": action.to_json(), "source": "bare_api"})
        return PlannerResult(
            raw_plan={"visual_state_description": result.raw.get("visible_state", ""),
                      "action": action.to_json()},
            action=action, planner_skill=result.raw, verifier={}, stage_outputs=attempts,
        )

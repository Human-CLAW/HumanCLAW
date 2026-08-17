"""Plan-and-skill planner with independently versioned prompt and verifier."""

from __future__ import annotations

import re
import time
from typing import Any

from PIL import Image

from humanclaw_bench.agent.prompts import resolve_prompt_version
from humanclaw_bench.agent.skills import SkillCall, skill_to_text
from humanclaw_bench.agent.types import (
    PlannerResult,
    PSVStageOutput,
)
from humanclaw_bench.agent.utils import (
    clip_text,
    image_path_to_data_url,
    image_to_data_url,
    parse_json_loose,
)
from humanclaw_bench.agent.verifiers import resolve_verifier_version
from humanclaw_bench.vlm.base import VLM

PLAN_HORIZON_STEPS = 6
VLM_STAGE_MAX_ATTEMPTS = 5
VLM_RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)


class HumanClawBenchPSVPlanSkillPlanner:
    """Two-stage HumanClaw plan-and-skill planner.

    Stage 1 chooses a mid-level goal and a proposed low-level action. Stage 2
    applies the v3 route selected for Walk, Stop/Stand, Climb upstairs, or the
    two sitting transitions, using the same current ego image.
    """

    def __init__(
        self,
        model: VLM,
        *,
        prompt_version: str = "v4",
        verifier_version: str = "v3",
        max_history: int = 10,
        use_feedback: bool = True,
        plan_horizon_steps: int = PLAN_HORIZON_STEPS,
    ) -> None:
        """Load the requested prompt/verifier versions and initialize per-episode planner state."""

        self.model = model
        self.prompts = resolve_prompt_version(prompt_version)
        self.verifier = resolve_verifier_version(verifier_version)
        self.prompt_version = str(prompt_version or "v4")
        self.verifier_version = str(getattr(self.verifier, "VERSION", verifier_version))
        self.max_history = max(int(max_history or 0), 10)
        self.use_feedback = use_feedback
        self.plan_horizon_steps = int(plan_horizon_steps)
        self.task: Any | None = None
        self.current_plan: str | None = None
        self.current_step = 0
        self.turn_for_sit_sequence_verified = False

    def reset(self, task: Any) -> None:
        """Clear plans, counters, and sitting-transition state for a new task."""

        self.task = task
        self.current_plan = None
        self.current_step = 0
        self.turn_for_sit_sequence_verified = False

    def _data_url(self, image: Image.Image | str) -> str:
        """Convert a PIL image or image path into the VLM adapter's data-URL format."""

        if isinstance(image, str):
            return image_path_to_data_url(image)
        return image_to_data_url(image)

    def _message(self, image: Image.Image | str, prompt: str) -> dict[str, Any]:
        """Build one multimodal user message containing the prompt and current ego image."""

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._data_url(image)}},
            ],
        }

    def _call(
        self,
        image: Image.Image | str,
        prompt: str,
        *,
        stage: str,
    ) -> PSVStageOutput:
        """Call the VLM once and preserve raw text, parsed JSON, usage, and parse errors."""

        messages = [self._message(image, prompt)]
        try:
            raw_output = self.model.respond(messages)
        except Exception as exc:  # noqa: BLE001
            return PSVStageOutput(
                stage=stage,
                raw={},
                raw_output="",
                prompt=prompt,
                error=f"{type(exc).__name__}: {exc}",
                usage={},
            )
        # Read usage immediately: adapters intentionally expose only the most
        # recent call, and a verifier call may follow in the same agent step.
        usage_value = getattr(self.model, "last_usage", {})
        usage = dict(usage_value) if isinstance(usage_value, dict) else {}
        try:
            raw = parse_json_loose(raw_output)
        except Exception as exc:  # noqa: BLE001
            return PSVStageOutput(
                stage=stage,
                raw={},
                raw_output=raw_output,
                prompt=prompt,
                error=f"{type(exc).__name__}: {exc}",
                usage=usage,
            )
        return PSVStageOutput(
            stage=stage,
            raw=raw,
            raw_output=raw_output,
            prompt=prompt,
            usage=usage,
        )

    def _call_with_retries(
        self,
        image: Image.Image | str,
        prompt: str,
        *,
        stage: str,
    ) -> list[PSVStageOutput]:
        """Retry one logical VLM stage without restarting the environment.

        Transport failures and invalid JSON are both retryable because neither
        produces an executable benchmark decision.  Returning every attempt
        keeps token accounting honest; the evaluator persists only the final
        response for this logical stage, so retries add no routine log clutter.
        """

        attempts: list[PSVStageOutput] = []
        for attempt_index in range(VLM_STAGE_MAX_ATTEMPTS):
            attempt = self._call(image, prompt, stage=stage)
            attempts.append(attempt)
            if not attempt.error:
                break
            if attempt_index < VLM_STAGE_MAX_ATTEMPTS - 1:
                # Provider overload and transient transport failures tend to
                # arrive in short correlated bursts.  Immediate retries make
                # all parallel episodes hit the same burst five times and
                # incorrectly trigger the recovery action.  Backoff changes
                # no successful request or model parameter; it only spaces
                # retries that would otherwise fail at the same environment
                # state.
                time.sleep(VLM_RETRY_BACKOFF_SECONDS[attempt_index])
        return attempts

    @staticmethod
    def _angle_bracket_tokens(action_name: str) -> list[str]:
        """Extract lower-cased angle-bracket arguments from an action name."""

        return [
            token.strip().lower().replace(" ", "_")
            for token in re.findall(r"<([^>]+)>", action_name)
        ]

    @staticmethod
    def _display_action_name(action_name: str) -> str:
        """Return the canonical action name shown in prompts and episode logs."""

        return action_name.replace("left_forward", "left forward").replace(
            "right_forward", "right forward"
        )

    @staticmethod
    def _clamp_degree(value: float) -> float:
        """Clamp a proposed turn angle to the skill's supported angular range."""

        return max(10.0, min(120.0, float(value)))

    @staticmethod
    def _clamp_short_distance(value: float) -> float:
        """Clamp short forward/backward distances to supported motion values."""

        return max(0.10, min(0.60, float(value)))

    @staticmethod
    def _clamp_side_step_distance(value: float) -> float:
        """Clamp lateral displacement to the side-step skill's supported range."""

        return max(0.10, min(0.50, float(value)))

    def _chooser_action(self, chooser_plan: dict[str, Any]) -> SkillCall:
        """Translate the chooser JSON action ID/name into a validated SkillCall."""

        action_name = str(chooser_plan.get("action_name") or "").strip()
        action_id = chooser_plan.get("action_id")
        try:
            # The prompt defines action_id as the stable machine contract.
            # Providers occasionally omit it, so action-name inference is a
            # bounded recovery path rather than a second action vocabulary.
            family_id = int(action_id)
        except Exception:
            lowered = action_name.lower()
            if "downstairs" in lowered or "climb down" in lowered:
                family_id = 7
            elif lowered.startswith("walk"):
                family_id = 0
            elif "stop" in lowered or "stand" in lowered:
                family_id = 1
            elif lowered.startswith("turn"):
                family_id = 2
            elif "climb" in lowered:
                family_id = 3
            elif "sit" in lowered:
                family_id = 4
            elif "step back" in lowered or lowered.startswith("back"):
                family_id = 5
            elif (
                "side step" in lowered
                or "sidestep" in lowered
                or "side walk" in lowered
            ):
                family_id = 6
            else:
                family_id = 1

        tokens = self._angle_bracket_tokens(action_name)
        if family_id == 0:
            speed = tokens[1] if len(tokens) >= 2 else "slow"
            speed = {"slow": "slow", "normal": "normal", "fast": "fast"}.get(
                speed,
                "slow",
            )
            distance = {"slow": 0.2, "normal": 0.4, "fast": 0.6}[speed]
            return SkillCall(
                skill="walk_forward",
                cond=[0.0, distance, 0.0],
                action_name=f"Walk<forward><{speed}>",
            )

        if family_id == 2:
            direction = tokens[0] if len(tokens) >= 1 else "left"
            direction = "right" if direction == "right" else "left"
            degree_text = tokens[1] if len(tokens) >= 2 else action_name
            match = re.search(r"-?\d+(?:\.\d+)?", degree_text)
            degree = self._clamp_degree(float(match.group(0))) if match else 10.0
            signed_degree = degree if direction == "left" else -degree
            degree_label = str(int(round(degree)))
            return SkillCall(
                skill="turn",
                cond=signed_degree,
                action_name=f"Turn<{direction}><{degree_label}>",
            )

        if family_id == 3:
            return SkillCall(
                skill="step_climb_up",
                cond=[0.28, 0.30],
                action_name="Climb upstairs<normal>",
            )

        if family_id == 4:
            match = re.search(r"\d+(?:\.\d+)?", action_name)
            target_height = float(match.group(0)) if match else 0.50
            target_height = max(0.15, min(0.85, target_height))
            return SkillCall(
                skill="sit",
                cond=target_height,
                action_id=4,
                action_name=f"Sit down<{target_height:.2f}>",
            )

        if family_id == 5:
            match = re.search(r"\d+(?:\.\d+)?", action_name)
            distance = (
                self._clamp_short_distance(float(match.group(0))) if match else 0.25
            )
            return SkillCall(
                skill="step_back",
                cond=[0.0, -distance],
                action_id=5,
                action_name=f"Step back<{distance:.2f}>",
            )

        if family_id == 6:
            direction = (
                "right"
                if "right" in tokens or "right" in action_name.lower()
                else "left"
            )
            number_text = " ".join(tokens[1:]) if len(tokens) >= 2 else action_name
            match = re.search(r"\d+(?:\.\d+)?", number_text)
            distance = (
                self._clamp_side_step_distance(float(match.group(0))) if match else 0.25
            )
            signed_x = distance if direction == "left" else -distance
            return SkillCall(
                skill="side_walk",
                cond=signed_x,
                action_id=6,
                action_name=f"Side step<{direction}><{distance:.2f}>",
            )

        if family_id == 7:
            return SkillCall(
                skill="step_climb_down",
                cond=[0.20, 0.40],
                action_id=7,
                action_name="Walk downstairs<normal>",
            )

        return SkillCall(skill="stand", cond=None, action_name="Stop/Stand")

    def _previous_plan_skill_output(self, item: dict[str, Any]) -> dict[str, Any]:
        """Return the latest successful planner-stage JSON for feedback prompting."""

        plan_skill = item.get("planner_skill")
        return plan_skill if isinstance(plan_skill, dict) else {}

    def _previous_verifier_rejection_summary(self, item: dict[str, Any]) -> str:
        """Summarize the latest rejected verifier decision for the next planner prompt."""

        verifier = item.get("verifier")
        if not isinstance(verifier, dict):
            return ""
        verdict = str(verifier.get("verdict") or "").strip().lower()
        if verdict != "replace":
            return ""
        proposed = verifier.get("proposed_action")
        final = verifier.get("final_action")
        proposed_name = ""
        final_name = ""
        if isinstance(proposed, dict):
            proposed_name = str(
                proposed.get("action_name") or proposed.get("skill") or ""
            ).strip()
        if isinstance(final, dict):
            final_name = str(
                final.get("action_name") or final.get("skill") or ""
            ).strip()
        reason = str(verifier.get("reason") or "").strip()
        pieces = ["Verifier replaced"]
        if proposed_name:
            pieces.append(proposed_name)
        if final_name:
            pieces.append(f"with {final_name}")
        if reason:
            pieces.append(f"because {reason}")
        return " ".join(pieces).strip()

    def _plan_skill_history(
        self, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Collect bounded prior planner outputs used as temporal context."""

        rows: list[dict[str, Any]] = []
        for item in history[-self.max_history :]:
            # Feed back only the semantic fields that help the next decision.
            # Raw provider text, token usage, and images belong in per-call
            # logs and would needlessly inflate every subsequent request.
            plan_skill = self._previous_plan_skill_output(item)
            row = {
                "step": item.get("step", "?"),
                "visible_state": clip_text(
                    plan_skill.get("visible_state")
                    or item.get("visual_state_description", ""),
                    220,
                ),
                "mid_level_progress_analysis": clip_text(
                    plan_skill.get("mid_level_progress_analysis")
                    or item.get("reasoning_and_reflection", ""),
                    220,
                ),
                "mid_level_goal": clip_text(
                    plan_skill.get("mid_level_goal")
                    or item.get("language_plan")
                    or item.get("current_subgoal", ""),
                    160,
                ),
                "low_level_action_reasoning": clip_text(
                    plan_skill.get("low_level_action_reasoning", ""),
                    180,
                ),
                "action_name": clip_text(
                    plan_skill.get("action_name") or item.get("action_text", ""),
                    80,
                ),
            }
            verifier_rejection = self._previous_verifier_rejection_summary(item)
            if verifier_rejection:
                row["verifier_rejection"] = clip_text(verifier_rejection, 240)
            rows.append(row)
        return rows

    def _plan_skill_prompt(
        self,
        history: list[dict[str, Any]],
        env_feedback: list[dict[str, Any]],
    ) -> str:
        """Assemble the versioned planner prompt from task, history, and feedback."""

        del env_feedback
        if self.task is None:
            raise RuntimeError("Call reset(task) before act().")
        input_obj = {
            "goal": self.task.instruction,
            "current_ego_view_image": "attached",
            "current_step": len(history),
            "history": self._plan_skill_history(history),
        }
        return self.prompts.render(input_obj)

    @staticmethod
    def _is_finish_goal(plan_skill_plan: dict[str, Any]) -> bool:
        """Return whether the planner's mid-level goal explicitly requests Stop/Stand."""

        goal = str(plan_skill_plan.get("mid_level_goal") or "").strip().lower()
        return goal in {"stop/stand", "stop stand", "stop", "stand"}

    def _planner_from_plan_skill(
        self,
        plan_skill_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert parsed planner JSON into the public PlannerResult fields."""

        goal = str(plan_skill_plan.get("mid_level_goal") or "").strip()
        if not goal:
            goal = self.current_plan or "Stop/Stand"
        return {
            "visible_state": str(plan_skill_plan.get("visible_state") or ""),
            "plan_status": "finish"
            if self._is_finish_goal(plan_skill_plan)
            else "continue",
            "plan": goal,
            "reason": str(plan_skill_plan.get("mid_level_progress_analysis") or ""),
        }

    @staticmethod
    def _clean_additional_info(value: Any) -> dict[str, Any]:
        """Normalize optional planner metadata into a JSON-safe dictionary."""

        return value if isinstance(value, dict) else {}

    @classmethod
    def _is_additional_info_true(cls, value: Any, key: str) -> bool:
        """Read a boolean-like flag from planner additional_info."""

        flag = cls._clean_additional_info(value).get(key)
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, str):
            return flag.strip().lower() == "true"
        return False

    @staticmethod
    def _skiller_from_plan_skill(
        plan_skill_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert planner-stage action fields into a validated executable skill."""

        return {
            "plan_following_state": str(
                plan_skill_plan.get("mid_level_progress_analysis") or ""
            ),
            "skill_reason": str(
                plan_skill_plan.get("low_level_action_reasoning") or ""
            ),
            "current_subgoal": str(plan_skill_plan.get("mid_level_goal") or ""),
            "action_id": plan_skill_plan.get("action_id"),
            "action_name": str(plan_skill_plan.get("action_name") or ""),
            "additional_info": HumanClawBenchPSVPlanSkillPlanner._clean_additional_info(
                plan_skill_plan.get("additional_info")
            ),
        }

    def _gate_turn_for_sit_verifier(
        self,
        skiller_plan: dict[str, Any],
        proposed_action: SkillCall,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Decide whether a turn-before-sit proposal requires verifier approval."""

        info = dict(self._clean_additional_info(skiller_plan.get("additional_info")))
        requested = proposed_action.skill == "turn" and self._is_additional_info_true(
            info, "turn_for_sit"
        )
        scheduler_state: dict[str, Any] = {
            "turn_for_sit_requested": requested,
            "turn_for_sit_sequence_verified_before": self.turn_for_sit_sequence_verified,
            "turn_for_sit_verifier": "not_requested",
        }
        if not requested:
            return skiller_plan, scheduler_state
        if not self.turn_for_sit_sequence_verified:
            scheduler_state["turn_for_sit_verifier"] = "verify_first_turn"
            return skiller_plan, scheduler_state

        gated = dict(skiller_plan)
        info.pop("turn_for_sit", None)
        gated["additional_info"] = info
        scheduler_state["turn_for_sit_verifier"] = "skipped_after_first_verified_turn"
        return gated, scheduler_state

    def _update_turn_for_sit_state(
        self,
        scheduler_state: dict[str, Any],
        final_action: SkillCall,
    ) -> None:
        """Track whether the preparatory turn in a sitting sequence was approved."""

        requested = bool(scheduler_state.get("turn_for_sit_requested"))
        if requested and final_action.skill == "turn":
            self.turn_for_sit_sequence_verified = True
        elif final_action.skill != "turn":
            self.turn_for_sit_sequence_verified = False
        else:
            self.turn_for_sit_sequence_verified = False
        scheduler_state["turn_for_sit_sequence_verified_after"] = (
            self.turn_for_sit_sequence_verified
        )

    def _apply_plan_skill_output(self, plan_skill_plan: dict[str, Any]) -> None:
        """Update current-plan state from a successful planner-stage response."""

        plan_text = str(
            plan_skill_plan.get("mid_level_goal") or self.current_plan or ""
        ).strip()
        if not plan_text:
            plan_text = "Stop/Stand"

        if (
            self.current_plan is None
            or plan_text != self.current_plan
            or self.current_step >= self.plan_horizon_steps
        ):
            # A changed goal or expired horizon starts a fresh local-plan age;
            # repeating the same goal merely advances that age after act().
            self.current_plan = plan_text
            self.current_step = 0
        else:
            self.current_plan = plan_text

    def _recent_skill_history_text(self, history: list[dict[str, Any]]) -> str:
        """Format recent executed skills for inclusion in a verifier prompt."""

        if not history:
            return "none"
        actions: list[str] = []
        for item in history[-self.max_history :]:
            text = str(item.get("action_text") or "").strip()
            if not text:
                action = item.get("action")
                if isinstance(action, dict):
                    text = str(
                        action.get("action_name") or action.get("skill") or ""
                    ).strip()
            if text:
                actions.append(clip_text(text, 80))
        if not actions:
            return "none"

        grouped: list[tuple[str, int]] = []
        for action in actions:
            if grouped and grouped[-1][0] == action:
                grouped[-1] = (action, grouped[-1][1] + 1)
            else:
                grouped.append((action, 1))
        return "; ".join(f"{action} x{count}" for action, count in grouped)

    def _verifier_prompt(
        self,
        planner_plan: dict[str, Any],
        skiller_plan: dict[str, Any],
        proposed_action: SkillCall,
        history: list[dict[str, Any]],
        env_feedback: list[dict[str, Any]],
    ) -> str:
        """Assemble the route-specific verifier prompt for the proposed action."""

        if self.task is None:
            raise RuntimeError("Call reset(task) before act().")
        action_name = proposed_action.action_name or skill_to_text(proposed_action)
        tokens = self._angle_bracket_tokens(action_name)
        proposed_action_name = self._display_action_name(action_name)
        input_obj = {
            "goal": self.task.instruction,
            "proposed_action": proposed_action_name,
        }
        additional_info = self._clean_additional_info(
            skiller_plan.get("additional_info")
        )
        if additional_info:
            input_obj["additional_info"] = additional_info
        if proposed_action.skill == "stand" and self._is_additional_info_true(
            additional_info, "stop_after_sit"
        ):
            input_obj["recent_actions"] = self._recent_skill_history_text(history)
        return self.verifier.verifier_prompt(
            proposed_action,
            proposed_action_name=proposed_action_name,
            input_obj=input_obj,
            action_name=action_name,
            tokens=tokens,
            planner_plan=planner_plan,
            skiller_plan=skiller_plan,
            history=history,
            env_feedback=env_feedback,
        )

    def _verifier_action(
        self,
        verifier_plan: dict[str, Any],
        proposed_action: SkillCall,
    ) -> SkillCall:
        """Convert verifier JSON into an approved, corrected, or rejected SkillCall."""

        return self.verifier.verifier_action(
            verifier_plan,
            proposed_action,
            self._chooser_action,
        )

    def _normalize_verifier_plan(
        self,
        verifier_plan: dict[str, Any],
        proposed_action: SkillCall,
        final_action: SkillCall,
    ) -> dict[str, Any]:
        """Normalize verifier fields while retaining both raw and executable decisions."""

        normalized = self.verifier.normalize_verifier_plan(
            verifier_plan,
            proposed_action,
            final_action,
        )
        normalized["verifier_version"] = self.verifier_version
        return normalized

    def _combined_raw_plan(
        self,
        plan_skill_plan: dict[str, Any],
        planner_plan: dict[str, Any],
        skiller_plan: dict[str, Any],
        verifier_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine planner and verifier records into one audit-friendly plan object."""

        reasoning_parts = [
            f"planner: {planner_plan.get('reason', '')}",
            f"skiller: {skiller_plan.get('skill_reason', skiller_plan.get('reasoning_and_reflection', ''))}",
            f"verifier: {verifier_plan.get('reason', verifier_plan.get('reasoning_and_reflection', ''))}",
        ]
        visual_state = (
            verifier_plan.get("visual_state_description")
            or skiller_plan.get("visual_state_description")
            or skiller_plan.get("plan_following_state")
            or planner_plan.get("visible_state")
            or ""
        )
        final_action = verifier_plan.get("final_action")
        if not isinstance(final_action, dict):
            final_action = {}
        action_name = str(
            final_action.get("action_name") or final_action.get("name") or ""
        ).strip()
        skill = str(final_action.get("skill") or "").strip()
        action_id = final_action.get("action_id")
        stand_action = skill == "stand" or action_name == "Stop/Stand" or action_id == 1
        finish_at_target = self._is_finish_goal(plan_skill_plan) and stand_action
        raw_plan = {
            "visual_state_description": visual_state,
            "reasoning_and_reflection": " | ".join(
                part for part in reasoning_parts if part.strip()
            ),
            "current_subgoal": (
                skiller_plan.get("current_subgoal")
                or skiller_plan.get("action_name")
                or planner_plan.get("plan", "")
            ),
            "language_plan": str(plan_skill_plan.get("mid_level_goal") or ""),
            "at_target": bool(
                verifier_plan.get("at_target", False) or finish_at_target
            ),
        }
        return raw_plan

    def act(
        self,
        image: Image.Image | str,
        history: list[dict[str, Any]],
        env_feedback: list[dict[str, Any]],
    ) -> PlannerResult:
        """Run planner and optional verifier stages, then return one executable skill."""

        if self.task is None:
            raise RuntimeError("Call reset(task) before act().")

        stage_outputs: list[PSVStageOutput] = []
        if hasattr(self.model, "current_episode_step"):
            self.model.current_episode_step = len(history)

        plan_skill_prompt = self._plan_skill_prompt(history, env_feedback)
        # Stage 1 always sees the current ego image and emits the complete
        # percept -> mid-level goal -> low-level action JSON contract.  A
        # Transport or parse failure is retried at this same state.  If all
        # five attempts fail, one short forward walk lets the closed loop reach
        # a new observation rather than restarting or abandoning the episode.
        plan_skill_attempts = self._call_with_retries(
            image,
            plan_skill_prompt,
            stage="percept_mid_low",
        )
        stage_outputs.extend(plan_skill_attempts)
        plan_skill_stage = plan_skill_attempts[-1]
        planner_fallback = bool(plan_skill_stage.error)
        if planner_fallback:
            plan_skill_stage.error = (
                f"{plan_skill_stage.error}; planner retry limit reached; "
                "executing Walk<forward><slow>"
            )
            plan_skill_plan = {
                "visible_state": "",
                "mid_level_progress_analysis": (
                    "Planner unavailable after five attempts; use one short "
                    "forward step and re-plan from the next observation."
                ),
                "mid_level_goal": self.current_plan or "Continue toward the task goal.",
                "low_level_action_reasoning": ("Retry-exhausted recovery action."),
                "action_id": 0,
                "action_name": "Walk<forward><slow>",
                "additional_info": {"fallback": "walk_slow_after_retry_exhausted"},
            }
        else:
            plan_skill_plan = plan_skill_stage.raw

        self._apply_plan_skill_output(plan_skill_plan)
        planner_plan = self._planner_from_plan_skill(plan_skill_plan)
        skiller_plan = self._skiller_from_plan_skill(plan_skill_plan)
        proposed_action = self._chooser_action(skiller_plan)
        skiller_plan, scheduler_state = self._gate_turn_for_sit_verifier(
            skiller_plan,
            proposed_action,
        )

        verifier_prompt = (
            ""
            if planner_fallback
            else self._verifier_prompt(
                planner_plan,
                skiller_plan,
                proposed_action,
                history,
                env_feedback,
            )
        )
        if planner_fallback:
            # This action exists specifically because no planner response was
            # available.  Calling the verifier through the same unavailable
            # transport would add five more failures without new information.
            verifier_plan = {
                "verdict": "accept",
                "reason": "planner retry exhausted; executed Walk<forward><slow>",
                "fallback": "walk_slow_after_retry_exhausted",
            }
            final_action = proposed_action
            verifier_plan = self._normalize_verifier_plan(
                verifier_plan,
                proposed_action,
                final_action,
            )
        elif not verifier_prompt:
            # Most action families need no second model call.  This synthetic
            # accept record has no PSVStageOutput, so no fake verifier log is
            # written and token accounting still reflects real API calls.
            verifier_plan = {
                "verdict": "accept",
                "reason": "no verifier for this action type",
            }
            final_action = proposed_action
            verifier_plan = self._normalize_verifier_plan(
                verifier_plan,
                proposed_action,
                final_action,
            )
        else:
            # Stage 2 uses the same current image but a route-specific
            # verifier prompt.  Its normalized decision is authoritative when
            # available.  If all five attempts fail, the planner proposal is
            # still a valid action, so the verifier fails open and execution
            # continues from the current state without restarting the episode.
            verifier_attempts = self._call_with_retries(
                image,
                verifier_prompt,
                stage="verifier",
            )
            stage_outputs.extend(verifier_attempts)
            verifier_stage = verifier_attempts[-1]
            if verifier_stage.error:
                verifier_plan = {
                    "verdict": "accept",
                    "reason": (
                        "verifier unavailable after 5 attempts; accepted the "
                        "planner-proposed action"
                    ),
                    "fallback": "accept_after_retry_exhausted",
                }
                final_action = proposed_action
            else:
                verifier_plan = verifier_stage.raw
                final_action = self._verifier_action(verifier_plan, proposed_action)
            verifier_plan = self._normalize_verifier_plan(
                verifier_plan,
                proposed_action,
                final_action,
            )

        self._update_turn_for_sit_state(scheduler_state, final_action)
        if self.current_plan:
            self.current_step += 1
        raw_plan = self._combined_raw_plan(
            plan_skill_plan,
            planner_plan,
            skiller_plan,
            verifier_plan,
        )
        return PlannerResult(
            raw_plan=raw_plan,
            action=final_action,
            planner_skill=plan_skill_plan,
            verifier=verifier_plan,
            stage_outputs=stage_outputs,
        )


__all__ = [
    "VLM_RETRY_BACKOFF_SECONDS",
    "VLM_STAGE_MAX_ATTEMPTS",
    "HumanClawBenchPSVPlanSkillPlanner",
]

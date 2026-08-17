"""Verifier v3 used for the full-validation evaluation.

The shared frame, action spaces, route-specific rules, questions, and output
schemas are separate components.  No complete prompt is copied once per route,
and this module does not inherit from or patch an older verifier version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..skills import SkillCall, skill_to_text

VERSION = "v3"

SYSTEM_PROMPT = (
    "You control a BLUE humanoid in a 3D home. You receive an ego-centric "
    "head-view image from the humanoid point of view. Your blue body and feet "
    "are at the bottom-center edge of the image; that bottom-center edge is "
    "your near-body position."
)

TASK_TEMPLATE = """\
You are the verifier for the proposed action:
{proposed_action_name}

Current ego image is attached.
Use the current image as the main evidence.

{action_space}

{action_guideline}

Input:
{input_json}

Questions:
{questions}

Return exactly this JSON:
{return_json}
"""

ACTION_SPACE = """\
Available HumanClaw skills for the next 0.5 seconds:

action id 0: Walk<forward><speed>
- direction: <forward>
- speed: <slow> 0.2m, <normal> 0.4m, or <fast> 0.6m

action id 2: Turn<direction><degree>
- direction: <left> or <right>
- degree: 10-120 degrees

action id 3: Climb upstairs<normal>

action id 5: Step back<distance>
- distance: 0.1-0.6m

action id 6: Side step<direction><distance>
- direction: <left> or <right>
- distance: 0.1-0.5m

action id 7: Walk downstairs<normal>
"""

SITTING_ACTION_SPACE = """\
Available actions: 0 Walk<forward><speed>; 2 Turn<direction><degree>; 3 Climb upstairs<normal>; 5 Step back<distance>; 6 Side step<direction><distance>; 7 Walk downstairs<normal>.
"""

COMMON_GUIDELINE = """\
Action guideline:
- Verify only the proposed action using the current ego image.
- If the proposed action is appropriate, keep it.
- If the proposed action is not appropriate, replace it with one final action from the action space above.
- Return the final action using the exact action_name format.
"""

WALK_GUIDELINE = """\
- For Walk: check the straight-forward lane. Accept Walk<forward> if that lane is clear enough and the action progresses the current plan.
- For Walk: reject Walk<forward> if the straight-forward lane leads into an immediate wall, closed door, furniture, or other obstacle; replace it with a Turn toward the clearest visible open side lane or Side step toward the clear direction. For a target object, touching the target is necessary.
"""

STOP_GUIDELINE = """\
- For Stop/Stand: first decide what the high-level goal is and whether it is fully completed, for example whether the humanoid is close enough to the target. Accept Stop/Stand only if the full goal is complete; otherwise replace it with one final non-Stop action from the action space that continues progress toward the goal.
"""

CLIMB_UP_GUIDELINE = """\
- For Climb upstairs: accept Climb only when a first riser or next upward step is at the feet/directly ahead and the humanoid is facing the stair direction, or when the humanoid is already on stairs with upward steps ahead.
- For Climb upstairs: if no upward riser/step is at the feet or directly ahead, replace Climb with the smallest Walk or Turn that aligns with the current plan.
"""

TURN_FOR_SIT_GUIDELINE = """\
- For turn_for_sit: accept the turn only if the target sitting surface is touching the body at zero distance and this Turn rotates the body away so the target will be behind for Sit down.
- Reject if there is still any distance to the sitting target; continue moving toward the target.
"""

STOP_AFTER_SIT_GUIDELINE = """\
- For stop_after_sit: accept if the image shows the back of the blue legs contacting the object rather than floating or unsupported.
- Do not reject only because ego-view body posture is ambiguous.
"""

WALK_RETURN_SCHEMA = """\
{
  "lane_checked": "{lane_checked}",
  "lane_observation": "<what is in the checked walking lane>",
  "nearest_obstacle_distance_m": <number>,
  "unnecessary_collision": true | false,
  "verdict": "accept" | "replace",
  "reason": "<short reason>",
  "final_action_id": 0 | 1 | 2 | 3,
  "final_action_name": "<final action>"
}
"""

STOP_RETURN_SCHEMA = """\
{
  "high_level_goal": "<goal stated by the instruction>",
  "goal_type": "navigation" | "environment_interaction",
  "goal_completed": true | false,
  "completion_reason": "<short reason explaining why goal_completed is true or false>",
  "navigation_target_found": true | false | null,
  "estimated_distance_to_target_m": <number or null>,
  "touching_or_zero_distance": true | false | null,
  "interaction_completed": true | false | null,
  "verdict": "accept" | "replace",
  "reason": "<short reason>",
  "final_action_id": 0 | 1 | 2 | 3,
  "final_action_name": "<final action; if verdict is replace, this must be a non-Stop action>"
}
"""

CLIMB_UP_RETURN_SCHEMA = """\
{
  "riser_edge_angle_deg_from_horizontal": <number>,
  "facing_stairs": "front" | "side",
  "first_riser_distance_m": <number>,
  "front_facing_angle_range_deg": "<angle range>",
  "verdict": "accept" | "replace",
  "reason": "<short reason>",
  "final_action_id": 0 | 1 | 2 | 3,
  "final_action_name": "<final action>"
}
"""

TURN_FOR_SIT_RETURN_SCHEMA = """\
{
  "sitting_target_zero_distance": true | false | "unclear",
  "turn_is_for_sit_setup": true | false,
  "verdict": "accept" | "replace",
  "reason": "<short reason>",
  "final_action_id": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7,
  "final_action_name": "<final action>"
}
"""

STOP_AFTER_SIT_RETURN_SCHEMA = """\
{
  "clear_sit_failure": true | false,
  "back_of_legs_contacting_object": true | false | "unclear",
  "verdict": "accept" | "replace",
  "reason": "<short reason>",
  "final_action_id": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7,
  "final_action_name": "<final action>"
}
"""


@dataclass(frozen=True)
class RoutePrompt:
    """All variable components for one verifier route."""

    name: str
    action_space: str
    guideline: str
    questions: str
    return_schema: str


def _truthy(value: Any) -> bool:
    """Interpret common boolean-like JSON values from verifier responses."""

    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _additional_info(skiller_plan: dict[str, Any] | None) -> dict[str, Any]:
    """Return verifier additional_info only when it is a JSON object."""

    if isinstance(skiller_plan, dict) and isinstance(
        skiller_plan.get("additional_info"), dict
    ):
        return skiller_plan["additional_info"]
    return {}


def route_prompt(
    proposed_action: SkillCall,
    *,
    tokens: list[str],
    skiller_plan: dict[str, Any] | None = None,
) -> RoutePrompt | None:
    """Select the verifier route and its independently composable sections."""

    info = _additional_info(skiller_plan)
    if proposed_action.skill == "turn" and _truthy(info.get("turn_for_sit")):
        return RoutePrompt(
            name="turn_for_sit",
            action_space=SITTING_ACTION_SPACE,
            guideline=COMMON_GUIDELINE + TURN_FOR_SIT_GUIDELINE,
            questions=(
                "1. Is the target sitting surface touching the body at zero distance, or is there still distance to cover?\n"
                "2. Does this Turn rotate the body away so the target will be behind for Sit down?\n"
                "3. Should the proposed Turn be accepted or replaced with a movement that continues toward the target?"
            ),
            return_schema=TURN_FOR_SIT_RETURN_SCHEMA,
        )
    if proposed_action.skill == "stand" and _truthy(info.get("stop_after_sit")):
        return RoutePrompt(
            name="stop_after_sit",
            action_space=SITTING_ACTION_SPACE,
            guideline=COMMON_GUIDELINE + STOP_AFTER_SIT_GUIDELINE,
            questions=(
                "1. Is there clear evidence that sitting failed?\n"
                "2. Does the image show the back of the blue legs contacting the object rather than floating or unsupported?\n"
                "3. Should Stop/Stand be accepted or replaced?"
            ),
            return_schema=STOP_AFTER_SIT_RETURN_SCHEMA,
        )
    if proposed_action.skill == "walk_forward":
        speed = tokens[1] if len(tokens) > 1 else "slow"
        return RoutePrompt(
            name="walk",
            action_space=ACTION_SPACE,
            guideline=COMMON_GUIDELINE + WALK_GUIDELINE,
            questions=(
                "1. What is in the straight-forward walking lane?\n"
                "2. How far is the nearest object or obstacle in that lane, in meters?\n"
                f"3. If the humanoid executes Walk<forward><{speed}> for the next 0.5 seconds, would it unnecessarily collide with anything?\n"
                "4. If this walk is not appropriate, what should the final action be?"
            ),
            return_schema=WALK_RETURN_SCHEMA.replace("{lane_checked}", "forward"),
        )
    if proposed_action.skill == "stand":
        return RoutePrompt(
            name="stop",
            action_space=ACTION_SPACE,
            guideline=COMMON_GUIDELINE + STOP_GUIDELINE,
            questions=(
                "1. What is the high-level goal in the instruction, and is it a navigation goal, or an environment-interaction goal?\n"
                "2. Has the high-level goal been fully completed according to the current image? Give the reason whether the answer is yes or no.\n"
                "3. If this is navigation: has the target been found? How far is it to the target? Are you touching it (~zero distance)?\n"
                "4. If this is environment interaction: has that interaction been completed?\n"
                "5. If the goal is complete, accept Stop/Stand. Otherwise, choose one other non-Stop skill from the action space that best continues the goal."
            ),
            return_schema=STOP_RETURN_SCHEMA,
        )
    if proposed_action.skill == "step_climb_up":
        return RoutePrompt(
            name="climb_up",
            action_space=ACTION_SPACE,
            guideline=COMMON_GUIDELINE + CLIMB_UP_GUIDELINE,
            questions=(
                "1. What is the approximate angle between the visible stair/riser edge and the horizontal image direction, in degrees?\n"
                "2. Is the humanoid facing the stairs from the front, or is it side-facing the stairs?\n"
                "3. How far is the first stair/riser from the feet, in meters?\n"
                "4. What angle range counts as facing the stairs from the front?"
            ),
            return_schema=CLIMB_UP_RETURN_SCHEMA,
        )
    return None


def render(
    route: RoutePrompt,
    *,
    proposed_action_name: str,
    input_obj: dict[str, Any],
) -> str:
    """Compose one verifier prompt, with the system text included once."""

    task = TASK_TEMPLATE.format(
        proposed_action_name=proposed_action_name,
        action_space=route.action_space,
        action_guideline=route.guideline,
        input_json=json.dumps(input_obj, ensure_ascii=False, indent=2),
        questions=route.questions,
        return_json=route.return_schema,
    )
    return SYSTEM_PROMPT + "\n\n" + task


def verifier_prompt(
    proposed_action: SkillCall,
    *,
    proposed_action_name: str,
    input_obj: dict[str, Any],
    action_name: str,
    tokens: list[str],
    planner_plan: dict[str, Any] | None = None,
    skiller_plan: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    env_feedback: list[dict[str, Any]] | None = None,
) -> str:
    """Select and render the v3 verifier template for the proposed skill route."""

    del action_name, planner_plan, history, env_feedback
    route = route_prompt(
        proposed_action,
        tokens=tokens,
        skiller_plan=skiller_plan,
    )
    if route is None:
        return ""
    return render(
        route,
        proposed_action_name=proposed_action_name,
        input_obj=input_obj,
    )


def verifier_action(
    verifier_plan: dict[str, Any],
    proposed_action: SkillCall,
    choose_action: Callable[[dict[str, Any]], SkillCall],
) -> SkillCall:
    """Map a v3 verifier decision to the approved or corrected action."""

    verdict = str(verifier_plan.get("verdict") or "accept").strip().lower()
    final_name = str(verifier_plan.get("final_action_name") or "").strip()
    if verdict == "replace" and final_name and "<final action>" not in final_name:
        try:
            return choose_action(
                {
                    "action_id": verifier_plan.get("final_action_id"),
                    "action_name": final_name,
                }
            )
        except Exception:
            try:
                return choose_action({"action_name": final_name})
            except Exception:
                pass
    return proposed_action


def normalize_verifier_plan(
    verifier_plan: dict[str, Any],
    proposed_action: SkillCall,
    final_action: SkillCall,
) -> dict[str, Any]:
    """Normalize route-specific verifier JSON into one stable result schema."""

    normalized = dict(verifier_plan)
    normalized.setdefault("verdict", "accept")
    normalized.setdefault("reason", "")
    normalized["visual_state_description"] = (
        normalized.get("lane_observation") or normalized.get("reason") or ""
    )
    normalized["reasoning_and_reflection"] = normalized.get("reason", "")
    normalized["executable_plan"] = [
        {
            "action_id": final_action.action_id,
            "action_name": final_action.action_name or skill_to_text(final_action),
        }
    ]
    normalized["at_target"] = False
    normalized["proposed_action"] = proposed_action.to_json()
    normalized["final_action"] = final_action.to_json()
    return normalized


__all__ = [
    "ACTION_SPACE",
    "COMMON_GUIDELINE",
    "RoutePrompt",
    "SITTING_ACTION_SPACE",
    "SYSTEM_PROMPT",
    "TASK_TEMPLATE",
    "VERSION",
    "normalize_verifier_plan",
    "render",
    "route_prompt",
    "verifier_action",
    "verifier_prompt",
]

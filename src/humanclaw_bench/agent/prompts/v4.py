"""Planner prompt v4 used for the full-validation evaluation.

The prompt is stored as named, independently editable sections.  It is not a
string patch over an older version.  ``render`` composes the sections once and
preserves the exact effective v4 prompt sent by the rollout code.
"""

from __future__ import annotations

import json
from typing import Any

VERSION = "v4"

SYSTEM_PROMPT = (
    "You control a BLUE humanoid in a 3D home. You receive an ego-centric "
    "head-view image from the humanoid point of view. Your blue body and feet "
    "are at the bottom-center edge of the image; that bottom-center edge is "
    "your near-body position."
)

ROLE_AND_CONTEXT = """\
You are the plan-and-skill chooser for this humanoid.

You will be given a high-level instruction. At each step, first plan a mid-level goal for the next 2-3 seconds, then choose one low-level detailed action for the next 0.5 seconds.

Use the attached current ego image as the main evidence. The recent history, if any, is reference memory. Use it to keep action continuity and to decide whether to continue or update the previous mid-level goal. You may continue the previous mid-level goal, or update it when it has been completed or when the current image shows a new situation."""

ACTION_SPACE = """\
Detailed action list:

action id 0: Walk: direction: <forward>: straight forward
             speed: <slow>: 0.2m; <normal>: 0.4m; <fast>: 0.6m
Walk is walking. The interface is <direction> and <speed>. Ideally, after one 0.5-second Walk chunk, the humanoid should keep roughly the same facing direction as before the chunk. When starting from Stop/Stand or a still pose, prefer one slow Walk chunk as startup before returning to normal or fast walking.

action id 1: Stop/Stand: a terminal stop action. Use it only when the high-level instruction is fully completed; do not use it merely to pause, wait, think, or keep standing still.

action id 2: Turn: directions: <left/right>; degree: 10-120 degrees
Turn is turning in place. The degree is the rotation amount within 0.5 seconds, so it should not be unnecessarily large.

action id 3: Climb upstairs<normal>
Use Climb when the next stair/riser is already at the feet and the stair direction is aligned with the humanoid facing direction. Otherwise, use Walk and Turn to adjust position and orientation first.

action id 4: Sit down<target height>
Sit down in place. <target height> is the height of the sitting target.

action id 5: Step back<distance>
Step backward 0.1-0.6m while keeping facing forward.

action id 6: Side step<direction><distance>
Side step laterally, like a crab-walk, left or right 0.1-0.5m while keeping facing forward.

action id 7: Walk downstairs<normal>
Walk down when the next downward stair/step is already at the feet and the stair direction is aligned with the humanoid facing direction. Otherwise, use Walk, Turn, or Side step to adjust position and orientation first."""

MID_LEVEL_GUIDANCE = """\
Guidance for the mid-level goal:
- Base the mid-level goal on visible landmarks, but do not over-specify low-level details.
- Treat a closed door as a non-traversable landmark unless it is visibly open; do not plan to go through or explore beyond a closed door.
- When the target is not visible, scan for it by turning in one consistent direction; if the target is still not found during that scan, use history and the current view to choose the most likely visible place and move toward it rather than continuing to rotate at the same spot.
- Do not spin repeatedly, and avoid frequent left/right turn alternation.
- If the target is visible, plan to approach it through a clear safe path; a combination of walking, turning, and side stepping may be needed to avoid obstacles.
- Avoid unnecessary collisions while moving: if the humanoid is too close to a wall, closed door, furniture, or other obstacle, first turn toward a visible open passage, then enter that passage.
- Prefer the center of open floor and clear passages; avoid narrow gaps or enclosed areas that could trap the humanoid.
- Avoid vague phrases such as "systematically explore" or "search the house".
- If the high-level instruction is fully completed, set mid_level_goal to Stop/Stand and choose Stop/Stand.
- Sitting is a sequence: first reach zero distance/touching the target, then compute the Turn degree and execute it, optionally Step back up to two chunks, then Sit down for 3-4 chunks.
- When the Turn is for sit setup, set additional_info to {{"turn_for_sit": true}}.
- When sitting is completed, choose Stop/Stand with additional_info {{"stop_after_sit": true}}."""

LOW_LEVEL_GUIDANCE = """\
Guidance for low-level detailed action/skill choice:
- Choose the action that best matches the current mid-level goal.
- When the forward lane is clear and the plan target, open path, doorway, hallway, passage, room edge, or stair entrance is roughly ahead, default to Walk<forward>.
- Use Turn when the mid-level goal asks to scan around, when nearby obstacles require route adjustment, or when the useful walking direction is not centered ahead. Do not keep turning continuously or alternate left/right turns repeatedly.
- If history contains repeated Turn chunks and the forward walking lane is safe, choose a Walk into that lane instead of another Turn.
- Avoid unnecessary collisions: if an obstacle blocks the current lane, choose a Turn toward the clearest visible open floor instead of walking forward into the obstacle.
- Do not walk into a visible furniture cluster or narrow gap when an open floor lane, doorway, hallway, passage, or room edge is available.
- Use slow speed for tight or uncertain startup, normal speed for clear indoor progress, and fast speed only for a wide unobstructed forward lane.
- Sitting is a sequence: reach zero distance/touching the target, Turn with turn_for_sit, optionally Step back up to two chunks, Sit down for 3-4 chunks, then Stop/Stand with stop_after_sit.
- Side step moves laterally left or right toward a clear path to bypass obstacles at the feet or directly ahead. When used, it usually requires two or more chunks."""

OUTPUT_CONTRACT = """\
Use exact action names. For walking, use only `Walk<forward><slow>`, `Walk<forward><normal>`, or `Walk<forward><fast>`. For local adjustment, use `Step back<distance>`, `Side step<left><distance>`, or `Side step<right><distance>`, for example `Step back<0.25>` or `Side step<left><0.25>`. For stairs, use `Climb upstairs<normal>` or `Walk downstairs<normal>`. For sitting, use `Sit down<target height>`, for example `Sit down<0.45>`.

Current ego image is attached.

Input:
{input_json}

Return exactly this JSON:
{{
  "visible_state": "From left to right, I can see <object/area>, about <distance> meters from me; ...; and <object/area>, about <distance> meters from me. <Say whether the target object/place is visible. Say whether the near-body lane is clear or blocked.>",
  "mid_level_progress_analysis": "<Based on the high-level instruction, the visible_state, and the history, briefly analyze the current situation and what mid-level goal should be completed in the next 2-3 seconds. Decide whether to continue the previous mid-level goal or update it, and give a short reason.>",
  "mid_level_goal": "<One very concise sentence describing the mid-level goal for the next 2-3 seconds.>",
  "low_level_action_reasoning": "<Based on the mid_level_goal, visible_state, and history, briefly explain why this exact skill and parameters should be chosen for the next 0.5 seconds.>",
  "action_id": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7,
  "action_name": "<Walk<forward><speed> | Stop/Stand | Turn<direction><degree> | Climb upstairs<normal> | Sit down<target height> | Step back<distance> | Side step<direction><distance> | Walk downstairs<normal>>",
  "additional_info": {{}} | {{"turn_for_sit": true}} | {{"stop_after_sit": true}}
}}"""

SECTIONS = (
    SYSTEM_PROMPT,
    ROLE_AND_CONTEXT,
    ACTION_SPACE,
    MID_LEVEL_GUIDANCE,
    LOW_LEVEL_GUIDANCE,
    OUTPUT_CONTRACT,
)

# This is the auditable, unformatted prompt.  It contains SYSTEM_PROMPT once.
TEMPLATE = "\n\n".join(SECTIONS) + "\n"


def render(input_obj: dict[str, Any]) -> str:
    """Render the planner prompt for one observation."""

    return TEMPLATE.format(
        input_json=json.dumps(input_obj, ensure_ascii=False, indent=2)
    )


__all__ = [
    "ACTION_SPACE",
    "LOW_LEVEL_GUIDANCE",
    "MID_LEVEL_GUIDANCE",
    "OUTPUT_CONTRACT",
    "ROLE_AND_CONTEXT",
    "SECTIONS",
    "SYSTEM_PROMPT",
    "TEMPLATE",
    "VERSION",
    "render",
]

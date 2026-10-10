"""Planner v5: action-conditioned feedback with proprioception and two frames."""

from __future__ import annotations

import json
from typing import Any

VERSION = "v5"
USES_FEEDBACK = True

SYSTEM_PROMPT = (
    "You control a BLUE humanoid in a 3D home. You receive an ego-centric "
    "head-view image from the humanoid point of view. Your blue body and feet "
    "are at the bottom-center edge of the image; that bottom-center edge is "
    "your near-body position."
)

ROLE_AND_CONTEXT = """\
You are the plan-and-skill chooser for this humanoid.
At each decision, first assess the last EXECUTED action, then update your state estimate, then choose a mid-level goal for the next 2-3 seconds and one skill for the next 0.5 seconds.

The labeled previous decision image is BEFORE the last action; the current image is AFTER it. At step zero only the current image exists. Feedback contains current realized own-body pose and core joint rotations, plus the last action's requested parameters, generated motion and actual motion. These are different quantities: requested and generated motion do not prove that the body moved. All distances use metres, rotations state their units, and motion deltas are expressed in the starting body's horizontal frame (left, forward, up). Joint rotation vectors are parent-local radians. Root tilt is not a contact sensor; position relative to reset is not distance above the floor.

Judge movement according to the executed skill: walking/back/side steps require directional displacement, turns require heading change, stairs require suitable horizontal/vertical progress, and sitting requires posture change. A stationary turn or a multi-chunk sit is not automatically stuck. Allow startup and partial completion. Separately judge progress toward the task: physical motion can occur without getting closer to the target.

Compare the previous action_intent.expected_effect against the paired images and actual motion. Similar images alone do not prove blockage, and a target leaving the image does not prove overshoot (turns, occlusion and sitting can cause it). State only suspected anomalies, cite concise evidence, and express low/medium/high confidence. Missing or conflicting evidence means unknown, not success. Target position and distance are visual estimates; no ground-truth target geometry is supplied. Keep historical target location explicitly marked as history rather than a current observation.

The previous_state_estimate and recent history are fallible memory. Assess only action step current_step-1; never attribute a verifier-rejected proposal to the executed body. last_recovery identifies the latest actual recovery attempt, even when older than the history window. Judge whether its intended problem improved, not merely whether it caused movement; keep pending/unknown when more observation is needed. If no attempt exists, use no_attempt. After repeated ineffective actions, choose a different evidence-supported adjustment or view instead of repeating them without a new reason. Set action_intent.recovery_for only when the next chosen action is intended to address stuck, overshoot or target_lost; ordinary turns and side steps need not be recoveries.

Return concise state evidence, then plan using the current image, body state and assessed progress. Do not invent contact, exact target distance, or confirmed task completion from proprioception alone."""

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

Current ego image and, when available, the previous decision image are attached and labeled. Feedback may be unavailable; mark unsupported judgments unknown.

Input:
{input_json}

Return this JSON structure. For each pipe-separated vocabulary choose ONE value, not the entire string. recovery.action_step is an integer or null. Evidence must be short (at most 240 characters each); expected_effect at most 200 characters. On the first step progress is unknown.

Return exactly this JSON:
{{
  "visible_state": "From left to right, I can see <object/area>, about <distance> meters from me; ...; and <object/area>, about <distance> meters from me. <Say whether the target object/place is visible. Say whether the near-body lane is clear or blocked.>",
  "state_estimate": {{
    "progress": {{"motion": "met|partial|not_met|unknown", "goal": "advanced|unchanged|regressed|unknown", "evidence": "<short evidence>", "confidence": "low|medium|high"}},
    "anomalies": {{
      "stuck": {{"status": "suspected|not_observed|unknown", "evidence": "<short evidence>", "confidence": "low|medium|high"}},
      "overshoot": {{"status": "suspected|not_observed|unknown", "evidence": "<short evidence>", "confidence": "low|medium|high"}},
      "target_lost": {{"status": "suspected|not_observed|unknown", "evidence": "<short evidence>", "confidence": "low|medium|high"}}
    }},
    "body_target_relation": {{"visibility": "visible|not_visible|unknown", "direction": "left|front|right|behind|unknown", "distance": "near|medium|far|unknown", "facing": "aligned|misaligned|unknown", "source": "current_view|history|unknown", "evidence": "<short evidence>", "confidence": "low|medium|high"}},
    "recovery": {{"action_step": <integer from last_recovery.action_step, or null>, "status": "no_attempt|pending|effective|ineffective|unknown", "evidence": "<short evidence>", "confidence": "low|medium|high"}}
  }},
  "mid_level_progress_analysis": "<Based on the high-level instruction, the visible_state, and the history, briefly analyze the current situation and what mid-level goal should be completed in the next 2-3 seconds. Decide whether to continue the previous mid-level goal or update it, and give a short reason.>",
  "mid_level_goal": "<One very concise sentence describing the mid-level goal for the next 2-3 seconds.>",
  "low_level_action_reasoning": "<Based on the mid_level_goal, visible_state, and history, briefly explain why this exact skill and parameters should be chosen for the next 0.5 seconds.>",
  "action_id": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7,
  "action_name": "<Walk<forward><speed> | Stop/Stand | Turn<direction><degree> | Climb upstairs<normal> | Sit down<target height> | Step back<distance> | Side step<direction><distance> | Walk downstairs<normal>>",
  "action_intent": {{"expected_effect": "<brief, observable expected result of this action>", "recovery_for": "none|stuck|overshoot|target_lost"}},
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

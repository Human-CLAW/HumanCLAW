"""Bounded proprioceptive evidence and uncertain, action-linked planner memory.

Only the humanoid's own realized pose enters this module. Target geometry,
semantic sensors, contact labels, and evaluation metrics are not inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .skills import SkillCall, skill_to_text
from .utils import clip_text

FEEDBACK_VERSION = "v1"
BODY_JOINT_NAMES = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
_ZUP_TO_YUP = Rotation.from_matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
_ISSUES = ("stuck", "overshoot", "target_lost")


def _mapping(value: Any) -> dict[str, Any]:
    """Accept an object without treating malformed model values as mappings."""

    return value if isinstance(value, dict) else {}


def _choice(value: Any, choices: tuple[str, ...], default: str = "unknown") -> str:
    """Normalize one small vocabulary, defaulting invalid values to unknown."""

    return value if isinstance(value, str) and value in choices else default


def _number(value: Any) -> float | None:
    """Round a finite measurement for prompts without emitting JSON NaNs."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return round(number, 3) if np.isfinite(number) else None


def _vector(value: Any, size: int = 3) -> list[float] | None:
    """Keep a small finite numeric vector, excluding arbitrary nested data."""

    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    result = [_number(item) for item in value]
    return result if all(item is not None for item in result) else None


def _motion_evidence(value: Any) -> dict[str, Any]:
    """Project motion evidence onto finite own-body measurements only."""

    value = _mapping(value)
    if value.get("available") is not True:
        return {"available": False}
    return {
        "available": True,
        **{
            key: _number(value.get(key))
            for key in (
                "forward_m",
                "left_m",
                "up_m",
                "planar_distance_m",
                "yaw_left_deg",
                "duration_s",
                "mean_planar_speed_m_s",
                "mean_joint_rotation_change_deg",
                "max_joint_rotation_change_deg",
            )
        },
    }


def _pose(value: Any) -> tuple[np.ndarray, Rotation, np.ndarray] | None:
    """Convert an explicitly framed own-body snapshot to Y-up SMPL-X."""

    value = _mapping(value)
    try:
        position = np.asarray(value.get("transl"), dtype=float)
        orient = np.asarray(value.get("global_orient"), dtype=float)
        joints = np.asarray(value.get("body_pose"), dtype=float)
        if position.shape != (3,) or orient.shape != (3,):
            return None
        if joints.ndim != 2 or joints.shape[1] != 3 or joints.shape[0] < 21:
            return None
        joints = joints[:21]
        if not all(np.isfinite(part).all() for part in (position, orient, joints)):
            return None
        rotation = Rotation.from_rotvec(orient)
        frame = value.get("coordinate_frame")
        if frame == "half_physics_z_up":
            position = _ZUP_TO_YUP.apply(position)
            rotation = _ZUP_TO_YUP * rotation
        elif frame != "smplx_y_up":
            return None
        # Joint rotations are parent-local, so the world-axis conversion must
        # never be applied to them.
        return position, rotation, joints
    except (TypeError, ValueError):
        return None


def _heading(rotation: Rotation) -> tuple[float, np.ndarray] | None:
    """Return left-positive heading and horizontal forward, if well defined."""

    forward = rotation.apply([0.0, 0.0, 1.0])
    forward[1] = 0.0
    length = np.linalg.norm(forward)
    if length < 1e-6:
        return None
    forward /= length
    return float(np.degrees(np.arctan2(forward[0], forward[2]))), forward


def summarize_body(current: Any, initial: Any) -> dict[str, Any]:
    """Describe current core joints and root pose using only proprioception."""

    pose, origin = _pose(current), _pose(initial)
    if pose is None:
        return {"available": False}
    position, rotation, joints = pose
    heading = _heading(rotation)
    up = rotation.apply([0.0, 1.0, 0.0])
    return {
        "available": True,
        "coordinate_frame": "SMPL-X Y-up; root position relative to episode reset",
        "position_from_reset_m": (
            [_number(x) for x in position - origin[0]] if origin is not None else None
        ),
        "root_orientation_rotvec_rad": [_number(x) for x in rotation.as_rotvec()],
        "heading_left_deg": _number(heading[0]) if heading is not None else None,
        "root_tilt_deg": _number(np.degrees(np.arccos(np.clip(up[1], -1, 1)))),
        "joint_rotvec_rad": {
            name: [_number(x) for x in row]
            for name, row in zip(BODY_JOINT_NAMES, joints)
        },
    }


def motion_delta(before: Any, after: Any, duration_s: float | None) -> dict[str, Any]:
    """Measure actual motion in the starting body's floor-projected frame."""

    start, end = _pose(before), _pose(after)
    if start is None or end is None:
        return {"available": False}
    delta = end[0] - start[0]
    heading_before, heading_after = _heading(start[1]), _heading(end[1])
    forward_m = left_m = yaw = None
    if heading_before is not None:
        forward = heading_before[1]
        left = np.cross([0.0, 1.0, 0.0], forward)
        forward_m, left_m = _number(delta @ forward), _number(delta @ left)
        if heading_after is not None:
            yaw = _number((heading_after[0] - heading_before[0] + 180) % 360 - 180)
    planar = float(np.linalg.norm(delta[[0, 2]]))
    joint_change = np.degrees(
        (
            Rotation.from_rotvec(end[2]) * Rotation.from_rotvec(start[2]).inv()
        ).magnitude()
    )
    duration = _number(duration_s)
    return {
        "available": True,
        "forward_m": forward_m,
        "left_m": left_m,
        "up_m": _number(delta[1]),
        "planar_distance_m": _number(planar),
        "yaw_left_deg": yaw,
        "duration_s": duration,
        "mean_planar_speed_m_s": (
            _number(planar / duration_s)
            if duration_s is not None and duration is not None and duration > 0
            else None
        ),
        "mean_joint_rotation_change_deg": _number(joint_change.mean()),
        "max_joint_rotation_change_deg": _number(joint_change.max()),
    }


def _generated_snapshot(frame: np.ndarray) -> dict[str, Any]:
    """Extract a generated Y-up pose without importing the motion runtime."""

    return {
        "coordinate_frame": "smplx_y_up",
        "transl": frame[:3],
        "global_orient": frame[3:6],
        "body_pose": frame[6:69].reshape(21, 3),
    }


def requested_motion(action: SkillCall) -> dict[str, Any]:
    """State command parameters as requests, never as achieved displacement."""

    cond = action.cond
    try:
        if action.skill == "walk_forward":
            return {"forward_m": _number(cond[1])}
        if action.skill == "step_back":
            return {"forward_m": _number(cond[1])}
        if action.skill == "side_walk":
            return {"left_m": _number(cond)}
        if action.skill == "turn":
            return {"yaw_left_deg": _number(cond)}
        if action.skill in {"step_climb_up", "step_climb_down"}:
            sign = 1 if action.skill == "step_climb_up" else -1
            return {
                "up_m": _number(sign * float(cond[0])),
                "forward_m": _number(cond[1]),
            }
        if action.skill == "sit":
            return {"seat_height_m": _number(cond), "posture": "sit"}
        if action.skill == "stand":
            return {"terminal_stop": True}
    except (TypeError, ValueError, IndexError):
        pass
    return {}


def normalize_action_intent(value: Any, action: SkillCall) -> dict[str, Any]:
    """Bind bounded semantic intent and deterministic expectations to an action."""

    value = _mapping(value)
    effect = value.get("expected_effect")
    return {
        "expected_effect": clip_text(effect, 200)
        if isinstance(effect, str) and effect.strip()
        else "unknown",
        "recovery_for": _choice(value.get("recovery_for"), _ISSUES, "none"),
        "requested_motion": requested_motion(action),
    }


def build_transition(
    *,
    step: int,
    action: SkillCall,
    action_intent: dict[str, Any],
    before: Any,
    after: Any,
    generated_motion: Any,
    fps: float,
) -> dict[str, Any]:
    """Join executed intent, generated motion, and realized motion by action step."""

    generated = {"available": False}
    duration = 0.0 if action.skill == "stand" else None
    if generated_motion is not None:
        try:
            frames = np.asarray(generated_motion, dtype=float)
            if frames.ndim == 2 and frames.shape[1] == 75 and len(frames) and fps > 0:
                duration = len(frames) / float(fps)
                generated = motion_delta(
                    _generated_snapshot(frames[0]),
                    _generated_snapshot(frames[-1]),
                    duration,
                )
        except (TypeError, ValueError):
            pass
    return {
        "action_step": step,
        "executed_action": action.to_json(),
        "action_intent": normalize_action_intent(action_intent, action),
        "generated_motion": generated,
        "actual_motion": motion_delta(before, after, duration),
    }


def step_feedback(
    step: int, current: Any, initial: Any, transition: Any
) -> dict[str, Any]:
    """Create the one current feedback item from explicit own-state inputs."""

    previous = _mapping(transition)
    return {
        "version": FEEDBACK_VERSION,
        "observation_step": step,
        "body": summarize_body(current, initial),
        "previous_transition": previous
        if previous.get("action_step") == step - 1
        else None,
    }


def current_feedback(
    values: Any, step: int, *, include_joints: bool = True
) -> dict[str, Any]:
    """Select aligned feedback and project it onto the documented evidence fields."""

    value = _mapping(values[-1]) if isinstance(values, list) and values else {}
    if (
        value.get("observation_step") != step
        or value.get("version") != FEEDBACK_VERSION
    ):
        return {
            "version": FEEDBACK_VERSION,
            "observation_step": step,
            "body": {"available": False},
            "previous_transition": None,
        }
    body = _mapping(value.get("body"))
    selected_body: dict[str, Any] = {"available": body.get("available") is True}
    if selected_body["available"]:
        selected_body.update(
            {
                "coordinate_frame": "SMPL-X Y-up; root position relative to episode reset",
                "position_from_reset_m": _vector(body.get("position_from_reset_m")),
                "root_orientation_rotvec_rad": _vector(
                    body.get("root_orientation_rotvec_rad")
                ),
                "heading_left_deg": _number(body.get("heading_left_deg")),
                "root_tilt_deg": _number(body.get("root_tilt_deg")),
            }
        )
    if include_joints and isinstance(body.get("joint_rotvec_rad"), dict):
        selected_body["joint_rotvec_rad"] = {
            name: _vector(body["joint_rotvec_rad"][name])
            for name in BODY_JOINT_NAMES
            if name in body["joint_rotvec_rad"]
        }
    transition = _mapping(value.get("previous_transition"))
    selected_transition = None
    if step > 0 and transition.get("action_step") == step - 1:
        executed = _mapping(transition.get("executed_action"))
        condition = executed.get("cond")
        if isinstance(condition, (list, tuple)):
            condition = (
                _vector(condition, len(condition)) if len(condition) <= 3 else None
            )
        else:
            condition = _number(condition)
        action = SkillCall(
            skill=clip_text(executed.get("skill"), 32),
            cond=condition,
            action_name=clip_text(executed.get("action_name"), 80),
        )
        selected_transition = {
            "action_step": step - 1,
            "executed_action": {
                key: action.to_json()[key] for key in ("skill", "cond", "action_name")
            },
            "action_intent": normalize_action_intent(
                transition.get("action_intent"), action
            ),
            "generated_motion": _motion_evidence(transition.get("generated_motion")),
            "actual_motion": _motion_evidence(transition.get("actual_motion")),
        }
    return {
        "version": FEEDBACK_VERSION,
        "observation_step": step,
        "body": selected_body or {"available": False},
        "previous_transition": selected_transition,
    }


def _judgment(value: Any) -> dict[str, Any]:
    """Keep short evidence and categorical, explicitly uncalibrated confidence."""

    value = _mapping(value)
    evidence = value.get("evidence")
    return {
        "evidence": clip_text(evidence, 240)
        if isinstance(evidence, str) and evidence.strip()
        else "unknown",
        "confidence": _choice(
            value.get("confidence"), ("low", "medium", "high"), "low"
        ),
    }


def _evidenced(value: Any) -> dict[str, Any]:
    """Treat missing or malformed supporting evidence as an unknown estimate."""

    value = _mapping(value)
    evidence = value.get("evidence")
    return (
        value
        if isinstance(evidence, str)
        and evidence.strip().lower() not in {"", "unknown", "none", "n/a"}
        else {}
    )


def normalize_state_estimate(
    value: Any, *, step: int, previous: Any = None, recovery_attempt: Any = None
) -> dict[str, Any]:
    """Normalize model estimates without promoting stale memory to observation."""

    value, previous = _mapping(value), _mapping(previous)
    progress = _evidenced(value.get("progress")) if step > 0 else {}
    result: dict[str, Any] = {
        "observation_step": step,
        "evaluated_action_step": step - 1 if step > 0 else None,
        "progress": {
            "motion": _choice(
                progress.get("motion"), ("met", "partial", "not_met", "unknown")
            ),
            "goal": _choice(
                progress.get("goal"), ("advanced", "unchanged", "regressed", "unknown")
            ),
            **_judgment(progress),
        },
        "anomalies": {},
    }
    for issue in _ISSUES:
        item = _evidenced(_mapping(value.get("anomalies")).get(issue))
        result["anomalies"][issue] = {
            "status": _choice(
                item.get("status"), ("suspected", "not_observed", "unknown")
            ),
            **_judgment(item),
        }
    relation = _evidenced(value.get("body_target_relation"))
    target = {
        "visibility": _choice(
            relation.get("visibility"), ("visible", "not_visible", "unknown")
        ),
        "direction": _choice(
            relation.get("direction"), ("left", "front", "right", "behind", "unknown")
        ),
        "distance": _choice(
            relation.get("distance"), ("near", "medium", "far", "unknown")
        ),
        "facing": _choice(relation.get("facing"), ("aligned", "misaligned", "unknown")),
        "source": _choice(
            relation.get("source"), ("current_view", "history", "unknown")
        ),
        **_judgment(relation),
    }
    last_seen = _mapping(previous.get("body_target_relation")).get("last_seen")
    if target["visibility"] == "visible" and target["source"] == "current_view":
        last_seen = {
            "step": step,
            **{
                key: target[key]
                for key in ("direction", "distance", "facing", "confidence")
            },
        }
    elif target["visibility"] == "visible":
        target["visibility"] = "unknown"
    target["last_seen"] = last_seen if isinstance(last_seen, dict) else None
    result["body_target_relation"] = target

    attempt = _mapping(recovery_attempt)
    recovery = _mapping(value.get("recovery"))
    if not attempt:
        result["recovery"] = {
            "action_step": None,
            "status": "no_attempt",
            **_judgment({}),
        }
    else:
        result["recovery"] = dict(attempt)
        if (
            recovery.get("action_step") == attempt.get("action_step")
            and attempt["action_step"] < step
        ):
            supported = _evidenced(recovery)
            result["recovery"].update(
                {
                    "status": _choice(
                        supported.get("status"),
                        ("pending", "effective", "ineffective", "unknown"),
                    ),
                    **_judgment(supported),
                }
            )
    return result


def recovery_record(
    step: int, action: SkillCall, intent: dict[str, Any]
) -> dict[str, Any] | None:
    """Start a recovery attempt only for the action that will actually execute."""

    if intent.get("recovery_for") not in _ISSUES:
        return None
    return {
        "action_step": step,
        "action_name": skill_to_text(action),
        "issue": intent["recovery_for"],
        "expected_effect": intent["expected_effect"],
        "status": "pending",
        **_judgment({}),
    }

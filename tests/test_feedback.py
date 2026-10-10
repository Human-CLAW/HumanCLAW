"""Numerical evidence and uncertain-memory regressions without Habitat or a VLM."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanclaw_bench.agent.feedback import (
    BODY_JOINT_NAMES,
    build_transition,
    current_feedback,
    motion_delta,
    normalize_action_intent,
    normalize_state_estimate,
    recovery_record,
    step_feedback,
    summarize_body,
)
from humanclaw_bench.agent.skills import SkillCall
from humanclaw_bench.envs.half_physics_env import (
    HalfPhysicsEnv,
    _proprioception_snapshot,
    xb75_yup_to_half_physics_pose,
)


def pose(position=(0, 1, 0), yaw=0, *, tilt=0, joints=None):
    return {
        "coordinate_frame": "smplx_y_up",
        "transl": np.array(position, dtype=float),
        "global_orient": Rotation.from_euler(
            "yx", [yaw, tilt], degrees=True
        ).as_rotvec(),
        "body_pose": np.zeros((54, 3)) if joints is None else joints,
    }


@pytest.mark.parametrize(
    "position, expected",
    [
        ((0, 1, 0.4), (0.4, 0, 0)),
        ((0, 1, -0.2), (-0.2, 0, 0)),
        ((0.3, 1, 0), (0, 0.3, 0)),
        ((-0.3, 1, 0), (0, -0.3, 0)),
        ((0, 1.2, 0.4), (0.4, 0, 0.2)),
        ((0, 0.8, 0.4), (0.4, 0, -0.2)),
    ],
)
def test_directional_motion_and_stairs(position, expected):
    measured = motion_delta(pose(), pose(position), 0.5)
    assert tuple(measured[k] for k in ("forward_m", "left_m", "up_m")) == pytest.approx(
        expected
    )
    assert measured["mean_planar_speed_m_s"] == pytest.approx(
        np.hypot(*expected[:2]) / 0.5, abs=0.001
    )


def test_motion_axes_follow_starting_body_not_world():
    measured = motion_delta(pose(yaw=90), pose((0.4, 1, 0), yaw=90), 0.5)
    assert measured["forward_m"] == 0.4
    assert measured["left_m"] == 0


@pytest.mark.parametrize(
    "start,end,expected", [(0, 40, 40), (0, -40, -40), (179, -179, 2), (-179, 179, -2)]
)
def test_turns_and_wrapped_heading(start, end, expected):
    measured = motion_delta(pose(yaw=start), pose(yaw=end), 0.5)
    assert measured["yaw_left_deg"] == expected
    assert measured["planar_distance_m"] == 0


def test_generated_to_physics_conversion_preserves_local_joint_rotations():
    xb = np.zeros(75)
    xb[:3] = [2, 1, 3]
    xb[3:6] = Rotation.from_euler("y", 35, degrees=True).as_rotvec()
    xb[6:9] = [0.1, 0.2, 0.3]
    z_pose = _proprioception_snapshot(*xb75_yup_to_half_physics_pose(xb))
    body = summarize_body(z_pose, z_pose)
    assert body["heading_left_deg"] == 35
    assert body["root_tilt_deg"] == 0
    assert body["position_from_reset_m"] == [0, 0, 0]
    assert body["joint_rotvec_rad"]["left_hip"] == [0.1, 0.2, 0.3]
    next_xb = xb.copy()
    next_xb[:3] += Rotation.from_euler("y", 35, degrees=True).apply([0, 0, 0.4])
    measured = motion_delta(
        z_pose, _proprioception_snapshot(*xb75_yup_to_half_physics_pose(next_xb)), 0.5
    )
    assert measured["forward_m"] == 0.4
    assert measured["left_m"] == 0
    assert measured["up_m"] == 0


def test_sitting_records_posture_change_without_requiring_horizontal_motion():
    joints = np.zeros((54, 3))
    joints[0, 0] = np.pi / 2
    measured = motion_delta(pose(), pose((0, 0.6, 0), joints=joints), 0.5)
    assert measured["up_m"] == -0.4
    assert measured["planar_distance_m"] == 0
    assert measured["max_joint_rotation_change_deg"] == 90
    assert "stuck" not in measured
    command = normalize_action_intent({}, SkillCall("sit", 0.45))
    assert command["requested_motion"] == {"seat_height_m": 0.45, "posture": "sit"}


def test_generated_displacement_is_not_reported_as_realized_progress():
    frames = np.zeros((15, 75))
    frames[:, 1] = 1
    frames[:, 2] = np.linspace(0, 0.32, 15)
    action = SkillCall("walk_forward", [0, 0.4, 0])
    result = build_transition(
        step=3,
        action=action,
        action_intent={},
        before=pose(),
        after=pose(),
        generated_motion=frames,
        fps=30,
    )
    assert result["action_intent"]["requested_motion"]["forward_m"] == 0.4
    assert result["generated_motion"]["forward_m"] == 0.32
    assert result["actual_motion"]["forward_m"] == 0
    assert result["actual_motion"]["duration_s"] == 0.5
    assert result["action_step"] == 3


def test_body_inputs_exclude_fingers_scene_geometry_and_metric_labels():
    current = pose(joints=np.ones((54, 3)))
    current.update(target_aabb="secret", object_states="secret", metric_frames="secret")
    feedback = step_feedback(0, current, pose(), None)
    serialized = json.dumps(feedback, allow_nan=False)
    assert "secret" not in serialized
    assert list(feedback["body"]["joint_rotvec_rad"]) == list(BODY_JOINT_NAMES)
    assert len(feedback["body"]["joint_rotvec_rad"]) == 21
    assert (
        "joint_rotvec_rad"
        not in current_feedback([feedback], 0, include_joints=False)["body"]
    )


@pytest.mark.parametrize(
    "bad", [None, {}, {"transl": [0, 0, 0]}, "bad", pose((np.nan, 0, 0))]
)
def test_missing_or_nonfinite_proprioception_is_unknown(bad):
    assert summarize_body(bad, pose()) == {"available": False}
    assert motion_delta(bad, pose(), 0.5) == {"available": False}


def test_vertical_body_has_no_fabricated_heading():
    body = summarize_body(pose(tilt=90), pose())
    assert body["heading_left_deg"] is None
    assert body["root_tilt_deg"] == 90
    measured = motion_delta(pose(tilt=90), pose((0, 1, 0.2), tilt=90), 0.5)
    assert measured["forward_m"] is None
    assert measured["planar_distance_m"] == 0.2


def test_stale_feedback_is_not_used_for_current_action():
    old = step_feedback(3, pose(), pose(), {"action_step": 0})
    assert old["previous_transition"] is None
    assert current_feedback([old], 4)["body"] == {"available": False}


def test_transition_projection_excludes_nested_oracle_fields():
    action = SkillCall("walk_forward", [0, 0.4, 0], action_name="Walk<forward><normal>")
    transition = build_transition(
        step=0,
        action=action,
        action_intent={},
        before=pose(),
        after=pose(),
        generated_motion=None,
        fps=30,
    )
    assert transition["actual_motion"]["duration_s"] is None
    for key in (
        "executed_action",
        "action_intent",
        "generated_motion",
        "actual_motion",
    ):
        transition[key]["target_truth"] = "PRIVATE_SENTINEL"
    value = step_feedback(1, pose(), pose(), transition)
    serialized = json.dumps(current_feedback([value], 1), allow_nan=False)
    assert "PRIVATE_SENTINEL" not in serialized


def test_success_without_evidence_is_normalized_to_unknown():
    state = normalize_state_estimate(
        {
            "progress": {
                "motion": "met",
                "goal": "advanced",
                "confidence": "high",
                "evidence": "unknown",
            }
        },
        step=1,
    )
    assert state["progress"]["motion"] == "unknown"
    assert state["progress"]["confidence"] == "low"


def test_target_loss_does_not_automatically_become_overshoot_or_stuck():
    state = normalize_state_estimate(
        {
            "progress": {
                "motion": "met",
                "goal": "unknown",
                "evidence": "heading changed as requested",
                "confidence": "high",
            },
            "anomalies": {
                "target_lost": {
                    "status": "suspected",
                    "evidence": "target left the frame during the turn",
                    "confidence": "low",
                }
            },
            "body_target_relation": {
                "visibility": "not_visible",
                "direction": "behind",
                "source": "history",
                "evidence": "last seen before turning",
                "confidence": "low",
            },
        },
        step=1,
    )
    assert state["progress"]["motion"] == "met"
    assert state["progress"]["goal"] == "unknown"
    assert state["anomalies"]["target_lost"]["status"] == "suspected"
    assert state["anomalies"]["stuck"]["status"] == "unknown"
    assert state["anomalies"]["overshoot"]["status"] == "unknown"


def test_physical_motion_can_succeed_while_task_progress_regresses():
    state = normalize_state_estimate(
        {
            "progress": {
                "motion": "met",
                "goal": "regressed",
                "evidence": "body moved forward but passed the visible target",
                "confidence": "medium",
            },
            "anomalies": {
                "overshoot": {
                    "status": "suspected",
                    "evidence": "target moved from beside the feet to behind",
                    "confidence": "medium",
                }
            },
        },
        step=2,
    )
    assert state["progress"]["motion"] == "met"
    assert state["progress"]["goal"] == "regressed"
    assert state["anomalies"]["overshoot"]["status"] == "suspected"


def test_state_fields_are_bounded_and_time_links_are_assigned_by_runtime():
    raw = {
        "evaluated_action_step": 999,
        "progress": {
            "motion": "met",
            "goal": "advanced",
            "evidence": "x" * 900,
            "confidence": "high",
        },
        "anomalies": {"stuck": {"status": ["bad"]}},
        "body_target_relation": "bad",
        "recovery": {"action_step": 0, "status": "effective"},
    }
    state = normalize_state_estimate(raw, step=2)
    assert state["evaluated_action_step"] == 1
    assert len(state["progress"]["evidence"]) <= 240
    assert state["anomalies"]["stuck"]["status"] == "unknown"
    assert state["body_target_relation"]["visibility"] == "unknown"
    assert state["recovery"]["status"] == "no_attempt"
    assert normalize_state_estimate(raw, step=0)["progress"]["motion"] == "unknown"


def test_target_memory_survives_missing_estimates_without_becoming_current_evidence():
    initial = normalize_state_estimate(
        {
            "body_target_relation": {
                "visibility": "visible",
                "direction": "left",
                "distance": "near",
                "source": "current_view",
                "evidence": "chair left",
                "confidence": "medium",
            }
        },
        step=0,
    )
    last = initial
    for step in range(1, 14):
        last = normalize_state_estimate({}, step=step, previous=last)
    assert last["body_target_relation"]["visibility"] == "unknown"
    assert last["body_target_relation"]["last_seen"]["step"] == 0
    assert last["body_target_relation"]["last_seen"]["direction"] == "left"


@pytest.mark.parametrize("status", ["pending", "effective", "ineffective", "unknown"])
def test_recovery_outcome_must_reference_an_executed_attempt(status):
    action = SkillCall("side_walk", 0.25, action_name="Side step<left><0.25>")
    intent = normalize_action_intent(
        {"recovery_for": "stuck", "expected_effect": "clear the obstacle"}, action
    )
    attempt = recovery_record(4, action, intent)
    state = normalize_state_estimate(
        {
            "recovery": {
                "action_step": 4,
                "status": status,
                "evidence": "side lane now clear",
                "confidence": "medium",
            }
        },
        step=5,
        recovery_attempt=attempt,
    )
    assert state["recovery"]["status"] == status
    assert state["recovery"]["action_step"] == 4
    wrong = normalize_state_estimate(
        {"recovery": {"action_step": 3, "status": "effective"}},
        step=5,
        recovery_attempt=attempt,
    )
    assert wrong["recovery"]["status"] == "pending"


def test_environment_observation_uses_realized_reset_and_final_motion_pose(monkeypatch):
    env = HalfPhysicsEnv(build_runtime=False)
    reset_pose = (np.array([1.0, 2.0, 3.0]), np.zeros(3), np.zeros((54, 3)))
    monkeypatch.setattr(
        env,
        "_reset_runtime_pose",
        lambda **_kwargs: {"ego_rgb": np.zeros((2, 2, 4), dtype=np.uint8)},
    )
    monkeypatch.setattr(env, "_read_agent_state", lambda: reset_pose)
    observation = env.reset(initial_transl=[9, 9, 9])
    assert observation.proprioception["transl"].tolist() == [1, 2, 3]
    reset_pose[0][0] = 99
    assert observation.proprioception["transl"][0] == 1
    body = {
        "transl": np.array([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]]),
        "global_orient": np.zeros((2, 3)),
        "body_pose": np.zeros((2, 54, 3)),
    }
    monkeypatch.setattr(
        env,
        "_step_runtime_motion",
        lambda **_kwargs: {
            "ego_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "body_state": body,
            "object_states": {},
        },
    )
    after, _, _, _ = env.step(np.zeros((2, 75)))
    assert after.proprioception["transl"].tolist() == [3, 2, 1]

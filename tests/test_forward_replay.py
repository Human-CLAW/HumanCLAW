"""Pure-Python regressions for the manual forward-replay entry point."""

from types import SimpleNamespace

import numpy as np

import runtime_forward_replay as forward_replay
from humanclaw_bench.envs.half_physics_env import (
    xb75_yup_to_half_physics_pose,
)


def _before_state() -> dict[str, np.ndarray]:
    """Return one nonzero saved state whose poses require reassignment."""

    return {
        "initial_xb_world_75": np.zeros(0, dtype=np.float32),
        "initial_human_transl": np.ones(3, dtype=np.float32),
        "initial_human_global_orient": np.zeros(3, dtype=np.float32),
        "initial_human_body_pose": np.ones((54, 3), dtype=np.float32),
        "initial_human_root_linear_velocity": np.asarray(
            [0.1, 0.2, 0.3], dtype=np.float32
        ),
        "initial_human_root_angular_velocity": np.asarray(
            [0.4, 0.5, 0.6], dtype=np.float32
        ),
        "initial_human_joint_velocities": np.asarray(
            [0.7, 0.8], dtype=np.float32
        ),
        "initial_object_names": np.asarray(["chair_:0000"]),
        "initial_object_ids": np.asarray([11], dtype=np.int32),
        "initial_object_position": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "initial_object_rotation": np.asarray(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "initial_object_linear_velocity": np.asarray(
            [[0.9, 1.0, 1.1]], dtype=np.float32
        ),
        "initial_object_angular_velocity": np.asarray(
            [[1.2, 1.3, 1.4]], dtype=np.float32
        ),
    }


def test_reset_human_pose_prefers_exact_motion_runner_input() -> None:
    """Avoid feeding a post-Habitat axis-angle round trip back into reset."""

    before = _before_state()
    initial_xb = np.linspace(-0.2, 0.2, 75, dtype=np.float32)
    before["initial_xb_world_75"] = initial_xb
    actual = forward_replay._reset_human_pose(before)
    expected = xb75_yup_to_half_physics_pose(initial_xb)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_reset_human_pose_keeps_old_archive_fallback() -> None:
    """Initial public archives without the 75-D seed remain readable."""

    before = _before_state()
    transl, orient, pose = forward_replay._reset_human_pose(before)
    np.testing.assert_array_equal(transl, before["initial_human_transl"])
    np.testing.assert_array_equal(orient, before["initial_human_global_orient"])
    np.testing.assert_array_equal(pose, before["initial_human_body_pose"])


def test_pose_reassignment_restores_saved_nonzero_velocities(monkeypatch) -> None:
    """Pose setters clear velocities, so replay must restore them afterwards."""

    before = _before_state()
    saved_human = {
        "transl": np.zeros(3, dtype=np.float32),
        "global_orient": np.zeros(3, dtype=np.float32),
        "body_pose": np.zeros((54, 3), dtype=np.float32),
        "root_linear_velocity": before["initial_human_root_linear_velocity"].copy(),
        "root_angular_velocity": before[
            "initial_human_root_angular_velocity"
        ].copy(),
        "joint_velocities": before["initial_human_joint_velocities"].copy(),
    }
    saved_object = {
        "position": np.zeros(3, dtype=np.float32),
        "rotation": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "linear_velocity": before["initial_object_linear_velocity"][0].copy(),
        "angular_velocity": before["initial_object_angular_velocity"][0].copy(),
    }
    agent = SimpleNamespace(
        root_linear_velocity=saved_human["root_linear_velocity"].copy(),
        root_angular_velocity=saved_human["root_angular_velocity"].copy(),
        joint_velocities=saved_human["joint_velocities"].copy().tolist(),
    )
    obj = SimpleNamespace(
        object_id=11,
        linear_velocity=saved_object["linear_velocity"].copy(),
        angular_velocity=saved_object["angular_velocity"].copy(),
    )

    class Magnum:
        """Minimal vector constructor used by the replay helper."""

        @staticmethod
        def Vector3(x: float, y: float, z: float) -> np.ndarray:
            return np.asarray([x, y, z], dtype=np.float64)

    env = SimpleNamespace(
        agent=agent,
        _require_runtime=lambda: SimpleNamespace(mn=Magnum),
        _tracked_dynamic_objects=lambda: {"chair_:0000": obj},
        replay_initial_state=lambda: {
            "human": saved_human,
            "objects": {"chair_:0000": saved_object},
        },
    )

    def clear_agent_velocities(*_args) -> None:
        agent.root_linear_velocity = np.zeros(3)
        agent.root_angular_velocity = np.zeros(3)
        agent.joint_velocities = [0.0, 0.0]

    def clear_object_velocities(*_args) -> bool:
        obj.linear_velocity = np.zeros(3)
        obj.angular_velocity = np.zeros(3)
        return True

    monkeypatch.setattr(forward_replay, "apply_agent_pose", clear_agent_velocities)
    monkeypatch.setattr(forward_replay, "apply_object_pose", clear_object_velocities)
    forward_replay._restore_initial_state(env, before)

    np.testing.assert_allclose(
        agent.root_linear_velocity, before["initial_human_root_linear_velocity"]
    )
    np.testing.assert_allclose(
        agent.root_angular_velocity, before["initial_human_root_angular_velocity"]
    )
    np.testing.assert_allclose(
        agent.joint_velocities, before["initial_human_joint_velocities"]
    )
    np.testing.assert_allclose(
        obj.linear_velocity, before["initial_object_linear_velocity"][0]
    )
    np.testing.assert_allclose(
        obj.angular_velocity, before["initial_object_angular_velocity"][0]
    )

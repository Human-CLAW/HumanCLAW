import hashlib
import json
from types import SimpleNamespace

import numpy as np

from humanclaw_bench.evaluation.trajectory import TrajectoryRecorder


def test_before_after_bundle_contains_replay_state_and_all_dynamic_objects(tmp_path):
    initial_objects = {
        "chair_:0000": {
            "object_id": 11,
            "motion_type": "DYNAMIC",
            "position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "linear_velocity": np.array([0.1, 0.0, 0.0], dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
        },
        "table_:0000": {
            "object_id": 12,
            "motion_type": "DYNAMIC",
            "position": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "linear_velocity": np.zeros(3, dtype=np.float32),
            "angular_velocity": np.array([0.0, 0.2, 0.0], dtype=np.float32),
        },
    }
    recorder = TrajectoryRecorder(
        metadata={
            "schema": "humanclaw_replay_v1",
            "episode": {"episode_id": "7"},
            "physics": {"fps": 30.0},
        },
        initial_xb_world_75=np.arange(75, dtype=np.float32),
        initial_state={
            "human": {
                "transl": np.zeros(3, dtype=np.float32),
                "global_orient": np.zeros(3, dtype=np.float32),
                "body_pose": np.zeros((54, 3), dtype=np.float32),
                "root_linear_velocity": np.zeros(3, dtype=np.float32),
                "root_angular_velocity": np.zeros(3, dtype=np.float32),
                "joint_velocities": np.zeros(324, dtype=np.float32),
            },
            "objects": initial_objects,
        },
    )
    action = SimpleNamespace(skill="walk_forward", cond=1.0)
    before_xb = np.arange(150, dtype=np.float32).reshape(2, 75)
    recorder.record_before(
        step=3,
        action=action,
        action_text="Walk<forward><1>",
        xb_world_75=before_xb,
    )
    recorder.record_after(
        step=3,
        body_state={
            "transl": np.zeros((2, 3), dtype=np.float32),
            "global_orient": np.zeros((2, 3), dtype=np.float32),
            "body_pose": np.zeros((2, 54, 3), dtype=np.float32),
        },
        object_states={
            name: {
                "position": np.repeat(row["position"][None, :], 2, axis=0),
                "rotation": np.repeat(row["rotation"][None, :], 2, axis=0),
            }
            for name, row in initial_objects.items()
        },
    )

    before_path, after_path, manifest_path = recorder.write(tmp_path)

    with np.load(before_path, allow_pickle=False) as before:
        np.testing.assert_array_equal(before["xb_world_75"], before_xb)
        assert "state_219" not in before
        assert "frame_step" not in before
        assert before["initial_object_names"].tolist() == [
            "chair_:0000",
            "table_:0000",
        ]
        assert before["initial_object_position"].shape == (2, 3)
        assert before["initial_object_linear_velocity"].shape == (2, 3)

    with np.load(after_path, allow_pickle=False) as after:
        np.testing.assert_array_equal(after["frame_step"], [3, 3])
        assert "step_action_skill" not in after
        assert after["object_names"].tolist() == [
            "chair_:0000",
            "table_:0000",
        ]
        assert after["object_000_position"].shape == (2, 3)
        assert after["object_001_rotation"].shape == (2, 4)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["initial_state"]["dynamic_object_count"] == 2
    assert (
        manifest["files"][before_path.name]["sha256"]
        == hashlib.sha256(before_path.read_bytes()).hexdigest()
    )
    assert (
        manifest["files"][after_path.name]["sha256"]
        == hashlib.sha256(after_path.read_bytes()).hexdigest()
    )
    assert not (tmp_path / "trajectory.npz").exists()

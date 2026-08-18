import json

import numpy as np
import pytest

from humanclaw_bench.main import _build_parser
from humanclaw_bench.rendering.batch import (
    discover_render_jobs,
    load_render_jobs,
)
from humanclaw_bench.rendering.saved_trajectory import (
    RenderContract,
    environment_kwargs,
    load_render_contract,
    load_saved_after_trajectory,
)


def _write_after(path, *, frames=3):
    np.savez_compressed(
        path,
        transl=np.zeros((frames, 3), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 54, 3), dtype=np.float32),
        frame_step=np.arange(frames, dtype=np.int32),
        fps=np.asarray(30.0, dtype=np.float32),
        object_names=np.asarray(["chair_:0000"]),
        object_000_position=np.zeros((frames, 3), dtype=np.float32),
        object_000_rotation=np.tile(
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            (frames, 1),
        ),
    )


def test_saved_after_loader_validates_all_frame_aligned_poses(tmp_path):
    path = tmp_path / "trajectory_after.npz"
    _write_after(path)
    trajectory = load_saved_after_trajectory(path, fallback_fps=24.0)
    assert trajectory.frame_count == 3
    assert trajectory.fps == 30.0
    assert set(trajectory.object_poses) == {"chair_:0000"}

    bad = tmp_path / "bad.npz"
    np.savez_compressed(
        bad,
        transl=np.zeros((2, 3), dtype=np.float32),
        global_orient=np.zeros((2, 3), dtype=np.float32),
        body_pose=np.zeros((1, 54, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="body_pose"):
        load_saved_after_trajectory(bad, fallback_fps=30.0)


def test_environment_kwargs_use_portable_scene_fallback(tmp_path):
    scene_config = tmp_path / "hssd-hab.scene_dataset_config.json"
    scene_config.write_text("{}")
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    scene = scenes / "scene_a.scene_instance.json"
    scene.write_text("{}")
    physics_config = tmp_path / "physics.json"
    urdf = tmp_path / "agent.urdf"
    shift = tmp_path / "shift.npy"
    for path in (physics_config, urdf, shift):
        path.write_bytes(b"x")

    contract = RenderContract(
        source=tmp_path / "replay_manifest.json",
        schema="humanclaw_replay_v1",
        profile={
            "benchmark": {"scene_dataset_config": str(scene_config)},
            "physics": {},
            "rendering": {},
        },
        episode={
            "scene_id": "/old/machine/scene_a.scene_instance.json",
            "scene_label": "scene_a",
        },
        physics={
            "backend": "hp",
            "physics_config": str(physics_config),
            "agent_urdf": str(urdf),
            "agent_shift_npy": str(shift),
            "fps": 30.0,
        },
        rendering={},
        assets={},
    )
    kwargs = environment_kwargs(contract)
    assert kwargs["scene_id"] == str(scene)
    assert kwargs["video_enabled"] is True
    assert kwargs["compute_metrics"] is False


def test_render_contract_loads_execution_flags_and_defaults_old_manifests(tmp_path):
    """New execution metadata is optional for initial public replay bundles."""

    rollout = tmp_path / "rollout"
    rollout.mkdir()
    manifest = {
        "schema": "humanclaw_replay_v1",
        "profile": {},
        "episode": {},
        "physics": {},
        "rendering": {},
        "assets": {},
        "execution": {"compute_metrics": True},
    }
    path = rollout / "replay_manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_render_contract(rollout).execution == {"compute_metrics": True}
    manifest.pop("execution")
    path.write_text(json.dumps(manifest))
    assert load_render_contract(rollout).execution == {}


def test_batch_discovery_preserves_rollout_tree(tmp_path):
    source = tmp_path / "source"
    rollout = source / "scene_ep0" / "ep_0_bed" / "rollout_00"
    rollout.mkdir(parents=True)
    _write_after(rollout / "trajectory_after.npz")
    output = tmp_path / "videos"
    jobs = discover_render_jobs(source, output)
    assert len(jobs) == 1
    assert jobs[0].rollout_dir == rollout
    assert jobs[0].output_dir == output / "scene_ep0/ep_0_bed/rollout_00"


def test_jsonl_manifest_supports_replacement_trajectory(tmp_path):
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    replacement = tmp_path / "modified.npz"
    _write_after(replacement)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode_key": "scene_ep0",
                "rollout_dir": "rollout",
                "trajectory_path": "modified.npz",
            }
        )
        + "\n"
    )
    jobs = load_render_jobs(manifest, tmp_path / "videos")
    assert jobs[0].rollout_dir == rollout
    assert jobs[0].trajectory_path == replacement
    assert jobs[0].output_dir == tmp_path / "videos/scene_ep0"


def test_render_cli_exposes_single_and_parallel_modes():
    parser = _build_parser()
    single = parser.parse_args(["render", "--rollout-dir", "episode"])
    assert single.preset == "veryfast"
    assert single.crf == 20
    batch = parser.parse_args(
        [
            "render-batch",
            "--input-root",
            "rollouts",
            "--output-root",
            "videos",
            "--max-parallel",
            "8",
            "--devices",
            "0,1",
        ]
    )
    assert batch.max_parallel == 8
    assert batch.devices == "0,1"
    compose = parser.parse_args(
        ["compose-video", "--rollout-dir", "episode/rollout_00"]
    )
    assert compose.output is None
    assert compose.threads == 2
    compose_batch = parser.parse_args(
        [
            "compose-video-batch",
            "--input-root",
            "rollouts",
            "--output-root",
            "composites",
            "--max-parallel",
            "8",
        ]
    )
    assert compose_batch.max_parallel == 8

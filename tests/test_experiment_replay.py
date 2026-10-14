"""Replay/rerun regressions that require neither GPU assets nor model calls."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode
from humanclaw_bench.config import load_config
from humanclaw_bench.experiments import physics_replay, replay
from humanclaw_bench.experiments.recorder import TraceRecorder, span


def _episode() -> HCFindNavInteractEpisode:
    """Provide a frozen episode whose instruction must survive unchanged."""
    return HCFindNavInteractEpisode(
        name="frozen", task_type="find_nav_interact", instruction="Saved instruction.",
        scene_id="/prepared/scenes/test.scene_instance.json", scene_label="test",
        scene_dataset_config="/prepared/hssd-hab.scene_dataset_config.json",
        episode_id="41", object_category="chair", object_label="chair",
        init_offset=(1.0, 2.0, 0.0), init_yaw=73.0, max_steps=17,
        goals=[], viewpoint_positions=[], goal_objects=[],
    )


def _run(tmp_path: Path) -> Path:
    """Create one real recorder attempt with frozen settings and a completed call."""
    profile = deepcopy(load_config("paper_fullval_v1").data)
    profile["agent"]["max_history"] = 3
    with TraceRecorder(tmp_path / "rollout", {
        "profile": profile, "episode": asdict(_episode()),
        "harness": "bare_api", "seed": 42, "rollout_index": 3,
        "effective_seed": 45, "compute_metrics": True, "save_video": False,
        "model_config": {"api_key": "do-not-reuse", "model": "old-model"},
        "config": {"device": "cpu"},
    }) as recorder:
        with span("model.respond", kind="model") as call:
            call.set_output({"text": "{}"})
        return recorder.directory


def _replay_files(directory: Path) -> Path:
    """Place the minimal replay filenames for dispatch tests."""
    output = directory / "replay"
    output.mkdir()
    (output / "replay_manifest.json").write_text(json.dumps({
        "schema": "humanclaw_replay_v1", "physics": {}, "rendering": {}, "assets": {},
    }))
    np.savez(output / "trajectory_before.npz", value=np.zeros(1))
    np.savez(output / "trajectory_after.npz", value=np.zeros(1))
    return output


def test_log_replay_is_read_only_and_tolerates_truncated_tail(tmp_path):
    """Interrupted JSONL is inspectable without modifying any source artifact."""
    directory = _run(tmp_path)
    events = directory / "events.jsonl"
    with events.open("a") as handle:
        handle.write('{"truncated":')
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result = replay.replay_run(directory)
    assert "model.respond" in result["timeline"]
    assert "Ignored truncated final record" in result["timeline"]
    assert result["validation"]["warnings"]
    assert before == {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_replay_export_rejects_original_tree_and_existing_output(tmp_path):
    """Replay export must never reuse a source or preexisting destination."""
    directory = _run(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        replay.replay_run(directory, output_dir=directory / "export")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        replay.replay_run(directory, output_dir=existing)
    result = replay.replay_run(directory, output_dir=tmp_path / "log-export")
    assert Path(result["output_dir"], "timeline.txt").is_file()


def test_physics_dispatch_uses_frozen_episode_and_records_mismatch(tmp_path, monkeypatch):
    """A trace's immutable snapshot and effective episode feed forward replay."""
    directory = _run(tmp_path)
    archive = _replay_files(directory)
    captured = {}

    def compare(path, **kwargs):
        """Capture dispatch without starting Habitat."""
        captured.update(path=path, **kwargs)
        return {"status": "mismatch", "passed": False, "replayed_steps": 2}

    monkeypatch.setattr(physics_replay, "replay_and_compare", compare)
    result = replay.replay_run(directory, mode="physics", max_steps=2)
    assert captured["path"] == archive
    assert captured["episode"].instruction == "Saved instruction."
    assert captured["max_steps"] == 2
    assert captured["fail_on_divergence"] is False
    assert result["status"] == "mismatch"
    assert json.loads(Path(result["output_dir"], "replay_report.json").read_text())["passed"] is False


def test_render_dispatch_restores_saved_poses(tmp_path, monkeypatch):
    """Render delegates to the existing renderer using a separate output tree."""
    from humanclaw_bench.rendering import saved_trajectory

    directory = _run(tmp_path)
    archive = _replay_files(directory)
    captured = {}

    def render(path, *, output_dir):
        """Capture the input snapshot and write no video in the CPU test."""
        captured.update(path=path, output=output_dir)
        return {"status": "complete", "physics_steps": 0, "vlm_calls": 0}

    monkeypatch.setattr(saved_trajectory, "render_saved_trajectory", render)
    result = replay.replay_run(directory, mode="render", output_dir=tmp_path / "render")
    assert captured["path"] == archive
    assert result["physics_steps"] == 0
    with pytest.raises(ValueError, match="only for physics"):
        replay.replay_run(directory, mode="render", max_steps=1)


def test_failed_physics_replay_keeps_failure_audit(tmp_path, monkeypatch):
    """A runtime failure remains visible and is not labeled a comparison pass."""
    directory = _run(tmp_path)
    _replay_files(directory)
    output = tmp_path / "failure"

    def unavailable(*args, **kwargs):
        """Model unavailable simulator inputs."""
        raise FileNotFoundError("scene missing")

    monkeypatch.setattr(physics_replay, "replay_and_compare", unavailable)
    with pytest.raises(FileNotFoundError, match="scene missing"):
        replay.replay_run(directory, mode="physics", output_dir=output)
    assert json.loads((output / "replay_report.json").read_text())["status"] == "failed"


def test_rerun_preserves_settings_uses_explicit_model_and_new_seed_index(tmp_path, monkeypatch):
    """A single rerun uses the saved effective seed, instruction and profile."""
    directory = _run(tmp_path)
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"backend": "test", "model": "new-model", "api_key": "not-logged"}))
    output = tmp_path / "rerun"
    captured = {}

    def execute(config):
        """Inspect constructed evaluator arguments without provider activity."""
        captured.update(config)
        return {"completed_rollouts": 1}

    monkeypatch.setattr(replay, "_execute_rerun", execute)
    result = replay.rerun_run(directory, model, output)
    assert captured["profile"].data["agent"]["max_history"] == 3
    assert captured["episode_override"].instruction == "Saved instruction."
    assert captured["max_steps"] == 17
    assert captured["seed"] == 45
    assert captured["n_rollouts"] == 1
    assert captured["harness"] == "bare_api"
    assert captured["compute_metrics"] is True
    assert captured["device"] == "cpu"
    assert captured["trace"] is True
    assert captured["model_config_path"] == str(model)
    assert captured["parent_run_id"] == json.loads((directory / "run.json").read_text())["run_id"]
    assert result["status"] == "completed"
    for path in output.rglob("*.json"):
        assert "not-logged" not in path.read_text()
        assert "do-not-reuse" not in path.read_text()
    with pytest.raises(FileExistsError):
        replay.rerun_run(directory, model, output)


def test_rerun_requires_model_and_rejects_original_subtree(tmp_path, monkeypatch):
    """No endpoint, source configuration, or destination is inferred silently."""
    directory = _run(tmp_path)
    model = tmp_path / "model.json"
    model.write_text("{}")
    with pytest.raises(ValueError, match="explicit"):
        replay.rerun_run(directory, "", tmp_path / "new")
    with pytest.raises(ValueError, match="outside"):
        replay.rerun_run(directory, model, directory / "rerun")
    assert not (directory / "rerun").exists()


def _physics_fixture(tmp_path: Path, monkeypatch, *, mismatch=False):
    """Provide two motion chunks and a deterministic fake physics environment."""
    before = {"step_starts": np.array([0, 1]), "step_lengths": np.array([1, 1]),
              "xb_world_75": np.zeros((2, 75)), "initial_human_transl": np.zeros(3),
              "initial_human_global_orient": np.zeros(3),
              "initial_human_body_pose": np.zeros((54, 3))}
    after = {"step_starts": np.array([0, 1]), "step_lengths": np.array([1, 1]),
             "transl": np.zeros((2, 3)), "global_orient": np.zeros((2, 3)),
             "body_pose": np.zeros((2, 54, 3)), "object_names": np.array([], dtype=str)}
    np.savez(tmp_path / "trajectory_before.npz", **before)
    np.savez(tmp_path / "trajectory_after.npz", **after)
    contract = SimpleNamespace(profile={"benchmark": {}}, episode={}, execution={
        "video_enabled": False, "compute_metrics": False, "pre_motion_metric_reset": False})
    monkeypatch.setattr(physics_replay, "load_render_contract", lambda _: contract)
    monkeypatch.setattr(physics_replay, "environment_kwargs", lambda _: {})
    monkeypatch.setattr(physics_replay, "_restore_initial_state", lambda *args: None)

    class Environment:
        """Return frame-aligned simulator outputs and record cleanup."""

        def __init__(self, **kwargs):
            """Start a fresh counted replay."""
            self.calls = 0
            self.closed = False

        def reset(self, *args, **kwargs):
            """Accept the frozen episode and exact initial pose."""

        def step(self, motion, **kwargs):
            """Return either matching or deliberately divergent translations."""
            self.calls += 1
            body = {"transl": np.full((len(motion), 3), 0.5 if mismatch else 0.0),
                    "global_orient": np.zeros((len(motion), 3)),
                    "body_pose": np.zeros((len(motion), 54, 3))}
            return None, 0, False, {"body_state": body, "object_states": {}}

        def close(self):
            """Record release of simulator resources."""
            self.closed = True

    env = Environment()
    monkeypatch.setattr(physics_replay, "HCFindNavInteractEnv", lambda **kwargs: env)
    return env


@pytest.mark.parametrize("mismatch", [False, True])
def test_physics_replays_chunks_compares_results_and_closes(tmp_path, monkeypatch, mismatch):
    """The promoted implementation executes input chunks and audits divergence."""
    env = _physics_fixture(tmp_path, monkeypatch, mismatch=mismatch)
    result = physics_replay.replay_and_compare(
        tmp_path, episode=_episode(), max_steps=1, fail_on_divergence=False,
    )
    assert result["passed"] is (not mismatch)
    assert result["replayed_steps"] == 1
    assert result["max_errors"]["human_translation_m"] == (0.5 if mismatch else 0)
    assert env.calls == 1 and env.closed


def test_physics_rejects_broken_boundaries_before_runtime(tmp_path, monkeypatch):
    """Invalid frame boundaries cannot produce an empty false-positive pass."""
    env = _physics_fixture(tmp_path, monkeypatch)
    path = tmp_path / "trajectory_before.npz"
    before = physics_replay._archive(path)
    before["step_starts"] = np.array([0, 0])
    np.savez(path, **before)
    with pytest.raises(RuntimeError, match="boundaries differ"):
        physics_replay.replay_and_compare(tmp_path, episode=_episode())
    assert env.calls == 0


def test_quaternion_comparison_accepts_sign_and_rejects_missing_values():
    """Equivalent quaternion signs match, but unavailable values cannot hide drift."""
    rotation = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert physics_replay._quaternion_error(rotation, -rotation) == 0.0
    with pytest.raises(RuntimeError, match="finite-value"):
        physics_replay._quaternion_error(np.full((1, 4), np.nan), rotation)

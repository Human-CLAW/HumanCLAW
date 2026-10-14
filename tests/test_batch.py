import copy
import json
from pathlib import Path

import pytest

from humanclaw_bench import batch
from humanclaw_bench.batch import _child_process_env
from humanclaw_bench.config import ReleaseConfig
from humanclaw_bench.experiments.recorder import TraceRecorder, span


def test_child_process_env_bounds_cpu_thread_pools() -> None:
    env = _child_process_env({"PATH": "/bin"})

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_child_process_env_preserves_explicit_overrides() -> None:
    env = _child_process_env(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "5",
        }
    )

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["NUMEXPR_NUM_THREADS"] == "5"


def _config(tmp_path, monkeypatch, *, episodes=None, trace=True):
    episodes = episodes or [{"scene_id": "scene", "episode_id": "7", "object_category": "chair"}]
    profile_data = {
        "profile": "fixture", "benchmark": {"dataset_dir": str(tmp_path / "dataset"), "split": "val", "max_steps": 100},
        "motion": {"verify_weights": False},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_data))
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"backend": "fixture", "model": "fixture-model", "api_key": "not-for-manifest"}))
    config = {
        "profile": ReleaseConfig(profile_path, profile_data),
        "model_config": str(model_path), "output_root": str(tmp_path / "output"),
        "max_parallel": 2, "devices": ("0", "1"), "trace": trace,
        "harness": "bare_api", "seed": 11, "max_steps": 4,
    }
    monkeypatch.setattr(batch, "list_episode_specs", lambda *args: copy.deepcopy(episodes))
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    return config, episodes


def _canonical_outputs(config, episode):
    rollout = Path(config["output_root"]) / batch._task_name(episode) / "rollout_00"
    rollout.mkdir(parents=True, exist_ok=True)
    for name in ("replay_manifest.json", "trajectory_before.npz", "trajectory_after.npz"):
        (rollout / name).write_bytes(b"fixture")
    if config.get("compute_metrics"):
        (rollout / "metrics.json").write_text("{}")
    if config.get("save_video"):
        for name in ("ego.mp4", "exo.mp4"):
            (rollout / name).write_bytes(b"fixture")
    return rollout


def _completed_fixture(config, episode, *, env=None, changes=None, success=True):
    expected = batch._trace_configuration(config, config["profile"], Path(config["model_config"]))
    metadata = {key: copy.deepcopy(value) for key, value in expected.items() if key != "max_steps"}
    metadata["episode"] = {**episode, "scene_label": episode["scene_id"], "max_steps": expected["max_steps"]}
    if env:
        metadata["batch_attempt_id"] = env["HUMANCLAW_BATCH_ATTEMPT_ID"]
        metadata["batch_child_id"] = env["HUMANCLAW_BATCH_CHILD_ID"]
    for dotted, value in (changes or {}).items():
        parts = dotted.split(".")
        target = metadata
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    with TraceRecorder(_canonical_outputs(config, episode), metadata, pricing=expected.get("pricing")) as recorder:
        with span("fixture.module", kind="module"):
            pass
        recorder.finish(outcome={"task_success": success})
    return recorder


class _ExitedProcess:
    pid = 123

    def __init__(self, code=0):
        self.code = code

    def poll(self):
        return self.code


def test_batch_forwards_experiment_flags_and_preserves_unique_attempt_membership(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch)
    prices = tmp_path / "pricing.json"
    prices.write_text('{"currency":"USD"}')
    config["pricing"] = str(prices)
    config["agent_asset"] = "finger_separated"
    launched = []

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        _completed_fixture(config, episodes[0], env=kwargs["env"])
        return _ExitedProcess()

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    first = batch.run_batch(config)
    first_path = Path(first["batch_manifest"])
    first_bytes = first_path.read_bytes()
    second = batch.run_batch(config)
    assert Path(second["batch_manifest"]) != first_path
    assert first_path.read_bytes() == first_bytes
    command, kwargs = launched[0]
    assert "--trace" in command
    for flag, value in (("--harness", "bare_api"), ("--seed", "11"), ("--max-steps", "4"), ("--pricing", str(prices)), ("--agent-asset", "finger_separated")):
        assert command[command.index(flag) + 1] == value
    manifest = json.loads(first_bytes)
    row = manifest["children"][0]
    assert kwargs["env"]["HUMANCLAW_BATCH_ATTEMPT_ID"] == manifest["batch_attempt_id"]
    assert kwargs["env"]["HUMANCLAW_BATCH_CHILD_ID"] == row["child_id"]
    assert row["status"] == "completed"
    assert Path(row["stderr_log"]).is_file()
    assert row["task_success"] is True
    assert len(row["trace_runs"]) == 1
    assert first["batch_outcomes"]["requested_episodes"] == 1
    assert second["experiments"]["aggregate"]["runs"] == 2
    assert second["batch_outcomes"]["scored_episodes"] == 1
    assert "previous batches" in second["experiments"]["scope"]
    capabilities = second["experiments"]["aggregate"]["capabilities"]
    entry = capabilities["entries"][0]
    assert entry["name"] == "fixture.module"
    assert entry["executed"] == entry["failure_rate_denominator"] == 2
    assert entry["failure_rate"] == 0
    assert entry["latency"]["closed_spans"] == 2
    saved = json.loads(Path(second["batch_manifest"]).read_text())
    assert saved["summary"]["experiments"]["aggregate"]["capabilities"] == capabilities
    assert "not-for-manifest" not in first_bytes.decode()


def test_traced_resume_rejects_old_artifacts_without_trace(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch)
    config["resume"] = True
    _canonical_outputs(config, episodes[0])
    launched = []
    monkeypatch.setattr(batch.subprocess, "Popen", lambda command, **kwargs: launched.append(command) or _ExitedProcess(6))
    result = batch.run_batch(config)
    assert len(launched) == 1
    assert result["skipped_complete"] == 0
    assert result["batch_outcomes"]["process_failures_without_trace"] == 1


@pytest.mark.parametrize("changes", [
    {"harness": "humanclaw_psv"}, {"seed": 99},
    {"model_config.model": "different-model"},
    {"effective_model_config.model": "different-effective-model"},
    {"source.files_sha256": "different-implementation"},
    {"profile.profile": "different-profile"}, {"episode.max_steps": 99},
    {"compute_metrics": True}, {"save_video": True},
    {"config.agent_asset": "different-body"},
])
def test_traced_resume_rejects_completed_mismatched_configuration(tmp_path, monkeypatch, changes):
    config, episodes = _config(tmp_path, monkeypatch)
    config["resume"] = True
    _completed_fixture(config, episodes[0], changes=changes)
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *args, **kwargs: _ExitedProcess(7))
    result = batch.run_batch(config)
    assert result["skipped_complete"] == 0
    assert result["launched"] == 1


def test_traced_resume_reuses_only_valid_matching_completed_trace(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch)
    config["resume"] = True
    prior = _completed_fixture(config, episodes[0], success=False)

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("Valid matching completed trace should be reused")

    monkeypatch.setattr(batch.subprocess, "Popen", unexpected_launch)
    result = batch.run_batch(config)
    manifest = json.loads(Path(result["batch_manifest"]).read_text())
    assert result["skipped_complete"] == 1
    assert result["launched"] == 0
    assert manifest["children"][0]["trace_runs"][0]["run_id"] == prior.run_id
    assert manifest["children"][0]["status"] == "skipped_complete"
    assert result["batch_outcomes"]["scored_episodes"] == 1
    assert result["batch_outcomes"]["task_successes"] == 0


def test_traced_resume_rejects_missing_event_endings(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch)
    config["resume"] = True
    prior = _completed_fixture(config, episodes[0])
    path = prior.directory / "events.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *args, **kwargs: _ExitedProcess(7))
    result = batch.run_batch(config)
    assert result["skipped_complete"] == 0


def test_traced_resume_rejects_changed_pricing_instead_of_reusing_old_cost(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch)
    config["resume"] = True
    prices = tmp_path / "pricing.json"
    prices.write_text('{"version":"one"}')
    config["pricing"] = str(prices)
    _completed_fixture(config, episodes[0])
    prices.write_text('{"version":"two"}')
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *args, **kwargs: _ExitedProcess(7))
    result = batch.run_batch(config)
    assert result["skipped_complete"] == 0


def test_startup_and_untraced_process_failures_remain_in_requested_denominator(tmp_path, monkeypatch):
    episodes = [
        {"scene_id": "scene", "episode_id": "1", "object_category": "chair"},
        {"scene_id": "scene", "episode_id": "2", "object_category": "bed"},
    ]
    config, _ = _config(tmp_path, monkeypatch, episodes=episodes)

    def launch(command, **kwargs):
        if command[command.index("--episode-id") + 1] == "1":
            raise OSError("process could not start")
        kwargs["stderr"].write(b"fixture startup error before recorder construction\n")
        kwargs["stderr"].flush()
        return _ExitedProcess(9)

    monkeypatch.setattr(batch.subprocess, "Popen", launch)
    result = batch.run_batch(config)
    manifest = json.loads(Path(result["batch_manifest"]).read_text())
    assert result["selected"] == 2
    assert result["launched"] == 1
    assert result["failed"] == 2
    assert [row["status"] for row in manifest["children"]] == ["startup_failed", "failed"]
    assert all(row["error"] for row in manifest["children"])
    assert "before recorder" in Path(manifest["children"][1]["stderr_log"]).read_text()
    assert result["batch_outcomes"]["requested_episodes"] == 2
    assert result["batch_outcomes"]["unscored_episodes"] == 2
    assert result["batch_outcomes"]["startup_failures"] == 1
    assert result["batch_outcomes"]["process_failures_without_trace"] == 1
    assert result["experiments"]["aggregate"]["runs"] == 0


def test_zero_exit_without_owned_trace_is_incomplete_not_success(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch)
    # An old, unrelated completed trace must not be attributed to the new child.
    _completed_fixture(config, episodes[0])
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *args, **kwargs: _ExitedProcess(0))
    result = batch.run_batch(config)
    manifest = json.loads(Path(result["batch_manifest"]).read_text())
    assert result["failed"] == 1 and result["completed"] == 0
    assert manifest["children"][0]["error"]["type"] == "IncompleteTrace"
    assert manifest["children"][0]["task_success"] is None


def test_invalid_model_configuration_still_preserves_requested_batch(tmp_path, monkeypatch):
    config, _ = _config(tmp_path, monkeypatch)
    Path(config["model_config"]).write_text("invalid json")
    with pytest.raises(ValueError, match="Cannot snapshot"):
        batch.run_batch(config)
    path = next((Path(config["output_root"]) / "batch_attempts").glob("*/batch.json"))
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["requested_count"] == 1
    assert manifest["children"][0]["status"] == "not_launched"
    assert manifest["children"][0]["error"]


def test_legacy_resume_keeps_artifact_only_behavior_without_batch_trace(tmp_path, monkeypatch):
    config, episodes = _config(tmp_path, monkeypatch, trace=False)
    config["resume"] = True
    _canonical_outputs(config, episodes[0])
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Unexpected launch"))
    result = batch.run_batch(config)
    assert result["skipped_complete"] == 1
    assert "batch_manifest" not in result
    assert not (Path(config["output_root"]) / "batch_attempts").exists()

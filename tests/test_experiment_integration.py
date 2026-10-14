"""Exercise real harness boundaries with deterministic fake providers and physics."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from humanclaw_bench.agent.bare_agent import BareAPIAgent
from humanclaw_bench.agent.ego_agent import PSVEgoAgent
from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode
from humanclaw_bench.config import load_config
from humanclaw_bench.evaluation.evaluator import HCFindNavInteractEvaluator
from humanclaw_bench.experiments.recorder import TraceRecorder
from humanclaw_bench.experiments.report import read_events, summarize_run, validate_trace


STOP = {"visible_state": "The target is next to the body.", "action_id": 1, "action_name": "Stop/Stand"}
WALK = {"visible_state": "Clear floor ahead.", "action_id": 0, "action_name": "Walk<forward><slow>"}
ACCEPT = {"verdict": "accept", "reason": "The current action is suitable."}
USAGE = {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
         "prompt_tokens_details": {"cached_tokens": 20},
         "completion_tokens_details": {"reasoning_tokens": 10}}


class Model:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []
        self.last_usage = {}
        self.model_name = "mock-model"

    def respond(self, messages):
        self.messages.append(messages)
        value, usage = self.replies.pop(0)
        self.last_usage = deepcopy(usage)
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value) if isinstance(value, dict) else value


class Env:
    fps = 30

    def __init__(self, failure=None):
        self.failure = failure
        self.actions = []
        self.closed = False
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)

    def step(self, action, reasoning=None):
        self.actions.append(action)
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(head_rgb=self.image), 0.0, True, {}

    def close(self):
        self.closed = True


class Motion:
    def __init__(self):
        self.calls = []
        self.unloads = 0

    def generate(self, skill, cond):
        self.calls.append((skill, cond))
        return SimpleNamespace(xb_world_75=np.zeros((15, 75), dtype=np.float32))

    def unload(self):
        self.unloads += 1


def _episode():
    return HCFindNavInteractEpisode(
        name="example", task_type="find_nav_interact", instruction="Find the bed.",
        scene_id="scene", scene_label="scene", scene_dataset_config="scene.json",
        episode_id="0", object_category="bed", object_label="bed",
        init_offset=(0.0, 0.0, 0.0), init_yaw=0.0, max_steps=2,
        goals=[], viewpoint_positions=[], goal_objects=[],
    )


def _evaluator(tmp_path, model, *, trace=True, bare=False, failure=None):
    evaluator = HCFindNavInteractEvaluator({
        "output_root": tmp_path, "profile": load_config(), "trace": trace,
        "harness": "bare_api" if bare else "humanclaw_psv",
    })
    evaluator.agent = BareAPIAgent(model) if bare else PSVEgoAgent(model)
    evaluator.env = Env(failure)
    evaluator.motion = Motion()
    evaluator._model_config = {"model": "mock-model", "api_key": "must-not-survive"}
    evaluator._initial_xb_world_75 = np.zeros(75, dtype=np.float32)
    evaluator._reset_env_for_rollout = lambda episode: SimpleNamespace(head_rgb=evaluator.env.image)
    return evaluator


def _run_dir(result):
    return Path(result["experiment_dir"])


def test_traced_stop_captures_retry_usage_real_hierarchy_and_unknown_success(tmp_path, monkeypatch):
    monkeypatch.setattr("humanclaw_bench.agent.planner.time.sleep", lambda seconds: None)
    model = Model([(RuntimeError("temporary provider error"), USAGE), (STOP, USAGE), (ACCEPT, USAGE)])
    evaluator = _evaluator(tmp_path, model)
    result = evaluator.run_rollout(_episode(), 0)
    directory = _run_dir(result)
    assert validate_trace(directory)["ok"]
    summary = summarize_run(directory)
    assert summary["status"] == "completed"
    assert summary["outcome"]["task_success"] is None
    assert summary["outcome"]["termination_reason"] == "stop"
    assert summary["model_usage"]["leaf_model_calls"] == 3
    assert summary["model_usage"]["status_counts"] == {"failed": 1, "completed": 2}
    assert summary["model_usage"]["tokens"]["input_tokens"]["total"] == 300
    assert summary["model_usage"]["tokens"]["output_tokens"]["total"] == 90
    assert summary["cost"]["total_usd"] is None
    events = read_events(directory)
    starts = {event["span_id"]: event for event in events if event["event"] == "span.start"}
    model_calls = [event for event in starts.values() if event["kind"] == "model"]
    assert [starts[event["parent_span_id"]]["name"] for event in model_calls] == ["percept_mid_low", "percept_mid_low", "verifier"]
    stage = starts[model_calls[0]["parent_span_id"]]
    assert starts[stage["parent_span_id"]]["name"] == "agent.act"
    tool_calls = [event for event in starts.values() if event["kind"] == "tool"]
    assert [event["name"] for event in tool_calls] == ["stand"]
    assert starts[tool_calls[0]["parent_span_id"]]["kind"] == "step"
    assert evaluator.motion.calls == []
    assert evaluator.env.closed
    assert evaluator.env.actions[0]["stop"] is True
    assert len(model.messages) == 3
    assert "must-not-survive" not in (directory / "run.json").read_text()
    assert all((directory / "replay" / name).is_file() for name in (
        "replay_manifest.json", "trajectory_before.npz", "trajectory_after.npz",
    ))


@pytest.mark.parametrize("failure,status", [(RuntimeError("physics failed"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_physics_failure_closes_spans_preserves_partial_replay_and_unknown_metrics(tmp_path, failure, status):
    model = Model([(WALK, USAGE), (ACCEPT, USAGE)])
    evaluator = _evaluator(tmp_path, model, failure=failure)
    with pytest.raises(type(failure)):
        evaluator.run_rollout(_episode(), 0)
    directories = list(tmp_path.rglob("experiment_runs/run_*"))
    assert len(directories) == 1
    directory = directories[0]
    summary = summarize_run(directory)
    assert summary["status"] == status
    assert summary["outcome"]["task_success"] is None
    assert summary["outcome"]["benchmark_metrics"] is None
    assert summary["latency"]["incomplete_spans"] == 0
    assert validate_trace(directory)["ok"]
    assert evaluator.env.closed
    assert len(evaluator.motion.calls) == 1
    assert (directory / "replay" / "trajectory_before.npz").is_file()
    assert any(event["event"] == "span.end" and event["kind"] == "environment"
               and event["data"]["status"] == status for event in read_events(directory))


def test_repeating_episode_retains_prior_trace_and_frozen_replay_profile(tmp_path):
    model = Model([(STOP, USAGE), (STOP, USAGE)])
    evaluator = _evaluator(tmp_path, model, bare=True)
    evaluator.profile.data["agent"]["max_history"] = 13
    first = _run_dir(evaluator.run_rollout(_episode(), 0))
    snapshot = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    frozen = json.loads((first / "replay" / "replay_manifest.json").read_text())
    assert frozen["profile"]["agent"]["max_history"] == 13
    evaluator.profile.data["agent"]["max_history"] = 99
    evaluator._initial_xb_world_75 = np.ones(75, dtype=np.float32)
    second = _run_dir(evaluator.run_rollout(_episode(), 0))
    assert first != second
    assert len(list(tmp_path.rglob("experiment_runs/run_*"))) == 2
    assert snapshot == {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    assert json.loads((second / "replay" / "replay_manifest.json").read_text())["profile"]["agent"]["max_history"] == 99


def test_disabled_trace_keeps_original_rollout_artifact_contract(tmp_path):
    model = Model([(STOP, USAGE), (ACCEPT, USAGE)])
    evaluator = _evaluator(tmp_path, model, trace=False)
    result = evaluator.run_rollout(_episode(), 0)
    assert "experiment_dir" not in result
    assert not list(tmp_path.rglob("events.jsonl"))
    assert not list(tmp_path.rglob("experiment_runs"))
    files = {path.name for path in Path(result["episode_dir"]).iterdir()}
    assert files == {"step000_percept_mid_low.json", "step000_verifier.json", "replay_manifest.json", "trajectory_before.npz", "trajectory_after.npz"}


def test_verifier_replacement_keeps_proposal_separate_from_executed_tool(tmp_path):
    replacement = {"verdict": "replace", "reason": "Forward lane blocked.",
                   "final_action_id": 2, "final_action_name": "Turn<right><45>"}
    model = Model([(WALK, USAGE), (replacement, USAGE)])
    evaluator = _evaluator(tmp_path, model)
    directory = _run_dir(evaluator.run_rollout(_episode(), 0))
    events = read_events(directory)
    requests = [event for event in events if event["event"] == "tool.requested"]
    assert [(event["name"], event["data"]["source"]) for event in requests] == [
        ("walk_forward", "planner"), ("turn", "verifier"),
    ]
    assert any(event["event"] == "tool.rejected" and event["name"] == "walk_forward" for event in events)
    executions = [event["name"] for event in events if event["event"] == "span.start" and event["kind"] == "tool"]
    assert executions == ["turn"]
    assert evaluator.motion.calls == [("turn", -45.0)]
    capabilities = {entry["name"]: entry for entry in summarize_run(directory)["capabilities"]["entries"]}
    assert capabilities["walk_forward"]["requested"] == 1
    assert capabilities["walk_forward"]["rejected"] == 1
    assert capabilities["walk_forward"]["executed"] == 0
    assert capabilities["turn"]["executed"] == 1


@pytest.mark.parametrize("action_id", [None, True, "1", -1, 8])
def test_bare_rejects_invalid_action_id_without_verifier_or_execution(tmp_path, action_id):
    model = Model([({**STOP, "action_id": action_id}, USAGE)])
    agent = BareAPIAgent(model)
    agent.reset(_episode())
    with TraceRecorder(tmp_path, {"harness": "bare_api"}) as recorder:
        with pytest.raises(ValueError, match="action_id"):
            agent.act(Image.new("RGB", (8, 8)), [{"reasoning": "SECRET_HISTORY"}], [{"privileged": "SECRET_FEEDBACK"}])
    events = read_events(recorder.directory)
    assert len(model.messages) == 1
    text = json.dumps(model.messages)
    assert "SECRET_HISTORY" not in text
    assert "SECRET_FEEDBACK" not in text
    assert not any(event["event"] == "span.start" and event["name"] in {"verifier", "percept_mid_low"} for event in events)
    assert any(event["event"] == "tool.rejected" and event["name"] == "invalid_action" for event in events)
    assert not any(event["event"] == "span.start" and event["kind"] == "tool" for event in events)


def test_bare_success_uses_single_stage_and_normalization_is_auditable(tmp_path):
    model = Model([(STOP, USAGE)])
    evaluator = _evaluator(tmp_path, model, bare=True)
    result = evaluator.run_rollout(_episode(), 0)
    events = read_events(_run_dir(result))
    assert len(model.messages) == 1
    assert any(event["event"] == "span.start" and event["name"] == "bare_api" for event in events)
    assert any(event["event"] == "action.normalized" for event in events)
    assert not any(event["name"] in {"verifier", "percept_mid_low"} for event in events)
    assert summarize_run(_run_dir(result))["model_usage"]["leaf_model_calls"] == 1


def test_live_rerun_uses_frozen_episode_profile_and_new_explicit_model_config(tmp_path, monkeypatch):
    from humanclaw_bench.experiments import replay

    model = Model([(STOP, USAGE)])
    evaluator = _evaluator(tmp_path / "original", model, bare=True)
    evaluator.profile.data["agent"]["max_history"] = 17
    source = _run_dir(evaluator.run_rollout(_episode(), 0))
    original_manifest = (source / "run.json").read_bytes()
    model_config = tmp_path / "explicit_model.json"
    model_config.write_text(json.dumps({"model": "new-explicit-model"}))
    captured = {}

    def execute(config):
        captured.update(config)
        return {"mock_live_attempt": True}

    monkeypatch.setattr(replay, "_execute_rerun", execute)
    evaluator.profile.data["agent"]["max_history"] = 99
    result = replay.rerun_run(source, model_config, tmp_path / "fresh-rerun")
    assert result["status"] == "completed"
    assert captured["profile"].data["agent"]["max_history"] == 17
    assert captured["episode_override"].instruction == "Find the bed."
    assert captured["model_config_path"] == str(model_config.resolve())
    assert captured["harness"] == "bare_api"
    assert captured["trace"] is True
    assert captured["source_run"] == str(source.resolve())
    assert (source / "run.json").read_bytes() == original_manifest
    assert (tmp_path / "fresh-rerun" / "frozen_profile.json").is_file()
    with pytest.raises(ValueError, match="outside"):
        replay.rerun_run(source, model_config, source / "overwrite")

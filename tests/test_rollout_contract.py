import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanclaw_bench.agent.skills import STAND_SKILL_CALL
from humanclaw_bench.agent.types import PlannerResult, PSVStageOutput
from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode
from humanclaw_bench.evaluation.evaluator import (
    HCFindNavInteractEvaluator,
    _clear_generated_rollout_artifacts,
    _write_step_vlm_records,
)


def _decision(*stages: PSVStageOutput) -> PlannerResult:
    return PlannerResult(
        raw_plan={},
        action=STAND_SKILL_CALL,
        planner_skill={},
        verifier={},
        stage_outputs=list(stages),
    )


def test_step_vlm_records_are_split_by_real_provider_call(tmp_path):
    planner_response = {
        "visible_state": "The bed is visible.",
        "action_id": 2,
        "action_name": "Turn<left><45>",
    }
    verifier_response = {"verdict": "accept", "reason": "clear"}
    decision = _decision(
        PSVStageOutput(
            stage="percept_mid_low",
            raw=planner_response,
            raw_output=json.dumps(planner_response),
            prompt="planner prompt",
        ),
        PSVStageOutput(
            stage="verifier",
            raw=verifier_response,
            raw_output=json.dumps(verifier_response),
            prompt="verifier prompt",
        ),
    )

    paths = _write_step_vlm_records(tmp_path, 7, decision)

    assert [path.name for path in paths] == [
        "step007_percept_mid_low.json",
        "step007_verifier.json",
    ]
    planner_log = json.loads(paths[0].read_text(encoding="utf-8"))
    assert planner_log == {
        "prompt": "planner prompt",
        "response": planner_response,
    }
    assert not (tmp_path / "episode_log.json").exists()
    assert not (tmp_path / "vlm_calls").exists()


def test_step_vlm_records_skip_verifier_when_no_provider_call_occurred(tmp_path):
    synthetic_verifier = PSVStageOutput(
        stage="verifier",
        raw={"verdict": "accept", "reason": "no verifier for this action type"},
        raw_output="",
        prompt="",
    )

    paths = _write_step_vlm_records(tmp_path, 0, _decision(synthetic_verifier))

    assert paths == []
    assert list(tmp_path.iterdir()) == []


def test_step_vlm_records_keep_only_the_final_retry_response(tmp_path):
    failed = PSVStageOutput(
        stage="percept_mid_low",
        raw={},
        raw_output='{"visible_state": "truncated"',
        prompt="same planner prompt",
        error="ValueError: invalid JSON",
    )
    recovered = PSVStageOutput(
        stage="percept_mid_low",
        raw={"visible_state": "chair visible", "action_id": 2},
        raw_output='{"visible_state": "chair visible", "action_id": 2}',
        prompt="same planner prompt",
    )

    paths = _write_step_vlm_records(tmp_path, 3, _decision(failed, recovered))

    assert [path.name for path in paths] == ["step003_percept_mid_low.json"]
    assert json.loads(paths[0].read_text(encoding="utf-8")) == {
        "prompt": "same planner prompt",
        "response": {"visible_state": "chair visible", "action_id": 2},
    }


def test_final_failed_provider_attempt_keeps_its_raw_response(tmp_path):
    stage = PSVStageOutput(
        stage="percept_mid_low",
        raw={},
        raw_output='{"visible_state": "truncated',
        prompt="exact planner prompt",
        error="ValueError: invalid JSON",
    )
    paths = _write_step_vlm_records(tmp_path, 12, _decision(stage))

    assert len(paths) == 1
    assert json.loads(paths[0].read_text(encoding="utf-8")) == {
        "prompt": "exact planner prompt",
        "response": '{"visible_state": "truncated',
        "error": "ValueError: invalid JSON",
    }

    empty = PSVStageOutput(
        stage="percept_mid_low",
        raw={},
        raw_output="",
        prompt="retry prompt",
        error="JSONDecodeError: empty response",
    )
    empty_path = _write_step_vlm_records(tmp_path, 13, _decision(empty))[0]
    assert json.loads(empty_path.read_text(encoding="utf-8"))["response"] == ""


def test_explicit_rerun_removes_only_generated_artifacts_from_incomplete_run(tmp_path):
    generated = (
        "step000_percept_mid_low.json",
        "step099_verifier.json",
        "ego.mp4",
        "exo.mp4",
        "metrics.json",
        "replay_manifest.json",
        "trajectory_before.npz",
        "trajectory_after.npz",
    )
    for name in generated:
        (tmp_path / name).write_text("stale", encoding="utf-8")
    user_file = tmp_path / "notes.txt"
    user_file.write_text("keep", encoding="utf-8")

    _clear_generated_rollout_artifacts(tmp_path)

    assert not any((tmp_path / name).exists() for name in generated)
    assert user_file.read_text(encoding="utf-8") == "keep"


def test_rollout_persists_vlm_records_and_replay_bundle(tmp_path):
    response = {"visible_state": "bed ahead", "action_name": "Stop/Stand"}
    decision = PlannerResult(
        raw_plan={"at_target": True},
        action=STAND_SKILL_CALL,
        planner_skill=response,
        verifier={},
        stage_outputs=[
            PSVStageOutput(
                stage="percept_mid_low",
                raw=response,
                raw_output=json.dumps(response),
                prompt="planner prompt",
            )
        ],
    )

    class Agent:
        def reset(self, _episode):
            pass

        def act(self, _image, _history, _feedback):
            return decision

    class Env:
        def step(self, _action, reasoning=None):
            del reasoning
            return SimpleNamespace(head_rgb=observation.head_rgb), 0.0, True, {}

        def close(self):
            pass

    observation = SimpleNamespace(head_rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    episode = HCFindNavInteractEpisode(
        name="example",
        task_type="find_nav_interact",
        instruction="Find the bed.",
        scene_id="scene",
        scene_label="scene",
        scene_dataset_config="scene.json",
        episode_id="0",
        object_category="bed",
        object_label="bed",
        init_offset=(0.0, 0.0, 0.0),
        init_yaw=0.0,
        max_steps=1,
        goals=[],
        viewpoint_positions=[],
        goal_objects=[],
    )
    evaluator = HCFindNavInteractEvaluator({"output_root": tmp_path})
    evaluator.output_root = tmp_path
    evaluator.agent = Agent()
    evaluator.env = Env()
    evaluator._reset_env_for_rollout = lambda _episode: observation

    evaluator.run_rollout(episode, 0)

    files = sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()
    )
    assert files == [
        Path("scene_ep0_bed/rollout_00/replay_manifest.json"),
        Path("scene_ep0_bed/rollout_00/step000_percept_mid_low.json"),
        Path("scene_ep0_bed/rollout_00/trajectory_after.npz"),
        Path("scene_ep0_bed/rollout_00/trajectory_before.npz"),
    ]
    assert not (tmp_path / "scene_ep0_bed/rollout_00/trajectory.npz").exists()

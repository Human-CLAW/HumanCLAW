"""Simulated rollout wiring and version isolation, without Habitat or providers."""

import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_feedback_planner import Model, images, prompt_input, proposal

from humanclaw_bench.agent.ego_agent import PSVEgoAgent
from humanclaw_bench.batch import _rollout_complete, _task_name
from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode
from humanclaw_bench.config import load_config
from humanclaw_bench.envs.find_nav_interact_env import HCFindNavInteractEnv
from humanclaw_bench.envs.half_physics_env import (
    HalfPhysicsObservation,
    _proprioception_snapshot,
    xb75_yup_to_half_physics_pose,
)
from humanclaw_bench.evaluation.compatibility import (
    runtime_contract,
    validate_existing_rollout,
)
from humanclaw_bench.evaluation.evaluator import HCFindNavInteractEvaluator
from humanclaw_bench.main import _build_parser


def test_default_profile_and_cli_use_feedback_but_paper_profile_is_pinned():
    profile = load_config()
    assert profile.profile == "feedback_v1"
    assert profile.section("agent")["prompt_version"] == "v5"
    assert profile.section("agent")["verifier_version"] == "v4"
    assert "extends" not in profile.data
    assert load_config("paper_fullval_v1").section("agent")["prompt_version"] == "v4"
    parser = _build_parser()
    for args in (
        ["config"],
        ["rollout", "--model-config", "m", "--scene-id", "s"],
        ["batch", "--model-config", "m", "--output-root", "o"],
    ):
        assert parser.parse_args(args).profile == "feedback_v1"


def test_resume_checks_contract_even_before_artifacts_are_complete(tmp_path):
    episode = {"scene_id": "s", "episode_id": "0", "object_category": "chair"}
    output = tmp_path / _task_name(episode) / "rollout_00"
    output.mkdir(parents=True)
    paper, feedback = load_config("paper_fullval_v1"), load_config()
    manifest = output / "replay_manifest.json"
    manifest.write_text(json.dumps({"profile": paper.profile}))
    validate_existing_rollout(output, paper)
    with pytest.raises(ValueError, match="separate output directory"):
        _rollout_complete(
            tmp_path, episode, save_video=False, compute_metrics=False, profile=feedback
        )
    manifest.write_text(
        json.dumps(
            {
                "profile": feedback.profile,
                "runtime_contract": runtime_contract(feedback),
            }
        )
    )
    assert not _rollout_complete(
        tmp_path, episode, save_video=False, compute_metrics=False, profile=feedback
    )
    for name in ("trajectory_before.npz", "trajectory_after.npz"):
        (output / name).write_bytes(b"test")
    assert _rollout_complete(
        tmp_path, episode, save_video=False, compute_metrics=False, profile=feedback
    )
    changed = replace(feedback, data=json.loads(json.dumps(feedback.data)))
    changed.data["agent"]["use_feedback"] = False
    with pytest.raises(ValueError, match="do not match"):
        validate_existing_rollout(output, changed)


def test_unidentified_or_corrupt_existing_artifacts_are_preserved(tmp_path):
    output = tmp_path / "rollout"
    output.mkdir()
    stage = output / "step000_percept_mid_low.json"
    stage.write_text("keep this")
    with pytest.raises(ValueError, match="Cannot identify"):
        validate_existing_rollout(output, load_config())
    (output / "replay_manifest.json").write_text("invalid json")
    with pytest.raises(ValueError, match="Cannot verify"):
        validate_existing_rollout(output, load_config())
    assert stage.read_text() == "keep this"


def test_version_conflict_closes_resources_without_clearing_prior_results(tmp_path):
    evaluator = HCFindNavInteractEvaluator(
        {"profile": load_config(), "output_root": tmp_path}
    )
    episode = SimpleNamespace(
        scene_label="scene", episode_id="0", object_category="chair"
    )
    output = evaluator.rollout_dir(episode, 0)
    output.mkdir(parents=True)
    manifest = output / "replay_manifest.json"
    original = json.dumps({"profile": "paper_fullval_v1"})
    manifest.write_text(original)
    cleanup = []
    evaluator.env = SimpleNamespace(close=lambda: cleanup.append("env"))
    evaluator.motion = SimpleNamespace(unload=lambda: cleanup.append("motion"))
    with pytest.raises(ValueError, match="do not match"):
        evaluator.run_rollout(episode, 0)
    assert cleanup == ["motion", "env"]
    assert manifest.read_text() == original


def test_lighting_rerender_preserves_paired_body_state(monkeypatch):
    env = HCFindNavInteractEnv(build_runtime=False)
    gfx = ModuleType("habitat_sim.gfx")
    gfx.LightInfo = lambda **kwargs: kwargs
    gfx.LightPositionModel = SimpleNamespace(Global=0)
    monkeypatch.setitem(sys.modules, "habitat_sim.gfx", gfx)
    mn = SimpleNamespace(Vector3=lambda *args: args, Vector4=lambda *args: args)
    monkeypatch.setattr(env, "_require_loaded_modules", lambda: SimpleNamespace(mn=mn))
    monkeypatch.setattr(env, "_update_cameras", lambda: None)
    env.sim = SimpleNamespace(
        set_light_setup=lambda _lights: None,
        get_sensor_observations=lambda: {"ego_rgb": np.ones((2, 2, 3), dtype=np.uint8)},
    )
    state = {"transl": [1, 2, 3]}
    fallback = HalfPhysicsObservation(np.zeros((2, 2, 3), dtype=np.uint8), state)
    refreshed = env._apply_lighting_and_rerender(fallback)
    assert refreshed.proprioception is state
    assert np.all(refreshed.head_rgb == 1)


def test_stop_preserves_the_synchronized_observation():
    env = HCFindNavInteractEnv(build_runtime=False)
    obs = HalfPhysicsObservation(np.zeros((2, 2, 3), dtype=np.uint8), {"own_state": 1})
    env._reset, env._last_obs = True, obs
    returned, _, done, info = env.step({"stop": True})
    assert done and returned is obs and info == {}


def test_rollout_records_blockage_recovery_and_terminal_measurement(tmp_path):
    no_progress = {
        "progress": {
            "motion": "not_met",
            "goal": "unchanged",
            "evidence": "requested 0.4m but root moved 0m and view unchanged",
            "confidence": "high",
        },
        "anomalies": {
            "stuck": {
                "status": "suspected",
                "evidence": "no progress after walk",
                "confidence": "medium",
            }
        },
    }
    recovered = {
        "progress": {
            "motion": "met",
            "goal": "advanced",
            "evidence": "body moved left and exit opened",
            "confidence": "medium",
        },
        "recovery": {
            "action_step": 1,
            "status": "effective",
            "evidence": "blocked path is cleared",
            "confidence": "medium",
        },
    }
    model = Model(
        [
            proposal(0, "Walk<forward><normal>"),
            {"verdict": "accept"},
            proposal(0, "Walk<forward><slow>", state_estimate=no_progress),
            {
                "verdict": "replace",
                "final_action_id": 6,
                "final_action_name": "Side step<left><0.25>",
                "action_intent": {
                    "expected_effect": "clear blocked path",
                    "recovery_for": "stuck",
                },
            },
            proposal(
                1, "Stop/Stand", mid_level_goal="Stop/Stand", state_estimate=recovered
            ),
            {"verdict": "accept"},
        ]
    )
    seed = np.zeros(75)
    seed[1] = 1

    class Env:
        fps = 30.0

        def __init__(self):
            self.step_index = 0
            self.current = seed.copy()
            self.closed = False

        def observation(self):
            state = _proprioception_snapshot(
                *xb75_yup_to_half_physics_pose(self.current)
            )
            state["target_geometry"] = "PRIVATE_TARGET_SENTINEL"
            return HalfPhysicsObservation(
                np.full((4, 4, 3), self.step_index * 20, dtype=np.uint8), state
            )

        def replay_initial_state(self):
            body = self.observation().proprioception
            return {
                "human": {k: body[k] for k in ("transl", "global_orient", "body_pose")},
                "objects": {},
            }

        def metric_find_observation(self):
            raise AssertionError("feedback must not request metric sensors")

        def step(self, action, reasoning=None):
            if isinstance(action, dict):
                return self.observation(), 0, True, {}
            if self.step_index == 1:
                self.current[0] += 0.25
            self.step_index += 1
            frames = np.repeat(self.current[None, :], 15, axis=0)
            transl, orient, pose = xb75_yup_to_half_physics_pose(frames)
            return (
                self.observation(),
                0,
                False,
                {
                    "body_state": {
                        "transl": transl,
                        "global_orient": orient,
                        "body_pose": pose,
                    },
                    "object_states": {},
                    "target_geometry": "PRIVATE_TARGET_SENTINEL",
                },
            )

        def close(self):
            self.closed = True

    class Motion:
        def generate(self, skill, cond):
            frames = np.repeat(seed[None, :], 15, axis=0)
            if skill == "walk_forward":
                frames[:, 2] += np.linspace(0, cond[1], 15)
            else:
                frames[:, 0] += np.linspace(0, cond, 15)
            return SimpleNamespace(xb_world_75=frames)

    episode = HCFindNavInteractEpisode(
        name="feedback",
        task_type="find_nav_interact",
        instruction="Find the chair.",
        scene_id="scene",
        scene_label="scene",
        scene_dataset_config="scene.json",
        episode_id="0",
        object_category="chair",
        object_label="chair",
        init_offset=(0, 0, 0),
        init_yaw=0,
        max_steps=3,
        goals=[],
        viewpoint_positions=[],
        goal_objects=[],
    )
    evaluator = HCFindNavInteractEvaluator(
        {"profile": load_config(), "output_root": tmp_path}
    )
    evaluator.env, evaluator.motion, evaluator.agent = (
        Env(),
        Motion(),
        PSVEgoAgent(model),
    )
    evaluator._initial_xb_world_75 = seed
    evaluator._reset_env_for_rollout = lambda _episode: evaluator.env.observation()
    result = evaluator.run_rollout(episode, 0)
    assert result["steps"] == 3 and evaluator.env.closed
    assert len(model.messages) == 6  # one planner and one verifier per routed action
    assert [len(images(model.messages[i])) for i in (0, 2, 4)] == [1, 2, 2]
    assert all(len(images(model.messages[i])) == 1 for i in (1, 3, 5))
    second = prompt_input(model.messages[2])
    assert second["feedback"]["previous_transition"]["actual_motion"]["forward_m"] == 0
    assert (
        second["feedback"]["previous_transition"]["generated_motion"]["forward_m"]
        == 0.4
    )
    third = prompt_input(model.messages[4])
    assert "Side step" in third["history"][-1]["executed_action"]
    assert third["last_recovery"]["action_step"] == 1
    assert third["history"][0]["assessment"]["evaluated_action_step"] == 0
    assert "PRIVATE_TARGET_SENTINEL" not in json.dumps(model.messages)
    output = evaluator.rollout_dir(episode, 0)
    final = json.loads((output / "step002_percept_mid_low.json").read_text())
    assert final["feedback"]["state_estimate"]["recovery"]["status"] == "effective"
    assert final["feedback"]["execution_result"]["actual_motion"]["duration_s"] == 0
    assert final["feedback"]["execution_result"]["executed_action"]["skill"] == "stand"
    manifest = json.loads((output / "replay_manifest.json").read_text())
    assert manifest["runtime_contract"] == runtime_contract(load_config())

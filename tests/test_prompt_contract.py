import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from humanclaw_bench.agent.planner import (
    VLM_RETRY_BACKOFF_SECONDS,
    VLM_STAGE_MAX_ATTEMPTS,
    HumanClawBenchPSVPlanSkillPlanner,
)
from humanclaw_bench.agent.prompts import resolve_prompt_version
from humanclaw_bench.agent.skills import SkillCall
from humanclaw_bench.agent.verifiers import resolve_verifier_version


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_planner_v4_is_composable_without_changing_the_effective_prompt():
    prompt = resolve_prompt_version("v4")
    assert prompt.VERSION == "v4"
    assert len(prompt.TEMPLATE) == 7326
    assert _digest(prompt.TEMPLATE) == (
        "2b7db9c7f19357a308f87883e844c71af639cfe3c0188423a6d4006633f0f502"
    )
    assert prompt.TEMPLATE.count(prompt.SYSTEM_PROMPT) == 1
    assert prompt.SYSTEM_PROMPT not in prompt.ROLE_AND_CONTEXT
    assert prompt.SYSTEM_PROMPT not in prompt.OUTPUT_CONTRACT
    rendered = prompt.render(
        {
            "goal": "find the chair",
            "current_ego_view_image": "attached",
            "current_step": 0,
            "history": [],
        }
    )
    assert rendered.count(prompt.SYSTEM_PROMPT) == 1
    assert '"goal": "find the chair"' in rendered


@pytest.mark.parametrize(
    (
        "skill",
        "action_name",
        "tokens",
        "additional_info",
        "route_name",
        "expected_length",
        "expected_digest",
    ),
    [
        (
            "walk_forward",
            "Walk<forward><slow>",
            ["forward", "slow"],
            {},
            "walk",
            2347,
            "fa0186775ea47bd4143f9efe41be582f636700361e4b7ce3aa19ce3a328949c0",
        ),
        (
            "stand",
            "Stop/Stand",
            [],
            {},
            "stop",
            2813,
            "4061995a18e06246cb2904bbac8d4face86cdb2f906869009d93d730edbfa2eb",
        ),
        (
            "step_climb_up",
            "Climb upstairs<normal>",
            ["normal"],
            {},
            "climb_up",
            2325,
            "f62cacd10863741e94ab0a579d6906fde73d45b9786b54ba37ad75e7162bc561",
        ),
        (
            "turn",
            "Turn<left><90>",
            ["left", "90"],
            {"turn_for_sit": True},
            "turn_for_sit",
            1806,
            "0a2c95d01fe8248515338ec5a8bce058c92145f55974f3b554d5b0a6316fe751",
        ),
        (
            "stand",
            "Stop/Stand",
            [],
            {"stop_after_sit": True},
            "stop_after_sit",
            1620,
            "fca8c1c195d6e9e317915ca6368e9c32429012127790c435bf606737dd007a49",
        ),
    ],
)
def test_verifier_v3_routes_are_explicit_and_include_system_once(
    skill,
    action_name,
    tokens,
    additional_info,
    route_name,
    expected_length,
    expected_digest,
):
    verifier = resolve_verifier_version("v3")
    action = SkillCall(skill=skill, cond=None, action_name=action_name)
    route = verifier.route_prompt(
        action,
        tokens=tokens,
        skiller_plan={"additional_info": additional_info},
    )
    assert route is not None
    assert route.name == route_name
    rendered = verifier.render(
        route,
        proposed_action_name=action_name,
        input_obj={"goal": "test", "proposed_action": action_name},
    )
    assert rendered.count(verifier.SYSTEM_PROMPT) == 1
    assert verifier.SYSTEM_PROMPT not in verifier.TASK_TEMPLATE
    # Pin the effective wire prompt, not merely its modular source sections.
    # Refactoring section boundaries is allowed only if these bytes stay the
    # same as the formally evaluated verifier-v3 prompts.
    assert len(rendered) == expected_length
    assert _digest(rendered) == expected_digest


def test_only_bundled_prompt_versions_resolve():
    with pytest.raises(ValueError, match="not bundled"):
        resolve_prompt_version("v999")
    with pytest.raises(ValueError, match="not bundled"):
        resolve_verifier_version("v999")


def test_five_failed_planner_attempts_fall_back_to_walk_slow(monkeypatch):

    sleeps = []
    monkeypatch.setattr("humanclaw_bench.agent.planner.time.sleep", sleeps.append)

    class FailingModel:
        calls = 0

        def respond(self, messages):
            del messages
            self.calls += 1
            raise RuntimeError("provider unavailable")

    model = FailingModel()
    planner = HumanClawBenchPSVPlanSkillPlanner(model)
    planner.reset(SimpleNamespace(instruction="Find the chair."))
    decision = planner.act(Image.new("RGB", (8, 8)), [], [])

    assert model.calls == VLM_STAGE_MAX_ATTEMPTS == 5
    assert len(decision.stage_outputs) == 5
    assert all(output.stage == "percept_mid_low" for output in decision.stage_outputs)
    assert decision.action.skill == "walk_forward"
    assert decision.action.action_name == "Walk<forward><slow>"
    assert decision.verifier["fallback"] == "walk_slow_after_retry_exhausted"
    assert "executing Walk<forward><slow>" in decision.stage_outputs[-1].error
    assert sleeps == list(VLM_RETRY_BACKOFF_SECONDS)


def test_planner_retries_only_the_current_stage_then_continues(monkeypatch):
    sleeps = []
    monkeypatch.setattr("humanclaw_bench.agent.planner.time.sleep", sleeps.append)
    valid = {
        "visible_state": "The chair is visible on the left.",
        "mid_level_progress_analysis": "Turn toward the chair.",
        "mid_level_goal": "Face the chair.",
        "low_level_action_reasoning": "The chair is left.",
        "action_id": 2,
        "action_name": "Turn<left><45>",
        "additional_info": {},
    }

    class RecoveringModel:
        def __init__(self):
            self.calls = 0

        def respond(self, messages):
            del messages
            self.calls += 1
            if self.calls == 1:
                return '{"visible_state": "truncated"'
            if self.calls == 2:
                raise RuntimeError("temporary transport failure")
            return json.dumps(valid)

    model = RecoveringModel()
    planner = HumanClawBenchPSVPlanSkillPlanner(model)
    planner.reset(SimpleNamespace(instruction="Find the chair."))

    decision = planner.act(Image.new("RGB", (8, 8)), [], [])

    assert model.calls == 3
    assert len(decision.stage_outputs) == 3
    assert decision.stage_outputs[0].error
    assert decision.stage_outputs[1].error
    assert decision.stage_outputs[2].error is None
    assert decision.action.skill == "turn"
    assert sleeps == list(VLM_RETRY_BACKOFF_SECONDS[:2])


def test_verifier_accepts_planner_action_after_five_failed_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr("humanclaw_bench.agent.planner.time.sleep", sleeps.append)
    valid_planner = {
        "visible_state": "The chair is straight ahead.",
        "mid_level_progress_analysis": "Approach the chair.",
        "mid_level_goal": "Move closer.",
        "low_level_action_reasoning": "The forward lane is clear.",
        "action_id": 0,
        "action_name": "Walk<forward><slow>",
        "additional_info": {},
    }

    class FailingVerifierModel:
        def __init__(self):
            self.calls = 0

        def respond(self, messages):
            del messages
            self.calls += 1
            if self.calls == 1:
                return json.dumps(valid_planner)
            raise RuntimeError("verifier unavailable")

    model = FailingVerifierModel()
    planner = HumanClawBenchPSVPlanSkillPlanner(model)
    planner.reset(SimpleNamespace(instruction="Find the chair."))

    decision = planner.act(Image.new("RGB", (8, 8)), [], [])

    assert model.calls == 1 + VLM_STAGE_MAX_ATTEMPTS
    assert decision.action.skill == "walk_forward"
    assert decision.verifier["verdict"] == "accept"
    assert decision.verifier["fallback"] == "accept_after_retry_exhausted"
    verifier_attempts = [
        output for output in decision.stage_outputs if output.stage == "verifier"
    ]
    assert len(verifier_attempts) == 5
    assert all(output.error for output in verifier_attempts)
    assert sleeps == list(VLM_RETRY_BACKOFF_SECONDS)

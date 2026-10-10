"""Exercise temporal inputs, final-action memory and feedback without API calls."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from humanclaw_bench.agent.feedback import step_feedback
from humanclaw_bench.agent.planner import HumanClawBenchPSVPlanSkillPlanner
from humanclaw_bench.agent.utils import image_to_data_url
from humanclaw_bench.evaluation.evaluator import _history_item


def proposal(action_id=2, action_name="Turn<left><30>", **extra):
    return {
        "visible_state": "A chair is near the left edge.",
        "mid_level_progress_analysis": "Check the open lane.",
        "mid_level_goal": "Approach the chair.",
        "low_level_action_reasoning": "Face the opening.",
        "action_id": action_id,
        "action_name": action_name,
        "additional_info": {},
        "action_intent": {
            "expected_effect": "opening moves to the center",
            "recovery_for": "none",
        },
        **extra,
    }


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def respond(self, messages):
        self.messages.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def prompt_input(messages):
    prompt = messages[0]["content"][0]["text"]
    return json.JSONDecoder().raw_decode(prompt.split("\nInput:\n", 1)[1].lstrip())[0]


def images(messages):
    return [
        part["image_url"]["url"]
        for part in messages[0]["content"]
        if part["type"] == "image_url"
    ]


def planner_for(responses, **kwargs):
    model = Model(responses)
    planner = HumanClawBenchPSVPlanSkillPlanner(model, **kwargs)
    planner.reset(SimpleNamespace(instruction="Find the chair."))
    return planner, model


def test_two_frames_are_chronological_and_retries_do_not_advance_them(monkeypatch):
    monkeypatch.setattr(
        "humanclaw_bench.agent.planner.time.sleep", lambda _seconds: None
    )
    planner, model = planner_for(
        [proposal(), RuntimeError("transient"), "invalid", proposal(), proposal()]
    )
    first = Image.new("RGB", (4, 4), "red")
    second = Image.new("RGB", (4, 4), "green")
    third = Image.new("RGB", (4, 4), "blue")
    decision0 = planner.act(first, [], [])
    history = [_history_item(0, decision0)]
    decision1 = planner.act(second, history, [])
    history.append(_history_item(1, decision1))
    planner.act(third, history, [])
    assert images(model.messages[0]) == [image_to_data_url(first)]
    for attempt in model.messages[1:4]:
        assert images(attempt) == [image_to_data_url(first), image_to_data_url(second)]
        assert "BEFORE action step 0" in attempt[0]["content"][1]["text"]
    assert images(model.messages[4]) == [
        image_to_data_url(second),
        image_to_data_url(third),
    ]
    assert decision1.state_estimate["evaluated_action_step"] == 0
    assert len(planner._frame_cache) == 2


@pytest.mark.parametrize(
    "new_intent",
    [None, {"expected_effect": "clear blocked lane", "recovery_for": "stuck"}],
)
def test_verifier_replacement_rebinds_intent_and_history_to_final_action(new_intent):
    verifier = {
        "verdict": "replace",
        "final_action_id": 6,
        "final_action_name": "Side step<left><0.25>",
    }
    if new_intent is not None:
        verifier["action_intent"] = new_intent
    state = {
        "progress": {
            "motion": "not_met",
            "goal": "unchanged",
            "evidence": "body hardly moved",
            "confidence": "high",
        }
    }
    planner, model = planner_for(
        [
            proposal(0, "Walk<forward><normal>", state_estimate=state),
            verifier,
            proposal(),
        ]
    )
    snapshot = {
        "coordinate_frame": "smplx_y_up",
        "transl": [0, 1, 0],
        "global_orient": [0, 0, 0],
        "body_pose": np.zeros((54, 3)),
    }
    decision = planner.act(
        Image.new("RGB", (4, 4)), [], [step_feedback(0, snapshot, snapshot, None)]
    )
    assert len(model.messages) == 2
    assert decision.action.skill == "side_walk"
    assert decision.action_intent["requested_motion"] == {"left_m": 0.25}
    assert decision.action_intent["expected_effect"] == (
        new_intent["expected_effect"] if new_intent else "unknown"
    )
    assert decision.action_intent["recovery_for"] == ("stuck" if new_intent else "none")
    verifier_input = prompt_input(model.messages[1])
    assert verifier_input["state_estimate"] == decision.state_estimate
    assert verifier_input["feedback"]["body"]["available"]
    assert "joint_rotvec_rad" not in verifier_input["feedback"]["body"]
    assert len(images(model.messages[1])) == 1
    planner.act(Image.new("RGB", (4, 4)), [_history_item(0, decision)], [])
    recorded = prompt_input(model.messages[2])["history"][0]
    assert "Side step<left><0.25>" in recorded["executed_action"]
    assert "Walk" not in recorded["executed_action"]
    assert recorded["expected_effect"] == decision.action_intent["expected_effect"]


def test_recovery_can_stay_pending_then_succeed_without_extra_calls():
    attempt = proposal(
        6,
        "Side step<left><0.25>",
        action_intent={
            "expected_effect": "open a path around the blockage",
            "recovery_for": "stuck",
        },
    )
    pending = proposal(
        state_estimate={
            "recovery": {
                "action_step": 0,
                "status": "pending",
                "evidence": "side motion occurred but exit is still occluded",
                "confidence": "low",
            }
        }
    )
    success = proposal(
        state_estimate={
            "recovery": {
                "action_step": 0,
                "status": "effective",
                "evidence": "the exit is now open ahead",
                "confidence": "medium",
            }
        }
    )
    planner, model = planner_for([attempt, pending, success])
    history = []
    decisions = []
    for step in range(3):
        decision = planner.act(Image.new("RGB", (4, 4)), history, [])
        history.append(_history_item(step, decision))
        decisions.append(decision)
    assert len(model.messages) == 3
    assert decisions[0].state_estimate["recovery"]["status"] == "no_attempt"
    assert decisions[1].state_estimate["recovery"]["status"] == "pending"
    assert decisions[2].state_estimate["recovery"]["status"] == "effective"
    assert prompt_input(model.messages[2])["last_recovery"]["action_step"] == 0


def test_history_bound_and_durable_memory_reset():
    seen = {
        "body_target_relation": {
            "visibility": "visible",
            "direction": "left",
            "distance": "near",
            "source": "current_view",
            "evidence": "chair visible",
            "confidence": "medium",
        }
    }
    first = proposal(
        state_estimate=seen,
        action_intent={
            "expected_effect": "see the target again",
            "recovery_for": "target_lost",
        },
    )
    planner, model = planner_for([first] + [proposal()] * 13)
    history = []
    for step in range(13):
        result = planner.act(Image.new("RGB", (4, 4)), history, [])
        history.append(_history_item(step, result))
    last_input = prompt_input(model.messages[-1])
    assert len(last_input["history"]) == 10
    assert last_input["last_recovery"]["action_step"] == 0
    assert (
        last_input["previous_state_estimate"]["body_target_relation"]["last_seen"][
            "step"
        ]
        == 0
    )
    assert "joint_rotvec_rad" not in json.dumps(last_input["history"])
    planner.reset(SimpleNamespace(instruction="Find the bed."))
    reset = planner.act(Image.new("RGB", (4, 4)), [], [])
    assert len(images(model.messages[-1])) == 1
    assert prompt_input(model.messages[-1])["last_recovery"] is None
    assert reset.state_estimate["body_target_relation"]["last_seen"] is None


@pytest.mark.parametrize("bad_state", [None, [], "broken", {"progress": []}])
def test_malformed_optional_estimate_does_not_retry_valid_action(bad_state):
    planner, model = planner_for(
        [proposal(state_estimate=bad_state, action_intent="bad")]
    )
    result = planner.act(Image.new("RGB", (4, 4)), [], [])
    assert len(model.messages) == 1
    assert result.action.skill == "turn"
    assert result.state_estimate["progress"]["motion"] == "unknown"
    assert result.action_intent["expected_effect"] == "unknown"


def test_paper_v4_keeps_single_image_and_original_history_contract():
    planner, model = planner_for(
        [proposal()], prompt_version="v4", verifier_version="v3"
    )
    history = [
        {
            "step": 0,
            "action_text": "Side step<left><0.25>",
            "planner_skill": proposal(0, "Walk<forward><slow>"),
        }
    ]
    result = planner.act(Image.new("RGB", (4, 4)), history, [{"secret": "ignored"}])
    data = prompt_input(model.messages[0])
    assert set(data) == {"goal", "current_ego_view_image", "current_step", "history"}
    assert data["history"][0]["action_name"] == "Walk<forward><slow>"
    assert len(model.messages[0][0]["content"]) == 2
    assert not planner._frame_cache
    assert result.state_estimate == result.action_intent == {}


def test_feedback_switch_disables_body_inputs_frame_cache_and_estimates():
    planner, model = planner_for([proposal(), proposal()], use_feedback=False)
    result = planner.act(Image.new("RGB", (4, 4)), [], [{"secret": "ignored"}])
    planner.act(Image.new("RGB", (4, 4)), [_history_item(0, result)], [])
    assert result.state_estimate == {}
    assert not planner._frame_cache
    assert len(images(model.messages[-1])) == 1
    assert "feedback" not in prompt_input(model.messages[-1])

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from humanclaw_bench.agent.types import PSVStageOutput
from humanclaw_bench.envs.half_physics_env import HalfPhysicsEnv
from humanclaw_bench.envs.runtime_records import _describe_sim_object
from humanclaw_bench.evaluation.metrics import collision as collision_metrics
from humanclaw_bench.evaluation.metrics.disturbance import DisturbanceTracker
from humanclaw_bench.evaluation.metrics.episode import (
    PaperMetricRecorder,
    aggregate_metric_files,
)
from humanclaw_bench.evaluation.metrics.report import format_metric_summary
from humanclaw_bench.evaluation.metrics.find import claims_target_visible
from humanclaw_bench.evaluation.metrics.geometry import (
    body_to_target_aabb_distance,
)
from humanclaw_bench.evaluation.metrics.jerk import (
    load_neutral_body22,
    root_rigid_motion_jerk,
)
from humanclaw_bench.evaluation.metrics.usage import UsageTracker, normalize_usage
from humanclaw_bench.evaluation.video import RolloutVideoWriter
from humanclaw_bench.vlm.filesystem_queue import _parse_usage


def test_final_find_rule_joins_render_and_non_negated_target_sentence():
    assert claims_target_visible("The mattress is visible on my left.", "bed")
    assert not claims_target_visible("The bed is not visible.", "bed")
    assert claims_target_visible("The bed is visible. No chair is nearby.", "bed")
    assert not claims_target_visible("I can see a table and no target.", "bed")


def test_body_to_target_aabb_distance_is_full_3d_minimum():
    points = [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
    aabbs = [([2.5, 0.5, -0.5], [3.0, 1.5, 0.5])]
    assert body_to_target_aabb_distance(points, aabbs) == pytest.approx(0.5)


def test_root_rigid_jerk_matches_cubic_translation():
    fps = 30.0
    frame = np.arange(50, dtype=np.float64)
    xb = np.zeros((frame.size, 75), dtype=np.float64)
    xb[:, 0] = (frame / fps) ** 3
    rest = np.zeros((22, 3), dtype=np.float64)
    score = root_rigid_motion_jerk(xb, rest, fps=fps, smooth_window=1, stride=2)
    assert score == pytest.approx(6.0, rel=1e-10, abs=1e-10)


def test_neutral_body_resource_is_exact_pelvis_relative_ssdmc_constant():
    rest = load_neutral_body22()
    assert rest.shape == (22, 3)
    assert rest[0].tolist() == [0.0, 0.0, 0.0]
    # Two nontrivial endpoints guard both scale and joint ordering.  Values
    # come from J_regressor @ v_template in the pinned neutral source model.
    assert rest[1] == pytest.approx(
        [0.06951973546651909, -0.09140621868807669, -0.006815336587141844]
    )
    assert rest[21] == pytest.approx(
        [-0.6824004497693242, 0.44289297304730024, -0.07489780563198079]
    )


def _dynamic(name):
    return {
        "type": "rigid_object",
        "name": name,
        "motion_type": "MotionType.DYNAMIC",
    }


def test_disturbance_propagates_on_same_or_later_frames_only():
    tracker = DisturbanceTracker()
    tracker.record_step(
        0,
        {
            "agent_contacts": [[{"other": _dynamic("chair_a")}], []],
            "dynamic_contacts": [
                [{"a": _dynamic("chair_a"), "b": _dynamic("chair_b")}],
                [],
            ],
        },
    )
    after = {
        "object_names": np.asarray(["chair_a", "chair_b"]),
        "object_000_position": np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float
        ),
        "object_001_position": np.asarray(
            [[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float
        ),
    }
    result = tracker.finalize(after)
    assert result["direct_dynamic_object_count"] == 1
    assert result["indirect_dynamic_object_count"] == 1
    assert result["affected_dynamic_object_count"] == 2
    assert result["affected_object_path_length_sum_m"] == pytest.approx(4.0)


def test_geo_interact_uses_the_landed_final_frame_of_a_sit_action(monkeypatch):
    """Match the paper's historical per-step ``last_frame_contacts`` rule."""

    env = SimpleNamespace(
        is_pelvis_target_contact=lambda contact: bool(contact.get("target"))
    )
    monkeypatch.setattr(
        collision_metrics,
        "_episode_floor_y",
        lambda *_args: (0.0, "test"),
    )
    monkeypatch.setattr(
        collision_metrics,
        "collect_metric_contacts",
        lambda *_args: ([], []),
    )
    recorder = PaperMetricRecorder(
        episode=SimpleNamespace(object_category="bed"),
        env=env,
        config={},
        profile_name="test",
        rollout_index=0,
    )
    recorder.record_reset()

    # Contact occurs transiently, then is lost before the Sit lands.
    recorder.record_motion(
        step=0,
        action_skill="sit",
        info={
            "metric_frames": {
                "agent_contacts": [[{"target": True}], []],
                "dynamic_contacts": [[], []],
            }
        },
    )
    assert recorder.n_sit == 1
    assert recorder.geo_interact_success is False

    # A later Sit lands with contact on its final frame and therefore counts.
    recorder.record_motion(
        step=1,
        action_skill="sit",
        info={
            "metric_frames": {
                "agent_contacts": [[], [{"target": True}]],
                "dynamic_contacts": [[], []],
            }
        },
    )
    assert recorder.n_sit == 2
    assert recorder.geo_interact_success is True


def test_contact_object_zero_is_not_confused_with_habitat_stage():
    rigid = SimpleNamespace(
        handle="first_dynamic_object",
        motion_type="MotionType.DYNAMIC",
    )

    class Manager:
        @staticmethod
        def get_library_has_id(object_id):
            return object_id == 0

        @staticmethod
        def get_object_by_id(object_id):
            assert object_id == 0
            return rigid

    manager = Manager()
    env = SimpleNamespace(
        agent=SimpleNamespace(object_id=99),
        sim=SimpleNamespace(
            get_rigid_object_manager=lambda: manager,
            get_articulated_object_manager=lambda: Manager(),
        ),
        _require_runtime=lambda: SimpleNamespace(
            habitat_sim=SimpleNamespace(stage_id=-1)
        ),
    )
    assert _describe_sim_object(env, -1)["type"] == "stage"
    described = _describe_sim_object(env, 0)
    assert described["type"] == "rigid_object"
    assert described["name"] == "first_dynamic_object"


def test_initial_penetration_is_queried_once_at_reset(monkeypatch):
    """Reset penetration is diagnostic and never removes an episode."""

    events = []
    monkeypatch.setattr(
        collision_metrics,
        "_episode_floor_y",
        lambda *_args: (events.append("floor") or (0.0, "test")),
    )

    def contacts(*_args):
        events.append("contacts")
        return ([{"contact_distance": -0.02}], [])

    monkeypatch.setattr(collision_metrics, "collect_metric_contacts", contacts)
    tracker = collision_metrics.CollisionTracker({})
    tracker.record_reset(SimpleNamespace())
    tracker.record_step(0, {"agent_contacts": [[]], "dynamic_contacts": [[]]})
    result = tracker.finalize()

    assert result["initial_penetration_detected"] is True
    assert result["initial_penetration_excluded"] is False
    assert result["collision_step_fraction"] == 0.0
    assert result["motion_steps"] == 1
    assert result["collision_contact_source"] == "post_physics_30hz"
    assert events == ["floor", "contacts"]


def test_collision_reuses_shared_landed_pose_contacts(monkeypatch):
    """Motion scoring must not issue a second per-frame collision query."""

    query_count = 0
    monkeypatch.setattr(
        collision_metrics,
        "_episode_floor_y",
        lambda *_args: (0.0, "test"),
    )

    def reset_contacts(*_args):
        nonlocal query_count
        query_count += 1
        return ([], [])

    monkeypatch.setattr(
        collision_metrics,
        "collect_metric_contacts",
        reset_contacts,
    )
    fixed_hit = {
        "body_part": "left_wrist",
        "other": {"type": "stage", "name": "stage"},
        "position_on_other_ws": [0.0, 0.2, 0.0],
    }
    floor_hit = {
        "body_part": "left_foot",
        "other": {"type": "stage", "name": "stage"},
        "position_on_other_ws": [0.0, 0.01, 0.0],
    }

    tracker = collision_metrics.CollisionTracker({})
    tracker.record_reset(SimpleNamespace())
    tracker.record_step(
        3,
        {"agent_contacts": [[floor_hit], [fixed_hit]], "dynamic_contacts": [[], []]},
    )
    tracker.record_step(
        4,
        {"agent_contacts": [[]], "dynamic_contacts": [[]]},
    )
    result = tracker.finalize()

    assert query_count == 1
    assert result["collision_steps"] == 1
    assert result["motion_steps"] == 2
    assert result["collision_step_fraction"] == pytest.approx(0.5)
    assert result["by_body_group_step_fraction"]["hand_arm"] == pytest.approx(0.5)


def test_usage_excludes_openai_reasoning_tokens_and_labels_fallback():
    normalized = normalize_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 15},
            "total_tokens": 140,
        }
    )
    assert normalized == {
        "input_tokens": 100,
        "visible_output_tokens": 25,
        "reasoning_tokens": 15,
        "total_tokens": 140,
    }

    tracker = UsageTracker()
    tracker.record(
        PSVStageOutput(
            stage="percept_mid_low",
            raw={},
            raw_output="12345678",
            prompt="12345678",
        )
    )
    summary = tracker.summary(1)
    assert summary["token_source"] == "estimated_chars_div_4"
    assert summary["input_tokens"] == 2
    assert summary["visible_output_tokens"] == 2

    no_calls = UsageTracker().summary(decision_steps=1)
    assert no_calls["token_source"] == "unavailable_no_calls"
    assert no_calls["vlm_calls"] == 0


def test_gemini_queue_usage_string_keeps_visible_and_reasoning_separate():
    usage = _parse_usage(
        "Usage(prompt_tokens=5197, completion_tokens=233, "
        "total_tokens=5715, num_reasoning_tokens=285)"
    )
    assert normalize_usage(usage) == {
        "input_tokens": 5197,
        "visible_output_tokens": 233,
        "reasoning_tokens": 285,
        "total_tokens": 5715,
    }


def test_gemini_queue_usage_mapping_and_pre_normalized_usage():
    queue_mapping = _parse_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "num_reasoning_tokens": 7,
            "total_tokens": 127,
        }
    )
    assert normalize_usage(queue_mapping) == {
        "input_tokens": 100,
        "visible_output_tokens": 20,
        "reasoning_tokens": 7,
        "total_tokens": 127,
    }
    assert normalize_usage(
        {
            "input_tokens": 50,
            "visible_output_tokens": 11,
            "reasoning_tokens": 3,
        }
    ) == {
        "input_tokens": 50,
        "visible_output_tokens": 11,
        "reasoning_tokens": 3,
        "total_tokens": 64,
    }


def _metric_row(index, *, interact=False, eligible=True):
    return {
        "schema": "humanclaw_paper_metrics_v1",
        "episode": {"episode_id": str(index)},
        "success": {
            "find_sr": index == 0,
            "geo_find_sr": True,
            "nav_sr_20cm": index == 0,
            "geo_nav_sr_20cm": True,
            "nav_sr_1m": True,
            "is_interact_episode": interact,
            "interact_sr": interact,
            "geo_interact_sr": interact,
        },
        "body_scene": {
            "physical_metrics_eligible": eligible,
            "collision_step_fraction": 0.25,
            "by_body_group_step_fraction": {
                "hand_arm": 0.1,
                "torso": 0.2,
                "leg": 0.3,
                "head": 0.4,
            },
            "affected_dynamic_object_count": 2,
            "mapped_affected_dynamic_object_count": 2,
            "affected_object_path_length_sum_m": 3.0,
        },
        "action_quality": {"motion_jerk_m_s3": 5.0},
        "cost": {
            "decision_steps": 10,
            "input_tokens": 100,
            "visible_output_tokens": 20,
            "token_source": "provider_exact",
        },
    }


def test_batch_aggregation_uses_all_physics_episodes(tmp_path):
    for index, row in enumerate(
        [_metric_row(0, interact=True), _metric_row(1)]
    ):
        directory = tmp_path / f"run_{index}"
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(row))
    result = aggregate_metric_files(tmp_path)
    assert result["counts"] == {
        "episodes": 2,
        "interact_episodes": 1,
        "physical_metric_episodes": 2,
        "initial_penetration_excluded": 0,
    }
    assert result["high_level_success_percent"]["find_sr"] == 50.0
    assert result["high_level_success_percent"]["interact_sr"] == 100.0
    assert result["body_scene"]["collision_step_percent"] == 25.0
    assert result["body_scene"]["disturbed_object_path_length_mean_m"] == 1.5
    assert (tmp_path / "metrics_summary.json").is_file()


def test_read_only_aggregation_formats_every_paper_table(tmp_path):
    directory = tmp_path / "shard00" / "episode"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps(_metric_row(0, interact=True)))

    result = aggregate_metric_files(tmp_path, write_summary=False)
    report = format_metric_summary(result)

    assert not (tmp_path / "metrics_summary.json").exists()
    assert "Episodes: 1 | Interact: 1 | Physical: 1" in report
    assert "| FindSR | NavSR@20cm | InteractSR |" in report
    assert "| GeoFindSR | GeoNavSR@20cm | NavSR@1m | GeoInteractSR |" in report
    assert "| Arm/Hand | Torso | Leg/Foot | Head |" in report
    assert "| 100.0% | 100.0% | 100.0% |" in report


def test_default_environment_builds_no_optional_metric_or_video_path():
    env = HalfPhysicsEnv(build_runtime=False)
    assert not env.ego_semantic_enabled
    assert not env.collect_metric_contacts
    assert not env.video_enabled
    assert env._video_frame_sink is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_video_writer_streams_two_final_mp4_files(tmp_path):
    writer = RolloutVideoWriter(tmp_path, fps=30)
    writer.append(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 127, dtype=np.uint8),
    )
    writer.close()
    assert (tmp_path / "ego.mp4").stat().st_size > 0
    assert (tmp_path / "exo.mp4").stat().st_size > 0

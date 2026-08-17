#!/usr/bin/env python3
"""Run one small, real-Habitat integration check on prepared HSSD.

This is intentionally a manual smoke test rather than a normal pytest test:
it requires Habitat-Sim, an authorized HSSD installation, and ffmpeg.  Unit
tests cover pure metric logic; this script checks the boundaries that mocks
cannot cover: scene loading, target-instance lookup, semantic rendering,
HalfPhysics, contact collection, dynamic-object recording, MP4 streaming, and
the terminal trajectory-derived metrics.

The script does not call a VLM or load motion-model weights.  It places the
bundled five-frame seed exactly as ``MotionSkillRunner.reset`` does and holds
that pose for a short motion chunk.  Consequently it validates runtime wiring
without spending API quota or requiring the separately downloaded weights.

Example (from the repository root)::

    PYTHONPATH=src python tests/runtime_habitat_smoke.py \
        --output /tmp/humanclaw_habitat_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanclaw_bench.agent.skills import STAND_SKILL_CALL, SkillCall
from humanclaw_bench.agent.types import PlannerResult, PSVStageOutput
from humanclaw_bench.benchmark.episodes import load_episode
from humanclaw_bench.config import load_config
from humanclaw_bench.envs.find_nav_interact_env import HCFindNavInteractEnv
from humanclaw_bench.envs.half_physics_env import (
    xb75_yup_to_half_physics_pose,
)
from humanclaw_bench.evaluation.metrics.episode import (
    PaperMetricRecorder,
    write_episode_metrics,
)
from humanclaw_bench.evaluation.trajectory import (
    TrajectoryRecorder,
    build_replay_metadata,
)
from humanclaw_bench.evaluation.video import RolloutVideoWriter
from humanclaw_bench.motion.conditioning import (
    load_seed_state,
    zero_neck_head_xb_state,
)
from humanclaw_bench.paths import repository_root, resolve_release_path


def _initial_seed_pose(
    seed_path: Path,
    init_offset: tuple[float, float, float],
    init_yaw_deg: float,
) -> np.ndarray:
    """Return the seed's first 75-D pose in episode coordinates.

    Motion generation normally performs this transform while loading the
    checkpoints.  The smoke test repeats only the small deterministic reset
    transform so it remains independent of those checkpoint files.
    """

    seed = zero_neck_head_xb_state(load_seed_state("pt", seed_pt=seed_path)).numpy()
    initial_canonical = seed[0, :75]

    reference_rotation = Rotation.from_rotvec(
        [0.0, np.radians(float(init_yaw_deg)), 0.0]
    ).as_matrix()
    offset = np.asarray(init_offset, dtype=np.float32)
    # Episode offsets use the same (x, horizontal-z, vertical-y) convention as
    # the paper rollout.  The motion state itself is conventional Y-up xyz.
    reference_translation = np.asarray(
        [offset[0], offset[2], -offset[1]], dtype=np.float32
    )
    translation = reference_rotation @ initial_canonical[:3]
    translation = translation + reference_translation
    orientation = Rotation.from_matrix(
        reference_rotation @ Rotation.from_rotvec(initial_canonical[3:6]).as_matrix()
    ).as_rotvec()
    world_pose = np.concatenate(
        [translation, orientation, initial_canonical[6:75]]
    ).astype(np.float32)
    return world_pose


def _decision(action: SkillCall, category: str, call_index: int) -> PlannerResult:
    """Build a provider-shaped decision to exercise FindSR and token usage."""

    visible_state = f"The target {category} is visible in the current ego view."
    stage = PSVStageOutput(
        stage="percept_mid_low",
        raw={"visible_state": visible_state},
        raw_output=json.dumps({"visible_state": visible_state}),
        prompt=f"runtime smoke prompt {call_index}",
        usage={
            "input_tokens": 5,
            "visible_output_tokens": 7,
            "reasoning_tokens": 2,
            "total_tokens": 14,
        },
    )
    return PlannerResult(
        raw_plan={"language_plan": action.action_name or action.skill},
        action=action,
        planner_skill={"visible_state": visible_state},
        verifier={},
        stage_outputs=[stage],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-id",
        default="102343992",
        help="One of the 41 short HSSD validation scene IDs.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frames",
        type=int,
        default=25,
        help="At least 25 frames are required to exercise stride-8 jerk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty directory for the smoke artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frames < 25:
        raise ValueError("--frames must be at least 25 to exercise Motion Jerk")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    profile = load_config("paper_fullval_v1")
    benchmark = profile.section("benchmark")
    motion = profile.section("motion")
    physics = profile.section("physics")
    rendering = profile.section("rendering")
    metric_config = profile.section("metrics")
    scene_config = resolve_release_path(benchmark["scene_dataset_config"])
    episode = load_episode(
        benchmark_dataset_dir=resolve_release_path(benchmark["dataset_dir"]),
        split=str(benchmark["split"]),
        scene_id=str(args.scene_id),
        scene_dataset_config=scene_config,
        episode_index=int(args.episode_index),
        object_category=None,
        max_steps=int(benchmark["max_steps"]),
    )

    initial_xb = _initial_seed_pose(
        resolve_release_path(motion["seed_pt"]),
        episode.init_offset,
        episode.init_yaw,
    )
    initial_transl, initial_orient, initial_body_pose = xb75_yup_to_half_physics_pose(
        initial_xb
    )
    physics_kwargs = {
        key: value
        for key, value in physics.items()
        if key not in {"backend", "agent_urdf", "agent_shift_npy", "physics_config"}
    }

    env = HCFindNavInteractEnv(
        scene_id=episode.scene_id,
        scene_dataset_config=episode.scene_dataset_config,
        half_physics_backend=str(physics["backend"]),
        agent_urdf=resolve_release_path(physics["agent_urdf"]),
        agent_shift_npy=resolve_release_path(physics["agent_shift_npy"]),
        physics_config=resolve_release_path(physics["physics_config"]),
        max_episode_steps=int(benchmark["max_steps"]),
        lighting=str(rendering["lighting"]),
        ambient_strength=float(rendering["ambient_strength"]),
        ego_resolution=tuple(rendering["ego_resolution"]),
        third_person_resolution=tuple(rendering["third_person_resolution"]),
        compute_metrics=True,
        video_enabled=True,
        **physics_kwargs,
    )
    # Check the values on the live articulated object, not only in JSON.  The
    # Python API exposes child links but not the pelvis/base friction, which is
    # why this count should match agent.num_links exactly.
    expected_friction = float(physics["friction"])
    link_frictions = [
        float(env.agent.get_link_friction(link_id))
        for link_id in range(int(env.agent.num_links))
    ]
    if not all(
        np.isclose(value, expected_friction, rtol=0.0, atol=1.0e-7)
        for value in link_frictions
    ):
        raise AssertionError(
            "Human child-link friction mismatch: "
            f"range={[min(link_frictions), max(link_frictions)]}"
        )
    if env.root_linear_xz_command_substeps != (0, 2):
        raise AssertionError(
            f"Unexpected root x/z schedule: {env.root_linear_xz_command_substeps}"
        )
    backend = env._require_runtime().hp
    if float(backend.ANGULAR_LIMIT_DEGREES_PER_FRAME) != 30.0:
        raise AssertionError("Half-Physics angular limit is not 30 degrees/frame")
    video = RolloutVideoWriter(output, float(env.fps))
    video_closed = False
    try:
        # The sink is attached before reset, matching the evaluator lifecycle;
        # the reset frame is emitted explicitly after target IDs and lighting
        # have been applied.
        env.set_video_frame_sink(video.append)
        observation = env.reset(
            episode,
            initial_transl=initial_transl,
            initial_global_orient=initial_orient,
            initial_body_pose=initial_body_pose,
        )
        env.emit_initial_video_frame()
        if observation.head_rgb.shape[:2] != tuple(rendering["ego_resolution"]):
            raise AssertionError(
                f"Unexpected ego frame shape: {observation.head_rgb.shape}"
            )

        trajectory = TrajectoryRecorder(
            metadata=build_replay_metadata(
                profile_name=profile.profile,
                episode=episode,
                rollout_index=0,
                env=env,
            ),
            initial_xb_world_75=initial_xb,
            initial_state=env.replay_initial_state(),
        )
        metrics = PaperMetricRecorder(
            episode=episode,
            env=env,
            config=metric_config,
            profile_name=profile.profile,
            rollout_index=0,
        )
        metrics.record_reset()

        motion_action = SkillCall(
            skill="walk_forward",
            cond=[0.0, 0.2, 0.0],
            action_name="Walk<forward><slow>",
        )
        metrics.record_decision(
            step=0,
            decision=_decision(motion_action, episode.object_category, 0),
            find_observation=env.metric_find_observation(),
        )
        # A constant requested pose is sufficient here: HalfPhysics still
        # advances gravity/contact dynamics, while the pre-physics signal has
        # a known zero root-rigid jerk value.
        requested_xb = np.repeat(initial_xb[None, :], args.frames, axis=0)
        trajectory.record_before(
            step=0,
            action=motion_action,
            action_text="Walk<forward><slow>",
            xb_world_75=requested_xb,
        )
        _observation, _reward, _done, info = env.step(requested_xb)
        trajectory.record_after(
            step=0,
            body_state=info["body_state"],
            object_states=info["object_states"],
        )
        metrics.record_motion(
            step=0,
            action_skill=motion_action.skill,
            info=info,
        )

        # A second decision verifies active-stop bookkeeping without running a
        # second physics chunk or writing an artificial after trajectory.
        metrics.record_decision(
            step=1,
            decision=_decision(STAND_SKILL_CALL, episode.object_category, 1),
            find_observation=env.metric_find_observation(),
        )
        env.step({"stop": True, "skill": "stand"})

        video.close()
        video_closed = True
        before, after = trajectory.materialize()
        result = metrics.finalize(before=before, after=after)
        write_episode_metrics(output / "metrics.json", result)
        trajectory.write(output)

        expected_frames = args.frames + 1
        if video.ego.frame_count != expected_frames:
            raise AssertionError(
                f"ego.mp4 has {video.ego.frame_count} input frames, "
                f"expected {expected_frames}"
            )
        if video.exo.frame_count != expected_frames:
            raise AssertionError(
                f"exo.mp4 has {video.exo.frame_count} input frames, "
                f"expected {expected_frames}"
            )
        required_files = (
            "ego.mp4",
            "exo.mp4",
            "trajectory_before.npz",
            "trajectory_after.npz",
            "replay_manifest.json",
            "metrics.json",
        )
        missing = [name for name in required_files if not (output / name).is_file()]
        if missing:
            raise AssertionError(f"Smoke artifacts missing: {missing}")
        if result.get("schema") != "humanclaw_paper_metrics_v1":
            raise AssertionError("Unexpected metrics schema")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "repository": str(repository_root()),
                    "scene_id": episode.scene_label,
                    "episode_id": episode.episode_id,
                    "category": episode.object_category,
                    "target_count": len(episode.goal_objects),
                    "dynamic_object_count": len(info["object_states"]),
                    "half_physics": {
                        "backend": env.half_physics_backend,
                        "physics_config": env.physics_config.name,
                        "angular_limit_degrees_per_frame": float(
                            backend.ANGULAR_LIMIT_DEGREES_PER_FRAME
                        ),
                        "root_linear_xz_command_substeps": list(
                            env.root_linear_xz_command_substeps
                        ),
                        "human_child_link_count": len(link_frictions),
                        "human_child_link_friction_range": [
                            min(link_frictions),
                            max(link_frictions),
                        ],
                    },
                    "video_frames_per_view": expected_frames,
                    "output": str(output),
                    "metrics": result,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        if not video_closed:
            video.close()
        env.close()


if __name__ == "__main__":
    main()

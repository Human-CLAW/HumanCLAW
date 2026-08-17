#!/usr/bin/env python3
"""Replay saved generated motion through Half-Physics and compare its output.

This is a manual integration check because it requires the prepared HSSD val
assets and the release Habitat-Sim build.  It does not call a VLM or load the
motion generator: ``trajectory_before.npz`` is the recorded motion-generator
output, and ``trajectory_after.npz`` is the reference physics result.

Example:

    PYTHONPATH=src python tests/runtime_forward_replay.py \
        outputs/<batch>/<episode>/ep_<id>_<category>/rollout_00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from humanclaw_bench.benchmark.episodes import (
    apply_instruction_version,
    load_episode,
)
from humanclaw_bench.envs.find_nav_interact_env import HCFindNavInteractEnv
from humanclaw_bench.evaluation.replay import apply_agent_pose, apply_object_pose
from humanclaw_bench.paths import resolve_release_path
from humanclaw_bench.rendering.saved_trajectory import (
    environment_kwargs,
    load_render_contract,
)


def _archive(path: Path) -> dict[str, np.ndarray]:
    """Load one non-pickled NPZ into ordinary arrays before opening Habitat."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _names(values: Any) -> list[str]:
    """Convert an NPZ string vector to stable Python strings."""

    return [str(value) for value in np.asarray(values).tolist()]


def _set_vector(runtime: Any, value: np.ndarray) -> Any:
    """Convert a saved xyz array to Habitat's Magnum vector type."""

    xyz = np.asarray(value, dtype=np.float64).reshape(3)
    return runtime.mn.Vector3(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _same_value(actual: Any, expected: Any, *, atol: float = 1.0e-7) -> bool:
    """Return whether two finite saved-state arrays already agree."""

    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    return left.shape == right.shape and bool(
        np.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True)
    )


def _same_quaternion(actual: Any, expected: Any, *, atol: float = 1.0e-7) -> bool:
    """Compare one quaternion while accepting Habitat's equivalent sign."""

    left = np.asarray(actual, dtype=np.float64).reshape(4)
    right = np.asarray(expected, dtype=np.float64).reshape(4)
    return _same_value(left, right, atol=atol) or _same_value(
        left, -right, atol=atol
    )


def _restore_initial_state(env: Any, before: dict[str, np.ndarray]) -> None:
    """Restore only saved state that differs from the freshly loaded scene.

    Reassigning an already-identical rigid pose can unnecessarily invalidate
    Bullet's initial broad-phase/contact cache.  The explicit release scene
    normally loads at the saved state, so preserving it is the most exact
    replay; the assignments below are a fallback for portable scene loaders.
    """

    runtime = env._require_runtime()
    current = env.replay_initial_state()
    current_human = dict(current["human"])
    pose_fields = (
        ("transl", "initial_human_transl"),
        ("global_orient", "initial_human_global_orient"),
        ("body_pose", "initial_human_body_pose"),
    )
    if not all(
        _same_value(current_human[field], before[key]) for field, key in pose_fields
    ):
        apply_agent_pose(
            env,
            np.asarray(before["initial_human_transl"], dtype=np.float32),
            np.asarray(before["initial_human_global_orient"], dtype=np.float32),
            np.asarray(before["initial_human_body_pose"], dtype=np.float32),
        )
    if not _same_value(
        current_human["root_linear_velocity"],
        before["initial_human_root_linear_velocity"],
    ):
        env.agent.root_linear_velocity = _set_vector(
            runtime, before["initial_human_root_linear_velocity"]
        )
    if not _same_value(
        current_human["root_angular_velocity"],
        before["initial_human_root_angular_velocity"],
    ):
        env.agent.root_angular_velocity = _set_vector(
            runtime, before["initial_human_root_angular_velocity"]
        )
    joint_velocity = np.asarray(
        before["initial_human_joint_velocities"], dtype=np.float64
    ).reshape(-1)
    if joint_velocity.size and not _same_value(
        current_human["joint_velocities"], joint_velocity
    ):
        env.agent.joint_velocities = joint_velocity.tolist()

    tracked = env._tracked_dynamic_objects()
    names = _names(before["initial_object_names"])
    expected_ids = np.asarray(before["initial_object_ids"], dtype=np.int64)
    positions = np.asarray(before["initial_object_position"], dtype=np.float32)
    rotations = np.asarray(before["initial_object_rotation"], dtype=np.float32)
    linear = np.asarray(before["initial_object_linear_velocity"], dtype=np.float32)
    angular = np.asarray(before["initial_object_angular_velocity"], dtype=np.float32)
    if set(names) != set(tracked):
        missing = sorted(set(names) - set(tracked))
        extra = sorted(set(tracked) - set(names))
        raise RuntimeError(f"Dynamic-object identity mismatch: missing={missing}, extra={extra}")

    for index, name in enumerate(names):
        obj = tracked[name]
        current_object = dict(current["objects"][name])
        actual_id = int(getattr(obj, "object_id", -1))
        if actual_id != int(expected_ids[index]):
            raise RuntimeError(
                f"Dynamic-object ID mismatch for {name}: {actual_id} != {expected_ids[index]}"
            )
        if not (
            _same_value(current_object["position"], positions[index])
            and _same_quaternion(current_object["rotation"], rotations[index])
        ):
            if not apply_object_pose(env, obj, positions[index], rotations[index]):
                raise RuntimeError(f"Initial object pose is non-finite: {name}")
        if not _same_value(current_object["linear_velocity"], linear[index]):
            obj.linear_velocity = _set_vector(runtime, linear[index])
        if not _same_value(current_object["angular_velocity"], angular[index]):
            obj.angular_velocity = _set_vector(runtime, angular[index])


def _finite_max_abs(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    """Return maximum absolute error while requiring identical finite masks."""

    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise RuntimeError(f"{label} shape mismatch: {actual.shape} != {expected.shape}")
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    if not np.array_equal(actual_finite, expected_finite):
        raise RuntimeError(f"{label} finite-value mask differs")
    if not actual_finite.any():
        return 0.0
    return float(np.max(np.abs(actual[actual_finite] - expected[expected_finite])))


def _quaternion_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Measure xyzw quaternion error without treating q and -q as different."""

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.shape[-1] != 4:
        raise RuntimeError(
            f"Quaternion shape mismatch: {actual.shape} != {expected.shape}"
        )
    finite = np.isfinite(actual).all(axis=-1) & np.isfinite(expected).all(axis=-1)
    if not finite.any():
        return 0.0
    direct = np.max(np.abs(actual[finite] - expected[finite]), axis=-1)
    negated = np.max(np.abs(actual[finite] + expected[finite]), axis=-1)
    return float(np.max(np.minimum(direct, negated)))


def replay_and_compare(
    rollout_dir: Path,
    *,
    max_steps: int,
    pose_atol: float,
    object_atol: float,
) -> dict[str, Any]:
    """Forward-replay saved chunks and fail if any recorded state diverges."""

    contract = load_render_contract(rollout_dir)
    before = _archive(rollout_dir / "trajectory_before.npz")
    after = _archive(rollout_dir / "trajectory_after.npz")
    kwargs = environment_kwargs(contract)
    # Match the original optional sensors/contact queries.  They should be
    # read-only with respect to Bullet, but matching the run makes this test a
    # stronger end-to-end determinism check than a physics-only approximation.
    kwargs["video_enabled"] = (rollout_dir / "ego.mp4").is_file()
    kwargs["compute_metrics"] = (rollout_dir / "metrics.json").is_file()
    benchmark = dict(contract.profile.get("benchmark") or {})
    episode = load_episode(
        benchmark_dataset_dir=resolve_release_path(benchmark["dataset_dir"]),
        split=str(benchmark["split"]),
        scene_id=str(contract.episode["scene_label"]),
        scene_dataset_config=Path(kwargs["scene_dataset_config"]),
        episode_id=str(contract.episode["episode_id"]),
        object_category=str(contract.episode["object_category"]),
        max_steps=int(contract.episode.get("max_steps", 100)),
    )
    episode = apply_instruction_version(
        episode, str(benchmark.get("instruction_version", "v0"))
    )
    env = HCFindNavInteractEnv(**kwargs)

    starts = np.asarray(before["step_starts"], dtype=np.int64)
    lengths = np.asarray(before["step_lengths"], dtype=np.int64)
    after_starts = np.asarray(after["step_starts"], dtype=np.int64)
    after_lengths = np.asarray(after["step_lengths"], dtype=np.int64)
    if not (np.array_equal(starts, after_starts) and np.array_equal(lengths, after_lengths)):
        raise RuntimeError("Before/after step boundaries differ")
    selected_steps = len(starts) if max_steps <= 0 else min(max_steps, len(starts))

    max_errors = {
        "human_translation_m": 0.0,
        "human_global_orient_rad": 0.0,
        "human_body_pose_rad": 0.0,
        "object_translation_m": 0.0,
        "object_quaternion_component": 0.0,
    }
    max_error_locations: dict[str, str] = {}
    try:
        env.reset(
            episode,
            initial_transl=before["initial_human_transl"],
            initial_global_orient=before["initial_human_global_orient"],
            initial_body_pose=before["initial_human_body_pose"],
        )
        _restore_initial_state(env, before)
        reference_names = _names(after["object_names"])

        for step_index in range(selected_steps):
            start = int(starts[step_index])
            stop = start + int(lengths[step_index])
            motion = np.asarray(before["xb_world_75"][start:stop], dtype=np.float32)
            _observation, _reward, _done, info = env.step(motion, reasoning={})
            body = dict(info["body_state"])
            max_errors["human_translation_m"] = max(
                max_errors["human_translation_m"],
                _finite_max_abs(body["transl"], after["transl"][start:stop], "transl"),
            )
            max_errors["human_global_orient_rad"] = max(
                max_errors["human_global_orient_rad"],
                _finite_max_abs(
                    body["global_orient"],
                    after["global_orient"][start:stop],
                    "global_orient",
                ),
            )
            max_errors["human_body_pose_rad"] = max(
                max_errors["human_body_pose_rad"],
                _finite_max_abs(
                    body["body_pose"], after["body_pose"][start:stop], "body_pose"
                ),
            )

            objects = dict(info["object_states"])
            if set(objects) != set(reference_names):
                raise RuntimeError("Replayed dynamic-object set differs from reference")
            for object_index, name in enumerate(reference_names):
                state = dict(objects[name])
                position_error = _finite_max_abs(
                    state["position"],
                    after[f"object_{object_index:03d}_position"][start:stop],
                    f"{name}.position",
                )
                if position_error > max_errors["object_translation_m"]:
                    max_errors["object_translation_m"] = position_error
                    max_error_locations["object_translation_m"] = (
                        f"step={step_index}, object={name}"
                    )
                rotation_error = _quaternion_error(
                    state["rotation"],
                    after[f"object_{object_index:03d}_rotation"][start:stop],
                )
                if rotation_error > max_errors["object_quaternion_component"]:
                    max_errors["object_quaternion_component"] = rotation_error
                    max_error_locations["object_quaternion_component"] = (
                        f"step={step_index}, object={name}"
                    )
    finally:
        env.close()

    pose_error = max(
        max_errors["human_translation_m"],
        max_errors["human_global_orient_rad"],
        max_errors["human_body_pose_rad"],
    )
    object_error = max(
        max_errors["object_translation_m"],
        max_errors["object_quaternion_component"],
    )
    if pose_error > pose_atol or object_error > object_atol:
        raise RuntimeError(
            f"Forward replay exceeded tolerance: errors={max_errors}, "
            f"locations={max_error_locations}, "
            f"pose_atol={pose_atol}, object_atol={object_atol}"
        )
    return {
        "status": "ok",
        "rollout_dir": str(rollout_dir),
        "replayed_steps": selected_steps,
        "replayed_frames": int(lengths[:selected_steps].sum()),
        "dynamic_objects": len(_names(after["object_names"])),
        "max_errors": max_errors,
        "max_error_locations": max_error_locations,
        "pose_atol": pose_atol,
        "object_atol": object_atol,
    }


def main() -> None:
    """Parse a saved rollout path, execute replay, and print its audit record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Motion chunks to replay; <=0 checks the complete trajectory.",
    )
    parser.add_argument(
        "--pose-atol",
        type=float,
        default=1.0e-4,
        help="Maximum pose/translation component error (default: 1e-4).",
    )
    parser.add_argument("--object-atol", type=float, default=5.0e-5)
    args = parser.parse_args()
    result = replay_and_compare(
        args.rollout_dir.expanduser().resolve(),
        max_steps=args.max_steps,
        pose_atol=args.pose_atol,
        object_atol=args.object_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Fast ego/exo rendering from ``trajectory_after.npz``.

The Habitat scene is loaded once.  Each output frame is produced by assigning
the recorded humanoid and every recorded dynamic-object pose, updating the two
cameras, and asking Habitat to rasterize.  No VLM, motion model, physics step,
contact query, semantic render, or image-frame directory is involved.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from humanclaw_bench.config import load_config
from humanclaw_bench.evaluation.replay import (
    apply_agent_pose,
    apply_object_pose,
    object_pose_arrays,
)
from humanclaw_bench.evaluation.video import RolloutVideoWriter
from humanclaw_bench.paths import resolve_release_path


@dataclass(frozen=True)
class RenderContract:
    """Resolved episode/runtime description used only to construct the scene."""

    source: Path
    schema: str
    profile: dict[str, Any]
    episode: dict[str, Any]
    physics: dict[str, Any]
    rendering: dict[str, Any]
    assets: dict[str, Any]


@dataclass(frozen=True)
class SavedAfterTrajectory:
    """Validated in-memory view of one post-physics trajectory."""

    path: Path
    transl: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    frame_step: np.ndarray | None
    object_poses: dict[str, tuple[np.ndarray, np.ndarray]]
    fps: float

    @property
    def frame_count(self) -> int:
        """Return the number of frame-aligned post-physics human poses."""

        return int(self.transl.shape[0])


def _read_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file and require an object at its root."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_render_contract(rollout_dir: str | Path) -> RenderContract:
    """Load the self-contained release replay manifest for one rollout."""

    directory = Path(rollout_dir).expanduser().resolve()
    replay_manifest = directory / "replay_manifest.json"
    if not replay_manifest.is_file():
        raise FileNotFoundError(f"Replay manifest not found: {replay_manifest}")
    value = _read_json_object(replay_manifest)
    if value.get("schema") != "humanclaw_replay_v1":
        raise ValueError(f"Unsupported replay manifest schema: {value.get('schema')}")
    profile_ref = value.get("profile", "paper_fullval_v1")
    profile = (
        dict(profile_ref)
        if isinstance(profile_ref, dict)
        else dict(load_config(str(profile_ref)).data)
    )
    physics = {
        **dict(profile.get("physics") or {}),
        **dict(value.get("physics") or {}),
    }
    rendering = {
        **dict(profile.get("rendering") or {}),
        **dict(value.get("rendering") or {}),
    }
    return RenderContract(
        source=replay_manifest,
        schema=str(value["schema"]),
        profile=profile,
        episode=dict(value.get("episode") or {}),
        physics=physics,
        rendering=rendering,
        assets=dict(value.get("assets") or {}),
    )


def _identity_path(value: Any) -> Any:
    """Extract the path field from an asset-identity object when necessary."""

    return value.get("path") if isinstance(value, dict) else value


def _path_candidate(value: Any) -> Path | None:
    """Resolve an optional absolute or release-relative asset path."""

    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else resolve_release_path(path)


def _require_existing(label: str, *values: Any) -> Path:
    """Return the first existing candidate for a required render asset."""

    attempted: list[str] = []
    for value in values:
        path = _path_candidate(value)
        if path is None:
            continue
        attempted.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve {label}; tried: {attempted}")


def environment_kwargs(contract: RenderContract) -> dict[str, Any]:
    """Resolve the minimal environment settings needed for RGB rendering."""

    benchmark = dict(contract.profile.get("benchmark") or {})
    physics = contract.physics
    rendering = contract.rendering
    episode = contract.episode
    assets = contract.assets

    scene_dataset_config = _require_existing(
        "scene dataset config",
        _identity_path(assets.get("scene_dataset_config")),
        episode.get("scene_dataset_config"),
        benchmark.get("scene_dataset_config"),
    )
    scene_label = str(episode.get("scene_label") or "").strip()
    raw_scene_id = str(episode.get("scene_id") or "").strip()
    if not scene_label and raw_scene_id:
        scene_label = Path(raw_scene_id).name.split(".scene_instance", 1)[0]
    portable_scene = (
        scene_dataset_config.parent / "scenes" / f"{scene_label}.scene_instance.json"
        if scene_label
        else None
    )
    scene_id = _require_existing(
        "scene instance",
        _identity_path(assets.get("scene_instance")),
        raw_scene_id,
        portable_scene,
    )

    return {
        "scene_id": str(scene_id),
        "scene_dataset_config": scene_dataset_config,
        "physics_config": _require_existing(
            "physics config",
            _identity_path(assets.get("physics_config")),
            physics.get("physics_config"),
        ),
        "agent_urdf": _require_existing(
            "agent URDF",
            _identity_path(assets.get("agent_urdf")),
            physics.get("agent_urdf"),
        ),
        "agent_shift_npy": _require_existing(
            "agent shift",
            _identity_path(assets.get("agent_shift_npy")),
            physics.get("agent_shift_npy"),
        ),
        "half_physics_backend": str(physics.get("backend", "hp")),
        "max_episode_steps": max(1, int(episode.get("max_steps", 100) or 100)),
        "fps": float(physics.get("fps", 30.0)),
        "root_gravity_scale": float(physics.get("root_gravity_scale", 1.0)),
        "root_gravity_mode": str(physics.get("root_gravity_mode", "midpoint")),
        "inherit_downward_root_y_velocity": bool(
            physics.get("inherit_downward_root_y_velocity", True)
        ),
        "pjsc_lambda": float(physics.get("pjsc_lambda", 1.0)),
        "pjsc_lambda_by_link": dict(physics.get("pjsc_lambda_by_link") or {}),
        "pjsc_substeps": int(physics.get("pjsc_substeps", 4)),
        "root_linear_xz_command_substeps": tuple(
            physics.get("root_linear_xz_command_substeps", (0, 2))
        ),
        "friction": float(physics.get("friction", 0.4)),
        "lighting": str(rendering.get("lighting", "ambient")),
        "ambient_strength": float(rendering.get("ambient_strength", 1.2)),
        "room_light_strength": float(rendering.get("room_light_strength", 1.0)),
        "ego_resolution": tuple(rendering.get("ego_resolution", (448, 448))),
        "third_person_resolution": tuple(
            rendering.get("third_person_resolution", (512, 512))
        ),
        # The exo sensor is needed, but metric-only sensors and contact traces
        # remain disabled.
        "video_enabled": True,
        "compute_metrics": False,
    }


def load_saved_after_trajectory(
    path: str | Path,
    *,
    fallback_fps: float,
) -> SavedAfterTrajectory:
    """Load and shape-check one compact post-physics NPZ."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Saved trajectory not found: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        after = {name: archive[name] for name in archive.files}

    required = ("transl", "global_orient", "body_pose")
    missing = [name for name in required if name not in after]
    if missing:
        raise ValueError(f"{resolved} is missing arrays: {missing}")
    transl = np.asarray(after["transl"], dtype=np.float32)
    orient = np.asarray(after["global_orient"], dtype=np.float32)
    pose = np.asarray(after["body_pose"], dtype=np.float32)
    frame_count = int(transl.shape[0]) if transl.ndim else 0
    if transl.shape != (frame_count, 3):
        raise ValueError(f"transl must have shape (T, 3), got {transl.shape}")
    if orient.shape != (frame_count, 3):
        raise ValueError(f"global_orient must have shape (T, 3), got {orient.shape}")
    if pose.shape != (frame_count, 54, 3):
        raise ValueError(f"body_pose must have shape (T, 54, 3), got {pose.shape}")
    if frame_count == 0:
        raise ValueError(f"Saved trajectory has no frames: {resolved}")
    if not (
        np.isfinite(transl).all()
        and np.isfinite(orient).all()
        and np.isfinite(pose).all()
    ):
        raise ValueError(
            f"Saved humanoid trajectory contains non-finite values: {resolved}"
        )

    frame_step: np.ndarray | None = None
    if "frame_step" in after:
        frame_step = np.asarray(after["frame_step"], dtype=np.int32)
        if frame_step.shape != (frame_count,):
            raise ValueError(
                f"frame_step must have shape ({frame_count},), got {frame_step.shape}"
            )
    objects = object_pose_arrays(after)
    for name, (positions, rotations) in objects.items():
        if positions.shape != (frame_count, 3):
            raise ValueError(
                f"dynamic object {name!r} position shape is {positions.shape}"
            )
        if rotations.shape != (frame_count, 4):
            raise ValueError(
                f"dynamic object {name!r} rotation shape is {rotations.shape}"
            )
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError(f"dynamic object {name!r} has non-finite saved poses")

    fps = float(np.asarray(after.get("fps", fallback_fps)).reshape(()))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid saved trajectory fps: {fps}")
    return SavedAfterTrajectory(
        path=resolved,
        transl=transl,
        global_orient=orient,
        body_pose=pose,
        frame_step=frame_step,
        object_poses=objects,
        fps=fps,
    )


def _rgb(value: Any) -> np.ndarray:
    """Normalize a Habitat sensor image to contiguous uint8 RGB."""

    rgb = np.asarray(value)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB sensor returned invalid shape: {rgb.shape}")
    return np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)


def _fingerprint(path: Path) -> dict[str, Any]:
    """Build a stable hashable summary of a completed render job."""

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _completed_report(
    output_dir: Path,
    *,
    source: dict[str, Any],
    preset: str,
    crf: int,
) -> dict[str, Any] | None:
    """Load a prior render report only when it represents a complete matching job."""

    report_path = output_dir / "render_report.json"
    try:
        report = _read_json_object(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    videos = (output_dir / "ego.mp4", output_dir / "exo.mp4")
    if not all(path.is_file() and path.stat().st_size > 0 for path in videos):
        return None
    if not (
        report.get("schema") == "humanclaw_saved_trajectory_render_v1"
        and report.get("source_trajectory") == source
        and report.get("encoder", {}).get("preset") == preset
        and report.get("encoder", {}).get("crf") == crf
    ):
        return None
    return report


def render_saved_trajectory(
    rollout_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    trajectory_path: str | Path | None = None,
    preset: str = "veryfast",
    crf: int = 20,
    force: bool = False,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Render synchronized ego/exo MP4s directly from saved after poses.

    Habitat/Bullet is initialized to load the articulated agent and scene, but
    simulation time is never advanced.  Each frame is a direct state restore
    followed by a two-camera RGB render.
    """

    rollout = Path(rollout_dir).expanduser().resolve()
    if not rollout.is_dir():
        raise NotADirectoryError(f"Rollout directory not found: {rollout}")
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else rollout / "rendered"
    )
    trajectory_file = (
        Path(trajectory_path).expanduser().resolve()
        if trajectory_path is not None
        else rollout / "trajectory_after.npz"
    )
    if not 0 <= int(crf) <= 51:
        raise ValueError(f"H.264 CRF must be in [0, 51], got {crf}")

    contract = load_render_contract(rollout)
    kwargs = environment_kwargs(contract)
    trajectory = load_saved_after_trajectory(
        trajectory_file,
        fallback_fps=float(kwargs["fps"]),
    )
    if abs(trajectory.fps - float(kwargs["fps"])) > 1e-4:
        raise ValueError(
            f"Trajectory fps {trajectory.fps} disagrees with contract fps {kwargs['fps']}"
        )
    source = _fingerprint(trajectory.path)
    if not force:
        existing = _completed_report(
            output,
            source=source,
            preset=str(preset),
            crf=int(crf),
        )
        if existing is not None:
            return {**existing, "status": "already_complete"}

    output.mkdir(parents=True, exist_ok=True)
    # Imported lazily so listing CLI help and validating manifests do not load
    # Habitat-Sim, Magnum, or the HalfPhysics backend.
    from humanclaw_bench.envs.find_nav_interact_env import HCFindNavInteractEnv

    env = HCFindNavInteractEnv(**kwargs)
    writer = RolloutVideoWriter(
        output,
        trajectory.fps,
        preset=str(preset),
        crf=int(crf),
    )
    started = time.perf_counter()
    render_error: BaseException | None = None
    try:
        env.reset(
            contract.episode,
            initial_transl=trajectory.transl[0],
            initial_global_orient=trajectory.global_orient[0],
            initial_body_pose=trajectory.body_pose[0],
        )
        tracked = env._tracked_dynamic_objects()
        saved_names = set(trajectory.object_poses)
        tracked_names = set(tracked)
        if saved_names != tracked_names:
            raise RuntimeError(
                "Saved/runtime dynamic-object sets differ: "
                f"missing_from_scene={sorted(saved_names - tracked_names)}, "
                f"missing_from_trajectory={sorted(tracked_names - saved_names)}"
            )

        for frame in range(trajectory.frame_count):
            for name, (positions, rotations) in trajectory.object_poses.items():
                if not apply_object_pose(
                    env,
                    tracked[name],
                    positions[frame],
                    rotations[frame],
                ):
                    raise RuntimeError(
                        f"Non-finite dynamic-object pose: {name}, frame {frame}"
                    )
            apply_agent_pose(
                env,
                trajectory.transl[frame],
                trajectory.global_orient[frame],
                trajectory.body_pose[frame],
            )
            env._update_cameras()
            observations = env.sim.get_sensor_observations()
            writer.append(
                _rgb(observations["ego_rgb"]),
                _rgb(observations["third_person_rgb"]),
            )
            rendered = frame + 1
            if progress_every > 0 and (
                rendered % progress_every == 0 or rendered == trajectory.frame_count
            ):
                print(
                    f"rendered {rendered}/{trajectory.frame_count}",
                    file=sys.stderr,
                    flush=True,
                )
    except BaseException as exc:  # preserve the rendering error after cleanup
        render_error = exc
    finally:
        try:
            writer.close()
        except BaseException as exc:
            if render_error is None:
                render_error = exc
        finally:
            env.close()
    if render_error is not None:
        raise render_error
    if (
        writer.ego.frame_count != trajectory.frame_count
        or writer.exo.frame_count != trajectory.frame_count
    ):
        raise RuntimeError(
            "Video frame mismatch: "
            f"ego={writer.ego.frame_count}, exo={writer.exo.frame_count}, "
            f"expected={trajectory.frame_count}"
        )

    elapsed = time.perf_counter() - started
    report = {
        "schema": "humanclaw_saved_trajectory_render_v1",
        "status": "complete",
        "source_rollout_dir": str(rollout),
        "source_trajectory": source,
        "source_contract": str(contract.source.resolve()),
        "contract_schema": contract.schema,
        "method": (
            "Direct per-frame restoration of the saved post-physics humanoid "
            "and every dynamic object, followed by ego/exo RGB rasterization."
        ),
        "physics_steps": 0,
        "vlm_calls": 0,
        "motion_generation_calls": 0,
        "contact_queries": 0,
        "semantic_renders": 0,
        "frames": trajectory.frame_count,
        "fps": trajectory.fps,
        "duration_s": trajectory.frame_count / trajectory.fps,
        "elapsed_s": elapsed,
        "rendered_frames_per_second": trajectory.frame_count / max(elapsed, 1e-9),
        "dynamic_objects_restored_per_frame": len(trajectory.object_poses),
        "ego_resolution": list(env.ego_camera.resolution),
        "exo_resolution": list(env.third_person_camera.resolution),
        "encoder": {"codec": "libx264", "preset": str(preset), "crf": int(crf)},
        "ego_video": str((output / "ego.mp4").resolve()),
        "exo_video": str((output / "exo.mp4").resolve()),
    }
    (output / "render_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "RenderContract",
    "SavedAfterTrajectory",
    "environment_kwargs",
    "load_render_contract",
    "load_saved_after_trajectory",
    "render_saved_trajectory",
]

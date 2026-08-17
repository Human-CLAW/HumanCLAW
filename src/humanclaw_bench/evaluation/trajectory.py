"""Compact before/after trajectories for deterministic HalfPhysics replay."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from humanclaw_bench.envs.half_physics_env import HALF_PHYSICS_ASSET_DIR


def _sha256(path: Path) -> str:
    """Stream a file through SHA-256 without loading the whole asset into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(value: Any) -> dict[str, Any]:
    """Describe a replay asset by path, byte size, and SHA-256 when present."""

    text = str(value or "")
    if not text:
        return {"path": ""}
    path = Path(text).expanduser()
    identity: dict[str, Any] = {"path": str(path)}
    if path.is_file():
        resolved = path.resolve()
        identity.update(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    return identity


def build_replay_metadata(
    *,
    profile_name: str,
    episode: Any,
    rollout_index: int,
    env: Any,
) -> dict[str, Any]:
    """Describe only the episode, assets, and physics needed for replay."""

    backend = str(getattr(env, "half_physics_backend", ""))
    assets = {
        "scene_dataset_config": _file_identity(
            getattr(
                env,
                "scene_dataset_config",
                getattr(episode, "scene_dataset_config", ""),
            )
        ),
        "scene_instance": _file_identity(
            getattr(env, "scene_id", getattr(episode, "scene_id", ""))
        ),
        "physics_config": _file_identity(getattr(env, "physics_config", "")),
        "agent_urdf": _file_identity(getattr(env, "agent_urdf", "")),
        "agent_shift_npy": _file_identity(getattr(env, "agent_shift_npy", "")),
        "half_physics_backend": _file_identity(
            HALF_PHYSICS_ASSET_DIR / f"{backend}.py"
        ),
    }
    ego_camera = getattr(env, "ego_camera", None)
    third_person_camera = getattr(env, "third_person_camera", None)
    return {
        "schema": "humanclaw_replay_v1",
        "profile": str(profile_name),
        "episode": {
            "episode_id": str(getattr(episode, "episode_id", "")),
            "scene_id": str(getattr(episode, "scene_id", "")),
            "scene_label": str(getattr(episode, "scene_label", "")),
            "scene_dataset_config": str(getattr(episode, "scene_dataset_config", "")),
            "object_category": str(getattr(episode, "object_category", "")),
            "init_offset": list(getattr(episode, "init_offset", (0.0, 0.0, 0.0))),
            "init_yaw": float(getattr(episode, "init_yaw", 0.0)),
            "max_steps": int(getattr(episode, "max_steps", 0)),
            "rollout_index": int(rollout_index),
        },
        "physics": {
            "backend": backend,
            "fps": float(getattr(env, "fps", 30.0)),
            "root_gravity_scale": float(getattr(env, "root_gravity_scale", 1.0)),
            "root_gravity_mode": str(getattr(env, "root_gravity_mode", "midpoint")),
            "inherit_downward_root_y_velocity": bool(
                getattr(env, "inherit_downward_root_y_velocity", True)
            ),
            "pjsc_lambda": float(getattr(env, "pjsc_lambda", 1.0)),
            "pjsc_lambda_by_link": dict(getattr(env, "pjsc_lambda_by_link", {}) or {}),
            "pjsc_substeps": int(getattr(env, "pjsc_substeps", 4)),
            "root_linear_xz_command_substeps": list(
                getattr(env, "root_linear_xz_command_substeps", (0, 2))
            ),
            "friction": float(getattr(env, "friction", 0.4)),
        },
        # Pin camera/light settings as part of replay metadata.  A delayed
        # renderer can therefore reproduce the rollout view even if a named
        # release profile is edited later.
        "rendering": {
            "lighting": str(getattr(env, "lighting", "ambient")),
            "ambient_strength": float(getattr(env, "ambient_strength", 1.2)),
            "room_light_strength": float(getattr(env, "room_light_strength", 1.0)),
            "ego_resolution": list(getattr(ego_camera, "resolution", (448, 448))),
            "third_person_resolution": list(
                getattr(third_person_camera, "resolution", (512, 512))
            ),
        },
        "assets": assets,
        "coordinate_frames": {
            "trajectory_before": "motion-generator Y-up SMPL-X",
            "trajectory_after": "HalfPhysics trajectory coordinates (Y-up SMPL-X)",
            "object_rotation": "xyzw quaternion",
        },
        "replay": {
            "input": "trajectory_before.npz",
            "reference_output": "trajectory_after.npz",
            "initial_state_location": "trajectory_before.npz",
        },
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy values to JSON-serializable Python values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _action_metadata(action: Any, action_text: str) -> dict[str, Any]:
    """Serialize the skill name, display text, and condition for one agent step."""

    condition = getattr(action, "cond", None)
    numeric_condition = (
        float(condition)
        if isinstance(condition, (int, float, np.number))
        and not isinstance(condition, (bool, np.bool_))
        else np.nan
    )
    return {
        "action_skill": str(getattr(action, "skill", "")),
        "action_text": str(action_text),
        "action_cond": numeric_condition,
        "action_cond_json": json.dumps(
            _jsonable(condition), ensure_ascii=False, separators=(",", ":")
        ),
    }


def _step_arrays(
    records: list[dict[str, Any]],
    length_key: str,
    *,
    include_actions: bool,
) -> dict[str, np.ndarray]:
    """Build compact step-boundary arrays for concatenated frame trajectories."""

    lengths = np.asarray([record[length_key] for record in records], dtype=np.int32)
    starts = (
        np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(lengths[:-1])]
        ).astype(np.int32)
        if records
        else np.zeros(0, dtype=np.int32)
    )
    arrays = {
        "step_indices": np.asarray(
            [record["step"] for record in records], dtype=np.int32
        ),
        "step_starts": starts,
        "step_lengths": lengths,
    }
    if not include_actions:
        return arrays
    arrays.update(
        {
            "step_action_skill": np.asarray(
                [record["action_skill"] for record in records]
            ),
            "step_action_text": np.asarray(
                [record["action_text"] for record in records]
            ),
            "step_action_cond": np.asarray(
                [record["action_cond"] for record in records], dtype=np.float32
            ),
            "step_action_cond_json": np.asarray(
                [record["action_cond_json"] for record in records]
            ),
        }
    )
    return arrays


class TrajectoryRecorder:
    """Collect compact replay inputs and post-physics reference states."""

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        initial_xb_world_75: Any = None,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        """Initialize compact pre-physics and post-physics trajectory buffers."""

        self.metadata = copy.deepcopy(metadata)
        self.initial_xb_world_75 = (
            np.asarray(initial_xb_world_75, dtype=np.float32).reshape(75)
            if initial_xb_world_75 is not None
            else np.zeros((0, 75), dtype=np.float32)
        )
        self.initial_state = dict(initial_state or {})
        self.before_records: list[dict[str, Any]] = []
        self.after_records: list[dict[str, Any]] = []
        # Concatenating long motion chunks is not free.  Metrics and trajectory
        # serialization share these two materialized dictionaries so each
        # episode performs the concatenation exactly once.
        self._materialized: tuple[dict[str, Any], dict[str, Any]] | None = None

    @property
    def fps(self) -> float:
        """Return the physics frame rate pinned in replay metadata."""

        return float((self.metadata.get("physics") or {}).get("fps", 30.0))

    def record_before(
        self,
        *,
        step: int,
        action: Any,
        action_text: str,
        xb_world_75: Any,
    ) -> None:
        """Append generated motion frames and action metadata before Half-Physics."""

        xb = np.asarray(xb_world_75, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != 75:
            raise ValueError(f"xb_world_75 must have shape (T, 75), got {xb.shape}")
        self._materialized = None
        self.before_records.append(
            {
                "step": int(step),
                "length": int(xb.shape[0]),
                "xb_world_75": xb,
                **_action_metadata(action, action_text),
            }
        )

    def record_after(
        self,
        *,
        step: int,
        body_state: dict[str, Any],
        object_states: dict[str, Any],
    ) -> None:
        """Append realized human and dynamic-object states after Half-Physics."""

        transl = np.asarray(body_state.get("transl"), dtype=np.float32)
        orient = np.asarray(body_state.get("global_orient"), dtype=np.float32)
        pose = np.asarray(body_state.get("body_pose"), dtype=np.float32)
        if transl.ndim != 2 or transl.shape[1] != 3:
            raise ValueError(f"after transl must have shape (T, 3), got {transl.shape}")
        if orient.shape != transl.shape:
            raise ValueError("after global_orient must match transl")
        if pose.shape != (transl.shape[0], 54, 3):
            raise ValueError(
                "after body_pose must have shape "
                f"({transl.shape[0]}, 54, 3), got {pose.shape}"
            )

        objects: dict[str, dict[str, np.ndarray]] = {}
        for name, state in (object_states or {}).items():
            # Every dynamic object must have exactly one pose per realized
            # human frame.  Rejecting partial arrays here prevents a renderer
            # from silently pairing poses from different times.
            position = np.asarray(state.get("position"), dtype=np.float32)
            rotation = np.asarray(state.get("rotation"), dtype=np.float32)
            if position.shape != (transl.shape[0], 3):
                raise ValueError(
                    f"dynamic object {name!r} position has shape {position.shape}"
                )
            if rotation.shape != (transl.shape[0], 4):
                raise ValueError(
                    f"dynamic object {name!r} rotation has shape {rotation.shape}"
                )
            objects[str(name)] = {"position": position, "rotation": rotation}

        self._materialized = None
        self.after_records.append(
            {
                "step": int(step),
                "length": int(transl.shape[0]),
                "transl": transl,
                "global_orient": orient,
                "body_pose": pose,
                "objects": objects,
            }
        )

    def _initial_arrays(self) -> dict[str, np.ndarray]:
        """Serialize the exact reset human/object state needed for deterministic replay."""

        human = dict(self.initial_state.get("human") or {})
        objects = dict(self.initial_state.get("objects") or {})
        names = sorted(str(name) for name in objects)

        def human_array(key: str, empty_shape: tuple[int, ...]) -> np.ndarray:
            """Shape-check one initial human vector or return an empty typed array."""

            value = human.get(key)
            if value is None:
                return np.zeros(empty_shape, dtype=np.float32)
            return np.asarray(value, dtype=np.float32)

        def object_stack(key: str, width: int) -> np.ndarray:
            """Stack one named initial-object field in stable object-name order."""

            if not names:
                return np.zeros((0, width), dtype=np.float32)
            return np.stack(
                [np.asarray(objects[name][key], dtype=np.float32) for name in names]
            )

        return {
            "initial_xb_world_75": self.initial_xb_world_75,
            "initial_human_transl": human_array("transl", (0, 3)),
            "initial_human_global_orient": human_array("global_orient", (0, 3)),
            "initial_human_body_pose": human_array("body_pose", (0, 54, 3)),
            "initial_human_root_linear_velocity": human_array(
                "root_linear_velocity", (0, 3)
            ),
            "initial_human_root_angular_velocity": human_array(
                "root_angular_velocity", (0, 3)
            ),
            "initial_human_joint_velocities": human_array("joint_velocities", (0,)),
            "initial_object_names": np.asarray(names),
            "initial_object_ids": np.asarray(
                [int(objects[name].get("object_id", -1)) for name in names],
                dtype=np.int32,
            ),
            "initial_object_motion_types": np.asarray(
                [str(objects[name].get("motion_type", "")) for name in names]
            ),
            "initial_object_position": object_stack("position", 3),
            "initial_object_rotation": object_stack("rotation", 4),
            "initial_object_linear_velocity": object_stack("linear_velocity", 3),
            "initial_object_angular_velocity": object_stack("angular_velocity", 3),
        }

    def _before_arrays(self) -> dict[str, Any]:
        """Materialize the concatenated replay input and its action boundaries once."""

        records = self.before_records
        xb = (
            np.concatenate([record["xb_world_75"] for record in records], axis=0)
            if records
            else np.zeros((0, 75), dtype=np.float32)
        )
        return {
            "xb_world_75": xb,
            "fps": np.asarray(self.fps, dtype=np.float32),
            **_step_arrays(records, "length", include_actions=True),
            **self._initial_arrays(),
            "meta_json": np.asarray(
                json.dumps(
                    {
                        "schema": "humanclaw_trajectory_before_v2",
                        "state_source": "generated_motion_before_half_physics",
                        "episode": self.metadata.get("episode", {}),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        }

    def _after_arrays(self) -> dict[str, Any]:
        """Materialize realized human/object pose arrays in frame-aligned form."""

        records = self.after_records
        transl = (
            np.concatenate([record["transl"] for record in records], axis=0)
            if records
            else np.zeros((0, 3), dtype=np.float32)
        )
        orient = (
            np.concatenate([record["global_orient"] for record in records], axis=0)
            if records
            else np.zeros((0, 3), dtype=np.float32)
        )
        pose = (
            np.concatenate([record["body_pose"] for record in records], axis=0)
            if records
            else np.zeros((0, 54, 3), dtype=np.float32)
        )
        frame_step = (
            np.concatenate(
                [
                    np.full(record["length"], record["step"], dtype=np.int32)
                    for record in records
                ]
            )
            if records
            else np.zeros(0, dtype=np.int32)
        )
        object_names = sorted(
            {name for record in records for name in record.get("objects", {}).keys()}
        )
        # Object names are stored once; per-object numeric arrays avoid pickle
        # and keep NPZ readable by standard NumPy with allow_pickle=False.
        arrays: dict[str, Any] = {
            "transl": transl,
            "global_orient": orient,
            "body_pose": pose,
            "frame_step": frame_step,
            "fps": np.asarray(self.fps, dtype=np.float32),
            **_step_arrays(records, "length", include_actions=False),
            "object_names": np.asarray(object_names),
            "meta_json": np.asarray(
                json.dumps(
                    {
                        "schema": "humanclaw_trajectory_after_v1",
                        "state_source": "actual_half_physics_simulator_state",
                        "episode": self.metadata.get("episode", {}),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        }
        for index, name in enumerate(object_names):
            positions: list[np.ndarray] = []
            rotations: list[np.ndarray] = []
            for record in records:
                state = record["objects"].get(name)
                if state is None:
                    # A dynamic object may be created/removed between actions.
                    # Preserve global frame alignment with NaNs rather than
                    # shifting the remainder of that object's trajectory.
                    positions.append(
                        np.full((record["length"], 3), np.nan, dtype=np.float32)
                    )
                    rotations.append(
                        np.full((record["length"], 4), np.nan, dtype=np.float32)
                    )
                else:
                    positions.append(state["position"])
                    rotations.append(state["rotation"])
            arrays[f"object_{index:03d}_position"] = np.concatenate(positions, axis=0)
            arrays[f"object_{index:03d}_rotation"] = np.concatenate(rotations, axis=0)
        return arrays

    def materialize(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return cached before/after arrays shared by metrics and NPZ output."""

        if self._materialized is None:
            self._materialized = (self._before_arrays(), self._after_arrays())
        return self._materialized

    def write(self, output_dir: Path) -> tuple[Path, Path, Path]:
        """Write replay manifest plus enabled before/after trajectory archives."""

        output_dir.mkdir(parents=True, exist_ok=True)
        before_path = output_dir / "trajectory_before.npz"
        after_path = output_dir / "trajectory_after.npz"
        manifest_path = output_dir / "replay_manifest.json"

        before, after = self.materialize()
        # Compression is paid once at episode finalization.  The in-memory
        # materialization above is shared with metrics, so metrics do not read
        # or decompress the files they are about to produce.
        np.savez_compressed(before_path, **before)
        np.savez_compressed(after_path, **after)

        manifest = copy.deepcopy(self.metadata)
        initial_objects = self.initial_state.get("objects") or {}
        manifest["initial_state"] = {
            "stored_in": before_path.name,
            "dynamic_object_count": len(initial_objects),
        }
        manifest["files"] = {
            before_path.name: {
                "size_bytes": before_path.stat().st_size,
                "sha256": _sha256(before_path),
            },
            after_path.name: {
                "size_bytes": after_path.stat().st_size,
                "sha256": _sha256(after_path),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return before_path, after_path, manifest_path


__all__ = ["TrajectoryRecorder", "build_replay_metadata"]

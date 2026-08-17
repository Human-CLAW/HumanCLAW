"""Shared pose-restoration primitives for saved HumanClaw trajectories.

These functions assign already-recorded post-physics states to Habitat objects.
They deliberately do not call ``sim.step_physics`` or ``HalfPhysicsEnv.step``.
Both delayed rendering and no-step metric checks use this same coordinate
conversion, so there is only one interpretation of ``trajectory_after.npz``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def decoded_names(values: Any) -> list[str]:
    """Decode an NPZ string array without requiring pickle support."""

    names: list[str] = []
    for value in np.asarray(values).tolist():
        names.append(
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else str(value)
        )
    return names


def object_pose_arrays(
    after: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return every saved dynamic object's position/quaternion arrays."""

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    names = decoded_names(after.get("object_names", []))
    if len(names) != len(set(names)):
        raise ValueError("trajectory_after contains duplicate dynamic-object names")
    for index, name in enumerate(names):
        position_key = f"object_{index:03d}_position"
        rotation_key = f"object_{index:03d}_rotation"
        if position_key not in after or rotation_key not in after:
            raise ValueError(
                f"trajectory_after is missing pose arrays for dynamic object {name!r}"
            )
        arrays[name] = (
            np.asarray(after[position_key], dtype=np.float32),
            np.asarray(after[rotation_key], dtype=np.float32),
        )
    return arrays


def apply_agent_pose(
    env: Any,
    transl: np.ndarray,
    orient: np.ndarray,
    pose: np.ndarray,
) -> None:
    """Assign one saved humanoid pose without advancing simulation time."""

    runtime = env._require_runtime()
    root_shift = env.world_transformation.apply(env.original_root_shift)
    env.agent.translation = root_shift + env.world_transformation.apply(transl)
    rotation = env.world_transformation * runtime.rotation_cls.from_rotvec(orient)
    quaternion = rotation.as_quat()
    env.agent.rotation = runtime.mn.Quaternion(
        ((quaternion[0], quaternion[1], quaternion[2]), quaternion[3])
    )
    reordered = pose[env.smplx2urdf]
    joints = runtime.rotation_cls.from_rotvec(reordered).as_quat()
    env.agent.joint_positions = joints.reshape(-1).tolist()
    env.agent.motion_type = runtime.motion_type.DYNAMIC

    # Velocities do not affect rasterization, but clearing them prevents stale
    # values from leaking into a caller that performs a contact query next.
    try:
        env.agent.root_linear_velocity = runtime.mn.Vector3(0.0, 0.0, 0.0)
        env.agent.root_angular_velocity = runtime.mn.Vector3(0.0, 0.0, 0.0)
        env.agent.joint_velocities = (
            np.zeros_like(np.asarray(env.agent.joint_velocities, dtype=np.float64))
            .reshape(-1)
            .tolist()
        )
    except Exception:
        pass


def apply_object_pose(
    env: Any,
    obj: Any,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> bool:
    """Assign one saved dynamic-object pose without a physics step."""

    if not np.isfinite(position).all() or not np.isfinite(quaternion_xyzw).all():
        return False
    runtime = env._require_runtime()
    obj.translation = env.world_transformation.apply(position)
    rotation = env.world_transformation * runtime.rotation_cls.from_quat(
        quaternion_xyzw
    )
    quaternion = rotation.as_quat()
    obj.rotation = runtime.mn.Quaternion(
        ((quaternion[0], quaternion[1], quaternion[2]), quaternion[3])
    )
    try:
        obj.linear_velocity = runtime.mn.Vector3(0.0, 0.0, 0.0)
        obj.angular_velocity = runtime.mn.Vector3(0.0, 0.0, 0.0)
        obj.awake = True
    except Exception:
        pass
    return True


__all__ = [
    "apply_agent_pose",
    "apply_object_pose",
    "decoded_names",
    "object_pose_arrays",
]

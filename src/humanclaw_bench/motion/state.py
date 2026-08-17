"""Inference-only helpers for the 219-D HumanClaw motion state."""

from __future__ import annotations

import torch

from .canonicalization import (
    axis_angle_to_rotation_matrix,
    extract_floor_projected_transform,
    inverse_transform,
    rotation_matrix_to_axis_angle,
)

XB_DIM = 75
JTS_START = 75
JTS_DIM = 72
VEL_START = 147
VEL_DIM = 72
N_JOINTS = 24
STATE_DIM = 219


def state_to_joints(state: torch.Tensor) -> torch.Tensor:
    """Extract per-frame joint positions from the 219-D motion representation."""

    return state[..., JTS_START : JTS_START + JTS_DIM].reshape(
        *state.shape[:-1], N_JOINTS, 3
    )


def state_to_xb(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the 219-D generated state to the 75-D SMPL-X xb representation."""

    return state[..., :3], state[..., 3:6], state[..., 6:XB_DIM]


def decanonicalize_joints(
    joints_canon: torch.Tensor, T_ref: torch.Tensor, floor_y: float
) -> torch.Tensor:
    """Transform canonical joint positions back into the current world frame."""

    joints = joints_canon.clone()
    joints[..., 1] += floor_y
    homogeneous = torch.cat([joints, torch.ones_like(joints[..., :1])], dim=-1)
    return (T_ref[None, None] @ homogeneous[..., None]).squeeze(-1)[..., :3]


def decanonicalize_xb(
    transl_canon: torch.Tensor,
    orient_canon: torch.Tensor,
    T_ref: torch.Tensor,
    floor_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform canonical SMPL-X translation/orientation back to world coordinates."""

    transl = transl_canon.clone()
    transl[:, 1] += floor_y
    rotation = T_ref[:3, :3]
    transl_world = (rotation @ transl.unsqueeze(-1)).squeeze(-1) + T_ref[:3, 3]
    orient_world = rotation_matrix_to_axis_angle(
        rotation.unsqueeze(0) @ axis_angle_to_rotation_matrix(orient_canon)
    )
    return transl_world, orient_world


def canonicalize_from_world(
    transl_world: torch.Tensor,
    orient_world: torch.Tensor,
    body_pose: torch.Tensor,
    joints_world: torch.Tensor,
    vel_canon_old: torch.Tensor,
    rotation_old: torch.Tensor,
    ref_frame: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalize a world motion chunk and return its next rolling reference frame."""

    n_frames = transl_world.shape[0]
    floor_transforms = extract_floor_projected_transform(transl_world, joints_world)
    transform = floor_transforms[ref_frame]
    inverse = inverse_transform(transform.unsqueeze(0))[0]
    rotation_inverse = inverse[:3, :3]

    orient = rotation_matrix_to_axis_angle(
        rotation_inverse.unsqueeze(0) @ axis_angle_to_rotation_matrix(orient_world)
    )
    transl = (
        rotation_inverse
        @ (transl_world - floor_transforms[ref_frame, :3, 3]).unsqueeze(-1)
    ).squeeze(-1)
    joints_h = torch.cat([joints_world, torch.ones_like(joints_world[..., :1])], dim=-1)
    joints = (inverse[None, None] @ joints_h[..., None]).squeeze(-1)[..., :3]

    floor_y = joints[ref_frame, :, 1].min()
    transl[:, 1] -= floor_y
    joints[..., 1] -= floor_y

    velocity_rotation = rotation_inverse @ rotation_old
    velocity = vel_canon_old.reshape(n_frames, N_JOINTS, 3)
    velocity = (velocity_rotation[None, None] @ velocity[..., None]).squeeze(-1)
    state = torch.cat(
        [
            transl,
            orient,
            body_pose,
            joints.reshape(n_frames, JTS_DIM),
            velocity.reshape(n_frames, VEL_DIM),
        ],
        dim=-1,
    )
    return state, transform, floor_y


__all__ = [
    "JTS_DIM",
    "N_JOINTS",
    "STATE_DIM",
    "VEL_DIM",
    "VEL_START",
    "canonicalize_from_world",
    "decanonicalize_joints",
    "decanonicalize_xb",
    "state_to_joints",
    "state_to_xb",
]

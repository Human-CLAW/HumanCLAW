"""Canonicalization utilities for motion data.

Implements ego-centric canonicalization centered on a reference frame:
1. Extract floor-projected root transform (XZ translation + yaw rotation)
2. Compute inverse at reference frame
3. Transform all frames to be relative to reference frame

Based on Chuan's approach (from_chuan/preprocessors_interact.py).
Adapted for SMPL-X (no pymomentum dependency).
"""

import torch


def axis_angle_to_rotation_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix using Rodrigues formula.

    Args:
        aa: [..., 3] axis-angle vectors

    Returns:
        R: [..., 3, 3] rotation matrices
    """
    theta = torch.norm(aa, dim=-1, keepdim=True)  # [..., 1]
    axis = aa / (theta + 1e-8)  # [..., 3]

    cos_t = torch.cos(theta).unsqueeze(-1)  # [..., 1, 1]
    sin_t = torch.sin(theta).unsqueeze(-1)  # [..., 1, 1]

    # Skew-symmetric matrix
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )  # [..., 3, 3]

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)

    # Handle zero rotation (theta ≈ 0)
    small = theta.squeeze(-1) < 1e-6
    if small.any():
        R[small] = torch.eye(3, device=aa.device, dtype=aa.dtype)

    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle.

    Args:
        R: [..., 3, 3] rotation matrices

    Returns:
        aa: [..., 3] axis-angle vectors
    """
    # Use the trace to get the angle
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)  # [...]

    # Axis from skew-symmetric part
    axis = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )  # [..., 3]
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)

    aa = axis * angle.unsqueeze(-1)

    # Handle small angles
    small = angle < 1e-6
    if small.any():
        aa[small] = torch.zeros(3, device=R.device, dtype=R.dtype)

    return aa


def extract_floor_projected_transform(
    transl: torch.Tensor,
    joints: torch.Tensor,
) -> torch.Tensor:
    """Extract floor-projected (XZ) root transform using hip joints.

    Uses left-right hip direction (PriMAL method) instead of rotation matrix
    Z-axis. More robust for non-upright poses (bridge, flipping, lying down).

    Forward direction = cross(hip_direction, world_up), projected to XZ plane.

    Args:
        transl: [T, 3] root translation (Y-up)
        joints: [T, J, 3] joint positions (Y-up), J >= 3 (need joints 1,2 = hips)

    Returns:
        T_floor: [T, 4, 4] floor-projected homogeneous transform
    """
    # SMPL joint 1 = left hip, joint 2 = right hip
    x_axis = joints[:, 1, :] - joints[:, 2, :]  # [T, 3] left-right direction
    x_axis[:, 1] = 0  # project to XZ plane (zero Y)
    x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)

    # Y-axis is world up
    y_axis = torch.zeros_like(x_axis)
    y_axis[:, 1] = 1

    # Z-axis (forward) = cross(x_axis, y_axis)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True) + 1e-8)

    # Build rotation matrix
    R_floor = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [T, 3, 3]

    # Floor-projected translation (zero Y)
    t_floor = transl.clone()
    t_floor[:, 1] = 0

    # Build 4x4 transform
    T = torch.zeros(transl.shape[0], 4, 4, device=transl.device, dtype=transl.dtype)
    T[:, :3, :3] = R_floor
    T[:, :3, 3] = t_floor
    T[:, 3, 3] = 1.0

    return T


def inverse_transform(T: torch.Tensor) -> torch.Tensor:
    """Compute inverse of a 4x4 homogeneous rigid transform.

    Args:
        T: [..., 4, 4]

    Returns:
        T_inv: [..., 4, 4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_t = R.transpose(-1, -2)

    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_t
    T_inv[..., :3, 3] = -(R_t @ t.unsqueeze(-1)).squeeze(-1)
    T_inv[..., 3, 3] = 1.0
    return T_inv


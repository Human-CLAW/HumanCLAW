"""Final paper Motion Jerk: root-rigid, denoised, decision-timescale.

The signal is generated motion *before* physics (`xb_world_75`).  Body
articulation is frozen at the neutral rest skeleton, so the score measures the
coherence of root translation and root rotation rather than joint-controller
noise.  This is SSDMC jerk-v2 with moving-average window 3 and stride 8 at
30 Hz (an effective 0.267 s timescale).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from humanclaw_bench.paths import resolve_release_path

_BODY22 = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

DEFAULT_NEUTRAL_BODY22 = "resources/metrics/smpl_neutral_body22.json"


def load_neutral_body22(
    path: str | Path = DEFAULT_NEUTRAL_BODY22,
) -> np.ndarray:
    """Load the exact pelvis-relative rest joints used by SSDMC jerk-v2.

    The source evaluator computes ``J_regressor @ v_template`` from its pinned
    neutral SMPL model and uses joints 0--21.  The release stores those 66
    resulting constants—not the full licensed body model—in a transparent
    JSON resource.  Validating the schema, order, units, shape, and pelvis row
    prevents a plausible-looking but numerically different skeleton from
    silently changing the geometry-weighted root-rotation score.
    """

    resolved = resolve_release_path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("schema") != "humanclaw_smpl_neutral_body22_v1":
        raise ValueError(f"Invalid neutral-body schema: {resolved}")
    if value.get("units") != "meters":
        raise ValueError(f"Neutral-body resource must use meters: {resolved}")
    if tuple(value.get("joint_names") or ()) != _BODY22:
        raise ValueError(f"Neutral-body joint order mismatch: {resolved}")
    positions = np.asarray(value.get("positions_m"), dtype=np.float64)
    if positions.shape != (22, 3) or not np.isfinite(positions).all():
        raise ValueError(f"Neutral-body positions must be finite (22, 3): {resolved}")
    if not np.array_equal(positions[0], np.zeros(3, dtype=np.float64)):
        raise ValueError(f"Neutral-body positions must be pelvis-relative: {resolved}")
    return positions


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to rotation matrices with Rodrigues' formula."""

    theta = np.linalg.norm(rotvec, axis=1, keepdims=True)
    safe_theta = np.where(theta < 1e-8, 1.0, theta)
    axis = rotvec / safe_theta
    skew = np.zeros((rotvec.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -axis[:, 2]
    skew[:, 0, 2] = axis[:, 1]
    skew[:, 1, 0] = axis[:, 2]
    skew[:, 1, 2] = -axis[:, 0]
    skew[:, 2, 0] = -axis[:, 1]
    skew[:, 2, 1] = axis[:, 0]
    sin = np.sin(theta)[:, :, None]
    cos = np.cos(theta)[:, :, None]
    rotations = np.eye(3)[None, :, :] + sin * skew + (1.0 - cos) * (skew @ skew)
    rotations[theta[:, 0] < 1e-8] = np.eye(3)
    return rotations


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Smooth a frame sequence along time with an edge-preserving moving average."""

    pad = window // 2
    padded = np.pad(signal, [(pad, pad), (0, 0), (0, 0)], mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.concatenate(
        [np.zeros((1,) + signal.shape[1:], dtype=np.float64), cumulative], axis=0
    )
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def root_rigid_motion_jerk(
    xb_world_75: np.ndarray,
    rest_joints: np.ndarray,
    *,
    fps: float = 30.0,
    smooth_window: int = 3,
    stride: int = 8,
) -> float | None:
    """Return the episode mean jerk magnitude in m/s^3."""

    xb = np.asarray(xb_world_75, dtype=np.float64)
    rest = np.asarray(rest_joints, dtype=np.float64)
    if xb.ndim != 2 or xb.shape[1] != 75:
        raise ValueError(f"xb_world_75 must have shape (T, 75), got {xb.shape}")
    if rest.shape != (22, 3):
        raise ValueError(f"rest_joints must have shape (22, 3), got {rest.shape}")
    if smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError("smooth_window must be a positive odd number")
    if stride < 1 or fps <= 0:
        raise ValueError("stride and fps must be positive")
    if xb.shape[0] < 3 * stride + 1:
        return None

    # With every local joint rotation frozen to identity, forward kinematics is
    # simply root rotation applied to the root-relative neutral skeleton.
    relative = rest - rest[:1]
    rotations = _rodrigues(xb[:, 3:6])
    positions = np.einsum("tij,kj->tki", rotations, relative)
    positions += xb[:, None, 0:3]
    positions = _moving_average(positions, smooth_window)

    h = float(stride) / float(fps)
    jerk = (
        positions[3 * stride :]
        - 3.0 * positions[2 * stride : -stride]
        + 3.0 * positions[stride : -2 * stride]
        - positions[: -3 * stride]
    ) / (h**3)
    return float(np.linalg.norm(jerk, axis=-1).mean())


__all__ = ["load_neutral_body22", "root_rigid_motion_jerk"]

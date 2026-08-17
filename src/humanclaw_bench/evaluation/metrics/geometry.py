"""Geometry primitives shared by Nav and Interact metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def normalize_aabb(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize supported AABB representations to minimum and maximum xyz arrays."""

    if isinstance(value, dict):
        minimum = value.get("min", value.get("min_xyz", value.get("mn")))
        maximum = value.get("max", value.get("max_xyz", value.get("mx")))
    else:
        minimum, maximum = value
    mn = np.asarray(minimum, dtype=np.float64).reshape(-1)[:3]
    mx = np.asarray(maximum, dtype=np.float64).reshape(-1)[:3]
    if mn.shape != (3,) or mx.shape != (3,):
        raise ValueError("AABB bounds must each contain three values")
    return np.minimum(mn, mx), np.maximum(mn, mx)


def body_to_target_aabb_distance(
    body_points: Sequence[Sequence[float]],
    target_aabbs: Sequence[Any],
) -> float:
    """Minimum 3D distance between any humanoid joint and target AABB."""

    points = np.asarray(body_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        raise ValueError(f"Body points must have shape (N, 3), got {points.shape}")
    points = points[:, :3]
    best = float("inf")
    for raw_aabb in target_aabbs:
        mn, mx = normalize_aabb(raw_aabb)
        delta = np.maximum(np.maximum(mn[None, :] - points, 0.0), points - mx[None, :])
        best = min(best, float(np.linalg.norm(delta, axis=1).min()))
    if not np.isfinite(best):
        raise ValueError("At least one target AABB is required")
    return best


__all__ = ["body_to_target_aabb_distance", "normalize_aabb"]

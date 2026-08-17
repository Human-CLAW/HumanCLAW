"""Paper-time seed and action conditioning contract, without training imports."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .networks.normalization import normalize_target

NECK_HEAD_BODY_POSE_JOINTS = (11, 14)


def zero_neck_head_body_pose(value: torch.Tensor | np.ndarray):
    """Zero neck and head axis-angle joints in a copied SMPL-X body pose."""

    if value.shape[-1] != 69:
        raise ValueError(f"Expected 69-D body pose, got {value.shape[-1]}")
    out = (
        value.clone() if isinstance(value, torch.Tensor) else np.array(value, copy=True)
    )
    for joint in NECK_HEAD_BODY_POSE_JOINTS:
        out[..., joint * 3 : joint * 3 + 3] = 0
    return out


def zero_neck_head_xb_state(value: torch.Tensor | np.ndarray):
    """Zero neck/head coordinates in a copied 75-D motion seed tensor."""

    if value.shape[-1] < 75:
        raise ValueError(f"Expected state dim >=75, got {value.shape[-1]}")
    out = (
        value.clone() if isinstance(value, torch.Tensor) else np.array(value, copy=True)
    )
    out[..., 6:75] = zero_neck_head_body_pose(out[..., 6:75])
    return out


def load_seed_state(
    seed_mode: str,
    *,
    seed_pt: str | Path | None = None,
    seed_pkl: str | Path | None = None,
) -> torch.Tensor:
    """Load and shape-check the configured deterministic motion seed."""

    if seed_mode in {"pt", "tpose_fk"}:
        if seed_pt is None:
            raise ValueError("seed_pt is required for pt/tpose_fk mode")
        try:
            seed = torch.load(seed_pt, map_location="cpu", weights_only=True)
        except TypeError:
            seed = torch.load(seed_pt, map_location="cpu")
        if isinstance(seed, dict):
            seed = seed["state"]
        result = torch.as_tensor(seed, dtype=torch.float32)
    elif seed_mode == "pkl":
        if seed_pkl is None:
            raise ValueError("seed_pkl is required for pkl mode")
        with Path(seed_pkl).open("rb") as handle:
            result = torch.tensor(
                pickle.load(handle)["chunks"][0][:5], dtype=torch.float32
            )
    elif seed_mode in {"zero", "none"}:
        result = torch.zeros(5, 219, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown seed mode: {seed_mode}")
    if tuple(result.shape) != (5, 219):
        raise ValueError(f"Expected seed shape (5, 219), got {tuple(result.shape)}")
    return result


def condition_args(skill: str, kind: str, condition: Any, device: str):
    """Convert a SkillCall condition into tensors expected by its motion network."""

    def tensor(values):
        """Create one float32 condition tensor on the motion runner's device."""

        return torch.tensor([values], dtype=torch.float32, device=device)

    if skill == "walk_forward" and kind == "fast_xzyaw":
        values = list(condition)
        if len(values) == 2:
            values.append(0.0)
        if len(values) < 3:
            raise ValueError(
                f"walk_forward expects [x,z] or [x,z,yaw], got {condition}"
            )
        return (tensor([float(x) for x in values[:3]]),)
    if skill == "side_walk":
        value = tensor([float(condition)])
        return (value if kind == "side_fourier" else value / 0.5,)
    if skill == "step_back":
        if isinstance(condition, (list, tuple)):
            x, z = condition[:2]
        else:
            x, z = 0.0, -float(condition)
        raw = tensor([float(x), float(z)])
        if kind == "step_back_raw_z":
            return (raw[:, 1:2] / 0.5,)
        return (raw if kind == "fourier" else normalize_target(raw),)
    if skill in {"step_climb_up", "step_climb_down"}:
        height, distance = condition
        raw = tensor([float(height), float(distance)])
        normalized = torch.cat(
            [4.0 * raw[:, :1] - 1.0, (raw[:, 1:2] - 0.5) / 0.3], dim=-1
        )
        return (normalized,)
    if skill == "turn":
        yaw = tensor([float(condition)])
        return (yaw if kind == "fourier" else yaw / 75.0,)
    if skill == "sit":
        height = condition[0] if isinstance(condition, (list, tuple)) else condition
        return (tensor([float(height)]),)
    if skill == "stand":
        return ()
    raise ValueError(f"Unsupported skill: {skill}")


__all__ = [
    "condition_args",
    "load_seed_state",
    "zero_neck_head_body_pose",
    "zero_neck_head_xb_state",
]

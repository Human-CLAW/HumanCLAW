"""Small conditioning transforms shared by the evaluated motion networks."""

from __future__ import annotations

import torch


def normalize_target(value: torch.Tensor) -> torch.Tensor:
    """Soft-saturate a 2-D displacement while preserving its direction.

    This is the exact transform used when the spatial ControlNets were trained:
    ``2 * (1 - exp(-||x||)) * x / ||x||``.  Keeping it in a tiny runtime module
    avoids shipping the unused training-only ``SpatialControlDiTConcat`` model.
    """

    norm = value.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return 2.0 * (1.0 - torch.exp(-norm)) * value / norm


__all__ = ["normalize_target"]

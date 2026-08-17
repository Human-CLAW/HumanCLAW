"""Flow-matching targets shared by base-MotionDiT and ControlNet training."""

from __future__ import annotations

import torch


def sample_flow_matching_batch(
    clean_motion: torch.Tensor,
    *,
    sigma_min: float = 1e-6,
    logit_normal_mu: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise clean motion and return ``(x_t, t, target_velocity)``.

    This is the same linear probability path used for the released base and
    eight skill ControlNets.  Time is uniform on ``[0, 1]`` unless a
    logit-normal mean is explicitly supplied.
    """

    batch_size = clean_motion.shape[0]
    noise = torch.randn_like(clean_motion)
    if logit_normal_mu is None:
        time = torch.rand(batch_size, device=clean_motion.device)
    else:
        time = torch.sigmoid(
            torch.randn(batch_size, device=clean_motion.device) + logit_normal_mu
        )
    expanded_time = time[:, None, None]
    noisy_motion = (
        1.0 - (1.0 - sigma_min) * expanded_time
    ) * noise + expanded_time * clean_motion
    target_velocity = clean_motion - (1.0 - sigma_min) * noise
    return noisy_motion, time, target_velocity


__all__ = ["sample_flow_matching_batch"]

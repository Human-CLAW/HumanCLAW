"""Midpoint flow-matching solver used by all paper-time motion skills."""

from __future__ import annotations

import torch


@torch.no_grad()
def denoise(model, history: torch.Tensor, condition_args: tuple) -> torch.Tensor:
    """Integrate the learned flow field from noise to a generated motion sequence."""

    device = next(model.parameters()).device
    batch_size = history.shape[0]
    sample = torch.randn(
        (batch_size, model.n_future, model.x_dim), device=device, dtype=history.dtype
    )
    nfe_steps = int(model.nfe_steps)
    dt = 1.0 / nfe_steps
    for time_value in torch.linspace(0, 1, nfe_steps + 1, device=device)[:-1]:
        time = torch.ones(batch_size, device=device) * time_value
        velocity = model._forward(history, sample, time, *condition_args)
        midpoint = sample + dt * 0.5 * velocity
        midpoint_time = torch.ones(batch_size, device=device) * (time_value + dt * 0.5)
        midpoint_velocity = model._forward(
            history, midpoint, midpoint_time, *condition_args
        )
        sample = sample + dt * midpoint_velocity
    return sample[0]


__all__ = ["denoise"]

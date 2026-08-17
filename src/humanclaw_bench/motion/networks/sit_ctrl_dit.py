"""Sit-down per-action ControlNet with a simple MLP scalar encoder.

Input condition:
  target_h_raw = minimum pelvis_y over the accepted sit-down segment

Normalization follows the existing sit Fourier path for consistency:
  h_norm = (h_raw - 0.45) / 0.4
"""

from __future__ import annotations

import copy

import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT


class SitCtrlDiT(nn.Module):
    """Condition a frozen MotionDiT on the requested sitting target."""

    def __init__(self, base_dit: MotionDiT, hidden_dim: int = 512):
        """Freeze the base denoiser and construct the SitCtrlDiT control layers."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.h_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        n_layers = len(base_dit.blocks)
        self.ctrl_blocks = nn.ModuleList([copy.deepcopy(b) for b in base_dit.blocks])
        self.zero_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        for lin in self.zero_linears:
            nn.init.constant_(lin.weight, 0)
            nn.init.constant_(lin.bias, 0)

    def forward(self, concat_input, timestep, target_h_raw):
        """Predict sitting motion conditioned on the requested pelvis height."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        target_h_norm = (target_h_raw - 0.45) / 0.4
        cond = self.h_encoder(target_h_norm)
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))
        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks, self.ctrl_blocks, self.zero_linears
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond.unsqueeze(1)
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)
        return self.base_dit.final_layer(tokens, c)

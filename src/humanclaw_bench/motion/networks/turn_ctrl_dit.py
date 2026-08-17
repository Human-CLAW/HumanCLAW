"""Turn per-action ControlNet (Step 13).

Single ctrl branch + 1-D yaw encoder (1-MLP from 1 → hidden_dim).
"""

import copy

import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT


class TurnCtrlDiT(nn.Module):
    """Condition a frozen MotionDiT on the requested turn angle."""

    def __init__(self, base_dit: MotionDiT, hidden_dim: int = 512):
        """Freeze the base denoiser and construct the TurnCtrlDiT control layers."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.yaw_encoder = nn.Sequential(
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

    def forward(self, concat_input, timestep, target_yaw_norm):
        """Predict turning motion conditioned on a signed yaw angle."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        cond = self.yaw_encoder(target_yaw_norm)
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))
        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks, self.ctrl_blocks, self.zero_linears
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond.unsqueeze(1)
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)
        return self.base_dit.final_layer(tokens, c)

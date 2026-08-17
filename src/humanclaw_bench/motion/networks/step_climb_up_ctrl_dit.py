"""Step climb up per-action ControlNet (Step 13).

Same architecture as WalkForwardCtrlDiT (1 ctrl branch + 2-MLP encoder + zero linears).
Input: (h_norm, d_norm) in [-1, +1].
"""

import copy

import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT


class StepClimbUpCtrlDiT(nn.Module):
    """Condition a frozen MotionDiT on the requested climb-up geometry."""

    def __init__(
        self,
        base_dit: MotionDiT,
        hidden_dim: int = 512,
    ):
        """Freeze the base denoiser and construct the StepClimbUpCtrlDiT control layers."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.hd_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
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

    def forward(self, concat_input, timestep, target_hd_norm):
        """Predict climb-up motion conditioned on the requested riser geometry."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        cond = self.hd_encoder(target_hd_norm)
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))
        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks, self.ctrl_blocks, self.zero_linears
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond.unsqueeze(1)
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)
        return self.base_dit.final_layer(tokens, c)

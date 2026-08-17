"""Step 13: per-action ControlNet with NO condition input.

Encoder = single learnable token (nn.Parameter). Used for stand/standup,
where the action has no parameters but still needs an action-specific
ctrl branch.
"""

import copy

import torch
import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT


class NonCondCtrlDiT(nn.Module):
    """Add a zero-initialized trainable control branch to a frozen MotionDiT."""

    def __init__(self, base_dit: MotionDiT, hidden_dim: int = 512):
        """Freeze the base denoiser and construct the NonCondCtrlDiT control layers."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.cond_token = nn.Parameter(torch.zeros(hidden_dim))

        n_layers = len(base_dit.blocks)
        self.ctrl_blocks = nn.ModuleList([copy.deepcopy(b) for b in base_dit.blocks])
        self.zero_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        for lin in self.zero_linears:
            nn.init.constant_(lin.weight, 0)
            nn.init.constant_(lin.bias, 0)

    def forward(self, concat_input, timestep):
        """Predict motion with an unconditioned trainable residual over the frozen base."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))
        cond = self.cond_token.unsqueeze(0).unsqueeze(0)  # [1, 1, h]
        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks, self.ctrl_blocks, self.zero_linears
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)
        return self.base_dit.final_layer(tokens, c)

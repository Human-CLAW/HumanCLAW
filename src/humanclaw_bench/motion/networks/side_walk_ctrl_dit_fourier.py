"""Side-walk per-action ControlNet with Fourier features on lateral x.

This mirrors the walk_forward Fourier style, but the condition is one scalar:
final-frame canonical pelvis x displacement. The encoder uses:
  [x / 0.5, sin(2^i*pi*x), cos(2^i*pi*x)] for i in [0, L).
"""

import copy
import math

import torch
import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT


class SideWalkFourierEncoder(nn.Module):
    """Encode lateral displacement with normalized and Fourier features."""

    def __init__(self, hidden_dim: int, n_freqs: int = 6):
        """Build lateral Fourier frequencies and the hidden projection layer."""

        super().__init__()
        self.n_freqs = n_freqs
        freqs = (
            torch.tensor([2.0**i for i in range(n_freqs)], dtype=torch.float32)
            * math.pi
        )
        self.register_buffer("freqs", freqs)
        in_dim = 1 + 2 * n_freqs
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, side_x_m: torch.Tensor) -> torch.Tensor:
        """Encode lateral meters into one hidden conditioning vector."""

        x_norm = side_x_m / 0.5
        scaled = side_x_m.unsqueeze(-1) * self.freqs
        fourier = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(1)
        feat = torch.cat([x_norm, fourier], dim=1)
        return self.ln(self.linear(feat))


class SideWalkCtrlDiTFourier(nn.Module):
    """Condition side-walk motion through a Fourier-feature control branch."""

    def __init__(self, base_dit: MotionDiT, hidden_dim: int = 512, n_freqs: int = 6):
        """Freeze the base DiT and build a Fourier-conditioned control branch."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.x_encoder = SideWalkFourierEncoder(hidden_dim, n_freqs=n_freqs)

        n_layers = len(base_dit.blocks)
        self.ctrl_blocks = nn.ModuleList([copy.deepcopy(b) for b in base_dit.blocks])
        self.zero_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        for lin in self.zero_linears:
            nn.init.constant_(lin.weight, 0)
            nn.init.constant_(lin.bias, 0)

    def forward(
        self,
        concat_input: torch.Tensor,
        timestep: torch.Tensor,
        target_side_x_m: torch.Tensor,
    ) -> torch.Tensor:
        """Predict side-walk motion from Fourier-encoded lateral displacement."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        cond = self.x_encoder(target_side_x_m)
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))

        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks,
            self.ctrl_blocks,
            self.zero_linears,
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond.unsqueeze(1)
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)

        return self.base_dit.final_layer(tokens, c)

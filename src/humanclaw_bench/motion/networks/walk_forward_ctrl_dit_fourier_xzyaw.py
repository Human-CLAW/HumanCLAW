"""Walk-forward per-action ControlNet with raw (x, z, yaw_deg) Fourier encoder."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT
from humanclaw_bench.motion.networks.normalization import normalize_target


class FourierXZYawEncoder(nn.Module):
    """Encode x/z displacement and yaw with Fourier features."""

    def __init__(self, hidden_dim: int, n_freqs: int = 6):
        """Build x/z/yaw Fourier frequencies and their hidden projection."""

        super().__init__()
        self.n_freqs = n_freqs
        freqs = (
            torch.tensor([2.0**i for i in range(n_freqs)], dtype=torch.float32)
            * math.pi
        )
        self.register_buffer("freqs", freqs)
        in_dim = (
            2 + 2 + 1 + 4 * n_freqs + 2 * n_freqs
        )  # norm_xz + (theta,v) + yaw_unit + Fourier(xz) + Fourier(yaw)
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, xzyaw_raw: torch.Tensor) -> torch.Tensor:
        """Encode requested x/z displacement and yaw into one hidden vector."""

        xz_raw = xzyaw_raw[:, 0:2]
        yaw_deg = xzyaw_raw[:, 2:3]

        x = xz_raw[:, 0:1]
        z = xz_raw[:, 1:2]
        norm_xz = normalize_target(xz_raw)
        theta = torch.atan2(x, z) / math.pi
        v = torch.linalg.norm(xz_raw, dim=1, keepdim=True) / 0.5
        yaw_unit = yaw_deg / 90.0

        xz_scaled = xz_raw.unsqueeze(-1) * self.freqs
        xz_fourier = torch.cat(
            [torch.sin(xz_scaled), torch.cos(xz_scaled)], dim=-1
        ).flatten(1)

        yaw_scaled = yaw_unit.unsqueeze(-1) * self.freqs
        yaw_fourier = torch.cat(
            [torch.sin(yaw_scaled), torch.cos(yaw_scaled)], dim=-1
        ).flatten(1)

        feat = torch.cat([norm_xz, theta, v, yaw_unit, xz_fourier, yaw_fourier], dim=1)
        return self.ln(self.linear(feat))


class WalkForwardCtrlDiTFourierXZYaw(nn.Module):
    """Condition walking motion on Fourier x/z and yaw features."""

    def __init__(self, base_dit: MotionDiT, hidden_dim: int = 512, n_freqs: int = 6):
        """Freeze the base DiT and build an x/z/yaw-conditioned walking branch."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.cond_encoder = FourierXZYawEncoder(hidden_dim, n_freqs=n_freqs)

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
        target_xzyaw_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Predict walking from Fourier-encoded displacement and yaw targets."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        cond = self.cond_encoder(target_xzyaw_raw)
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))

        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks, self.ctrl_blocks, self.zero_linears
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond.unsqueeze(1)
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)

        return self.base_dit.final_layer(tokens, c)

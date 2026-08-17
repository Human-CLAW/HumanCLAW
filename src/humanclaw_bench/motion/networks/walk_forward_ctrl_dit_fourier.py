"""Walk forward per-action ControlNet — Fourier-feature encoder.

Encoder input (28 dim):
  - normalize_target(xz) : 2
  - (theta, v)           : 2   (theta = atan2(x,z)/pi in [-1,1], v = |xz_raw|/0.5)
  - Fourier(x_raw, z_raw): 2 vars * L freqs * (sin,cos) = 4L  (L=6 -> 24)
Then Linear(28, hidden) + LayerNorm.

Raw xz comes in as meters; encoder computes normalize/theta/v/fourier internally.
"""

import copy
import math

import torch
import torch.nn as nn

from humanclaw_bench.motion.networks.motion_dit import MotionDiT
from humanclaw_bench.motion.networks.normalization import normalize_target


class FourierXZEncoder(nn.Module):
    """Encode an x/z displacement with normalized and Fourier features."""

    def __init__(self, hidden_dim: int, n_freqs: int = 6):
        """Build x/z Fourier frequencies and the hidden projection layer."""

        super().__init__()
        self.n_freqs = n_freqs
        freqs = (
            torch.tensor([2.0**i for i in range(n_freqs)], dtype=torch.float32)
            * math.pi
        )
        self.register_buffer("freqs", freqs)
        in_dim = 2 + 2 + 2 * n_freqs * 2  # 2 norm + 2 (theta, v) + 4L fourier
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, xz_raw: torch.Tensor) -> torch.Tensor:
        """Encode requested x/z displacement into one hidden conditioning vector."""

        x = xz_raw[:, 0:1]
        z = xz_raw[:, 1:2]
        norm = normalize_target(xz_raw)  # [B, 2]
        theta = torch.atan2(x, z) / math.pi  # [B, 1] in [-1,1]
        v = torch.linalg.norm(xz_raw, dim=1, keepdim=True) / 0.5  # [B, 1]
        # Fourier on (x, z): [B, 2, L] -> sin/cos -> [B, 4L]
        scaled = xz_raw.unsqueeze(-1) * self.freqs  # [B, 2, L]
        fourier = torch.cat(
            [torch.sin(scaled), torch.cos(scaled)], dim=-1
        )  # [B, 2, 2L]
        fourier = fourier.flatten(1)  # [B, 4L]
        feat = torch.cat([norm, theta, v, fourier], dim=1)  # [B, 28]
        return self.ln(self.linear(feat))


class WalkForwardCtrlDiTFourier(nn.Module):
    """Condition walking motion through a Fourier-feature x/z control branch."""

    def __init__(self, base_dit: MotionDiT, hidden_dim: int = 512, n_freqs: int = 6):
        """Freeze the base DiT and build a Fourier-conditioned walking branch."""

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_dit = base_dit
        for p in self.base_dit.parameters():
            p.requires_grad = False

        self.xz_encoder = FourierXZEncoder(hidden_dim, n_freqs=n_freqs)

        n_layers = len(base_dit.blocks)
        self.ctrl_blocks = nn.ModuleList([copy.deepcopy(b) for b in base_dit.blocks])
        self.zero_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        for lin in self.zero_linears:
            nn.init.constant_(lin.weight, 0)
            nn.init.constant_(lin.bias, 0)

    def forward(self, concat_input, timestep, target_xz_raw):
        """Predict forward walking from Fourier-encoded x/z displacement."""

        t_emb = self.base_dit.t_embedder(timestep)
        c = t_emb
        cond = self.xz_encoder(target_xz_raw)
        tokens = self.base_dit.pos_enc(self.base_dit.in_fc(concat_input))
        for base_block, ctrl_block, zero_lin in zip(
            self.base_dit.blocks, self.ctrl_blocks, self.zero_linears
        ):
            h_base = base_block(tokens, c)
            tokens_ctrl = tokens + cond.unsqueeze(1)
            h_ctrl = ctrl_block(tokens_ctrl, c)
            tokens = h_base + zero_lin(h_ctrl)
        return self.base_dit.final_layer(tokens, c)

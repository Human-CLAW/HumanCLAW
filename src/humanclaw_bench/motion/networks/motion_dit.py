"""MotionDiT: DiT-style denoiser for motion generation with adaLN-Zero conditioning."""

import numpy as np
import torch
import torch.nn as nn


class TimeEmbedder(nn.Module):
    """Sinusoidal time embedding + MLP, following DiT."""

    def __init__(self, d_model: int = 512, max_len: int = 5000):
        """Build the sinusoidal time table and its learned projection MLP."""

        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer or normalized continuous diffusion times.

        Args:
            t: [B] — if integer, direct lookup; if float in [0,1], interpolated.
        Returns:
            [B, d_model]
        """
        if not torch.is_floating_point(t):
            t_int = t.clamp(0, self.pe.shape[0] - 1)
            x_emb = self.pe[t_int]
        else:
            t_scaled = t.float() * (self.pe.shape[0] - 1)
            t0 = torch.floor(t_scaled).long().clamp(0, self.pe.shape[0] - 1)
            t1 = torch.ceil(t_scaled).long().clamp(0, self.pe.shape[0] - 1)
            w1 = (t_scaled - t0.float()).unsqueeze(-1)
            x_emb = (1.0 - w1) * self.pe[t0] + w1 * self.pe[t1]

        return self.mlp(x_emb)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        """Precompute sinusoidal sequence positions and dropout."""

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positions to a batch-first token sequence."""

        x = x + self.pe[:, : x.size(1), : x.size(2)]
        return self.dropout(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN shift and scale parameters to batch-first token features."""

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning and optional QK-norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        use_qk_norm: bool = False,
    ):
        """Build one adaLN-Zero attention/MLP residual block."""

        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = (
            QKNormAttention(hidden_size, num_heads)
            if use_qk_norm
            else SelfAttention(hidden_size, num_heads)
        )

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply conditioned attention and MLP residuals to the motion tokens."""

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class SelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int):
        """Build standard multi-head self-attention projections."""

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply standard multi-head self-attention to batch-first tokens."""

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class QKNormAttention(nn.Module):
    """Multi-head self-attention with QK normalization."""

    def __init__(self, dim: int, num_heads: int):
        """Build multi-head attention with per-head query/key normalization."""

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply query/key-normalized multi-head self-attention."""

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation + linear projection."""

    def __init__(self, hidden_size: int, output_dim: int):
        """Build the adaLN-modulated projection to motion features."""

        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply final adaptive normalization and project to motion features."""

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class MotionDiT(nn.Module):
    """DiT-style denoiser for motion generation.

    Conditioning (time + history) is injected via adaLN-Zero, not as extra tokens.

    Args:
        input_dim: per-frame motion feature dimension (219 for smplx_jts_locs_velocity)
        output_dim: per-frame output dimension (same as input_dim)
        hidden_dim: transformer hidden dimension
        num_layers: number of DiT blocks
        num_heads: number of attention heads
        mlp_ratio: MLP hidden dim = hidden_dim * mlp_ratio
        n_time_embeddings: max time embedding entries
        use_qk_norm: whether to use QK normalization in attention
    """

    def __init__(
        self,
        input_dim: int = 219,
        output_dim: int = 219,
        hidden_dim: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        n_time_embeddings: int = 1000,
        use_qk_norm: bool = False,
    ):
        """Build the time-conditioned DiT denoiser and initialize its weights."""

        super().__init__()
        self.hidden_dim = hidden_dim

        self.t_embedder = TimeEmbedder(d_model=hidden_dim, max_len=n_time_embeddings)
        self.pos_enc = PositionalEncoder(hidden_dim, dropout=0.0)
        self.in_fc = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim, num_heads, mlp_ratio=mlp_ratio, use_qk_norm=use_qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize projections and preserve adaLN-Zero's zero-residual start."""

        def _basic_init(module):
            """Apply Xavier initialization to one linear submodule."""

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Input projection
        nn.init.xavier_uniform_(self.in_fc.weight.view(self.in_fc.weight.shape[0], -1))
        nn.init.constant_(self.in_fc.bias, 0)

        # Time embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        c_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity field for noisy future motion.

        Args:
            sample: [B, T, input_dim] — noisy future frames
            timestep: [B] — flow matching time in [0, 1]
            c_emb: [B, hidden_dim] — history condition embedding
        Returns:
            [B, T, output_dim] — predicted velocity field
        """
        t_emb = self.t_embedder(timestep)  # [B, hidden_dim]
        c = t_emb + c_emb  # [B, hidden_dim]

        tokens = self.pos_enc(self.in_fc(sample))  # [B, T, hidden_dim]

        for block in self.blocks:
            tokens = block(tokens, c)

        output = self.final_layer(tokens, c)  # [B, T, output_dim]
        return output

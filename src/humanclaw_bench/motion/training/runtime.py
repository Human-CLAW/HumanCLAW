"""Shared CUDA, checkpoint, and model-construction helpers for training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as torch_functional

from humanclaw_bench.motion.networks.motion_dit import (
    MotionDiT,
    QKNormAttention,
    SelfAttention,
    TimeEmbedder,
)


def configure_cuda_training() -> None:
    """Enable the CUDA kernels used by the historical fast skill trainers."""

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def resolve_dtype(name: str) -> torch.dtype:
    """Translate the profile's stable dtype name into a Torch dtype."""

    choices = {"fp32": torch.float32, "bf16": torch.bfloat16}
    try:
        return choices[name]
    except KeyError as error:
        raise ValueError(f"Unsupported training dtype: {name!r}") from error


def _self_attention_sdpa(self: SelfAttention, value: torch.Tensor) -> torch.Tensor:
    """Run the original self-attention projections through fused SDPA."""

    batch_size, sequence_length, hidden_dim = value.shape
    qkv = self.qkv(value).reshape(
        batch_size, sequence_length, 3, self.num_heads, self.head_dim
    )
    query, key, attention_value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    output = torch_functional.scaled_dot_product_attention(
        query,
        key,
        attention_value,
        dropout_p=0.0,
        is_causal=False,
        scale=self.scale,
    )
    return self.proj(output.transpose(1, 2).reshape(value.shape))


def _qk_norm_attention_sdpa(
    self: QKNormAttention, value: torch.Tensor
) -> torch.Tensor:
    """Run query/key-normalized attention through fused SDPA."""

    batch_size, sequence_length, hidden_dim = value.shape
    qkv = self.qkv(value).reshape(
        batch_size, sequence_length, 3, self.num_heads, self.head_dim
    )
    query, key, attention_value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    output = torch_functional.scaled_dot_product_attention(
        self.q_norm(query),
        self.k_norm(key),
        attention_value,
        dropout_p=0.0,
        is_causal=False,
        scale=self.scale,
    )
    return self.proj(output.transpose(1, 2).reshape(value.shape))


def _dtype_preserving_time_embedding(
    self: TimeEmbedder, time: torch.Tensor
) -> torch.Tensor:
    """Interpolate time embeddings without silently promoting BF16 models."""

    if not torch.is_floating_point(time):
        embedding = self.pe[time.clamp(0, self.pe.shape[0] - 1)]
    else:
        table_dtype = self.pe.dtype
        scaled = time.to(dtype=table_dtype) * (self.pe.shape[0] - 1)
        lower = torch.floor(scaled).long().clamp(0, self.pe.shape[0] - 1)
        upper = torch.ceil(scaled).long().clamp(0, self.pe.shape[0] - 1)
        upper_weight = (scaled - lower.to(dtype=table_dtype)).unsqueeze(-1)
        embedding = (
            (1.0 - upper_weight) * self.pe[lower]
            + upper_weight * self.pe[upper]
        )
    return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


def patch_fast_kernels(module: torch.nn.Module, *, use_sdpa: bool = True) -> int:
    """Apply the two forward patches used by the paper's fast trainers.

    The return value is the number of attention modules switched to SDPA.
    Time embedders are always patched because this is required when parameters
    are held in BF16.
    """

    patched_attention = 0
    for child in module.modules():
        if use_sdpa and isinstance(child, SelfAttention):
            child.forward = _self_attention_sdpa.__get__(child, child.__class__)
            patched_attention += 1
        elif use_sdpa and isinstance(child, QKNormAttention):
            child.forward = _qk_norm_attention_sdpa.__get__(child, child.__class__)
            patched_attention += 1
        if isinstance(child, TimeEmbedder):
            child.forward = _dtype_preserving_time_embedding.__get__(
                child, child.__class__
            )
    return patched_attention


def new_motion_dit(architecture: Mapping[str, Any]) -> MotionDiT:
    """Construct MotionDiT from a portable architecture mapping."""

    return MotionDiT(
        input_dim=int(architecture["x_dim"]),
        output_dim=int(architecture["x_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        num_layers=int(architecture["num_layers"]),
        num_heads=int(architecture["num_heads"]),
        mlp_ratio=float(architecture.get("mlp_ratio", 2.0)),
        use_qk_norm=bool(architecture.get("use_qk_norm", False)),
    )


def load_torch_mapping(path: str | Path) -> dict[str, Any]:
    """Load a tensor checkpoint on CPU with Torch's restricted unpickler."""

    resolved = Path(path).expanduser().resolve()
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(resolved, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is not a mapping: {resolved}")
    return value


def load_base_motion_dit(
    path: str | Path, architecture: Mapping[str, Any]
) -> MotionDiT:
    """Load a base from release, refactored-training, or Lightning format."""

    checkpoint = load_torch_mapping(path)
    state: Mapping[str, torch.Tensor]
    if checkpoint.get("schema") == "humanclaw_motion_dit_state_v1":
        state = checkpoint["state_dict"]
    elif checkpoint.get("schema") == "humanclaw_base_training_v1":
        state = checkpoint["model_state"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        # Original MotionDiTConcatLightning checkpoints prefix the denoiser
        # keys.  Other Lightning state (optimizer/callbacks) is intentionally
        # ignored when starting a ControlNet run.
        state = {
            key.removeprefix("denoiser."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("denoiser.")
        }
    elif all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        state = checkpoint
    else:
        raise ValueError(f"Unsupported base checkpoint format: {path}")
    model = new_motion_dit(architecture)
    model.load_state_dict(state, strict=True)
    return model


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return only parameters that the optimizer is allowed to update."""

    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def make_batch_indices(
    sample_count: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Sample one no-replacement batch exactly as the fast trainers did."""

    if batch_size > sample_count:
        raise ValueError(
            f"batch_size={batch_size} exceeds training samples={sample_count}"
        )
    return torch.randperm(sample_count, device=device)[:batch_size]


def load_resume_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a ControlNet run and return its completed global step."""

    checkpoint = load_torch_mapping(path)
    raw_state = checkpoint.get("model_state")
    if not isinstance(raw_state, dict):
        raise ValueError(f"Resume checkpoint has no model_state: {path}")
    model.load_state_dict(raw_state, strict=True)
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError(f"Resume checkpoint has no optimizer_state: {path}")
    optimizer.load_state_dict(optimizer_state)
    return int(checkpoint.get("global_step", 0))


def save_training_checkpoint(
    directory: str | Path,
    *,
    step: int,
    schema: str,
    model_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    extra_state: Mapping[str, Any],
) -> Path:
    """Atomically write one checkpoint and update a lightweight ``last.pt`` link.

    Historical scripts wrote the same hundreds of megabytes twice—once as a
    numbered checkpoint and once as ``last.pt``.  This implementation writes
    the tensors once and makes ``last.pt`` a relative symlink.
    """

    checkpoint_dir = Path(directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"step{step:08d}.pt"
    temporary = checkpoint_dir / f".{destination.name}.tmp"
    payload = {
        "schema": schema,
        "global_step": int(step),
        "model_state": dict(model_state),
        "optimizer_state": optimizer.state_dict(),
        "config": dict(config),
        "extra_state": dict(extra_state),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    last = checkpoint_dir / "last.pt"
    link_temporary = checkpoint_dir / ".last.pt.tmp"
    link_temporary.unlink(missing_ok=True)
    link_temporary.symlink_to(destination.name)
    os.replace(link_temporary, last)
    return destination


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a stable, human-readable JSON mapping."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def format_duration(seconds: float) -> str:
    """Format a non-negative duration as ``HH:MM:SS``."""

    value = max(0, int(seconds))
    return f"{value // 3600:02d}:{(value % 3600) // 60:02d}:{value % 60:02d}"


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    """Count scalar elements across a parameter iterable."""

    return sum(parameter.numel() for parameter in parameters)


__all__ = [
    "configure_cuda_training",
    "count_parameters",
    "format_duration",
    "load_base_motion_dit",
    "load_resume_checkpoint",
    "load_torch_mapping",
    "make_batch_indices",
    "new_motion_dit",
    "patch_fast_kernels",
    "resolve_dtype",
    "save_training_checkpoint",
    "trainable_parameters",
    "write_json",
]

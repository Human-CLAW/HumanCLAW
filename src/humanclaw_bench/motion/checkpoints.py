"""Strict, inference-only loader for the compact paper checkpoints.

The training checkpoints repeated the frozen MotionDiT inside every ControlNet.
Four skills captured the full-precision base, while four captured the exact
``FP32 -> BF16 -> FP32`` round-trip of that same base.  The release stores the
full-precision tensors once, reconstructs the rounded variant deterministically,
and stores only each skill's control branch.  This preserves the evaluated
weights bit-for-bit without shipping eight redundant MotionDiT copies.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from humanclaw_bench.assets import sha256_file, weight_entries
from humanclaw_bench.paths import repository_root, resolve_release_path

from .networks.motion_dit import MotionDiT
from .networks.noncond_ctrl_dit import NonCondCtrlDiT
from .networks.side_walk_ctrl_dit_fourier import SideWalkCtrlDiTFourier
from .networks.sit_ctrl_dit import SitCtrlDiT
from .networks.step_climb_down_ctrl_dit import StepClimbDownCtrlDiT
from .networks.step_climb_up_ctrl_dit import StepClimbUpCtrlDiT
from .networks.turn_ctrl_dit import TurnCtrlDiT
from .networks.walk_forward_ctrl_dit_fourier import WalkForwardCtrlDiTFourier
from .networks.walk_forward_ctrl_dit_fourier_xzyaw import (
    WalkForwardCtrlDiTFourierXZYaw,
)

NETWORKS = {
    "WalkForwardCtrlDiTFourierXZYaw": WalkForwardCtrlDiTFourierXZYaw,
    "SideWalkCtrlDiTFourier": SideWalkCtrlDiTFourier,
    "WalkForwardCtrlDiTFourier": WalkForwardCtrlDiTFourier,
    "TurnCtrlDiT": TurnCtrlDiT,
    "StepClimbUpCtrlDiT": StepClimbUpCtrlDiT,
    "StepClimbDownCtrlDiT": StepClimbDownCtrlDiT,
    "NonCondCtrlDiT": NonCondCtrlDiT,
    "SitCtrlDiT": SitCtrlDiT,
}


def _torch_load(path: Path) -> dict[str, Any]:
    """Load a Torch checkpoint on CPU while supporting older Torch keyword sets."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return value


def _verified_path(root: Path, entry: dict[str, Any], verify: bool) -> Path:
    """Resolve a pinned checkpoint path and verify its exact SHA-256."""

    path = (root / str(entry["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pinned checkpoint missing: {path}")
    if path.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"Checkpoint size mismatch: {path}")
    if verify and sha256_file(path) != str(entry["sha256"]).lower():
        raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
    return path


def load_weight_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the weight manifest and validate its exact-step selection policy."""

    resolved = (
        resolve_release_path(path)
        if path is not None
        else repository_root() / "resources" / "weights" / "paper_fullval_v1.json"
    )
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if manifest.get("selection_policy") != "exact_step_only":
        raise ValueError("Only exact_step_only weight manifests are accepted")
    return manifest


def _manifest_path(path: str | Path | None) -> Path:
    """Resolve the default or caller-supplied manifest path once."""

    return (
        resolve_release_path(path)
        if path is not None
        else repository_root() / "resources" / "weights" / "paper_fullval_v1.json"
    )


def _batch_parent_verified(root: Path, manifest_path: Path) -> bool:
    """Recognize a manifest that the current batch parent already verified.

    A batch launches one subprocess per episode. Rehashing the same gigabyte
    of immutable weights in every child adds no reproducibility value, so the
    parent passes the resolved root and small manifest digest through its child
    environment after checking every weight once. Standalone rollouts have no
    marker and retain normal full verification.
    """

    expected_root = os.environ.get("HUMANCLAW_BATCH_VERIFIED_WEIGHTS_ROOT")
    expected_manifest = os.environ.get(
        "HUMANCLAW_BATCH_VERIFIED_WEIGHT_MANIFEST_SHA256"
    )
    return bool(
        expected_root
        and expected_manifest
        and Path(expected_root).resolve() == root
        and expected_manifest == sha256_file(manifest_path)
    )


def _new_base(architecture: Mapping[str, Any]) -> MotionDiT:
    """Instantiate the one MotionDiT architecture used by every skill."""

    return MotionDiT(
        input_dim=int(architecture["x_dim"]),
        output_dim=int(architecture["x_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        num_layers=int(architecture["num_layers"]),
        num_heads=int(architecture["num_heads"]),
        mlp_ratio=float(architecture["mlp_ratio"]),
        use_qk_norm=bool(architecture["use_qk_norm"]),
    )


def _state_payload(
    root: Path,
    entry: dict[str, Any],
    *,
    verify: bool,
    expected_schema: str,
) -> OrderedDict[str, torch.Tensor]:
    """Load one compact tensor mapping after size/hash and schema validation."""

    path = _verified_path(root, entry, verify)
    checkpoint = _torch_load(path)
    if checkpoint.get("schema") != expected_schema:
        raise ValueError(
            f"Checkpoint {path} has schema {checkpoint.get('schema')!r}; "
            f"expected {expected_schema!r}"
        )
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in raw_state.items()
    ):
        raise ValueError(f"Checkpoint has no tensor state_dict: {path}")
    return OrderedDict(raw_state)


def _base_variant(
    canonical_state: Mapping[str, torch.Tensor],
    architecture: Mapping[str, Any],
    transform: str,
) -> MotionDiT:
    """Build one manifest-declared base variant without mutating another skill."""

    if transform == "identity":
        state = canonical_state
    elif transform == "bf16_roundtrip":
        # This is not an approximation chosen for the release.  It exactly
        # reproduces the base tensors embedded in the four historical paper
        # checkpoints, all 112 of which equal this deterministic round-trip.
        state = OrderedDict(
            (key, value.to(torch.bfloat16).to(torch.float32))
            if value.is_floating_point()
            else (key, value)
            for key, value in canonical_state.items()
        )
    else:
        raise ValueError(f"Unsupported MotionDiT base transform: {transform!r}")
    model = _new_base(architecture)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _attach_forward(model, architecture: dict[str, Any], nfe_steps: int):
    """Bind a checkpoint-compatible forward adapter to a skill control model."""

    model.n_history = int(architecture["n_history"])
    model.n_future = int(architecture["n_future"])
    model.x_dim = int(architecture["x_dim"])
    model.nfe_steps = int(nfe_steps)

    def forward(history, noisy_future, timestep, *condition_args):
        """Run the checkpoints forward pass."""

        output = model(
            torch.cat([history, noisy_future], dim=1), timestep, *condition_args
        )
        return output[:, model.n_history :]

    model._forward = forward
    return model


def load_skill_models(
    skills: Sequence[str],
    *,
    weights_root: str | Path,
    manifest_path: str | Path | None = None,
    device: str = "cuda",
    verify: bool = True,
) -> dict[str, tuple[Any, str]]:
    """Load exact evaluated models from one base plus control-only checkpoints."""

    resolved_manifest = _manifest_path(manifest_path)
    manifest = load_weight_manifest(resolved_manifest)
    root = Path(weights_root).expanduser().resolve()
    requested = tuple(skills)
    unknown = sorted(set(requested) - set(manifest["skills"]))
    if unknown:
        raise ValueError(f"Skills are not pinned in the weight manifest: {unknown}")
    # Validate the complete paper set before unpickling any checkpoint.
    if verify and not _batch_parent_verified(root, resolved_manifest):
        for entry in weight_entries(manifest):
            _verified_path(root, entry, True)
    architecture = manifest["base"]["architecture"]
    canonical_state = _state_payload(
        root,
        manifest["base"],
        verify=False,
        expected_schema="humanclaw_motion_dit_state_v1",
    )
    variants = manifest.get("base_variants")
    if not isinstance(variants, dict):
        raise ValueError("Weight manifest has no base_variants mapping")
    requested_variants = {
        str(manifest["skills"][skill].get("base_variant") or "") for skill in requested
    }
    bases: dict[str, MotionDiT] = {}
    for variant_name in sorted(requested_variants):
        variant = variants.get(variant_name)
        if not isinstance(variant, dict):
            raise ValueError(f"Unknown base variant: {variant_name!r}")
        bases[variant_name] = _base_variant(
            canonical_state,
            architecture,
            str(variant.get("transform") or ""),
        )

    # Construct every model on CPU first.  Models in the same variant group
    # deliberately reference one immutable base module, so GPU memory also
    # contains two bases rather than eight copies.
    pending: dict[str, tuple[Any, str]] = {}
    for skill in requested:
        entry = manifest["skills"][skill]
        variant_name = str(entry["base_variant"])
        base = bases[variant_name]
        cls = NETWORKS[str(entry["network"])]
        kwargs: dict[str, Any] = {
            "base_dit": base,
            "hidden_dim": int(architecture["hidden_dim"]),
        }
        if "n_freqs" in entry:
            kwargs["n_freqs"] = int(entry["n_freqs"])
        model = cls(**kwargs)
        state = _state_payload(
            root,
            entry,
            verify=False,
            expected_schema="humanclaw_control_state_v1",
        )
        incompatible = model.load_state_dict(state, strict=False)
        expected_missing = {f"base_dit.{key}" for key in base.state_dict()}
        if set(incompatible.missing_keys) != expected_missing:
            raise ValueError(
                f"Control state for {skill} has unexpected missing keys: "
                f"{sorted(set(incompatible.missing_keys) ^ expected_missing)}"
            )
        if incompatible.unexpected_keys:
            raise ValueError(
                f"Control state for {skill} has unexpected keys: "
                f"{incompatible.unexpected_keys}"
            )
        model = _attach_forward(model, architecture, int(entry["nfe_steps"]))
        pending[skill] = (model, str(entry["kind"]))

    del canonical_state
    loaded: dict[str, tuple[Any, str]] = {}
    for skill, (model, kind) in pending.items():
        loaded[skill] = (model.to(device).eval(), kind)
    return loaded


__all__ = ["load_skill_models", "load_weight_manifest"]

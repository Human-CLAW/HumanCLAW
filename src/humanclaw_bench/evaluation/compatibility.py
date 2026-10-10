"""Identify effective agent inputs and prevent cross-profile output reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from humanclaw_bench.config import ReleaseConfig, load_config


def runtime_contract(profile: ReleaseConfig) -> dict[str, Any]:
    """Record version choices plus a digest of the complete experiment profile."""

    agent = profile.section("agent")
    # Keep dispatch/resume usable without importing the rollout dependencies.
    # v5 is the bundled proprioceptive input protocol; v4 is the paper protocol.
    feedback = agent["prompt_version"] == "v5" and bool(agent.get("use_feedback", True))
    return {
        "profile": profile.profile,
        "profile_sha256": hashlib.sha256(
            json.dumps(profile.data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_version": agent["prompt_version"],
        "verifier_version": agent["verifier_version"],
        "feedback_version": "v1" if feedback else None,
        "max_history": max(int(agent["max_history"] or 0), 10),
        "plan_horizon_steps": int(agent["plan_horizon_steps"]),
    }


def validate_existing_rollout(output_dir: Path, profile: ReleaseConfig) -> None:
    """Refuse incompatible or unidentifiable output before skipping or clearing it."""

    manifest_path = output_dir / "replay_manifest.json"
    if not manifest_path.exists():
        has_outputs = any(output_dir.glob("step???_*.json")) or any(
            (output_dir / name).exists()
            for name in ("trajectory_before.npz", "trajectory_after.npz")
        )
        if not has_outputs:
            return
        raise ValueError(
            f"Cannot identify existing rollout without replay_manifest.json: {output_dir}. "
            "Use a separate output directory."
        )
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("manifest must be an object")
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot verify existing rollout {output_dir}; use a separate output directory"
        ) from error
    stored = metadata.get("runtime_contract")
    if stored is None:
        # Historical paper archives predate version metadata. Only the exact
        # bundled paper profile may reuse them; a new profile cannot infer it.
        paper = load_config("paper_fullval_v1")
        if metadata.get("profile") == paper.profile and profile.data == paper.data:
            return
    elif stored == runtime_contract(profile):
        return
    raise ValueError(
        f"Existing rollout profile/agent versions do not match {profile.profile!r}: {output_dir}. "
        "Use a separate output directory for this experiment."
    )

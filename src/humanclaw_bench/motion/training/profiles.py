"""Load and validate versioned motion-training profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from humanclaw_bench.paths import repository_root, resolve_release_path

PROFILE_SCHEMA = "humanclaw_motion_training_profile_v1"


def default_profile_path() -> Path:
    """Return the release-pinned paper training profile."""

    return (
        repository_root()
        / "resources"
        / "motion"
        / "training"
        / "paper_training_v1.json"
    )


def load_training_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load a profile and reject unknown schemas or incomplete skill entries."""

    profile_path = default_profile_path() if path is None else resolve_release_path(path)
    value = json.loads(profile_path.read_text(encoding="utf-8"))
    if value.get("schema") != PROFILE_SCHEMA:
        raise ValueError(
            f"Training profile {profile_path} has schema {value.get('schema')!r}; "
            f"expected {PROFILE_SCHEMA!r}"
        )
    if not isinstance(value.get("architecture"), dict):
        raise ValueError(f"Training profile has no architecture: {profile_path}")
    skills = value.get("skills")
    if not isinstance(skills, dict) or not skills:
        raise ValueError(f"Training profile has no skills: {profile_path}")
    for skill_name, skill in skills.items():
        if not isinstance(skill, dict):
            raise ValueError(f"Skill profile is not a mapping: {skill_name}")
        for field in ("network", "manifest", "expected_samples", "repeat_factor"):
            if field not in skill:
                raise ValueError(f"Skill {skill_name} lacks {field}")
    value["_path"] = str(profile_path)
    return value


def canonical_skill_name(name: str) -> str:
    """Map the legacy internal action key to the public Stop name."""

    normalized = name.strip().lower().replace("-", "_")
    return "stop" if normalized == "stand" else normalized


def selected_skill_profile(
    profile: dict[str, Any], name: str
) -> tuple[str, dict[str, Any]]:
    """Return one canonical skill name and its merged default configuration."""

    skill_name = canonical_skill_name(name)
    try:
        specific = dict(profile["skills"][skill_name])
    except KeyError as error:
        choices = ", ".join(sorted(profile["skills"]))
        raise ValueError(f"Unknown skill {name!r}; choose one of: {choices}") from error
    merged = dict(profile.get("control_defaults", {}))
    merged.update(specific)
    return skill_name, merged


__all__ = [
    "canonical_skill_name",
    "default_profile_path",
    "load_training_profile",
    "selected_skill_profile",
]

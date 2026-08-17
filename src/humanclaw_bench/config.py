"""Load complete, non-inheriting release profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import repository_root, resolve_release_path

# A release profile is a complete experiment definition, not an inheritance
# fragment.  Model credentials remain external, but every benchmark/runtime
# policy—including optional metric thresholds—is present in this one file.
REQUIRED_SECTIONS = {
    "benchmark",
    "agent",
    "vlm",
    "motion",
    "physics",
    "rendering",
    "metrics",
}


@dataclass(frozen=True)
class ReleaseConfig:
    """Immutable view of one complete release profile and its source path."""

    path: Path
    data: dict[str, Any]

    @property
    def profile(self) -> str:
        """Return the profile's stable public name."""

        return str(self.data["profile"])

    def section(self, name: str) -> dict[str, Any]:
        """Return one required profile section as a dictionary."""

        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Profile section {name!r} is not an object")
        return value

    def path_value(self, section: str, key: str) -> Path:
        """Resolve a required path-valued setting relative to the release root."""

        value = self.section(section).get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing path {section}.{key}")
        return resolve_release_path(value)


def load_config(profile_or_path: str | Path = "paper_fullval_v1") -> ReleaseConfig:
    """Load and validate one complete, non-inheriting release profile."""

    candidate = Path(profile_or_path)
    if not candidate.suffix and not candidate.exists():
        candidate = repository_root() / "configs" / f"{candidate.name}.json"
    elif not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Release profile not found: {candidate}")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("profile"):
        raise ValueError(f"Invalid release profile: {candidate}")
    missing = sorted(REQUIRED_SECTIONS - set(data))
    if missing:
        raise ValueError(f"Profile is incomplete; missing sections: {missing}")
    if "extends" in data:
        raise ValueError(
            "Release profiles must be self-contained and may not use 'extends'"
        )
    metrics = data.get("metrics")
    required_metrics = {
        "find_pixel_threshold",
        "nav_distance_m",
        "nav_relaxed_distance_m",
        "collision_contact_source",
        "fixed_contact_min_height_m",
        "initial_penetration_threshold_m",
        "jerk_neutral_body22",
        "jerk_smooth_window",
        "jerk_stride",
    }
    if not isinstance(metrics, dict):
        raise ValueError("Profile metrics section must be an object")
    missing_metrics = sorted(required_metrics - set(metrics))
    if missing_metrics:
        raise ValueError(
            "Profile metrics section is incomplete; missing: "
            + ", ".join(missing_metrics)
        )
    if metrics.get("collision_contact_source") != "post_physics_30hz":
        raise ValueError(
            "metrics.collision_contact_source must be 'post_physics_30hz'"
        )
    neutral_resource = metrics.get("jerk_neutral_body22")
    if not isinstance(neutral_resource, str) or not neutral_resource:
        raise ValueError("metrics.jerk_neutral_body22 must be a release path")
    neutral_path = resolve_release_path(neutral_resource)
    if not neutral_path.is_file():
        raise FileNotFoundError(
            f"Motion Jerk neutral-body resource not found: {neutral_path}"
        )
    return ReleaseConfig(candidate, data)

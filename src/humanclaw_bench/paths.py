"""Portable repository and resource path discovery."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    """Locate the release root by walking upward from the installed source file."""

    override = os.environ.get("HUMANCLAW_BENCH_HOME")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(
                f"HUMANCLAW_BENCH_HOME is not a release root: {root}"
            )
        return root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "resources").is_dir():
            return parent
    raise RuntimeError("Cannot locate HumanClawBench root; set HUMANCLAW_BENCH_HOME")


def resolve_release_path(value: str | Path) -> Path:
    """Resolve an absolute or release-root-relative path without changing it."""

    path = Path(value).expanduser()
    return (
        path.resolve() if path.is_absolute() else (repository_root() / path).resolve()
    )

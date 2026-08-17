"""Versioned action verifiers."""

from __future__ import annotations

import importlib
import re
from types import ModuleType

_VERSION = re.compile(r"v[0-9]+")


def resolve_verifier_version(version: str) -> ModuleType:
    """Load one self-contained verifier module by version."""

    name = str(version)
    if _VERSION.fullmatch(name) is None:
        raise ValueError(f"Invalid verifier version: {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as error:
        if error.name == f"{__name__}.{name}":
            raise ValueError(f"Verifier version is not bundled: {name}") from error
        raise
    required = ("verifier_prompt", "verifier_action", "normalize_verifier_plan")
    if getattr(module, "VERSION", None) != name or not all(
        callable(getattr(module, item, None)) for item in required
    ):
        raise ValueError(f"Invalid verifier module: {module.__name__}")
    return module


__all__ = ["resolve_verifier_version"]

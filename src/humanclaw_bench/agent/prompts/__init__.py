"""Versioned planner prompts.

Each version is a self-contained module.  A new prompt can be added as, for
example, ``v5.py`` without modifying the planner or inheriting and patching an
older prompt at import time.
"""

from __future__ import annotations

import importlib
import re
from types import ModuleType

_VERSION = re.compile(r"v[0-9]+")


def resolve_prompt_version(version: str) -> ModuleType:
    """Load one self-contained planner-prompt module by version."""

    name = str(version)
    if _VERSION.fullmatch(name) is None:
        raise ValueError(f"Invalid planner prompt version: {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as error:
        if error.name == f"{__name__}.{name}":
            raise ValueError(
                f"Planner prompt version is not bundled: {name}"
            ) from error
        raise
    if getattr(module, "VERSION", None) != name or not callable(
        getattr(module, "render", None)
    ):
        raise ValueError(f"Invalid planner prompt module: {module.__name__}")
    return module


__all__ = ["resolve_prompt_version"]

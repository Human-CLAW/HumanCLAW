"""Lazy environment exports so setup commands do not import Habitat."""

from typing import Any

__all__ = ["HalfPhysicsEnv", "HCFindNavInteractEnv"]


def __getattr__(name: str) -> Any:
    """Lazily expose environment classes without importing Habitat eagerly."""

    if name == "HalfPhysicsEnv":
        from .half_physics_env import HalfPhysicsEnv

        return HalfPhysicsEnv
    if name == "HCFindNavInteractEnv":
        from .find_nav_interact_env import HCFindNavInteractEnv

        return HCFindNavInteractEnv
    raise AttributeError(name)

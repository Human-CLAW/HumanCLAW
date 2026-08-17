"""Small, in-memory contact records for optional benchmark metrics.

Habitat exposes Bullet contact objects whose fields are tied to the current
simulator frame.  The metric path needs a stable Python representation, but the
normal rollout path should not pay for contact detection or serialization.
Consequently this module contains only stateless helpers and is imported by
``HalfPhysicsEnv`` when ``compute_metrics`` is enabled.

Nothing produced here is written as a contact trace.  The episode metric
recorder consumes these records immediately for disturbance and interaction
scores, then discards them.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _vec3(value: Any) -> np.ndarray:
    """Convert Magnum, list, or numpy vectors to a float64 xyz vector."""

    if hasattr(value, "x"):
        return np.asarray([value.x, value.y, value.z], dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] < 3:
        raise ValueError(f"Expected a 3-vector, got shape={array.shape}")
    return array[:3]


def _xyz(value: Any) -> list[float]:
    """Convert a vector-like contact value into a JSON-safe xyz list."""

    return [float(component) for component in _vec3(value)]


def _describe_sim_object(env: Any, object_id: int) -> dict[str, Any]:
    """Resolve a Bullet object id without relying on private scene metadata."""

    object_id = int(object_id)
    if object_id == int(env.agent.object_id):
        return {"object_id": object_id, "type": "agent", "name": "agent"}

    # Habitat exposes the stage ID as a module-level runtime constant; its
    # numeric value differs across builds.  Reading the active build's value
    # keeps object classification portable instead of assuming literal 0.
    runtime = env._require_runtime()
    stage_id = int(getattr(runtime.habitat_sim, "stage_id", -1))
    if object_id == stage_id:
        return {"object_id": object_id, "type": "stage", "name": "stage"}

    for manager_name, getter in (
        ("rigid_object", env.sim.get_rigid_object_manager),
        ("articulated_object", env.sim.get_articulated_object_manager),
    ):
        try:
            manager = getter()
            if not manager.get_library_has_id(object_id):
                continue
            obj = manager.get_object_by_id(object_id)
            row = {
                "object_id": object_id,
                "type": manager_name,
                "name": str(getattr(obj, "handle", object_id)),
            }
            if manager_name == "rigid_object":
                row["motion_type"] = str(getattr(obj, "motion_type", ""))
            return row
        except Exception:
            continue
    return {"object_id": object_id, "type": "unknown", "name": str(object_id)}


def _agent_contact(env: Any, point: Any, frame_index: int) -> dict[str, Any] | None:
    """Convert one active contact involving the humanoid into a compact row."""

    agent_id = int(env.agent.object_id)
    object_a = int(point.object_id_a)
    object_b = int(point.object_id_b)
    if object_a == agent_id:
        agent_link = int(point.link_id_a)
        other_id = object_b
        other_link = int(point.link_id_b)
        on_agent = point.position_on_a_in_ws
        on_other = point.position_on_b_in_ws
    elif object_b == agent_id:
        agent_link = int(point.link_id_b)
        other_id = object_a
        other_link = int(point.link_id_a)
        on_agent = point.position_on_b_in_ws
        on_other = point.position_on_a_in_ws
    else:
        return None

    return {
        "frame_index": int(frame_index),
        "body_part": env._link_id_to_name.get(agent_link, f"link_{agent_link}"),
        "other": _describe_sim_object(env, other_id),
        "other_object_id": int(other_id),
        "other_link_id": int(other_link),
        "position_on_agent_ws": _xyz(on_agent),
        "position_on_other_ws": _xyz(on_other),
        "contact_distance": float(point.contact_distance),
        "normal_force": float(point.normal_force),
    }


def _dynamic_pair(point: Any, frame_index: int, env: Any) -> dict[str, Any] | None:
    """Return a row only for a dynamic-rigid to dynamic-rigid contact."""

    side_a = _describe_sim_object(env, int(point.object_id_a))
    side_b = _describe_sim_object(env, int(point.object_id_b))

    def is_dynamic(side: dict[str, Any]) -> bool:
        """Return whether a contact side describes a dynamic rigid object."""

        return (
            side.get("type") == "rigid_object"
            and "DYNAMIC" in str(side.get("motion_type", "")).upper()
        )

    if not (is_dynamic(side_a) and is_dynamic(side_b)):
        return None
    return {
        "frame_index": int(frame_index),
        "a": side_a,
        "b": side_b,
    }


def collect_metric_contacts(
    env: Any,
    frame_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one discrete query and share its result across all metric consumers.

    A single call feeds fixed-geometry collision, interaction contact,
    movable-object disturbance, and the reset-time penetration check. Keeping
    this as one query is important: repeated
    ``perform_discrete_collision_detection`` calls were a measurable source of
    rollout overhead in the earlier research code.
    """

    env.sim.perform_discrete_collision_detection()
    agent_rows: list[dict[str, Any]] = []
    dynamic_rows: list[dict[str, Any]] = []
    for point in env.sim.get_physics_contact_points():
        if not bool(point.is_active):
            continue
        agent_row = _agent_contact(env, point, frame_index)
        if agent_row is not None:
            agent_rows.append(agent_row)
        dynamic_row = _dynamic_pair(point, frame_index, env)
        if dynamic_row is not None:
            dynamic_rows.append(dynamic_row)
    return agent_rows, dynamic_rows


__all__ = ["collect_metric_contacts"]

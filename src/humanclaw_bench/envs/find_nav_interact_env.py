"""Find/Nav/Interact task semantics on top of HalfPhysics.

Normal rollouts use this class only for lighting and explicit Stop/Stand.  When
``compute_metrics`` is enabled, the same class also resolves the episode's goal
instances, assigns temporary semantic IDs for FindSR, exposes terminal AABB
geometry for NavSR, and identifies pelvis-to-target mesh contacts for
InteractSR.  None of that target work runs in the default path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from humanclaw_bench.envs.half_physics_env import (
    HalfPhysicsEnv,
    HalfPhysicsObservation,
    _as_rgb_array,
)
from humanclaw_bench.evaluation.metrics.geometry import (
    body_to_target_aabb_distance,
)

INTERACT_CATEGORIES = {"bed", "couch", "toilet"}


def _field(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """Read the first available field from either a mapping or attribute object."""

    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _template_hash(value: Any) -> str:
    """Normalize a Habitat object handle to its source template hash."""

    text = str(value or "").strip()
    if ":" in text:
        text = text.split(":", 1)[0]
    return text.rstrip("_")


def _vec3(value: Any) -> np.ndarray:
    """Convert a Magnum or array-like vector into three float64 components."""

    if hasattr(value, "x"):
        return np.asarray([value.x, value.y, value.z], dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape[0] < 3:
        raise ValueError(f"Expected a 3-vector, got {array.shape}")
    return array[:3]


def action_requests_stop(action: Any, reasoning: Any = None) -> bool:
    """Return whether an action explicitly commits Stop/Stand."""

    def one(value: Any) -> bool:
        """Evaluate one candidate value for the enclosing operation."""

        if value is None:
            return False
        if isinstance(value, dict):
            if bool(value.get("stop", False)) or bool(value.get("at_target", False)):
                return True
            skill = str(value.get("skill") or value.get("action_name") or "").lower()
            if skill in {"stand", "stop", "stop/stand"}:
                return True
            nested = value.get("action")
            return one(nested) if nested is not None else False
        if bool(getattr(value, "stop", False)) or bool(
            getattr(value, "at_target", False)
        ):
            return True
        skill = str(
            getattr(value, "skill", "") or getattr(value, "action_name", "")
        ).lower()
        return skill in {"stand", "stop", "stop/stand"}

    return one(action) or one(reasoning)


class HCFindNavInteractEnv(HalfPhysicsEnv):
    """HalfPhysics plus benchmark lighting and optional paper metrics."""

    def __init__(
        self,
        *,
        lighting: str = "ambient",
        ambient_strength: float = 1.2,
        room_light_strength: float = 1.0,
        compute_metrics: bool = False,
        target_semantic_id_base: int = 900000,
        **kwargs: Any,
    ) -> None:
        """Configure optional paper metrics and task-specific lighting around HalfPhysics."""

        self.compute_metrics = bool(compute_metrics)
        if self.compute_metrics:
            kwargs["ego_semantic_enabled"] = True
            kwargs["collect_metric_contacts"] = True
        super().__init__(**kwargs)
        self.lighting = str(lighting)
        self.ambient_strength = float(ambient_strength)
        self.room_light_strength = float(room_light_strength)
        self.target_semantic_id_base = int(target_semantic_id_base)
        self._category = ""
        self._target_refs: list[dict[str, str]] = []
        self._target_semantic_ids: list[int] = []

    def reset(self, episode: Any = None, **kwargs: Any) -> HalfPhysicsObservation:
        # Resolve and label targets before the first semantic render.  The
        # prepared scene handles contain the source template hash, and the goal
        # center disambiguates repeated instances of the same template.
        """Resolve metric targets, reset physics, apply lighting, and return the first ego view."""

        if self.compute_metrics and episode is not None:
            self._require_runtime()
            self._category = str(
                _field(episode, ("object_category", "category"), "")
            ).lower()
            specs = list(_field(episode, ("goal_objects",), []) or [])
            self._target_refs = self._resolve_target_refs(specs)
            if not self._target_refs:
                raise ValueError(
                    "Metric mode could not resolve any goal object for "
                    f"scene={Path(self.scene_id).name}, category={self._category}"
                )
            self._assign_target_semantic_ids()
        else:
            self._category = ""
            self._target_refs = []
            self._target_semantic_ids = []

        observation = super().reset(episode, **kwargs)
        if self.sim is not None:
            observation = self._apply_lighting_and_rerender(observation)
        return observation

    def _apply_lighting_and_rerender(
        self, fallback: HalfPhysicsObservation
    ) -> HalfPhysicsObservation:
        """Apply the requested scene lighting and refresh the initial sensor observation."""

        if self.lighting == "original":
            return fallback
        runtime = self._require_loaded_modules()
        mn = runtime.mn
        from habitat_sim.gfx import LightInfo, LightPositionModel

        if self.lighting == "ambient":
            strength = self.ambient_strength
            lights = [
                LightInfo(
                    vector=mn.Vector4(0.0, -1.0, 0.0, 0.0),
                    color=mn.Vector3(0.92 * strength),
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=mn.Vector4(0.0, -0.45, -1.0, 0.0),
                    color=mn.Vector3(0.69 * strength),
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=mn.Vector4(0.0, -0.45, 1.0, 0.0),
                    color=mn.Vector3(0.46 * strength),
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=mn.Vector4(-1.0, -0.35, 0.0, 0.0),
                    color=mn.Vector3(0.37 * strength),
                    model=LightPositionModel.Global,
                ),
                LightInfo(
                    vector=mn.Vector4(1.0, -0.35, 0.0, 0.0),
                    color=mn.Vector3(0.37 * strength),
                    model=LightPositionModel.Global,
                ),
            ]
        elif self.lighting == "room-lamps":
            root = _vec3(self.agent.translation)
            strength = self.room_light_strength
            color = mn.Vector3(1.0 * strength, 0.96 * strength, 0.86 * strength)
            positions = (
                (root[0] - 2.0, root[1] + 2.8, root[2] - 2.0),
                (root[0] + 2.0, root[1] + 2.8, root[2] - 2.0),
                (root[0] - 2.0, root[1] + 2.8, root[2] + 2.0),
                (root[0] + 2.0, root[1] + 2.8, root[2] + 2.0),
            )
            lights = [
                LightInfo(
                    vector=mn.Vector4(float(x), float(y), float(z), 1.0),
                    color=color,
                    model=LightPositionModel.Global,
                )
                for x, y, z in positions
            ]
        else:
            raise ValueError(f"Unknown lighting mode: {self.lighting}")

        self.sim.set_light_setup(lights)
        self._update_cameras()
        sensors = dict(self.sim.get_sensor_observations())
        self._last_obs = HalfPhysicsObservation(
            head_rgb=_as_rgb_array(sensors["ego_rgb"])
        )
        self._last_semantic = (
            np.asarray(sensors["ego_semantic"]) if "ego_semantic" in sensors else None
        )
        self._last_third_person_rgb = (
            _as_rgb_array(sensors["third_person_rgb"])
            if "third_person_rgb" in sensors
            else None
        )
        return self._last_obs

    def _object_managers(self) -> list[tuple[str, Any]]:
        """Return Habitat rigid and articulated object managers when available."""

        return [
            ("rigid", self.sim.get_rigid_object_manager()),
            ("articulated", self.sim.get_articulated_object_manager()),
        ]

    def _resolve_target_refs(
        self, specs: Sequence[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Match episode goal specifications to concrete Habitat object handles."""

        manager_handles: list[tuple[str, Any, str]] = []
        for manager_name, manager in self._object_managers():
            try:
                handles = list(manager.get_object_handles())
            except Exception:
                handles = []
            for handle in handles:
                manager_handles.append((manager_name, manager, str(handle)))

        refs: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for spec in specs:
            template = str(spec.get("template_hash") or "").strip()
            if not template:
                template = _template_hash(spec.get("object_name"))
            if not template:
                continue
            center = np.asarray(spec.get("center"), dtype=np.float64).reshape(-1)
            best: tuple[float, str, str] | None = None
            for manager_name, manager, handle in manager_handles:
                if template not in handle:
                    continue
                distance = 0.0
                if center.shape[0] >= 3:
                    obj = manager.get_object_by_handle(handle)
                    position = _vec3(obj.translation)
                    distance = float(
                        (position[0] - center[0]) ** 2 + (position[2] - center[2]) ** 2
                    )
                if best is None or distance < best[0]:
                    best = (distance, manager_name, handle)
            if best is None:
                continue
            ref = {"manager": best[1], "handle": best[2]}
            key = (ref["manager"], ref["handle"])
            if key not in seen:
                refs.append(ref)
                seen.add(key)
        return refs

    def _object_for_ref(self, ref: dict[str, str]) -> Any | None:
        """Resolve one stored target reference back to its live Habitat object."""

        try:
            manager = dict(self._object_managers())[ref["manager"]]
            return manager.get_object_by_handle(ref["handle"])
        except Exception:
            return None

    def _target_objects(self) -> list[tuple[dict[str, str], Any]]:
        """Return all currently resolved live target objects."""

        result: list[tuple[dict[str, str], Any]] = []
        for ref in self._target_refs:
            obj = self._object_for_ref(ref)
            if obj is not None:
                result.append((ref, obj))
        return result

    def _assign_target_semantic_ids(self) -> None:
        """Assign stable temporary semantic IDs to every resolved target instance."""

        self._target_semantic_ids = []
        for index, (_ref, obj) in enumerate(self._target_objects()):
            semantic_id = self.target_semantic_id_base + index
            obj.semantic_id = semantic_id
            self._target_semantic_ids.append(semantic_id)

    def metric_find_observation(self) -> dict[str, Any]:
        """Count goal pixels in the semantic image already rendered for VLM input."""

        if not self.compute_metrics:
            raise RuntimeError("metric_find_observation requires compute_metrics")
        if self._last_semantic is None or not self._target_semantic_ids:
            return {"available": False, "target_pixel_count": 0}
        semantic = np.asarray(self._last_semantic)
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[:, :, 0]
        count = int(np.isin(semantic, np.asarray(self._target_semantic_ids)).sum())
        return {"available": True, "target_pixel_count": count}

    def _target_aabbs(self) -> list[tuple[list[float], list[float]]]:
        """Return world-space AABBs for all resolved target instances."""

        runtime = self._require_runtime()
        result: list[tuple[list[float], list[float]]] = []
        for _ref, obj in self._target_objects():
            try:
                node = obj.root_scene_node
                bounds = runtime.habitat_sim.geo.get_transformed_bb(
                    node.cumulative_bb,
                    node.absolute_transformation(),
                )
                result.append(
                    (
                        [float(bounds.min[i]) for i in range(3)],
                        [float(bounds.max[i]) for i in range(3)],
                    )
                )
            except Exception:
                continue
        return result

    def _body_points(self) -> list[list[float]]:
        """Collect world-space human link points used by navigation distance metrics."""

        points: list[list[float]] = []

        def add(value: Any) -> None:
            """Append one finite three-dimensional point to the body-point list."""

            try:
                point = _vec3(value)
            except Exception:
                return
            points.append([float(component) for component in point])

        # Match the paper evaluator: include pelvis, articulated root, and each
        # link origin when measuring body-to-target AABB distance.
        add(
            _vec3(self.agent.translation)
            - self.world_transformation.apply(self.original_root_shift)
        )
        add(self.agent.translation)
        for link_id in range(self.agent.num_links):
            try:
                node = self.agent.get_link_scene_node(link_id)
                transform = (
                    node.absolute_transformation()
                    if hasattr(node, "absolute_transformation")
                    else node.transformation
                )
                add(transform.translation)
            except Exception:
                continue
        return points

    def metric_target_geometry(self) -> dict[str, Any]:
        """Return the final body-to-target distance used by both Nav variants."""

        aabbs = self._target_aabbs()
        if not aabbs:
            raise RuntimeError("Metric mode lost all resolved target AABBs")
        points = self._body_points()
        return {
            "body_target_aabb_distance_m": body_to_target_aabb_distance(points, aabbs),
            "body_point_count": len(points),
            "target_aabb_count": len(aabbs),
        }

    def is_interact_episode(self) -> bool:
        """Return whether this target category supports the Sit interaction task."""

        return self._category in INTERACT_CATEGORIES

    def is_pelvis_target_contact(self, contact: dict[str, Any]) -> bool:
        """Match a pelvis contact to an exact resolved goal mesh instance."""

        if str(contact.get("body_part") or "").lower() not in {"pelvis", "base"}:
            return False
        target_ids: set[int] = set()
        target_handles: set[str] = set()
        target_templates: set[str] = set()
        for ref, obj in self._target_objects():
            target_handles.add(ref["handle"])
            target_templates.add(_template_hash(ref["handle"]))
            try:
                target_ids.add(int(obj.object_id))
            except Exception:
                pass
        other = contact.get("other") or {}
        try:
            if (
                int(contact.get("other_object_id", other.get("object_id")))
                in target_ids
            ):
                return True
        except (TypeError, ValueError):
            pass
        handle = str(other.get("name") or "")
        return handle in target_handles or any(
            template and template in handle for template in target_templates
        )

    def step(
        self,
        action: Any,
        reasoning: Any = None,
        i_flag: int | None = None,
    ) -> tuple[HalfPhysicsObservation, float, bool, dict[str, Any]]:
        """Execute a motion action and attach Find/Nav/Interact metric observations when enabled."""

        if action_requests_stop(action, reasoning):
            return self._stop_step()
        return super().step(action, reasoning=reasoning, i_flag=i_flag)

    def _stop_step(
        self,
    ) -> tuple[HalfPhysicsObservation, float, bool, dict[str, Any]]:
        """Commit an explicit Stop/Stand without generating another motion chunk."""

        if not self._reset or self._last_obs is None:
            raise RuntimeError("Reset env before stepping.")
        self._current_step += 1
        return self._last_obs, 0.0, True, {}


__all__ = ["HCFindNavInteractEnv", "action_requests_stop"]

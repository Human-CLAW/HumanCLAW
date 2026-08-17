"""Collision metric from realized 30 Hz post-physics poses.

HalfPhysics already performs one discrete contact query after every requested
30 Hz motion frame when metric mode is enabled.  The collision, interaction,
and disturbance metrics consume that same in-memory contact stream.  No
second pose interpolation pass, physics replay, or contact trace is needed.

A motion decision is marked as colliding when any of its realized 30 Hz poses
has a fixed-geometry contact sufficiently above the episode floor.  The
episode score remains a fraction of motion decisions, matching the public
``collision_step_fraction`` field.
"""

from __future__ import annotations

from typing import Any

from humanclaw_bench.envs.runtime_records import collect_metric_contacts

ARM_LINKS = {
    "left_collar",
    "right_collar",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    *{
        f"{side}_{finger}{joint}"
        for side in ("left", "right")
        for finger in ("index", "middle", "pinky", "ring", "thumb")
        for joint in (1, 2, 3)
    },
}
HEAD_LINKS = {"head", "jaw", "left_eye_smplhf", "right_eye_smplhf"}
LEG_LINKS = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
}


def body_group(link_name: str) -> str:
    """Map an articulated link name to the paper's body-part group."""

    name = str(link_name)
    if name in HEAD_LINKS:
        return "head"
    if name in ARM_LINKS:
        return "hand_arm"
    if name in LEG_LINKS:
        return "leg"
    return "torso"


def _agent_min_y(env: Any) -> float:
    """Return the lowest finite world-y point on the articulated human."""

    runtime = env._require_runtime()
    minimum: float | None = None
    nodes = []
    try:
        nodes.append(env.agent.root_scene_node)
    except Exception:
        pass
    for link_id in range(env.agent.num_links):
        try:
            nodes.append(env.agent.get_link_scene_node(link_id))
        except Exception:
            pass
    for node in nodes:
        try:
            transform = (
                node.absolute_transformation()
                if hasattr(node, "absolute_transformation")
                else node.transformation
            )
            bounds = runtime.habitat_sim.geo.get_transformed_bb(
                node.cumulative_bb, transform
            )
            value = float(bounds.min[1])
            minimum = value if minimum is None else min(minimum, value)
        except Exception:
            continue
    return float(env.agent.translation.y) if minimum is None else minimum


def _episode_floor_y(
    env: Any, max_distance: float, epsilon: float
) -> tuple[float, str]:
    """Cast one spawn ray and reuse that deterministic floor for the episode."""

    runtime = env._require_runtime()
    reference = _agent_min_y(env)
    origin = runtime.mn.Vector3(
        float(env.agent.translation.x),
        float(env.agent.translation.y) + 50.0,
        float(env.agent.translation.z),
    )
    ray = runtime.habitat_sim.geo.Ray(origin, runtime.mn.Vector3(0.0, -1.0, 0.0))
    try:
        result = env.sim.cast_ray(ray, float(max_distance))
    except TypeError:
        try:
            result = env.sim.cast_ray(ray)
        except Exception:
            result = None
    except Exception:
        result = None
    hits: list[float] = []
    agent_id = int(env.agent.object_id)
    for hit in getattr(result, "hits", []):
        if int(getattr(hit, "object_id", -999)) == agent_id:
            continue
        try:
            hits.append(float(hit.point[1]))
        except Exception:
            pass
    eligible = [height for height in hits if height <= reference + float(epsilon)]
    if eligible:
        return max(eligible), "spawn_raycast"
    return reference, "agent_min_y_fallback"


def _fixed_contact(contact: dict[str, Any]) -> bool:
    """Return whether a contact is against stage or non-dynamic geometry."""

    other = contact.get("other") or {}
    if (
        other.get("type") == "rigid_object"
        and "DYNAMIC" in str(other.get("motion_type", "")).upper()
    ):
        return False
    return str(other.get("type") or "") in {
        "stage",
        "rigid_object",
        "articulated_object",
        "unknown",
    }


def _contact_height(contact: dict[str, Any]) -> float | None:
    """Measure a contact's world-y position on either contact side."""

    for key in ("position_on_other_ws", "position_on_agent_ws"):
        value = contact.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[1])
    return None


class CollisionTracker:
    """Accumulate collision step sets from the shared 30 Hz contact stream."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Store thresholds; the reset state is recorded explicitly later."""

        self.threshold = float(config.get("initial_penetration_threshold_m", 0.01))
        self.height_threshold = float(
            config.get("fixed_contact_min_height_m", 0.0205)
        )
        self.floor_max_distance = float(
            config.get("floor_ray_max_distance_m", 100.0)
        )
        self.floor_epsilon = float(config.get("floor_select_epsilon_m", 0.05))
        self.floor_y: float | None = None
        self.floor_source = ""
        self.initial_depth: float | None = None
        self.motion_steps: set[int] = set()
        self.collision_steps: set[int] = set()
        self.group_steps = {
            group: set() for group in ("hand_arm", "torso", "leg", "head")
        }

    def record_reset(self, env: Any) -> None:
        """Measure floor and initial penetration before the first physics frame."""

        if self.initial_depth is not None:
            raise RuntimeError("Collision reset state was recorded more than once")
        self.floor_y, self.floor_source = _episode_floor_y(
            env,
            self.floor_max_distance,
            self.floor_epsilon,
        )
        contacts, _ = collect_metric_contacts(env, 0)
        self.initial_depth = max(
            (
                max(0.0, -float(row.get("contact_distance", 0.0)))
                for row in contacts
            ),
            default=0.0,
        )

    def record_step(
        self,
        step: int,
        metric_frames: dict[str, Any] | None,
    ) -> None:
        """Consume contacts already queried at each realized 30 Hz pose."""

        step = int(step)
        self.motion_steps.add(step)
        agent_frames = list(dict(metric_frames or {}).get("agent_contacts") or [])
        if self.floor_y is None:
            raise RuntimeError("Record the collision reset state before motion")
        for contacts in agent_frames:
            for contact in contacts or []:
                height = _contact_height(contact)
                if (
                    not _fixed_contact(contact)
                    or height is None
                    or height - self.floor_y <= self.height_threshold
                ):
                    continue
                self.collision_steps.add(step)
                group = body_group(str(contact.get("body_part") or ""))
                self.group_steps[group].add(step)

    def finalize(self) -> dict[str, Any]:
        """Return the compact episode result without another simulator pass."""

        if self.initial_depth is None or self.floor_y is None:
            raise RuntimeError("Collision reset state was not recorded")
        detected = self.initial_depth > self.threshold
        base = {
            "initial_penetration_depth_m": float(self.initial_depth),
            "initial_penetration_threshold_m": self.threshold,
            "initial_penetration_detected": bool(detected),
            # Full-val always uses all 1,218 episodes.  The reset measurement
            # remains visible for diagnosis but never changes a denominator.
            "initial_penetration_excluded": False,
            "episode_floor_y": float(self.floor_y),
            "episode_floor_source": self.floor_source,
            "collision_contact_source": "post_physics_30hz",
        }
        denominator = len(self.motion_steps)
        return {
            **base,
            "collision_step_fraction": (
                len(self.collision_steps) / denominator if denominator else 0.0
            ),
            "collision_steps": int(len(self.collision_steps)),
            "motion_steps": int(denominator),
            "fixed_contact_min_height_m": self.height_threshold,
            "by_body_group_step_fraction": {
                group: (len(steps) / denominator if denominator else 0.0)
                for group, steps in self.group_steps.items()
            },
        }


__all__ = ["CollisionTracker", "body_group"]

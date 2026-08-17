"""One flag-gated recorder for all metrics reported in the paper.

The recorder is intentionally the only orchestration layer for metrics.  It
receives semantic counts, VLM decisions, and one shared 30 Hz contact stream
while the rollout runs.  Collision, interaction, and disturbance update small
in-memory accumulators from that stream; no contact rows are serialized and no
second collision pass runs at episode end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from humanclaw_bench.evaluation.metrics.collision import CollisionTracker
from humanclaw_bench.evaluation.metrics.disturbance import DisturbanceTracker
from humanclaw_bench.evaluation.metrics.find import claims_target_visible
from humanclaw_bench.evaluation.metrics.jerk import (
    load_neutral_body22,
    root_rigid_motion_jerk,
)
from humanclaw_bench.evaluation.metrics.usage import UsageTracker


def _finite_mean(values: Iterable[Any]) -> float | None:
    """Average finite numeric values and return ``None`` when none are available."""

    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return fmean(numbers) if numbers else None


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    """Convert a boolean count and denominator into a percentage rate."""

    if not rows:
        return None
    return 100.0 * sum(bool(row.get(key)) for row in rows) / len(rows)


class PaperMetricRecorder:
    """Collect paper-aligned episode metrics without writing intermediate data."""

    def __init__(
        self,
        *,
        episode: Any,
        env: Any,
        config: dict[str, Any],
        profile_name: str,
        rollout_index: int,
    ) -> None:
        """Initialize shared online state for every paper metric enabled in one episode."""

        self.episode = episode
        self.env = env
        self.config = dict(config)
        self.profile_name = str(profile_name)
        self.rollout_index = int(rollout_index)
        self.category = str(getattr(episode, "object_category", ""))
        self.pixel_threshold = int(self.config.get("find_pixel_threshold", 100))

        self.decision_steps = 0
        self.active_stop = False
        self.find_success = False
        self.geo_find_success = False
        self.max_target_pixels = 0
        self.n_sit = 0
        self.geo_interact_success = False
        self._last_motion_pelvis_target_contact = False
        self.collision = CollisionTracker(self.config)
        self.disturbance = DisturbanceTracker()
        self.usage = UsageTracker()

    def record_reset(self) -> None:
        """Capture floor and initial penetration before any physics frame runs."""

        self.collision.record_reset(self.env)

    def record_decision(
        self,
        *,
        step: int,
        decision: Any,
        find_observation: dict[str, Any] | None,
    ) -> None:
        """Join the current ego render with the response generated from it."""

        del step  # Ordering is represented by call order; no per-step log is kept.
        self.decision_steps += 1
        for output in getattr(decision, "stage_outputs", ()):
            self.usage.record(output)

        sample = dict(find_observation or {})
        # This semantic render is the same frame attached to the planner call.
        # Pair it with that response now; a later frame could make a visibility
        # claim appear correct even though the model never saw the target.
        pixel_count = int(sample.get("target_pixel_count") or 0)
        rendered = bool(sample.get("available", False)) and (
            pixel_count >= self.pixel_threshold
        )
        self.max_target_pixels = max(self.max_target_pixels, pixel_count)
        self.geo_find_success = self.geo_find_success or rendered

        planner = getattr(decision, "planner_skill", {})
        visible_state = (
            planner.get("visible_state", "") if isinstance(planner, dict) else ""
        )
        acknowledged = claims_target_visible(visible_state, self.category)
        self.find_success = self.find_success or (rendered and acknowledged)
        self.active_stop = self.active_stop or (
            str(getattr(getattr(decision, "action", None), "skill", "")) == "stand"
        )

    def record_motion(
        self,
        *,
        step: int,
        action_skill: str,
        info: dict[str, Any] | None,
    ) -> None:
        """Consume shared contacts once for disturbance and mesh interaction."""

        metric_frames = dict((info or {}).get("metric_frames") or {})
        # One environment collision query supplies all three contact
        # consumers. Collision sees fixed geometry, disturbance sees dynamic
        # object edges, and interaction sees pelvis/target mesh contacts.
        self.collision.record_step(step, metric_frames)
        self.disturbance.record_step(step, metric_frames)
        agent_frames = list(metric_frames.get("agent_contacts") or [])
        last_frame_hits = [
            bool(self.env.is_pelvis_target_contact(contact))
            for contact in (agent_frames[-1] if agent_frames else [])
        ]
        last_frame_contact = any(last_frame_hits)
        self._last_motion_pelvis_target_contact = last_frame_contact
        if str(action_skill) == "sit":
            self.n_sit += 1
            # The final paper variant evaluates the landed pose of each Sit
            # decision.  A transient contact earlier inside the 15-frame Sit
            # chunk does not count unless it remains on that action's last
            # physics frame (the historical ``last_frame_contacts`` field).
            self.geo_interact_success = (
                self.geo_interact_success or last_frame_contact
            )

    def finalize(
        self,
        *,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute terminal geometry and trajectory-derived scores."""

        geometry = self.env.metric_target_geometry()
        distance = float(geometry["body_target_aabb_distance_m"])
        nav_threshold = float(self.config.get("nav_distance_m", 0.2))
        relaxed_threshold = float(self.config.get("nav_relaxed_distance_m", 1.0))
        is_interact = bool(self.env.is_interact_episode())

        collision = self.collision.finalize()
        # Collision and disturbance use realized 30 Hz states and their one
        # shared contact query. Motion Jerk intentionally uses generated
        # pre-physics motion. These sources must not be collapsed into one
        # generic trajectory.
        # Every full-val episode contributes to physical metrics.  Reset-time
        # penetration is retained only as a diagnostic in the collision row.
        physical_eligible = True
        disturbance = self.disturbance.finalize(after)
        jerk = root_rigid_motion_jerk(
            before["xb_world_75"],
            load_neutral_body22(
                self.config.get(
                    "jerk_neutral_body22",
                    "resources/metrics/smpl_neutral_body22.json",
                )
            ),
            fps=float(before.get("fps", 30.0)),
            smooth_window=int(self.config.get("jerk_smooth_window", 3)),
            stride=int(self.config.get("jerk_stride", 8)),
        )

        # InteractSR is an active committed stop after at least one Sit, with
        # pelvis-to-target mesh contact on the final frame of the last motion.
        interact_success = bool(
            is_interact
            and self.active_stop
            and self.n_sit > 0
            and self._last_motion_pelvis_target_contact
        )
        return {
            "schema": "humanclaw_paper_metrics_v1",
            "profile": self.profile_name,
            "episode": {
                "episode_id": str(getattr(self.episode, "episode_id", "")),
                "scene_id": str(getattr(self.episode, "scene_label", "")),
                "object_category": self.category,
                "rollout_index": self.rollout_index,
            },
            "success": {
                "find_sr": bool(self.find_success),
                "geo_find_sr": bool(self.geo_find_success),
                "max_target_pixel_count": int(self.max_target_pixels),
                "find_pixel_threshold": self.pixel_threshold,
                "nav_sr_20cm": bool(self.active_stop and distance <= nav_threshold),
                "geo_nav_sr_20cm": bool(distance <= nav_threshold),
                "nav_sr_1m": bool(self.active_stop and distance < relaxed_threshold),
                "final_body_target_aabb_distance_m": distance,
                "active_stop": bool(self.active_stop),
                "is_interact_episode": is_interact,
                "interact_sr": interact_success,
                "geo_interact_sr": bool(is_interact and self.geo_interact_success),
                "n_sit": int(self.n_sit),
            },
            "body_scene": {
                "physical_metrics_eligible": physical_eligible,
                **collision,
                **disturbance,
            },
            "action_quality": {
                "motion_jerk_m_s3": jerk,
            },
            "cost": self.usage.summary(self.decision_steps),
        }


def write_episode_metrics(path: Path, value: dict[str, Any]) -> Path:
    """Write the single compact metric artifact for one rollout."""

    path = Path(path)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def aggregate_metric_files(
    output_root: str | Path,
    *,
    write_summary: bool = True,
) -> dict[str, Any]:
    """Aggregate every nested ``metrics.json`` and optionally save the summary.

    Recursive discovery intentionally supports both a normal output directory
    and a parent directory containing multiple distributed-run shards.  The
    standalone metrics CLI sets ``write_summary=False`` by default so printing
    an existing run is a read-only operation.  Batch evaluation keeps the
    historical default and writes ``metrics_summary.json`` when it finishes.
    """

    root = Path(output_root)
    metric_paths = sorted(root.rglob("metrics.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    success_rows = [dict(row.get("success") or {}) for row in rows]
    interact_rows = [row for row in success_rows if row.get("is_interact_episode")]
    physical_rows = [
        row
        for row in rows
        if bool((row.get("body_scene") or {}).get("physical_metrics_eligible"))
    ]
    body_rows = [dict(row.get("body_scene") or {}) for row in physical_rows]
    quality_rows = [dict(row.get("action_quality") or {}) for row in physical_rows]
    cost_rows = [dict(row.get("cost") or {}) for row in rows]

    mapped_objects = sum(
        int(row.get("mapped_affected_dynamic_object_count") or 0) for row in body_rows
    )
    path_length_sum = sum(
        float(row.get("affected_object_path_length_sum_m") or 0.0) for row in body_rows
    )
    total_steps = sum(int(row.get("decision_steps") or 0) for row in cost_rows)
    total_input = sum(int(row.get("input_tokens") or 0) for row in cost_rows)
    total_output = sum(int(row.get("visible_output_tokens") or 0) for row in cost_rows)
    token_sources = sorted({str(row.get("token_source")) for row in cost_rows})

    summary = {
        "schema": "humanclaw_paper_metrics_summary_v1",
        "counts": {
            "episodes": len(rows),
            "interact_episodes": len(interact_rows),
            "physical_metric_episodes": len(physical_rows),
            "initial_penetration_excluded": len(rows) - len(physical_rows),
        },
        "high_level_success_percent": {
            "find_sr": _rate(success_rows, "find_sr"),
            "geo_find_sr": _rate(success_rows, "geo_find_sr"),
            "nav_sr_20cm": _rate(success_rows, "nav_sr_20cm"),
            "geo_nav_sr_20cm": _rate(success_rows, "geo_nav_sr_20cm"),
            "nav_sr_1m": _rate(success_rows, "nav_sr_1m"),
            "interact_sr": _rate(interact_rows, "interact_sr"),
            "geo_interact_sr": _rate(interact_rows, "geo_interact_sr"),
        },
        "body_scene": {
            "collision_step_percent": (
                None
                if not body_rows
                else 100.0
                * float(
                    _finite_mean(
                        row.get("collision_step_fraction") for row in body_rows
                    )
                    or 0.0
                )
            ),
            "collision_by_body_group_percent": {
                group: (
                    None
                    if not body_rows
                    else 100.0
                    * float(
                        _finite_mean(
                            (row.get("by_body_group_step_fraction") or {}).get(group)
                            for row in body_rows
                        )
                        or 0.0
                    )
                )
                for group in ("hand_arm", "torso", "leg", "head")
            },
            "disturbed_objects_per_episode": _finite_mean(
                row.get("affected_dynamic_object_count") for row in body_rows
            ),
            "disturbed_object_path_length_mean_m": (
                path_length_sum / mapped_objects if mapped_objects else None
            ),
        },
        "action_quality": {
            "motion_jerk_m_s3": _finite_mean(
                row.get("motion_jerk_m_s3") for row in quality_rows
            ),
        },
        "cost": {
            "average_steps": _finite_mean(
                row.get("decision_steps") for row in cost_rows
            ),
            "input_tokens_per_step": (
                total_input / total_steps if total_steps else None
            ),
            "visible_output_tokens_per_step": (
                total_output / total_steps if total_steps else None
            ),
            "token_sources": token_sources,
        },
    }
    if write_summary:
        path = root / "metrics_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


__all__ = [
    "PaperMetricRecorder",
    "aggregate_metric_files",
    "write_episode_metrics",
]

"""Movable-object disturbance metric from the final SSDMC evaluator.

The humanoid can disturb an object directly, or indirectly by pushing one
dynamic object into another.  Contacts are consumed in chronological order so
that propagation can only travel through an object that was already affected
at the same or an earlier physics frame.  We retain just the first affected
frame per object; the large raw contact trace is never written to disk.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _dynamic_name(side: Any) -> str | None:
    """Return the handle only when ``side`` is a dynamic rigid object."""

    if not isinstance(side, dict):
        return None
    if side.get("type") != "rigid_object":
        return None
    if "DYNAMIC" not in str(side.get("motion_type", "")).upper():
        return None
    name = str(side.get("name") or "")
    return name or None


def _decoded_names(values: Any) -> list[str]:
    """Decode stored NumPy string arrays into ordinary Python object names."""

    names: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="replace"))
        else:
            names.append(str(value))
    return names


class DisturbanceTracker:
    """Track the affected-object graph while an episode is running."""

    def __init__(self, escape_drop_threshold_m: float = 5.0) -> None:
        """Capture initial dynamic-object poses used as disturbance references.

        ``escape_drop_threshold_m`` is the depth below its own initial height
        at which an affected object is considered to have left the scene.  A
        thin floor-level object such as a mat or a stepping stone can, in rare
        cases, pass through the static scene geometry under the humanoid's
        feet and then fall freely for the rest of the episode.  From that
        point its motion no longer describes a disturbance of the scene, so
        the object is not counted.  Five metres is deeper than any drop inside
        an HSSD house, so ordinary falls are unaffected.
        """

        self._affected: dict[str, dict[str, Any]] = {}
        self._direct: set[str] = set()
        self._next_absolute_frame = 0
        self.escape_drop_threshold_m = float(escape_drop_threshold_m)

    def _mark(self, name: str, step: int, frame: int, source: str) -> None:
        """Record one object's displacement state at a realized physics frame."""

        current = self._affected.get(name)
        if current is None or frame < int(current["first_affected_frame"]):
            self._affected[name] = {
                "first_affected_step": int(step),
                "first_affected_frame": int(frame),
                "source": str(source),
            }

    def record_step(self, step: int, metric_frames: dict[str, Any] | None) -> None:
        """Consume one motion chunk's shared per-frame contact records."""

        frames = dict(metric_frames or {})
        agent_frames = list(frames.get("agent_contacts") or [])
        dynamic_frames = list(frames.get("dynamic_contacts") or [])
        frame_count = max(len(agent_frames), len(dynamic_frames))

        for relative_frame in range(frame_count):
            absolute_frame = self._next_absolute_frame + relative_frame

            # Human-to-dynamic contact is the only direct disturbance seed.
            if relative_frame < len(agent_frames):
                for contact in agent_frames[relative_frame] or []:
                    name = _dynamic_name(contact.get("other"))
                    if name is None:
                        continue
                    self._direct.add(name)
                    self._mark(name, step, absolute_frame, "direct")

            # Dynamic-to-dynamic contacts form an undirected graph for this
            # frame.  Start BFS only from nodes affected by now; this preserves
            # the time direction of the original SSDMC implementation.
            adjacency: dict[str, set[str]] = defaultdict(set)
            if relative_frame < len(dynamic_frames):
                for contact in dynamic_frames[relative_frame] or []:
                    name_a = _dynamic_name(contact.get("a"))
                    name_b = _dynamic_name(contact.get("b"))
                    if name_a is None or name_b is None or name_a == name_b:
                        continue
                    adjacency[name_a].add(name_b)
                    adjacency[name_b].add(name_a)
            queue = deque(name for name in adjacency if name in self._affected)
            seen = set(queue)
            while queue:
                name = queue.popleft()
                for neighbor in adjacency[name]:
                    if neighbor not in self._affected:
                        self._mark(neighbor, step, absolute_frame, "indirect")
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)

        self._next_absolute_frame += frame_count

    @staticmethod
    def _path_length(positions: Any, first_frame: int) -> float | None:
        """Sum frame-to-frame translation distance for one dynamic object."""

        xyz = np.asarray(positions, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[0] == 0:
            return None
        xyz = xyz[:, :3]
        finite = np.all(np.isfinite(xyz), axis=1)
        valid = np.flatnonzero(finite)
        if valid.size == 0:
            return None
        end = int(valid[-1])
        start = min(max(0, int(first_frame)), end)
        if not finite[start]:
            later = valid[valid >= start]
            start = int(later[0]) if later.size else end
        segment = xyz[start : end + 1]
        segment = segment[np.all(np.isfinite(segment), axis=1)]
        if segment.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())

    @staticmethod
    def _vertical_drop(positions: Any) -> float | None:
        """Largest descent below the object's first finite recorded height (Z-up)."""

        xyz = np.asarray(positions, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[0] == 0:
            return None
        z = xyz[:, 2]
        z = z[np.isfinite(z)]
        if z.size == 0:
            return None
        return float(z[0] - z.min())

    def finalize(self, after: dict[str, Any]) -> dict[str, Any]:
        """Map affected handles to trajectories and compute their path length.

        Affected objects that left the scene (see ``escape_drop_threshold_m``)
        are not counted.  They are listed separately, as is every counted
        object with its own path length, so the aggregate stays reproducible
        from the per-episode record.
        """

        names = _decoded_names(after.get("object_names", []))
        indices = {name: index for index, name in enumerate(names)}

        # Record every dynamic object that left the scene during the episode,
        # touched or not.  This list is informational and never enters the
        # metric.
        scene_escaped: list[str] = []
        for index, name in enumerate(names):
            key = f"object_{index:03d}_position"
            if key not in after:
                continue
            drop = self._vertical_drop(after[key])
            if drop is not None and drop > self.escape_drop_threshold_m:
                scene_escaped.append(name)

        kept: list[dict[str, Any]] = []
        escaped: list[dict[str, Any]] = []
        path_lengths: list[float] = []
        for name, info in self._affected.items():
            record = {
                "name": name,
                "source": str(info["source"]),
                "first_affected_step": int(info["first_affected_step"]),
            }
            index = indices.get(name)
            key = f"object_{index:03d}_position" if index is not None else None
            if key is None or key not in after:
                kept.append({**record, "path_length_m": None})
                continue
            positions = after[key]
            drop = self._vertical_drop(positions)
            if drop is not None and drop > self.escape_drop_threshold_m:
                escaped.append({**record, "drop_m": drop})
                continue
            length = self._path_length(positions, int(info["first_affected_frame"]))
            if length is not None:
                path_lengths.append(length)
            kept.append({**record, "path_length_m": length})

        kept_names = {item["name"] for item in kept}
        direct_names = self._direct & kept_names
        mapped_count = len(path_lengths)
        path_sum = float(sum(path_lengths))
        return {
            "affected_dynamic_object_count": int(len(kept_names)),
            "direct_dynamic_object_count": int(len(direct_names)),
            "indirect_dynamic_object_count": int(len(kept_names - direct_names)),
            "mapped_affected_dynamic_object_count": int(mapped_count),
            "affected_object_path_length_sum_m": path_sum,
            "affected_object_path_length_mean_m": (
                path_sum / mapped_count if mapped_count else None
            ),
            "affected_dynamic_objects": kept,
            "escape_drop_threshold_m": self.escape_drop_threshold_m,
            "escaped_affected_dynamic_object_count": int(len(escaped)),
            "escaped_affected_dynamic_objects": escaped,
            "scene_escaped_dynamic_object_count": int(len(scene_escaped)),
            "scene_escaped_dynamic_objects": scene_escaped,
        }


__all__ = ["DisturbanceTracker"]

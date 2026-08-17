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

    def __init__(self) -> None:
        """Capture initial dynamic-object poses used as disturbance references."""

        self._affected: dict[str, dict[str, Any]] = {}
        self._direct: set[str] = set()
        self._next_absolute_frame = 0

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

    def finalize(self, after: dict[str, Any]) -> dict[str, Any]:
        """Map affected handles to trajectories and compute their path length."""

        names = _decoded_names(after.get("object_names", []))
        indices = {name: index for index, name in enumerate(names)}
        path_lengths: list[float] = []
        for name, info in self._affected.items():
            index = indices.get(name)
            if index is None:
                continue
            key = f"object_{index:03d}_position"
            if key not in after:
                continue
            length = self._path_length(after[key], int(info["first_affected_frame"]))
            if length is not None:
                path_lengths.append(length)

        affected_count = len(self._affected)
        mapped_count = len(path_lengths)
        path_sum = float(sum(path_lengths))
        return {
            "affected_dynamic_object_count": int(affected_count),
            "direct_dynamic_object_count": int(len(self._direct)),
            "indirect_dynamic_object_count": int(
                len(set(self._affected) - self._direct)
            ),
            "mapped_affected_dynamic_object_count": int(mapped_count),
            "affected_object_path_length_sum_m": path_sum,
            "affected_object_path_length_mean_m": (
                path_sum / mapped_count if mapped_count else None
            ),
        }


__all__ = ["DisturbanceTracker"]

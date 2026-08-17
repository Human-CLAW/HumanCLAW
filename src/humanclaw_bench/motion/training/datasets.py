"""Datasets for the 20-frame base and curated skill training corpora."""

from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

N_HISTORY = 5
N_FUTURE = 15
CHUNK_FRAMES = N_HISTORY + N_FUTURE
STATE_DIM = 219
PELVIS_X_INDEX = 75
PELVIS_Z_INDEX = 77
ORIENTATION_START = 3
JOINTS_START = 75
LEFT_FOOT_INDEX = 10
RIGHT_FOOT_INDEX = 11


@dataclass(frozen=True)
class SkillArrays:
    """A curated skill corpus held in the historical GPU-resident layout."""

    chunks: np.ndarray
    condition: np.ndarray | None
    manifest_rows: tuple[dict[str, str], ...]


def _pickle_payload(path: Path) -> dict[str, Any]:
    """Read one trusted, locally generated AMASS chunk pickle."""

    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or "chunks" not in value:
        raise ValueError(f"Chunk pickle has no chunks mapping: {path}")
    return value


def _validated_chunks(value: Any, source: Path) -> np.ndarray:
    """Convert and shape-check a pickle's motion chunk tensor."""

    chunks = np.asarray(value, dtype=np.float32)
    if chunks.ndim != 3 or chunks.shape[1:] != (CHUNK_FRAMES, STATE_DIM):
        raise ValueError(
            f"Expected [N,{CHUNK_FRAMES},{STATE_DIM}] chunks in {source}, "
            f"got {chunks.shape}"
        )
    return chunks


class ChunkTreeDataset(Dataset):
    """Load the complete nested per-sequence chunk tree used by base training."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        relative_files: Sequence[str] | None = None,
        normalize: bool = False,
    ) -> None:
        """Read selected pickle files and concatenate their 20-frame chunks."""

        super().__init__()
        root = Path(data_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if relative_files is None:
            paths = sorted(root.rglob("*.pkl"))
        else:
            paths = [_safe_relative_path(root, name) for name in relative_files]
        arrays = [
            _validated_chunks(_pickle_payload(path)["chunks"], path) for path in paths
        ]
        if not arrays:
            raise ValueError(f"No chunk pickle files found under {root}")
        self.chunks = np.concatenate(arrays, axis=0)
        self.normalize = bool(normalize)
        if self.normalize:
            self.mean = self.chunks.mean(axis=(0, 1), dtype=np.float64).astype(
                np.float32
            )
            self.std = (
                self.chunks.std(axis=(0, 1), dtype=np.float64) + 1e-8
            ).astype(np.float32)
        else:
            self.mean = np.zeros(STATE_DIM, dtype=np.float32)
            self.std = np.ones(STATE_DIM, dtype=np.float32)
        self.relative_files = tuple(str(path.relative_to(root)) for path in paths)

    def __len__(self) -> int:
        """Return the number of 20-frame training chunks."""

        return int(self.chunks.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Split one chunk into five clean and fifteen target frames."""

        chunk = self.chunks[index]
        if self.normalize:
            chunk = (chunk - self.mean) / self.std
        tensor = torch.from_numpy(np.asarray(chunk, dtype=np.float32))
        return {
            "history": tensor[:N_HISTORY],
            "future": tensor[N_HISTORY:],
        }


def _safe_relative_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path while rejecting traversal outside its chunk root."""

    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"Manifest path must be relative: {relative}")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes chunk root: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def read_skill_manifest(
    path: str | Path, *, max_samples: int | None = None
) -> list[dict[str, str]]:
    """Read and minimally validate a transparent per-skill CSV list."""

    manifest = Path(path).expanduser().resolve()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    required = {"rel_pkl", "source_chunk_idx"}
    if not rows:
        raise ValueError(f"Skill manifest is empty: {manifest}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Skill manifest {manifest} lacks columns: {sorted(missing)}")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        rows = rows[:max_samples]
    return rows


def _axis_angle_rotation(axis_angle: np.ndarray) -> np.ndarray:
    """Convert one NumPy axis-angle vector into a 3x3 rotation matrix."""

    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(axis_angle, dtype=np.float64) / angle
    x_value, y_value, z_value = axis
    skew = np.asarray(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def final_yaw_degrees(chunk: np.ndarray) -> float:
    """Measure final canonical body yaw using the historical convention."""

    orientation = np.asarray(
        chunk[-1, ORIENTATION_START : ORIENTATION_START + 3], dtype=np.float64
    )
    forward = _axis_angle_rotation(orientation) @ np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float64
    )
    return math.degrees(math.atan2(float(forward[0]), float(forward[2])))


def step_height_and_depth(chunk: np.ndarray) -> tuple[float, float]:
    """Compute the toe-separation stair condition used for both climb skills."""

    joints = chunk[:, JOINTS_START : JOINTS_START + 72].reshape(
        CHUNK_FRAMES, 24, 3
    )
    left = joints[:, LEFT_FOOT_INDEX]
    right = joints[:, RIGHT_FOOT_INDEX]
    height = float(np.abs(left[:, 1] - right[:, 1]).max())
    depth = float(
        np.sqrt(
            np.square(left[:, 0] - right[:, 0])
            + np.square(left[:, 2] - right[:, 2])
        ).max()
    )
    return height, depth


def _row_float(row: Mapping[str, str], names: Iterable[str]) -> float:
    """Read the first populated floating-point field from a manifest row."""

    for name in names:
        value = row.get(name, "").strip()
        if value:
            return float(value)
    raise ValueError(f"Manifest row lacks any of these fields: {list(names)}")


def _condition_for(
    skill: str, chunk: np.ndarray, row: Mapping[str, str]
) -> np.ndarray | None:
    """Recover the exact raw condition expected by one released skill network."""

    x_value = float(chunk[-1, PELVIS_X_INDEX])
    z_value = float(chunk[-1, PELVIS_Z_INDEX])
    if skill == "walk_forward":
        return np.asarray(
            [x_value, z_value, final_yaw_degrees(chunk)], dtype=np.float32
        )
    if skill == "side_walk":
        # The historical export rounded this value to six decimals before
        # storing it in each subset pickle.  The distributed manifest carries
        # that exact stored condition rather than silently recomputing it.
        side_x = _row_float(row, ("target_side_x_m",))
        return np.asarray([side_x], dtype=np.float32)
    if skill == "step_back":
        return np.asarray([x_value, z_value], dtype=np.float32)
    if skill == "turn":
        return np.asarray([final_yaw_degrees(chunk)], dtype=np.float32)
    if skill in {"step_climb_up", "step_climb_down"}:
        return np.asarray(step_height_and_depth(chunk), dtype=np.float32)
    if skill in {"stop", "stand"}:
        return None
    if skill == "sit":
        target_height = _row_float(
            row,
            ("target_h_raw", "segment_min_pelvis_y_m", "target_sit_h_m"),
        )
        return np.asarray([target_height], dtype=np.float32)
    raise ValueError(f"Unknown motion skill: {skill!r}")


def load_skill_arrays(
    skill: str,
    *,
    chunk_root: str | Path,
    manifest_path: str | Path,
    max_samples: int | None = None,
) -> SkillArrays:
    """Materialize only manifest-selected chunks and their raw conditions."""

    root = Path(chunk_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = read_skill_manifest(manifest_path, max_samples=max_samples)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for row_index, row in enumerate(rows):
        grouped.setdefault(row["rel_pkl"], []).append((row_index, row))

    chunks_by_row: list[np.ndarray | None] = [None] * len(rows)
    conditions_by_row: list[np.ndarray | None] = [None] * len(rows)
    for relative, indexed_rows in grouped.items():
        path = _safe_relative_path(root, relative)
        source_chunks = _validated_chunks(_pickle_payload(path)["chunks"], path)
        for row_index, row in indexed_rows:
            source_index = int(row["source_chunk_idx"])
            if source_index < 0 or source_index >= source_chunks.shape[0]:
                raise IndexError(
                    f"{relative} has {source_chunks.shape[0]} chunks, "
                    f"manifest requests {source_index}"
                )
            chunk = source_chunks[source_index]
            chunks_by_row[row_index] = chunk
            conditions_by_row[row_index] = _condition_for(skill, chunk, row)

    if any(value is None for value in chunks_by_row):
        raise RuntimeError("Internal manifest materialization error")
    chunks = np.stack(chunks_by_row, axis=0).astype(np.float32, copy=False)
    if skill in {"stop", "stand"}:
        condition = None
    else:
        if any(value is None for value in conditions_by_row):
            raise RuntimeError(f"Missing condition while loading skill {skill}")
        condition = np.stack(conditions_by_row, axis=0).astype(np.float32, copy=False)
    return SkillArrays(chunks, condition, tuple(rows))


def repeat_skill_arrays(arrays: SkillArrays, factor: int) -> SkillArrays:
    """Repeat a small curated corpus exactly as the two paper profiles did."""

    if factor < 1:
        raise ValueError(f"repeat factor must be >= 1, got {factor}")
    if factor == 1:
        return arrays
    chunks = np.tile(arrays.chunks, (factor, 1, 1))
    condition = (
        None
        if arrays.condition is None
        else np.tile(arrays.condition, (factor, 1))
    )
    return SkillArrays(chunks, condition, arrays.manifest_rows * factor)


def write_base_file_list(data_root: str | Path, output: str | Path) -> None:
    """Write the deterministic relative pickle list consumed by base training."""

    root = Path(data_root).expanduser().resolve()
    paths = sorted(root.rglob("*.pkl"))
    if not paths:
        raise ValueError(f"No chunk pickle files found under {root}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(f"{path.relative_to(root)}\n" for path in paths), encoding="utf-8"
    )


def read_base_file_list(path: str | Path) -> list[str]:
    """Read a newline-delimited base-training pickle list."""

    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


__all__ = [
    "CHUNK_FRAMES",
    "ChunkTreeDataset",
    "N_FUTURE",
    "N_HISTORY",
    "STATE_DIM",
    "SkillArrays",
    "final_yaw_degrees",
    "load_skill_arrays",
    "read_base_file_list",
    "read_skill_manifest",
    "repeat_skill_arrays",
    "step_height_and_depth",
    "write_base_file_list",
]

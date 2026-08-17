"""Convert a reviewed legacy subset or CSV into a portable skill manifest."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..datasets import (
    CHUNK_FRAMES,
    PELVIS_X_INDEX,
    PELVIS_Z_INDEX,
    STATE_DIM,
    final_yaw_degrees,
    step_height_and_depth,
)
from ..profiles import canonical_skill_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one of the two historical-list conversion modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subset-root")
    source.add_argument("--input-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    return parser.parse_args(argv)


def _payload(path: Path) -> dict[str, Any]:
    """Read one trusted legacy subset pickle."""

    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or "chunks" not in value:
        raise ValueError(f"Invalid subset pickle: {path}")
    return value


def _aligned_array(
    payload: Mapping[str, Any], key: str, sample_count: int
) -> np.ndarray | None:
    """Return a per-chunk metadata array only when its first axis is aligned."""

    value = payload.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    return array if array.shape[:1] == (sample_count,) else None


def _float_text(value: float | np.floating[Any]) -> str:
    """Serialize a float32 value with enough digits for exact round-trip."""

    return np.format_float_positional(np.float32(value), unique=True, trim="k")


def _row_for_chunk(
    skill: str,
    *,
    relative_pickle: str,
    source_index: int,
    local_index: int,
    chunk: np.ndarray,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Build one portable row while preserving condition values when required."""

    row = {
        "rel_pkl": relative_pickle,
        "source_chunk_idx": str(source_index),
    }
    x_value = float(chunk[-1, PELVIS_X_INDEX])
    z_value = float(chunk[-1, PELVIS_Z_INDEX])
    yaw_value = final_yaw_degrees(chunk)
    if skill in {"walk_forward", "step_back"}:
        row.update(
            {
                "target_x_m": _float_text(x_value),
                "target_z_m": _float_text(z_value),
                "target_yaw_deg": _float_text(yaw_value),
            }
        )
    elif skill == "turn":
        row["target_yaw_deg"] = _float_text(yaw_value)
    elif skill == "side_walk":
        stored = _aligned_array(payload, "target_side_x_m", len(payload["chunks"]))
        if stored is None:
            raise ValueError("Side-walk subset lacks target_side_x_m")
        row["target_side_x_m"] = _float_text(stored[local_index])
        row["target_z_m"] = _float_text(z_value)
        row["target_yaw_deg"] = _float_text(yaw_value)
    elif skill in {"step_climb_up", "step_climb_down"}:
        height, depth = step_height_and_depth(chunk)
        row["target_step_h_m"] = _float_text(height)
        row["target_step_d_m"] = _float_text(depth)
    elif skill == "sit":
        stored = _aligned_array(payload, "target_h_raw", len(payload["chunks"]))
        if stored is None:
            stored = _aligned_array(
                payload, "segment_min_pelvis_y_m", len(payload["chunks"])
            )
        if stored is None:
            raise ValueError("Sit subset lacks target_h_raw metadata")
        row["target_h_raw"] = _float_text(stored[local_index])
        row["target_yaw_deg"] = _float_text(yaw_value)
    elif skill != "stop":
        raise ValueError(f"Unsupported skill: {skill}")
    return row


def _skill_filter_keeps(skill: str, chunk: np.ndarray) -> bool:
    """Apply the final yaw thresholds associated with a curated skill list."""

    absolute_yaw = abs(final_yaw_degrees(chunk))
    if skill == "walk_forward":
        return absolute_yaw <= 90.0
    if skill == "step_back":
        return absolute_yaw <= 5.0
    if skill == "turn":
        return 5.0 <= absolute_yaw <= 120.0
    return True


def rows_from_subset(skill: str, root: str | Path) -> list[dict[str, str]]:
    """Extract source indices and conditions from a mirrored legacy subset."""

    subset_root = Path(root).expanduser().resolve()
    paths = sorted(subset_root.rglob("*.pkl"))
    if not paths:
        raise ValueError(f"No subset pickle files under {subset_root}")
    rows: list[dict[str, str]] = []
    for path in paths:
        payload = _payload(path)
        chunks = np.asarray(payload["chunks"], dtype=np.float32)
        if chunks.ndim != 3 or chunks.shape[1:] != (CHUNK_FRAMES, STATE_DIM):
            raise ValueError(f"Unexpected chunk shape {chunks.shape}: {path}")
        indices = _aligned_array(payload, "source_chunk_indices", len(chunks))
        if indices is None:
            indices = np.arange(len(chunks), dtype=np.int64)
        relative = str(path.relative_to(subset_root))
        for local_index, chunk in enumerate(chunks):
            if not _skill_filter_keeps(skill, chunk):
                continue
            rows.append(
                _row_for_chunk(
                    skill,
                    relative_pickle=relative,
                    source_index=int(indices[local_index]),
                    local_index=local_index,
                    chunk=chunk,
                    payload=payload,
                )
            )
    return rows


def rows_from_manifest(path: str | Path) -> list[dict[str, str]]:
    """Strip a historical CSV down to portable source-chunk identifiers."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: list[dict[str, str]] = []
    for row in rows:
        output.append(
            {
                "rel_pkl": row["rel_pkl"],
                "source_chunk_idx": row["source_chunk_idx"],
            }
        )
    return output


def write_manifest(
    rows: list[dict[str, str]], output: str | Path, *, expected_samples: int | None
) -> None:
    """Write stable CSV columns after checking the required row count."""

    if not rows:
        raise ValueError("Refusing to write an empty skill manifest")
    if expected_samples is not None and len(rows) != expected_samples:
        raise ValueError(
            f"Recovered {len(rows)} rows but expected {expected_samples}"
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """Export one exact training list without copying its motion arrays."""

    args = parse_args(argv)
    skill = canonical_skill_name(args.skill)
    rows = (
        rows_from_subset(skill, args.subset_root)
        if args.subset_root
        else rows_from_manifest(args.input_manifest)
    )
    write_manifest(rows, args.output, expected_samples=args.expected_samples)


if __name__ == "__main__":
    main()

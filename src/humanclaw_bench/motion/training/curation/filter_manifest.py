"""Apply explicit numeric motion filters to a candidate chunk manifest."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import (
    CHUNK_FRAMES,
    PELVIS_X_INDEX,
    PELVIS_Z_INDEX,
    STATE_DIM,
    final_yaw_degrees,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse only explicit thresholds; no hidden skill preset is applied."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-abs-final-yaw-deg", type=float, default=None)
    parser.add_argument("--max-abs-final-yaw-deg", type=float, default=None)
    parser.add_argument("--max-abs-z-m", type=float, default=None)
    parser.add_argument("--min-abs-x-m", type=float, default=None)
    parser.add_argument("--min-abs-x-over-abs-z", type=float, default=None)
    parser.add_argument("--max-planar-displacement-m", type=float, default=None)
    parser.add_argument(
        "--z-direction", choices=("either", "positive", "negative"), default="either"
    )
    return parser.parse_args(argv)


def _load_chunk(root: Path, row: dict[str, str]) -> np.ndarray:
    """Load one manifest-addressed chunk from a trusted generated pickle."""

    relative = Path(row["rel_pkl"])
    if relative.is_absolute():
        raise ValueError(f"Manifest path must be relative: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest path escapes chunk root: {relative}") from error
    with path.open("rb") as handle:
        payload: Any = pickle.load(handle)
    chunks = np.asarray(payload["chunks"], dtype=np.float32)
    if chunks.ndim != 3 or chunks.shape[1:] != (CHUNK_FRAMES, STATE_DIM):
        raise ValueError(f"Unexpected chunk shape {chunks.shape}: {path}")
    return chunks[int(row["source_chunk_idx"])]


def _passes(chunk: np.ndarray, args: argparse.Namespace) -> tuple[bool, dict[str, str]]:
    """Evaluate every requested threshold and return transparent measurements."""

    x_value = float(chunk[-1, PELVIS_X_INDEX])
    z_value = float(chunk[-1, PELVIS_Z_INDEX])
    absolute_yaw = abs(final_yaw_degrees(chunk))
    displacement = math.hypot(x_value, z_value)
    ratio = abs(x_value) / max(abs(z_value), 1e-8)
    measurements = {
        "measured_final_x_m": f"{x_value:.9g}",
        "measured_final_z_m": f"{z_value:.9g}",
        "measured_abs_final_yaw_deg": f"{absolute_yaw:.9g}",
        "measured_planar_displacement_m": f"{displacement:.9g}",
        "measured_abs_x_over_abs_z": f"{ratio:.9g}",
    }
    checks = [
        args.min_abs_final_yaw_deg is None
        or absolute_yaw >= args.min_abs_final_yaw_deg,
        args.max_abs_final_yaw_deg is None
        or absolute_yaw <= args.max_abs_final_yaw_deg,
        args.max_abs_z_m is None or abs(z_value) <= args.max_abs_z_m,
        args.min_abs_x_m is None or abs(x_value) >= args.min_abs_x_m,
        args.min_abs_x_over_abs_z is None
        or ratio >= args.min_abs_x_over_abs_z,
        args.max_planar_displacement_m is None
        or displacement <= args.max_planar_displacement_m,
        args.z_direction == "either"
        or (args.z_direction == "positive" and z_value > 0)
        or (args.z_direction == "negative" and z_value < 0),
    ]
    return all(checks), measurements


def main(argv: list[str] | None = None) -> None:
    """Filter a CSV list without copying or modifying the underlying chunks."""

    args = parse_args(argv)
    root = Path(args.chunk_root).expanduser().resolve()
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        candidates = [dict(row) for row in csv.DictReader(handle)]
    kept: list[dict[str, str]] = []
    for row in candidates:
        passes, measurements = _passes(_load_chunk(root, row), args)
        if passes:
            kept.append({**row, **measurements})
    if not kept:
        raise ValueError("Numeric filters removed every candidate chunk")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(kept[0])
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    print(f"candidate_rows={len(candidates)} kept_rows={len(kept)}", flush=True)


if __name__ == "__main__":
    main()

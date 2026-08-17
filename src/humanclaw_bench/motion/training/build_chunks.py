"""Build the paper's 30 Hz, 20-frame, 219-D chunks from licensed AMASS data."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from humanclaw_bench.motion.canonicalization import (
    axis_angle_to_rotation_matrix,
    extract_floor_projected_transform,
    inverse_transform,
    rotation_matrix_to_axis_angle,
)

from .body_model import SMPLForwardKinematics
from .datasets import CHUNK_FRAMES, STATE_DIM

TARGET_FPS = 30
SOURCE_FRAMES = CHUNK_FRAMES + 1
SOURCE_STRIDE = 5
REFERENCE_FRAME = 4
FK_BATCH_SIZE = 4096
_BODY_MODEL_CACHE: dict[tuple[str, str], SMPLForwardKinematics] = {}


def z_up_to_y_up(
    translation: torch.Tensor, global_orientation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate raw AMASS world coordinates by -90 degrees about the X axis."""

    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        device=translation.device,
        dtype=translation.dtype,
    )
    converted_translation = (rotation @ translation.unsqueeze(-1)).squeeze(-1)
    converted_orientation = rotation_matrix_to_axis_angle(
        rotation.unsqueeze(0)
        @ axis_angle_to_rotation_matrix(global_orientation)
    )
    return converted_translation, converted_orientation


def canonicalize_source_frames(
    translation: torch.Tensor,
    global_orientation: torch.Tensor,
    body_pose: torch.Tensor,
    joints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express 21 source frames in the ego frame of history frame four."""

    floor_transforms = extract_floor_projected_transform(translation, joints)
    inverse_reference = inverse_transform(
        floor_transforms[REFERENCE_FRAME : REFERENCE_FRAME + 1]
    )[0]
    inverse_rotation = inverse_reference[:3, :3]
    orientation = rotation_matrix_to_axis_angle(
        inverse_rotation.unsqueeze(0)
        @ axis_angle_to_rotation_matrix(global_orientation)
    )
    reference_translation = floor_transforms[REFERENCE_FRAME, :3, 3]
    translated = (
        inverse_rotation @ (translation - reference_translation).unsqueeze(-1)
    ).squeeze(-1)
    homogeneous_joints = torch.cat(
        [joints, torch.ones_like(joints[..., :1])], dim=-1
    )
    canonical_joints = (
        inverse_reference[None, None] @ homogeneous_joints[..., None]
    ).squeeze(-1)[..., :3]
    floor_height = canonical_joints[REFERENCE_FRAME, :, 1].min()
    translated[:, 1] -= floor_height
    canonical_joints[..., 1] -= floor_height
    body = torch.cat([translated, orientation, body_pose], dim=-1)
    return body, canonical_joints


def build_state_vector(body: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Create 20 states by attaching joint positions and one-frame velocities."""

    flattened_joints = joints.reshape(joints.shape[0], -1)
    velocity = flattened_joints[1:] - flattened_joints[:-1]
    state = torch.cat([body[:-1], flattened_joints[:-1], velocity], dim=-1)
    if tuple(state.shape) != (CHUNK_FRAMES, STATE_DIM):
        raise ValueError(
            f"Expected ({CHUNK_FRAMES}, {STATE_DIM}) state, got {tuple(state.shape)}"
        )
    return state


def process_amass_sequence(
    npz_path: str | Path,
    body_model: SMPLForwardKinematics,
    device: torch.device,
) -> list[np.ndarray]:
    """Convert one AMASS sequence into overlapping canonical motion chunks."""

    archive = np.load(npz_path, allow_pickle=True)
    frame_rate = float(archive["mocap_framerate"])
    poses = np.asarray(archive["poses"])
    raw_translation = np.asarray(archive["trans"])
    frame_count = poses.shape[0]
    sampling_ratio = max(1, int(round(frame_rate / TARGET_FPS)))
    source_span = (SOURCE_FRAMES - 1) * sampling_ratio
    maximum_start = frame_count - 1 - source_span
    if maximum_start < 0:
        return []

    body_pose = np.concatenate(
        [poses[:, 3:66], np.zeros((frame_count, 6), dtype=poses.dtype)], axis=-1
    )
    translation = torch.from_numpy(raw_translation).float().to(device)
    orientation = torch.from_numpy(poses[:, :3]).float().to(device)
    translation, orientation = z_up_to_y_up(translation, orientation)
    body_pose_tensor = torch.from_numpy(body_pose).float().to(device)

    joint_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, frame_count, FK_BATCH_SIZE):
            stop = start + FK_BATCH_SIZE
            joint_parts.append(
                body_model(
                    translation[start:stop],
                    orientation[start:stop],
                    body_pose_tensor[start:stop],
                )
            )
    joints = torch.cat(joint_parts).cpu()
    translation = translation.cpu()
    orientation = orientation.cpu()
    body_pose_tensor = body_pose_tensor.cpu()

    chunks: list[np.ndarray] = []
    for start in range(0, maximum_start + 1, SOURCE_STRIDE):
        indices = [start + offset * sampling_ratio for offset in range(SOURCE_FRAMES)]
        body, canonical_joints = canonicalize_source_frames(
            translation[indices],
            orientation[indices],
            body_pose_tensor[indices],
            joints[indices],
        )
        chunks.append(build_state_vector(body, canonical_joints).numpy())
    return chunks


def collect_amass_files(root: str | Path) -> list[Path]:
    """Return every AMASS ``*_poses.npz`` path in deterministic order."""

    data_root = Path(root).expanduser().resolve()
    return sorted(data_root.rglob("*_poses.npz"))


def output_path_for(source: Path, source_root: Path, output_root: Path) -> Path:
    """Map an AMASS source path to its mirrored per-sequence pickle path."""

    relative = source.relative_to(source_root)
    filename = relative.name.removesuffix("_poses.npz") + ".pkl"
    return output_root / relative.parent / filename


def _cached_body_model(model_path: str, device: torch.device) -> SMPLForwardKinematics:
    """Reuse one loaded SMPL model per worker and device."""

    cache_key = (str(Path(model_path).resolve()), str(device))
    if cache_key not in _BODY_MODEL_CACHE:
        _BODY_MODEL_CACHE[cache_key] = SMPLForwardKinematics(model_path).to(device).eval()
    return _BODY_MODEL_CACHE[cache_key]


def process_one_file(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """Worker entry point that atomically writes one per-sequence pickle."""

    source_name, source_root_name, output_name, model_name, device_name = task
    source = Path(source_name)
    destination = Path(output_name)
    if destination.is_file():
        return {"source": source_name, "chunks": -1, "seconds": 0.0}
    started = time.perf_counter()
    device = torch.device(device_name)
    body_model = _cached_body_model(model_name, device)
    chunks = process_amass_sequence(source, body_model, device)
    if not chunks:
        return {
            "source": source_name,
            "chunks": 0,
            "seconds": time.perf_counter() - started,
        }
    array = np.stack(chunks).astype(np.float32, copy=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = {
        "chunks": array,
        "source": str(source.relative_to(Path(source_root_name))),
        "n_chunks": int(array.shape[0]),
        "chunk_len": CHUNK_FRAMES,
        "state_dim": STATE_DIM,
        "target_fps": TARGET_FPS,
        "source_stride": SOURCE_STRIDE,
        "reference_frame": REFERENCE_FRAME,
    }
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)
    return {
        "source": source_name,
        "chunks": int(array.shape[0]),
        "seconds": time.perf_counter() - started,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the portable chunk-builder command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-root", required=True)
    parser.add_argument("--smpl-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Visible CUDA devices to use; pass 0 for CPU preprocessing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build every missing per-sequence chunk pickle in parallel."""

    args = parse_args(argv)
    source_root = Path(args.amass_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sources = collect_amass_files(source_root)
    if not sources:
        raise ValueError(f"No *_poses.npz files found under {source_root}")
    if args.workers < 1 or args.gpus < 0:
        raise ValueError("workers must be >=1 and gpus must be >=0")
    tasks: list[tuple[str, str, str, str, str]] = []
    skipped = 0
    for index, source in enumerate(sources):
        destination = output_path_for(source, source_root, output_root)
        if destination.is_file():
            skipped += 1
            continue
        device = "cpu" if args.gpus == 0 else f"cuda:{index % args.gpus}"
        tasks.append(
            (
                str(source),
                str(source_root),
                str(destination),
                str(Path(args.smpl_model).expanduser().resolve()),
                device,
            )
        )
    print(
        f"AMASS sequences={len(sources)} pending={len(tasks)} existing={skipped}",
        flush=True,
    )
    if not tasks:
        return
    started = time.perf_counter()
    completed = 0
    chunk_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_file, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            chunk_count += max(0, int(result["chunks"]))
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                print(
                    f"[{completed}/{len(tasks)}] chunks={chunk_count} "
                    f"wall={time.perf_counter() - started:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()

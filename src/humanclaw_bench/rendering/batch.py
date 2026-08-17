"""Process-isolated parallel rendering for saved HumanClaw trajectories."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from humanclaw_bench.paths import repository_root


@dataclass(frozen=True)
class RenderJob:
    """One subprocess-safe saved-trajectory render request."""

    key: str
    rollout_dir: Path
    output_dir: Path
    trajectory_path: Path | None = None


def _resolved_from(value: Any, base: Path) -> Path:
    """Resolve a manifest path relative to the manifest that declared it."""

    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_relative(value: str) -> Path:
    """Return a safe relative output path without parent-directory traversal."""

    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Render output key must be a safe relative path: {value!r}")
    return path


def discover_render_jobs(
    input_root: str | Path,
    output_root: str | Path,
) -> list[RenderJob]:
    """Find standard rollout directories below an output root."""

    source_root = Path(input_root).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Input root not found: {source_root}")
    jobs: list[RenderJob] = []
    for trajectory in sorted(source_root.rglob("trajectory_after.npz")):
        rollout = trajectory.parent
        relative = rollout.relative_to(source_root)
        jobs.append(
            RenderJob(
                key=relative.as_posix(),
                rollout_dir=rollout,
                output_dir=destination_root / relative,
            )
        )
    return jobs


def load_render_jobs(
    manifest: str | Path,
    output_root: str | Path,
) -> list[RenderJob]:
    """Load JSONL (or a JSON list) with rollout and optional trajectory paths."""

    path = Path(manifest).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        rows = value.get("jobs", []) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Render manifest must contain JSON objects: {path}")

    base = path.parent
    destination_root = Path(output_root).expanduser().resolve()
    jobs: list[RenderJob] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        raw_rollout = row.get("rollout_dir") or row.get("source_rollout_dir")
        if not raw_rollout:
            raise ValueError(f"Manifest row {index} has no rollout_dir")
        rollout = _resolved_from(raw_rollout, base)
        key = str(row.get("episode_key") or row.get("output_key") or rollout.name)
        if key in seen:
            raise ValueError(f"Duplicate render job key: {key!r}")
        seen.add(key)
        relative = _safe_relative(key)
        raw_trajectory = row.get("trajectory_path")
        jobs.append(
            RenderJob(
                key=key,
                rollout_dir=rollout,
                output_dir=destination_root / relative,
                trajectory_path=(
                    _resolved_from(raw_trajectory, base) if raw_trajectory else None
                ),
            )
        )
    return jobs


def render_saved_batch(
    jobs: list[RenderJob],
    *,
    max_parallel: int = 1,
    devices: tuple[str, ...] = (),
    preset: str = "veryfast",
    crf: int = 20,
    force: bool = False,
) -> dict[str, Any]:
    """Render jobs in separate processes, one OpenGL context per episode."""

    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    pending = deque(enumerate(jobs))
    running: list[tuple[subprocess.Popen[bytes], RenderJob, str | None]] = []
    completed: list[str] = []
    failed: list[str] = []
    started = time.perf_counter()

    while pending or running:
        while pending and len(running) < max_parallel:
            launch_index, job = pending.popleft()
            command = [
                sys.executable,
                "-m",
                "humanclaw_bench",
                "render",
                "--rollout-dir",
                str(job.rollout_dir),
                "--output-dir",
                str(job.output_dir),
                "--preset",
                str(preset),
                "--crf",
                str(int(crf)),
                "--progress-every",
                "0",
            ]
            if job.trajectory_path is not None:
                command.extend(["--trajectory-path", str(job.trajectory_path)])
            if force:
                command.append("--force")

            env = os.environ.copy()
            device: str | None = None
            if devices:
                device = str(devices[launch_index % len(devices)])
                env["CUDA_VISIBLE_DEVICES"] = device
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env.setdefault(name, "1")
            process = subprocess.Popen(
                command,
                cwd=str(repository_root()),
                env=env,
                stdout=subprocess.DEVNULL,
            )
            running.append((process, job, device))

        remaining: list[tuple[subprocess.Popen[bytes], RenderJob, str | None]] = []
        for process, job, device in running:
            return_code = process.poll()
            if return_code is None:
                remaining.append((process, job, device))
                continue
            if return_code == 0:
                completed.append(job.key)
            else:
                failed.append(job.key)
            print(
                f"[{len(completed) + len(failed)}/{len(jobs)}] {job.key} "
                f"{'complete' if return_code == 0 else 'failed'}"
                + (f" gpu={device}" if device is not None else ""),
                flush=True,
            )
        running = remaining
        if running:
            time.sleep(0.2)

    return {
        "selected": len(jobs),
        "completed": len(completed),
        "failed": len(failed),
        "failed_jobs": failed,
        "elapsed_s": time.perf_counter() - started,
    }


__all__ = [
    "RenderJob",
    "discover_render_jobs",
    "load_render_jobs",
    "render_saved_batch",
]

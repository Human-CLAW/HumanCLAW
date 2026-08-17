"""Minimal local dispatcher for HumanClawBench episodes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from .assets import sha256_file, verify_weights
from .benchmark.episodes import list_episode_specs
from .config import ReleaseConfig
from .paths import repository_root, resolve_release_path


# Each rollout process owns an independent simulator and motion runtime.  BLAS
# libraries otherwise default to every host core, so a 32-process batch can
# create more than a thousand runnable CPU threads before the first episode
# starts.  The archived paper launchers set the same four variables to one.
# Respect an explicit user value, but keep the safe production default here so
# the Python dispatcher is self-contained.
_PARALLEL_CPU_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _child_process_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a rollout environment with bounded CPU-library thread pools."""

    env = dict(os.environ if base is None else base)
    for name in _PARALLEL_CPU_THREAD_ENV:
        env.setdefault(name, "1")
    return env


def resolve_devices(
    value: str,
    *,
    environ: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve an explicit GPU list or detect the GPUs visible to this shell.

    ``CUDA_VISIBLE_DEVICES`` takes precedence in auto mode, so a cluster job or
    container remains inside its assigned GPU set.  When that variable is
    absent, ``nvidia-smi`` supplies the physical indices.  The returned values
    remain strings because CUDA also accepts UUID and MIG identifiers.
    """

    text = str(value or "auto").strip()
    if text.lower() != "auto":
        devices = tuple(item.strip() for item in text.split(",") if item.strip())
    else:
        environment = os.environ if environ is None else environ
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            devices = tuple(
                item.strip()
                for item in visible.split(",")
                if item.strip() and item.strip() not in {"-1", "NoDevFiles"}
            )
        else:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index",
                        "--format=csv,noheader",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (FileNotFoundError, subprocess.SubprocessError):
                devices = ()
            else:
                devices = (
                    tuple(
                        line.strip()
                        for line in result.stdout.splitlines()
                        if line.strip()
                    )
                    if result.returncode == 0
                    else ()
                )
    if len(devices) != len(set(devices)):
        raise ValueError(f"GPU list contains duplicates: {devices}")
    return devices


def _task_name(episode: dict[str, Any]) -> str:
    """Build a stable human-readable name for one batch rollout task."""

    raw = (
        f"{episode['scene_id']}_ep{episode['episode_id']}_{episode['object_category']}"
    )
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in raw)


def _load_episode_subset(
    path: str | Path,
    canonical: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Load an inspectable episode list and validate it against full validation."""

    resolved = resolve_release_path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    rows = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Episode list must contain an episodes array: {resolved}")

    available = {
        (row["scene_id"], row["episode_id"], row["object_category"]): row
        for row in canonical
    }
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Episode list row {index} is not an object: {resolved}")
        key = (
            str(row.get("scene_id") or ""),
            str(row.get("episode_id") or ""),
            str(row.get("object_category") or ""),
        )
        if key in seen:
            raise ValueError(f"Duplicate episode-list row: {key}")
        if key not in available:
            raise ValueError(f"Episode-list row is not in full validation: {key}")
        seen.add(key)
        selected.append(available[key])

    declared_count = value.get("n_episodes")
    if declared_count is not None and int(declared_count) != len(selected):
        raise ValueError(
            f"Episode list declares {declared_count} rows but contains {len(selected)}"
        )
    return selected


def _select_episode_identity(
    episodes: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Optionally select one exact scene/episode/category from a split."""

    selectors = {
        "scene_id": config.get("scene_id"),
        "episode_id": config.get("episode_id"),
        "object_category": config.get("object_category"),
    }
    active = {key: str(value) for key, value in selectors.items() if value is not None}
    if not active:
        return episodes
    selected = [
        row
        for row in episodes
        if all(str(row.get(key)) == value for key, value in active.items())
    ]
    if bool(config.get("require_unique_episode", False)) and len(selected) != 1:
        rendered = ", ".join(f"{key}={value}" for key, value in active.items())
        raise ValueError(
            f"Expected exactly one episode for {rendered}; found {len(selected)}"
        )
    return selected


def _rollout_complete(
    output_root: Path,
    episode: dict[str, Any],
    *,
    save_video: bool,
    compute_metrics: bool,
) -> bool:
    """Return whether every requested final artifact exists and is non-empty."""

    rollout = (
        output_root
        / _task_name(episode)
        / "rollout_00"
    )
    required = [
        rollout / "replay_manifest.json",
        rollout / "trajectory_before.npz",
        rollout / "trajectory_after.npz",
    ]
    if save_video:
        required.extend((rollout / "ego.mp4", rollout / "exo.mp4"))
    if compute_metrics:
        required.append(rollout / "metrics.json")
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _least_loaded_device(
    devices: tuple[str, ...],
    active_devices: list[str],
    launch_index: int,
) -> str | None:
    """Choose the least-populated GPU, rotating ties across launch order.

    Episode durations vary substantially.  Plain round-robin assignment can
    therefore drift from 16/16 active processes to an unsafe 29/22 split even
    though total concurrency is unchanged.  Counting live children preserves
    balanced memory pressure without querying vendor-specific GPU tools.
    """

    if not devices:
        return None
    counts = {device: active_devices.count(device) for device in devices}
    pivot = int(launch_index) % len(devices)
    tie_order = devices[pivot:] + devices[:pivot]
    return min(tie_order, key=counts.__getitem__)


def run_batch(config: dict[str, Any]) -> dict[str, Any]:
    """Run an episode slice and optionally aggregate its paper metrics."""

    profile = config.get("profile")
    if not isinstance(profile, ReleaseConfig):
        raise TypeError("config['profile'] must be a loaded ReleaseConfig")
    max_parallel = int(config.get("max_parallel", 1))
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")

    benchmark = profile.section("benchmark")
    episodes = list_episode_specs(
        resolve_release_path(benchmark["dataset_dir"]),
        str(benchmark["split"]),
    )
    offset = int(config.get("offset", 0))
    limit = int(config.get("limit", 0))
    episode_list = config.get("episode_list")
    selected = (
        _load_episode_subset(episode_list, episodes) if episode_list else episodes
    )
    selected = _select_episode_identity(selected, config)
    selected = selected[offset:]
    if limit > 0:
        selected = selected[:limit]

    model_config = Path(config["model_config"]).expanduser().resolve()
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config}")
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requested = list(selected)
    skipped = 0
    if bool(config.get("resume", False)):
        selected = [
            episode
            for episode in requested
            if not _rollout_complete(
                output_root,
                episode,
                save_video=bool(config.get("save_video", False)),
                compute_metrics=bool(config.get("compute_metrics", False)),
            )
        ]
        skipped = len(requested) - len(selected)
    devices = tuple(config.get("devices") or ())
    scene_dataset_config = config.get("scene_dataset_config")

    # Verify external weights once in the dispatcher. Episode subprocesses
    # inherit an exact manifest/root marker and still check every file size,
    # but avoid hashing the same immutable files 1,218 times.
    child_weight_env: dict[str, str] = {}
    motion = profile.section("motion")
    if bool(motion.get("verify_weights", True)):
        weights_root = resolve_release_path(motion["weights_root"])
        weights_manifest = resolve_release_path(motion["weights_manifest"])
        weight_results = verify_weights(weights_manifest, weights_root)
        failures = [row for row in weight_results if not row["ok"]]
        if failures:
            raise ValueError(f"Motion weight verification failed: {failures}")
        child_weight_env = {
            "HUMANCLAW_BATCH_VERIFIED_WEIGHTS_ROOT": str(weights_root),
            "HUMANCLAW_BATCH_VERIFIED_WEIGHT_MANIFEST_SHA256": sha256_file(
                weights_manifest
            ),
        }

    pending = deque(selected)
    running: list[tuple[subprocess.Popen, dict[str, Any], str | None]] = []
    completed = 0
    failed = 0
    launch_index = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            episode = pending.popleft()
            # Another dispatcher may finish a disjoint slice while this long
            # batch is still draining its queue.  Recheck at launch time so a
            # resumed full-val dispatcher never overwrites that completed
            # episode with a duplicate rollout.
            if bool(config.get("resume", False)) and _rollout_complete(
                output_root,
                episode,
                save_video=bool(config.get("save_video", False)),
                compute_metrics=bool(config.get("compute_metrics", False)),
            ):
                skipped += 1
                continue
            command = [
                sys.executable,
                "-m",
                "humanclaw_bench",
                "rollout",
                "--profile",
                str(profile.path.resolve()),
                "--model-config",
                str(model_config),
                "--scene-id",
                str(episode["scene_id"]),
                "--episode-id",
                str(episode["episode_id"]),
                "--object-category",
                str(episode["object_category"]),
                "--output-root",
                str(output_root),
            ]
            if scene_dataset_config is not None:
                command += [
                    "--scene-dataset-config",
                    str(Path(scene_dataset_config).expanduser().resolve()),
                ]
            if bool(config.get("save_video", False)):
                command.append("--save-video")
            if bool(config.get("compute_metrics", False)):
                command.append("--compute-metrics")
            env = _child_process_env()
            env.update(child_weight_env)
            device = _least_loaded_device(
                devices,
                [active_device for _, _, active_device in running if active_device],
                launch_index,
            )
            if device is not None:
                env["CUDA_VISIBLE_DEVICES"] = device
            process = subprocess.Popen(
                command,
                cwd=str(repository_root()),
                env=env,
            )
            running.append((process, episode, device))
            launch_index += 1

        remaining: list[tuple[subprocess.Popen, dict[str, Any], str | None]] = []
        for process, episode, device in running:
            return_code = process.poll()
            if return_code is None:
                remaining.append((process, episode, device))
            elif return_code == 0:
                completed += 1
            else:
                failed += 1
        running = remaining
        if running:
            time.sleep(1.0)

    summary: dict[str, Any] = {
        "selected": len(requested),
        "skipped_complete": skipped,
        "launched": launch_index,
        "completed": completed,
        "failed": failed,
        "devices": list(devices),
        "max_parallel": max_parallel,
    }
    if bool(config.get("compute_metrics", False)):
        from .evaluation.metrics import aggregate_metric_files

        summary["metrics"] = aggregate_metric_files(output_root)
        summary["metrics_path"] = str(output_root / "metrics_summary.json")
    return summary


__all__ = ["resolve_devices", "run_batch"]

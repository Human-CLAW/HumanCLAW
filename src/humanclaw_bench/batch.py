"""Minimal local dispatcher for HumanClawBench episodes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
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


def _snapshot(value: Any) -> Any:
    """Copy JSON configuration with the same credential-key policy as traces."""

    from .experiments.recorder import _credential_key

    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _credential_key(str(key)) else _snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def _model_snapshot(path: Path) -> dict[str, Any]:
    """Match the rollout adapter's normalization without loading the adapter."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Model configuration must be a JSON object")
    if value.get("backend") == "filesystem_queue" and value.get("queue_dir"):
        queue = Path(str(value["queue_dir"])).expanduser()
        if not queue.is_absolute():
            value["queue_dir"] = str((path.parent / queue).resolve())
    return _snapshot(value)


def _trace_configuration(
    config: dict[str, Any], profile: ReleaseConfig, model_config: Path,
) -> dict[str, Any]:
    """Describe the configuration that a resumable child trace must match."""

    from .experiments.integration import resolved_model_settings, source_fingerprint

    max_steps = config.get("max_steps")
    scene_override = config.get("scene_dataset_config")
    model_error = None
    try:
        model = _model_snapshot(model_config)
    except (OSError, ValueError) as exc:
        model = {}
        model_error = {"type": type(exc).__name__, "message": str(exc)}
    pricing, pricing_error = None, None
    if config.get("pricing"):
        try:
            pricing = json.loads(Path(config["pricing"]).expanduser().read_text(encoding="utf-8"))
            if not isinstance(pricing, dict):
                raise ValueError("Pricing configuration must be a JSON object")
        except (OSError, ValueError) as exc:
            pricing = None
            pricing_error = {"type": type(exc).__name__, "message": str(exc)}
    result = {
        "profile": _snapshot(profile.data),
        "model_config": model,
        "effective_model_config": _snapshot(resolved_model_settings(model)),
        "source": source_fingerprint(),
        "pricing": _snapshot(pricing),
        "harness": config.get("harness", "humanclaw_psv"),
        "seed": config.get("seed"),
        "effective_seed": config.get("seed"),
        "rollout_index": 0,
        "max_steps": int(max_steps if max_steps is not None else profile.section("benchmark")["max_steps"]),
        "compute_metrics": bool(config.get("compute_metrics", False)),
        "save_video": bool(config.get("save_video", False)),
        "config": {
            "agent_asset": config.get("agent_asset"),
            "scene_dataset_config": str(Path(scene_override).expanduser().resolve()) if scene_override else None,
            "max_steps": max_steps,
        },
    }
    if model_error is not None:
        result["model_snapshot_error"] = model_error
    if pricing_error is not None:
        result["pricing_snapshot_error"] = pricing_error
    return result


def _trace_matches(
    manifest: dict[str, Any], episode: dict[str, Any], expected: dict[str, Any],
) -> bool:
    """Conservatively match actual child configuration and episode identity."""

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return False
    if manifest.get("pricing") != expected.get("pricing"):
        return False
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("files_sha256") != expected["source"].get("files_sha256"):
        return False
    for key in (
        "profile", "model_config", "effective_model_config", "harness", "seed", "effective_seed",
        "rollout_index", "compute_metrics", "save_video",
    ):
        if key not in metadata or metadata[key] != expected[key]:
            return False
    actual_episode = metadata.get("episode")
    if not isinstance(actual_episode, dict):
        return False
    actual_scene = actual_episode.get("scene_label", actual_episode.get("scene_id"))
    if (
        str(actual_scene) != str(episode["scene_id"])
        or str(actual_episode.get("episode_id")) != str(episode["episode_id"])
        or actual_episode.get("object_category") != episode["object_category"]
        or actual_episode.get("max_steps") != expected["max_steps"]
    ):
        return False
    overrides = metadata.get("config")
    if not isinstance(overrides, dict):
        return False
    return all(overrides.get(key) == value for key, value in expected["config"].items())


def _rollout_trace_directory(output_root: Path, episode: dict[str, Any]) -> Path:
    """Locate immutable attempts belonging to one rollout output directory."""

    return output_root / _task_name(episode) / "rollout_00" / "experiment_runs"


def _completed_trace(
    output_root: Path, episode: dict[str, Any], expected: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a valid matching completed trace; old output files alone never qualify."""

    from .experiments.report import validate_trace

    if not _rollout_complete(
        output_root, episode, save_video=expected["save_video"],
        compute_metrics=expected["compute_metrics"],
    ):
        return None
    candidates = sorted(_rollout_trace_directory(output_root, episode).glob("*/run.json"))
    for path in reversed(candidates):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(manifest, dict) and manifest.get("status") == "completed"
                and _trace_matches(manifest, episode, expected)
            ):
                validation = validate_trace(path.parent)
                if not validation["ok"] or validation.get("warnings"):
                    continue
                return {"run_dir": str(path.parent), "run_id": manifest.get("run_id"),
                        "status": "completed", "outcome": manifest.get("outcome", {})}
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def _batch_utc() -> str:
    """Return explicit UTC for dispatcher lifecycle records."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _BatchAttempt:
    """Preserve one dispatcher's complete denominator in its own fresh directory."""

    def __init__(
        self, output_root: Path, requested: list[dict[str, Any]],
        config: dict[str, Any], profile: ReleaseConfig, model_config: Path,
        expected: dict[str, Any],
    ) -> None:
        """Persist all requested episodes before any child can fail to start."""

        self.output_root = output_root
        self.expected = expected
        self.batch_id = f"batch_{uuid.uuid4().hex}"
        self.directory = output_root / "batch_attempts" / self.batch_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "batch.json"
        self.rows = {
            _task_name(episode): {
                "child_id": f"child_{uuid.uuid4().hex}", "episode": dict(episode),
                "status": "pending", "return_code": None, "error": None,
                "trace_runs": [], "task_success": None,
            }
            for episode in requested
        }
        self.manifest = {
            "schema": "humanclaw_batch_attempt_v1", "batch_attempt_id": self.batch_id,
            "status": "running", "started_at_utc": _batch_utc(), "finished_at_utc": None,
            "requested_count": len(requested), "configuration": _snapshot(expected),
            "configuration_sources": {"profile": str(profile.path.resolve()), "model_config": str(model_config)},
            "dispatch": _snapshot({key: config.get(key) for key in (
                "offset", "limit", "episode_list", "max_parallel", "devices", "resume", "pricing",
            )}),
            "children": list(self.rows.values()),
            "aggregate_scope": "This batch counts each requested episode once, including startup failures and reused matching traces.",
        }
        self.flush()

    def flush(self) -> None:
        """Atomically checkpoint the current attempt without touching older attempts."""

        from .experiments.recorder import TraceRecorder

        payload = json.dumps(_snapshot(self.manifest), indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        TraceRecorder._atomic_bytes(self.path, payload)

    def row(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Find the unique requested row for one scene/episode/category."""

        return self.rows[_task_name(episode)]

    def skip(self, episode: dict[str, Any], trace: dict[str, Any]) -> None:
        """Record precisely which matching prior attempt satisfies resume."""

        self.row(episode).update(
            status="skipped_complete", trace_runs=[trace],
            task_success=trace.get("outcome", {}).get("task_success"),
            trace_membership="reused_matching_trace", finished_at_utc=_batch_utc(),
        )
        self.flush()

    def collect(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        """Associate traces using explicit batch/child IDs, never directory timing."""

        from .experiments.report import validate_trace

        row = self.row(episode)
        traces = []
        for path in sorted(_rollout_trace_directory(self.output_root, episode).glob("*/run.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                metadata = manifest.get("metadata", {})
                if metadata.get("batch_attempt_id") != self.batch_id or metadata.get("batch_child_id") != row["child_id"]:
                    continue
                validation = validate_trace(path.parent)
                traces.append({
                    "run_id": manifest.get("run_id"), "run_dir": str(path.parent),
                    "status": manifest.get("status"), "outcome": manifest.get("outcome", {}),
                    "valid": validation["ok"] and not validation.get("warnings"), "diagnostics": validation,
                    "configuration_matches": _trace_matches(manifest, episode, self.expected),
                })
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        row.update(trace_runs=traces, trace_membership="explicit_batch_child_ids")
        return traces

    def child_exit(self, episode: dict[str, Any], return_code: int) -> bool:
        """Retain process errors and reject apparent success without a valid trace."""

        row = self.row(episode)
        traces = self.collect(episode)
        valid = (
            len(traces) == 1 and traces[0]["status"] == "completed"
            and traces[0]["valid"] and traces[0]["configuration_matches"]
        )
        complete = return_code == 0 and valid
        error = None
        if return_code != 0:
            error = {"type": "ChildProcessError", "message": f"Rollout process exited with code {return_code}"}
        elif not valid:
            error = {"type": "IncompleteTrace", "message": "Expected one completed, valid, matching trace owned by this child"}
        success = traces[0].get("outcome", {}).get("task_success") if complete else None
        row.update(
            status="completed" if complete else "failed", return_code=return_code,
            error=error, finished_at_utc=_batch_utc(),
            process_failure_without_trace=return_code != 0 and not traces,
            task_success=success if isinstance(success, bool) else None,
        )
        self.flush()
        return complete

    def finish(self, summary: dict[str, Any], *, status: str, error: Any = None) -> None:
        """Publish batch membership separately from the all-attempt collection."""

        from .experiments.report import summarize_runs

        rows = list(self.rows.values())
        scored = [row for row in rows if isinstance(row.get("task_success"), bool)]
        successes = sum(row["task_success"] for row in scored)
        membership = {
            "requested_episodes": len(rows), "scored_episodes": len(scored),
            "unscored_episodes": len(rows) - len(scored), "task_successes": successes,
            "task_success_rate_all_requested": successes / len(rows) if rows else None,
            "task_success_rate_scored": successes / len(scored) if scored else None,
            "outcome_coverage": len(scored) / len(rows) if rows else None,
            "startup_failures": sum(row["status"] == "startup_failed" for row in rows),
            "not_launched": sum(row["status"] == "not_launched" for row in rows),
            "process_failures_without_trace": sum(bool(row.get("process_failure_without_trace")) for row in rows),
        }
        summary.update(batch_attempt_id=self.batch_id, batch_manifest=str(self.path), batch_outcomes=membership)
        self.manifest.update(status=status, finished_at_utc=_batch_utc(), summary=summary, error=error)
        self.flush()
        try:
            collection = summarize_runs(self.output_root, write=True)
            aggregate = {
                "path": str(self.output_root / "experiment_summary.json"),
                "scope": "All recorded trace attempts under output_root, including previous batches; not the current batch denominator. Children without traces exist only in batch manifests.",
                "aggregate": collection["aggregate"],
            }
            summary["experiments"] = aggregate
        except Exception as exc:
            summary["experiment_summary_error"] = {"type": type(exc).__name__, "message": str(exc)}
        self.manifest["summary"] = summary
        self.flush()


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
    if not model_config.is_file() and not bool(config.get("trace", False)):
        raise FileNotFoundError(f"Model config not found: {model_config}")
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requested = list(selected)
    traced = bool(config.get("trace", False))
    expected = _trace_configuration(config, profile, model_config) if traced else None
    attempt = (
        _BatchAttempt(output_root, requested, config, profile, model_config, expected)
        if expected is not None else None
    )
    summary: dict[str, Any] = {
        "selected": len(requested), "skipped_complete": 0, "launched": 0,
        "completed": 0, "failed": 0, "devices": list(config.get("devices") or ()),
        "max_parallel": max_parallel,
    }
    terminal_status = "completed"
    terminal_error = None
    try:
        if expected is not None and expected.get("model_snapshot_error"):
            raise ValueError(f"Cannot snapshot model configuration: {expected['model_snapshot_error']}")
        if expected is not None and expected.get("pricing_snapshot_error"):
            raise ValueError(f"Cannot snapshot pricing configuration: {expected['pricing_snapshot_error']}")
        _dispatch_batch(config, profile, model_config, output_root, requested, summary, attempt)
        if bool(config.get("compute_metrics", False)):
            from .evaluation.metrics import aggregate_metric_files

            summary["metrics"] = aggregate_metric_files(output_root)
            summary["metrics_path"] = str(output_root / "metrics_summary.json")
        if summary["failed"]:
            terminal_status = "completed_with_failures"
    except BaseException as exc:
        terminal_status = "failed" if isinstance(exc, Exception) else "interrupted"
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        if attempt is not None:
            for row in attempt.rows.values():
                if row["status"] in {"pending", "launching", "running"}:
                    row.update(
                        status="not_launched" if row["status"] == "pending" else terminal_status,
                        error=terminal_error, finished_at_utc=_batch_utc(),
                    )
        raise
    finally:
        if attempt is not None:
            attempt.finish(summary, status=terminal_status, error=terminal_error)
    return summary


def _dispatch_batch(
    config: dict[str, Any], profile: ReleaseConfig, model_config: Path,
    output_root: Path, requested: list[dict[str, Any]], summary: dict[str, Any],
    attempt: _BatchAttempt | None,
) -> None:
    """Launch bounded children and checkpoint every exit or startup failure."""

    def skip_if_complete(episode: dict[str, Any]) -> bool:
        """Resume only an artifact-complete episode satisfying this mode's contract."""

        if not bool(config.get("resume", False)):
            return False
        if attempt is not None:
            trace = _completed_trace(output_root, episode, attempt.expected)
            if trace is None:
                return False
            attempt.skip(episode, trace)
        elif not _rollout_complete(
            output_root, episode, save_video=bool(config.get("save_video", False)),
            compute_metrics=bool(config.get("compute_metrics", False)),
        ):
            return False
        summary["skipped_complete"] += 1
        return True

    selected = [episode for episode in requested if not skip_if_complete(episode)]
    devices = tuple(config.get("devices") or ())
    max_parallel = int(config.get("max_parallel", 1))
    scene_dataset_config = config.get("scene_dataset_config")
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
            "HUMANCLAW_BATCH_VERIFIED_WEIGHT_MANIFEST_SHA256": sha256_file(weights_manifest),
        }

    pending = deque(selected)
    running: list[tuple[subprocess.Popen, dict[str, Any], str | None]] = []
    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                episode = pending.popleft()
                # Recheck before launch in case another dispatcher completed it.
                if skip_if_complete(episode):
                    continue
                command = [
                    sys.executable, "-m", "humanclaw_bench", "rollout",
                    "--profile", str(profile.path.resolve()),
                    "--model-config", str(model_config),
                    "--scene-id", str(episode["scene_id"]),
                    "--episode-id", str(episode["episode_id"]),
                    "--object-category", str(episode["object_category"]),
                    "--output-root", str(output_root),
                ]
                if scene_dataset_config is not None:
                    command += ["--scene-dataset-config", str(Path(scene_dataset_config).expanduser().resolve())]
                if bool(config.get("save_video", False)):
                    command.append("--save-video")
                if bool(config.get("compute_metrics", False)):
                    command.append("--compute-metrics")
                if config.get("agent_asset"):
                    command += ["--agent-asset", str(config["agent_asset"])]
                if bool(config.get("trace", False)):
                    command.append("--trace")
                if "harness" in config or bool(config.get("trace", False)):
                    command += ["--harness", str(config.get("harness", "humanclaw_psv"))]
                if config.get("seed") is not None:
                    command += ["--seed", str(config["seed"])]
                if config.get("max_steps") is not None:
                    command += ["--max-steps", str(config["max_steps"])]
                if config.get("pricing") is not None:
                    command += ["--pricing", str(Path(config["pricing"]).expanduser().resolve())]
                env = _child_process_env()
                env.update(child_weight_env)
                device = _least_loaded_device(
                    devices, [active_device for _, _, active_device in running if active_device],
                    summary["launched"],
                )
                if device is not None:
                    env["CUDA_VISIBLE_DEVICES"] = device
                if attempt is not None:
                    row = attempt.row(episode)
                    row.update(status="launching", command=command, device=device, started_at_utc=_batch_utc())
                    env["HUMANCLAW_BATCH_ATTEMPT_ID"] = attempt.batch_id
                    env["HUMANCLAW_BATCH_CHILD_ID"] = row["child_id"]
                    attempt.flush()
                stderr_handle = None
                try:
                    popen_options: dict[str, Any] = {"cwd": str(repository_root()), "env": env}
                    if attempt is not None:
                        logs = attempt.directory / "children"
                        logs.mkdir(exist_ok=True)
                        stderr_path = logs / f"{attempt.row(episode)['child_id']}.stderr.log"
                        stderr_handle = stderr_path.open("wb")
                        popen_options["stderr"] = stderr_handle
                        attempt.row(episode)["stderr_log"] = str(stderr_path)
                        attempt.flush()
                    process = subprocess.Popen(command, **popen_options)
                except OSError as exc:
                    if attempt is None:
                        raise
                    attempt.row(episode).update(
                        status="startup_failed", error={"type": type(exc).__name__, "message": str(exc)},
                        failure_stage="process_launch", finished_at_utc=_batch_utc(),
                    )
                    summary["failed"] += 1
                    attempt.flush()
                    continue
                finally:
                    if stderr_handle is not None:
                        # Popen duplicates the descriptor; the child retains it.
                        stderr_handle.close()
                running.append((process, episode, device))
                summary["launched"] += 1
                if attempt is not None:
                    attempt.row(episode).update(status="running", pid=getattr(process, "pid", None))
                    attempt.flush()

            remaining: list[tuple[subprocess.Popen, dict[str, Any], str | None]] = []
            for process, episode, device in running:
                return_code = process.poll()
                if return_code is None:
                    remaining.append((process, episode, device))
                    continue
                complete = attempt.child_exit(episode, return_code) if attempt is not None else return_code == 0
                summary["completed" if complete else "failed"] += 1
            running = remaining
            if running:
                time.sleep(1.0)
    finally:
        # An interrupted dispatcher owns its children; do not leave them running
        # after publishing an interrupted batch manifest.
        for process, _, _ in running:
            try:
                if process.poll() is None:
                    process.terminate()
            except OSError:
                pass
        deadline = time.monotonic() + 5.0
        for process, episode, _ in running:
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
            if attempt is not None:
                attempt.collect(episode)


__all__ = ["resolve_devices", "run_batch"]

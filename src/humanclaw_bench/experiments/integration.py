"""HumanCLAW boundaries shared by the instrumented harnesses."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .recorder import event, get_recorder, span

SKILLS = (
    "walk_forward", "stand", "turn", "step_climb_up", "sit", "step_back",
    "side_walk", "step_climb_down",
)


def resolved_model_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve public environment overrides without reading credential values."""
    settings = dict(config)
    if settings.get("model_env"):
        settings["model"] = os.environ.get(str(settings["model_env"]), settings.get("model"))
    if settings.get("backend") == "azure_openai":
        for key, default_env in (("azure_endpoint", "AZURE_OPENAI_ENDPOINT"),
                                 ("api_version", "AZURE_OPENAI_API_VERSION")):
            settings[key] = settings.get(key) or os.environ.get(str(settings.get(key + "_env", default_env)))
    elif settings.get("backend") == "openai_compatible":
        settings["base_url"] = settings.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return settings


def source_fingerprint() -> dict[str, Any]:
    """Hash the actual source tree, including untracked experiment modules."""
    from humanclaw_bench.paths import repository_root

    source_files = {}
    for folder, pattern in (("src", "*.py"), ("configs", "*.json")):
        for path in sorted((repository_root() / folder).rglob(pattern)):
            source_files[str(path.relative_to(repository_root()))] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    return {"files_sha256": source_digest, "files": source_files}


def register_capabilities(harness: str, *, compute_metrics: bool = False) -> None:
    """Register action tools separately from native provider tool calling."""
    for name in SKILLS:
        event("capability.registered", kind="tool", name=name, data={
            "interface": "structured_action_json", "namespace": "environment_skill",
            "available": True,
        })
    modules = ["agent.act", "motion.generate", "environment.reset", "history.update",
               "metrics.record_decision", "metrics.record_motion", "artifacts.finalize"]
    modules += ["bare_api"] if harness == "bare_api" else ["percept_mid_low", "verifier"]
    for name in modules:
        event("capability.registered", kind="module", name=name,
              data={"observability": "instrumented_boundary",
                    "available": compute_metrics if name.startswith("metrics.") else True})
    event("capability.registered", kind="environment", name="environment.step",
          data={"observability": "instrumented_boundary", "available": True})


def call_model(model: Any, messages: list[dict[str, Any]]) -> str:
    """Observe one adapter call; provider/SDK-internal operations remain opaque."""
    with span("model.respond", kind="model", input={"messages": messages}, attributes={
        "model": getattr(model, "model_name", type(model).__name__),
        "adapter": type(model).__name__, "boundary": "adapter_respond",
    }) as call:
        try:
            output = model.respond(messages)
        except BaseException:
            # Adapters clear last_usage on entry. Avoid inventing usage for failures.
            call.set_usage(dict(getattr(model, "last_trace_usage", getattr(model, "last_usage", {})) or {}))
            raise
        call.set_usage(dict(getattr(model, "last_trace_usage", getattr(model, "last_usage", {})) or {}))
        call.set_output({"text": output})
        return output


def seed_runtime(seed: int | None) -> None:
    """Seed host and, when installed, motion RNGs without demanding CUDA."""
    if seed is None:
        return
    if not 0 <= seed < 2**32:
        raise ValueError("Effective seed must be in [0, 2**32)")
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_metadata(evaluator: Any, episode: Any, rollout_index: int) -> dict[str, Any]:
    """Snapshot configuration, seed, software identity, and observable boundaries."""
    from humanclaw_bench.paths import repository_root
    profile = getattr(evaluator.profile, "data", {})
    versions = {}
    for package in ("humanclaw-bench", "numpy", "Pillow", "scipy", "torch", "openai", "habitat-sim"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root(), text=True,
            stderr=subprocess.DEVNULL, timeout=3,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "configs"], cwd=repository_root(),
            stderr=subprocess.DEVNULL, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        revision, diff = None, b""
    return {
        "benchmark": "HumanCLAW", "harness": evaluator.config.get("harness", "humanclaw_psv"),
        "episode_id": str(episode.episode_id), "episode": vars(episode),
        "rollout_index": rollout_index, "profile": profile,
        "model_config": evaluator._model_config or {},
        "effective_model_config": resolved_model_settings(evaluator._model_config or {}),
        "seed": evaluator.config.get("seed"),
        "effective_seed": (int(evaluator.config["seed"]) + rollout_index
                           if evaluator.config.get("seed") is not None else None),
        "compute_metrics": evaluator.compute_metrics, "save_video": evaluator.save_video,
        "config": {key: evaluator.config.get(key) for key in (
            "agent_asset", "device", "scene_dataset_config", "max_steps",
            "parent_run_id", "source_run",
        )},
        "batch_attempt_id": os.environ.get("HUMANCLAW_BATCH_ATTEMPT_ID"),
        "batch_child_id": os.environ.get("HUMANCLAW_BATCH_CHILD_ID"),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": versions},
        "source": {"commit": revision, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                   **source_fingerprint()},
        "observability": {
            "model": "adapter_calls", "tools": "structured_environment_actions",
            "modules": "instrumented_boundaries", "provider_internal": "unavailable",
            "sdk_internal_retries": "unavailable",
        },
        "latency_scope": "traced episode execution and archival; excludes process/model construction and report generation",
        "task_success_definition": "interact_sr for interaction episodes, nav_sr_20cm otherwise",
    }


def preserve_replay(output_dir: Path) -> dict[str, Any]:
    """Archive each attempt's physics data before a future rerun overwrites it."""
    from humanclaw_bench.assets import sha256_file

    recorder = get_recorder()
    if recorder is None:
        return {}
    replay = recorder.directory / "replay"
    replay.mkdir(exist_ok=True)
    refs = {}
    for name in ("replay_manifest.json", "trajectory_before.npz", "trajectory_after.npz"):
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, replay / name)
            if name == "replay_manifest.json":
                snapshot = json.loads(source.read_text())
                profile = recorder.manifest["metadata"].get("profile")
                if profile:
                    snapshot["profile"] = profile
                (replay / name).write_text(json.dumps(snapshot, indent=2) + "\n")
            refs[name] = {"artifact": str(Path("replay") / name),
                          "sha256": sha256_file(replay / name),
                          "media_type": "application/json" if name.endswith(".json") else "application/x-npz"}
    return refs


def benchmark_outcome(output_dir: Path, *, steps: int, termination_reason: str,
                      artifacts_ready: bool = True) -> dict[str, Any]:
    """Read benchmark outcomes and archive replay data without masking run failures."""
    path = output_dir / "metrics.json"
    diagnostics = []
    try:
        metrics = json.loads(path.read_text()) if artifacts_ready and path.is_file() else None
        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError("metrics root must be an object")
    except (OSError, ValueError) as exc:
        metrics = None
        diagnostics.append(f"metrics unavailable: {exc}")
    try:
        replay = preserve_replay(output_dir) if artifacts_ready else {}
    except (OSError, ValueError, TypeError) as exc:
        replay = {}
        diagnostics.append(f"replay archive failed: {exc}")
    success = (metrics or {}).get("success", {})
    key = "interact_sr" if success.get("is_interact_episode") else "nav_sr_20cm"
    value = success.get(key) if termination_reason not in {"error", "interrupted"} else None
    return {"steps": steps, "termination_reason": termination_reason,
            "task_success": value if isinstance(value, bool) else None,
            "benchmark_metrics": metrics, "replay": replay, "diagnostics": diagnostics}

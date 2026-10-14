"""Inspect, replay or rerun one immutable experiment attempt.

Log inspection uses only the standard library. Simulator/model imports happen
only after the corresponding mode is selected. Physics replay executes saved
generated motion; a rerun executes the agent anew and can make paid model calls.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Any
import uuid


def _object(path: Path) -> dict[str, Any]:
    """Read a JSON object without interpreting it as executable configuration."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _run(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Resolve one trace directory and validate its manifest envelope."""
    directory = Path(run_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Run directory not found: {directory}")
    manifest = _object(directory / "run.json")
    if manifest.get("schema") != "humanclaw_experiment_run_v1":
        raise ValueError(f"Unsupported experiment manifest: {manifest.get('schema')}")
    if not isinstance(manifest.get("metadata"), dict):
        raise ValueError("Run manifest has no structured metadata")
    return directory, manifest


def _fresh_output(source: Path, output_dir: str | Path | None, mode: str) -> Path:
    """Create a new destination outside the immutable source attempt."""
    output = (Path(output_dir).expanduser().resolve() if output_dir is not None
              else source.with_name(f"{source.name}_{mode}_{uuid.uuid4().hex[:8]}"))
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the original run directory")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    return output


def _write_report(output: Path, name: str, result: dict[str, Any]) -> None:
    """Write a JSON audit record with non-finite numeric values rejected."""
    (output / name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frozen_episode(metadata: dict[str, Any]) -> Any:
    """Reconstruct a stored episode without reloading mutable benchmark files."""
    from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode

    value = metadata.get("episode")
    if not isinstance(value, dict):
        raise ValueError("Run metadata does not include the complete frozen episode")
    allowed = {field.name for field in fields(HCFindNavInteractEpisode)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Frozen episode contains unsupported fields: {unknown}")
    try:
        episode = HCFindNavInteractEpisode(**deepcopy(value))
    except TypeError as exc:
        raise ValueError("Frozen episode is incomplete") from exc
    if not isinstance(episode.max_steps, int) or isinstance(episode.max_steps, bool) or episode.max_steps <= 0:
        raise ValueError("Frozen episode max_steps must be a positive integer")
    return episode


def replay_run(
    run_dir: str | Path,
    mode: str = "log",
    output_dir: str | Path | None = None,
    max_steps: int | None = None,
    *,
    pose_atol: float = 1.0e-4,
    object_atol: float = 5.0e-5,
) -> dict[str, Any]:
    """Inspect log events, compare forward physics, or render saved poses.

    ``max_steps`` limits physics replay only and must be positive when supplied.
    Physics/render modes create a fresh sibling output directory by default.
    Log mode is read-only unless an explicit output directory is supplied.
    Render is direct saved-pose rasterization; it does not execute physics.
    Legacy rollout directories without a trace remain usable in physics/render.
    """
    if mode not in {"log", "physics", "render"}:
        raise ValueError(f"Unknown replay mode: {mode}")
    if max_steps is not None:
        if mode != "physics":
            raise ValueError("max_steps is supported only for physics replay")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
    if not all(math.isfinite(value) and value >= 0 for value in (pose_atol, object_atol)):
        raise ValueError("Replay tolerances must be finite and nonnegative")
    directory = Path(run_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Run directory not found: {directory}")

    if mode == "log":
        from .report import format_timeline, validate_trace

        # Reader diagnostics intentionally permit inspection of a truncated run.
        if not (directory / "events.jsonl").is_file():
            raise FileNotFoundError(f"Trace events not found: {directory / 'events.jsonl'}")
        result = {"mode": mode, "source_run": str(directory),
                  "timeline": format_timeline(directory), "validation": validate_trace(directory)}
        if output_dir is not None:
            output = _fresh_output(directory, output_dir, mode)
            result["output_dir"] = str(output)
            (output / "timeline.txt").write_text(result["timeline"] + "\n", encoding="utf-8")
            _write_report(output, "replay_report.json", result)
        return result

    manifest: dict[str, Any] = {}
    if (directory / "run.json").is_file():
        directory, manifest = _run(directory)
        replay = directory / "replay"
    else:
        replay = directory
    required = ["replay_manifest.json", "trajectory_after.npz"]
    if mode == "physics":
        required.append("trajectory_before.npz")
    for name in required:
        if not (replay / name).is_file():
            raise FileNotFoundError(f"Saved replay input is missing: {replay / name}")
    episode = _frozen_episode(manifest["metadata"]) if manifest and mode == "physics" else None
    output = _fresh_output(directory, output_dir, mode)
    try:
        if mode == "physics":
            from .physics_replay import replay_and_compare

            report = replay_and_compare(
                replay, max_steps=max_steps or 0, pose_atol=pose_atol,
                object_atol=object_atol, episode=episode, fail_on_divergence=False,
            )
        else:
            from humanclaw_bench.rendering.saved_trajectory import render_saved_trajectory

            report = render_saved_trajectory(replay, output_dir=output)
        result = {"mode": mode, "source_run": str(directory),
                  "output_dir": str(output), **report}
        _write_report(output, "replay_report.json", result)
        return result
    except BaseException as exc:
        # Keep the failed attempt reviewable without exposing arbitrary provider
        # error text; callers still receive the original exception.
        _write_report(output, "replay_report.json", {
            "mode": mode, "source_run": str(directory), "output_dir": str(output),
            "status": "failed" if isinstance(exc, Exception) else "interrupted",
            "error_type": type(exc).__name__,
        })
        raise


def _execute_rerun(config: dict[str, Any]) -> dict[str, Any]:
    """Construct the real evaluator only for an explicitly requested live rerun."""
    from humanclaw_bench.evaluation.evaluator import HCFindNavInteractEvaluator

    evaluator = HCFindNavInteractEvaluator(config)
    evaluator.check_config_valid()
    return evaluator.evaluate_main()


def rerun_run(
    run_dir: str | Path,
    model_config: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute a fresh agent attempt from the saved episode/profile/harness/seed.

    This makes real model calls. Model connection settings and credentials must
    come from an explicitly supplied external config, never the sanitized trace.
    The source remains immutable; output must be a new directory outside it.
    A matching seed does not imply deterministic model/provider or GPU behavior.
    """
    source, manifest = _run(run_dir)
    metadata = manifest["metadata"]
    saved_config = metadata.get("config", {})
    if not isinstance(saved_config, dict):
        raise ValueError("Saved runtime config must be a JSON object")
    profile = deepcopy(metadata.get("profile"))
    if not isinstance(profile, dict) or not profile.get("profile"):
        raise ValueError("Rerun requires a complete frozen profile in run metadata")
    harness = metadata.get("harness")
    if harness not in {"humanclaw_psv", "bare_api"}:
        raise ValueError(f"Unsupported saved harness: {harness}")
    episode = _frozen_episode(metadata)
    seed = metadata.get("effective_seed")
    if "effective_seed" not in metadata and metadata.get("seed") is not None:
        seed = int(metadata["seed"]) + int(metadata.get("rollout_index", 0))
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("Saved effective seed must be a nonnegative integer or null")
    if not model_config or not output_dir:
        raise ValueError("Rerun requires explicit model_config and output_dir paths")
    model_path = Path(model_config).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Explicit model config not found: {model_path}")
    _object(model_path)  # Validate structure without copying or printing credentials.

    # The execution manifest captures optional agent asset overrides and any
    # effective physics/rendering settings in addition to the named profile.
    replay_manifest = source / "replay" / "replay_manifest.json"
    if replay_manifest.is_file():
        saved = _object(replay_manifest)
        for section in ("physics", "rendering"):
            profile[section] = {**profile.get(section, {}), **saved.get(section, {})}
        for key in ("agent_urdf", "agent_shift_npy", "physics_config"):
            identity = saved.get("assets", {}).get(key)
            value = identity.get("path") if isinstance(identity, dict) else identity
            if value:
                profile["physics"][key] = value
    from humanclaw_bench.config import load_config

    output = _fresh_output(source, output_dir, "rerun")
    profile_path = output / "frozen_profile.json"
    _write_report(output, profile_path.name, profile)
    try:
        frozen_profile = load_config(profile_path)
        config = {
            "profile": frozen_profile, "episode_override": episode,
            "model_config_path": str(model_path), "output_root": str(output),
            "scene_id": episode.scene_id, "scene_dataset_config": episode.scene_dataset_config,
            "episode_id": episode.episode_id, "object_category": episode.object_category,
            "max_steps": episode.max_steps, "n_rollouts": 1,
            "harness": harness, "seed": seed,
            "compute_metrics": bool(metadata.get("compute_metrics", False)),
            "save_video": bool(metadata.get("save_video", False)), "trace": True,
            "parent_run_id": manifest.get("run_id"), "source_run": str(source),
        }
        selected_device = device if device is not None else saved_config.get("device")
        if selected_device is not None:
            config["device"] = selected_device
        if not replay_manifest.is_file() and saved_config.get("agent_asset"):
            config["agent_asset"] = saved_config["agent_asset"]
        if isinstance(manifest.get("pricing"), dict):
            pricing_path = output / "frozen_pricing.json"
            _write_report(output, pricing_path.name, manifest["pricing"])
            config["pricing"] = str(pricing_path)
        result = {
            "mode": "rerun", "source_run": str(source),
            "parent_run_id": manifest.get("run_id"), "output_dir": str(output),
            "harness": harness, "effective_seed": seed,
            "seed_recorded": seed is not None,
            "model_config_path": str(model_path),
            "reproducibility": "frozen episode and profile with explicit model config; model/GPU determinism not guaranteed",
        }
        _write_report(output, "rerun_request.json", result)
        outcome = _execute_rerun(config)
        result.update(status="completed", outcome=outcome)
        _write_report(output, "rerun_report.json", result)
        return result
    except BaseException as exc:
        _write_report(output, "rerun_report.json", {
            "mode": "rerun", "source_run": str(source), "output_dir": str(output),
            "status": "failed" if isinstance(exc, Exception) else "interrupted",
            "error_type": type(exc).__name__,
        })
        raise


__all__ = ["replay_run", "rerun_run"]

#!/usr/bin/env python3
"""Validate one real traced episode, its videos and optional full GPU replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write(path: Path, value: dict[str, Any]) -> None:
    """Checkpoint a readable validation report without non-finite JSON values."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _decode_video(path: Path) -> dict[str, Any]:
    """Decode the entire video to verify its frame count and dimensions."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        frames = sum(1 for _frame in reader)
    finally:
        reader.close()
    if frames == 0:
        raise ValueError("Video contains no decodable frames")
    return {"path": str(path), "frames": frames, "size": metadata["size"],
            "fps": metadata["fps"], "size_bytes": path.stat().st_size}


def validate_episode(output_root: Path, validation_dir: Path, *, replay: bool) -> dict[str, Any]:
    """Audit the single attempt and preserve independent replay failure reports."""
    from humanclaw_bench.experiments.replay import replay_run
    from humanclaw_bench.experiments.report import summarize_run, validate_trace

    root = output_root.resolve()
    runs = sorted(root.glob("*/rollout_00/experiment_runs/*/run.json"))
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one experiment run, found {len(runs)}")
    run = runs[0].parent
    validation_dir = validation_dir.resolve()
    if validation_dir == run or validation_dir.is_relative_to(run):
        raise ValueError("Validation outputs must be outside the original trace")
    validation_dir.mkdir(parents=True, exist_ok=False)
    report_path = validation_dir / "validation.json"
    result: dict[str, Any] = {
        "schema": "humanclaw_episode_validation_v1", "status": "running",
        "run_dir": str(run), "model_api_calls_during_validation": 0,
        "requested_replay": replay, "checks": {}, "errors": [],
    }

    def record_error(stage: str, error_type: str) -> None:
        """Keep failure types without copying arbitrary provider error messages."""
        result["errors"].append({"stage": stage, "type": error_type})

    _write(report_path, result)
    trace = validate_trace(run)
    result["checks"]["trace"] = trace
    if not trace["ok"] or trace["warnings"]:
        record_error("trace", "InvalidOrIncompleteTrace")
    summary = summarize_run(run, write=True)
    result["episode_status"] = summary["status"]
    result["outcome"] = summary["outcome"]
    result["model_usage"] = summary["model_usage"]
    result["cost"] = summary["cost"]
    result["latency"] = summary["latency"]
    result["capabilities"] = summary["capabilities"]
    if summary["status"] != "completed":
        record_error("episode", "EpisodeDidNotComplete")
    usage = summary["model_usage"]
    if not usage["trace_accounting_complete"] or usage["status_counts"].get("completed", 0) < 1:
        record_error("model_usage", "NoCompleteRealModelCallTrace")
    capabilities = summary["capabilities"]["entries"]
    if not any(row["kind"] == "tool" and row["executed"] for row in capabilities):
        record_error("capabilities", "NoExecutedToolRecorded")
    if not any(row["name"] == "motion.generate" and row["completed"] for row in capabilities):
        record_error("capabilities", "NoCompletedMotionGeneration")
    rollout = run.parent.parent
    required = [rollout / "metrics.json", run / "replay/replay_manifest.json",
                run / "replay/trajectory_before.npz", run / "replay/trajectory_after.npz"]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    result["checks"]["required_artifacts"] = {"ok": not missing, "missing": missing}
    if missing:
        record_error("artifacts", "MissingRequiredArtifact")
    result["checks"]["rollout_videos"] = {}
    for view in ("ego", "exo"):
        try:
            result["checks"]["rollout_videos"][view] = _decode_video(rollout / f"{view}.mp4")
        except Exception as exc:
            record_error(f"rollout_video_{view}", type(exc).__name__)
    _write(report_path, result)

    for mode in (("log", "physics", "render") if replay else ("log",)):
        try:
            replay_report = replay_run(run, mode=mode, output_dir=validation_dir / mode)
            if mode == "log":
                result["checks"][mode] = {"validation": replay_report["validation"],
                                          "output_dir": replay_report["output_dir"]}
                if not replay_report["validation"]["ok"]:
                    record_error(mode, "LogReplayValidationFailed")
            else:
                result["checks"][mode] = replay_report
                if mode == "physics" and replay_report.get("passed") is not True:
                    record_error(mode, "PhysicsReplayDiverged")
                if mode == "render":
                    videos = {view: _decode_video(validation_dir / mode / f"{view}.mp4")
                              for view in ("ego", "exo")}
                    result["checks"]["render_videos"] = videos
                    expected = replay_report["frames"]
                    if any(video["frames"] != expected for video in videos.values()):
                        record_error(mode, "RenderedVideoFrameCountMismatch")
                    original = result["checks"]["rollout_videos"]
                    if any(video["frames"] != expected + 1 for video in original.values()):
                        record_error("rollout_video", "RolloutVideoFrameCountMismatch")
        except Exception as exc:
            record_error(mode, type(exc).__name__)
        _write(report_path, result)
    result["status"] = "passed" if not result["errors"] else "failed"
    _write(report_path, result)
    return result


def main() -> int:
    """Run validation once in a new output directory without invoking a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--replay", action="store_true", help="Also run full physics and render replay on a GPU")
    args = parser.parse_args()
    destination = args.validation_output or args.output_root / "validation"
    result = validate_episode(args.output_root, destination, replay=args.replay)
    print(json.dumps({"status": result["status"], "report": str(destination.resolve() / "validation.json"),
                      "errors": result["errors"], "task_success": result["outcome"].get("task_success"),
                      "model_calls": result["model_usage"]["leaf_model_calls"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

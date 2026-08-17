"""Compose saved ego/exo streams and exact per-step VLM text into one MP4.

This is a presentation-only pass over completed rollout artifacts.  It never
loads Habitat, advances physics, generates motion, or contacts a VLM.  Timing
comes from ``trajectory_before.npz`` so every text panel covers exactly the
motion frames produced by that decision.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageFont


PERCEPT_RE = re.compile(r"step(\d{3})_percept_mid_low\.json$")
VERIFIER_RE = re.compile(r"step(\d{3})_verifier\.json$")
ACTION_PREFIX_RE = re.compile(r"^action id \d+:\s*", re.IGNORECASE)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_SIZE = 17
LINE_HEIGHT = 23
VIEW_SIZE = 448
OUTPUT_WIDTH = VIEW_SIZE * 2

WHITE = (238, 241, 246)
MUTED = (178, 184, 195)
CYAN = (125, 220, 245)
GREEN = (144, 231, 165)
PURPLE = (208, 177, 255)
YELLOW = (255, 220, 110)
ORANGE = (255, 165, 100)
RED = (255, 115, 105)


@dataclass(frozen=True)
class CompositeStep:
    """Frame interval and saved model records for one executed motion decision."""

    step: int
    index: int
    start_frame: int
    frame_count: int
    trajectory_action: str
    percept_json: Path
    verifier_json: Path | None


@dataclass(frozen=True)
class CompositeSource:
    """Validated synchronized inputs needed for one composite video."""

    rollout_dir: Path
    ego_video: Path
    exo_video: Path
    fps: float
    video_frames: int
    trajectory_frames: int
    initial_frames: int
    steps: tuple[CompositeStep, ...]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object with an actionable type error."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _executable(name: str) -> str:
    """Resolve a required system video executable."""

    value = shutil.which(name)
    if value:
        return value
    raise RuntimeError(
        f"Composite video generation requires system `{name}`. Install an "
        "ffmpeg build that includes ffprobe and the libass subtitles filter."
    )


def _probe_video(path: Path) -> dict[str, Any]:
    """Return exact timing and geometry for one encoded video stream."""

    result = subprocess.run(
        [
            _executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed: {path}")
    streams = json.loads(result.stdout).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["r_frame_rate"])),
        "frames": int(stream["nb_frames"]),
    }


def _decode_strings(values: Any) -> list[str]:
    """Decode a NumPy string array without enabling pickle loading."""

    decoded: list[str] = []
    for item in np.asarray(values).tolist():
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(item))
    return decoded


def _indexed_paths(rollout: Path, pattern: str, regex: re.Pattern[str]) -> dict[int, Path]:
    """Index step artifact paths by the zero-padded step number in each name."""

    indexed: dict[int, Path] = {}
    for path in sorted(rollout.glob(pattern)):
        match = regex.search(path.name)
        if match:
            indexed[int(match.group(1))] = path
    return indexed


def load_composite_source(rollout_dir: str | Path) -> CompositeSource:
    """Validate one rollout and recover exact decision-to-frame alignment."""

    rollout = Path(rollout_dir).expanduser().resolve()
    if not rollout.is_dir():
        raise NotADirectoryError(f"Rollout directory not found: {rollout}")
    ego = rollout / "ego.mp4"
    exo = rollout / "exo.mp4"
    trajectory = rollout / "trajectory_before.npz"
    for path in (ego, exo, trajectory):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required composite-video input is missing: {path}")

    with np.load(trajectory, allow_pickle=False) as before:
        trajectory_frames = int(np.asarray(before["xb_world_75"]).shape[0])
        fps = float(np.asarray(before["fps"]).item())
        step_indices = np.asarray(before["step_indices"], dtype=np.int64)
        step_starts = np.asarray(before["step_starts"], dtype=np.int64)
        step_lengths = np.asarray(before["step_lengths"], dtype=np.int64)
        step_actions = _decode_strings(before["step_action_text"])

    count = len(step_indices)
    if not (len(step_starts) == len(step_lengths) == len(step_actions) == count):
        raise ValueError("Trajectory step arrays have inconsistent lengths")
    if int(step_lengths.sum()) != trajectory_frames:
        raise ValueError(
            "Trajectory step lengths do not cover every generated motion frame: "
            f"{int(step_lengths.sum())} != {trajectory_frames}"
        )

    percept_by_step = _indexed_paths(
        rollout, "step???_percept_mid_low.json", PERCEPT_RE
    )
    verifier_by_step = _indexed_paths(rollout, "step???_verifier.json", VERIFIER_RE)
    steps: list[CompositeStep] = []
    for index, (step, start, length, action) in enumerate(
        zip(step_indices, step_starts, step_lengths, step_actions)
    ):
        step_number = int(step)
        percept = percept_by_step.get(step_number)
        if percept is None:
            raise FileNotFoundError(
                f"Motion step {step_number} has no saved percept record in {rollout}"
            )
        steps.append(
            CompositeStep(
                step=step_number,
                index=index,
                start_frame=int(start),
                frame_count=int(length),
                trajectory_action=str(action),
                percept_json=percept,
                verifier_json=verifier_by_step.get(step_number),
            )
        )

    ego_info = _probe_video(ego)
    exo_info = _probe_video(exo)
    if ego_info["frames"] != exo_info["frames"]:
        raise ValueError(
            "Ego/exo videos are not frame synchronized: "
            f"{ego_info['frames']} != {exo_info['frames']}"
        )
    if abs(float(ego_info["fps"]) - fps) > 1e-4 or abs(
        float(exo_info["fps"]) - fps
    ) > 1e-4:
        raise ValueError(
            f"Video/trajectory fps disagree: ego={ego_info['fps']}, "
            f"exo={exo_info['fps']}, trajectory={fps}"
        )
    video_frames = int(ego_info["frames"])
    initial_frames = video_frames - trajectory_frames
    # Online --video output has one post-reset frame.  Delayed rendering from
    # trajectory_after contains motion frames only.  Both are public inputs.
    if initial_frames not in (0, 1):
        raise ValueError(
            "Video frame count must equal trajectory frames or trajectory frames + 1: "
            f"video={video_frames}, trajectory={trajectory_frames}"
        )
    return CompositeSource(
        rollout_dir=rollout,
        ego_video=ego,
        exo_video=exo,
        fps=fps,
        video_frames=video_frames,
        trajectory_frames=trajectory_frames,
        initial_frames=initial_frames,
        steps=tuple(steps),
    )


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the release font used to compute deterministic line wrapping."""

    try:
        return ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def _text_width(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> float:
    """Measure one candidate line with the font used by the subtitle renderer."""

    left, _top, right, _bottom = font.getbbox(text)
    return float(right - left)


def _wrap(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width_px: int = OUTPUT_WIDTH - 28,
) -> list[str]:
    """Wrap normalized model text to the fixed-width reasoning panel."""

    clean = " ".join(str(text or "").split())
    if not clean:
        return ["—"]
    lines: list[str] = []
    current = ""
    for word in clean.split(" "):
        candidate = word if not current else f"{current} {word}"
        if current and _text_width(candidate, font) > width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _normalize_action(value: Any) -> str:
    """Remove an optional action-id prefix before comparing display strings."""

    return ACTION_PREFIX_RE.sub("", str(value or "").strip())


def _decision(step: CompositeStep) -> dict[str, Any]:
    """Join planner, optional verifier, and executed action for one panel."""

    percept = _read_json(step.percept_json)
    planner = percept.get("response") or {}
    if not isinstance(planner, dict):
        raise TypeError(f"Planner response must be a JSON object: {step.percept_json}")
    proposed = str(planner.get("action_name") or "")
    verdict = "not_called"
    verifier_reason = "Verifier was not called for this action."
    final_action = proposed
    if step.verifier_json is not None:
        verifier = _read_json(step.verifier_json)
        response = verifier.get("response") or {}
        if not isinstance(response, dict):
            raise TypeError(
                f"Verifier response must be a JSON object: {step.verifier_json}"
            )
        verdict = str(response.get("verdict") or "unavailable")
        verifier_reason = str(response.get("reason") or "")
        final_action = str(response.get("final_action_name") or proposed)
    return {
        "visible": str(planner.get("visible_state") or ""),
        "progress": str(planner.get("mid_level_progress_analysis") or ""),
        "goal": str(planner.get("mid_level_goal") or ""),
        "low_level": str(planner.get("low_level_action_reasoning") or ""),
        "proposed": proposed,
        "verdict": verdict,
        "verifier_reason": verifier_reason,
        "final": final_action,
        "actual": step.trajectory_action,
        "verifier_trajectory_text_differs": (
            _normalize_action(final_action) != _normalize_action(step.trajectory_action)
        ),
    }


def _append_wrapped(
    lines: list[tuple[str, tuple[int, int, int], bool]],
    label: str,
    value: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Append one labeled, wrapped decision field to an ASS panel."""

    lines.extend((line, color, False) for line in _wrap(f"{label}: {value}", font))


def _panel_lines(
    step: CompositeStep,
    *,
    motion_count: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> list[tuple[str, tuple[int, int, int], bool]]:
    """Build the colored percept/reasoning/action lines for one motion chunk."""

    decision = _decision(step)
    lines: list[tuple[str, tuple[int, int, int], bool]] = [
        (
            f"STEP {step.step}  |  MOTION CHUNK {step.index + 1}/{motion_count}",
            YELLOW,
            True,
        )
    ]
    _append_wrapped(lines, "PERCEPT / visible_state", decision["visible"], CYAN, font)
    _append_wrapped(
        lines, "THINK / mid-level progress", decision["progress"], WHITE, font
    )
    _append_wrapped(lines, "THINK / mid-level goal", decision["goal"], GREEN, font)
    _append_wrapped(
        lines, "THINK / low-level reasoning", decision["low_level"], PURPLE, font
    )
    changed = str(decision["verdict"]).lower() == "replace"
    verifier = (
        f"{str(decision['verdict']).upper()} | proposed {decision['proposed']} -> "
        f"final {decision['final']} | {decision['verifier_reason']}"
    )
    _append_wrapped(lines, "VERIFIER", verifier, RED if changed else MUTED, font)
    _append_wrapped(
        lines,
        "FINAL ACTION EXECUTED",
        decision["actual"],
        ORANGE if changed or decision["verifier_trajectory_text_differs"] else YELLOW,
        font,
    )
    return lines


def _ass_color(rgb: tuple[int, int, int]) -> str:
    """Convert RGB to ASS subtitle BGR notation."""

    red, green, blue = rgb
    return f"&H00{blue:02X}{green:02X}{red:02X}&"


def _ass_escape(text: str) -> str:
    """Escape free-form provider text before placing it in an ASS event."""

    return (
        str(text)
        .replace("\\", "／")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _ass_time(seconds: float) -> str:
    """Format seconds using ASS centisecond timestamps."""

    centiseconds = max(0, int(round(float(seconds) * 100.0)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cents = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cents:02d}"


def _event_text(lines: list[tuple[str, tuple[int, int, int], bool]]) -> str:
    """Encode colored panel lines as one ASS dialogue payload."""

    rendered: list[str] = []
    for value, color, bold in lines:
        rendered.append(
            "{" + f"\\c{_ass_color(color)}\\b{1 if bold else 0}" + "}"
            + _ass_escape(value)
        )
    return r"\N".join(rendered)


def _write_ass(
    path: Path,
    *,
    source: CompositeSource,
    output_height: int,
    panels: dict[int, list[tuple[str, tuple[int, int, int], bool]]],
) -> None:
    """Write frame-aligned ASS events for all executed motion decisions."""

    duration = source.video_frames / source.fps
    panel_y = VIEW_SIZE + 12
    header_end = _ass_time(duration + 1.0 / source.fps)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {OUTPUT_WIDTH}",
        f"PlayResY: {output_height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Panel,DejaVu Sans,17,&H00EEF1F6,&H00FFFFFF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        "Style: Header,DejaVu Sans,15,&H00EEF1F6,&H00FFFFFF,&H00000000,"
        "&HAA000000,0,0,0,0,100,100,0,0,3,1,0,7,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
        (
            f"Dialogue: 2,0:00:00.00,{header_end},Header,,0,0,0,,"
            r"{\an7\pos(8,5)}EGO (model input)"
        ),
        (
            f"Dialogue: 2,0:00:00.00,{header_end},Header,,0,0,0,,"
            rf"{{\an7\pos({VIEW_SIZE + 8},5)}}EXO (observer view)"
        ),
    ]
    if source.initial_frames:
        lines.append(
            (
                f"Dialogue: 1,0:00:00.00,{_ass_time(1.0 / source.fps)},"
                "Panel,,0,0,0,,"
                rf"{{\an7\pos(14,{panel_y})\c{_ass_color(MUTED)}}}"
                "POST-RESET FRAME | no motion action has executed yet"
            )
        )
    for step in source.steps:
        start = (source.initial_frames + step.start_frame) / source.fps
        end = (
            source.initial_frames + step.start_frame + step.frame_count
        ) / source.fps
        lines.append(
            f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Panel,,0,0,0,,"
            rf"{{\an7\pos(14,{panel_y})}}{_event_text(panels[step.step])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_filter_path(path: Path) -> str:
    """Escape an absolute filename for ffmpeg's subtitles filter syntax."""

    return (
        str(path.resolve())
        .replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )


def compose_ego_exo_reasoning(
    rollout_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    preset: str = "veryfast",
    crf: int = 20,
    threads: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """Create one synchronized ego/exo/reasoning MP4 from a saved rollout."""

    if threads < 1:
        raise ValueError("ffmpeg thread count must be positive")
    if not 0 <= int(crf) <= 51:
        raise ValueError("H.264 CRF must be in [0, 51]")
    source = load_composite_source(rollout_dir)
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.rollout_dir / "full_ego_exo_reasoning.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and not force:
        existing = _probe_video(output)
        if int(existing["frames"]) == source.video_frames:
            return {
                "status": "already_complete",
                "rollout_dir": str(source.rollout_dir),
                "output": str(output),
                "frames": source.video_frames,
                "fps": source.fps,
            }

    font = _font()
    panels = {
        step.step: _panel_lines(step, motion_count=len(source.steps), font=font)
        for step in source.steps
    }
    max_lines = max((len(value) for value in panels.values()), default=1)
    panel_height = 12 + max_lines * LINE_HEIGHT + 12
    if panel_height % 2:
        panel_height += 1
    output_height = VIEW_SIZE + panel_height
    temporary_video = output.with_name(f".{output.stem}.tmp{output.suffix}")
    temporary_video.unlink(missing_ok=True)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="humanclaw-video-") as temporary_dir:
        ass_path = Path(temporary_dir) / "decision_overlay.ass"
        _write_ass(
            ass_path,
            source=source,
            output_height=output_height,
            panels=panels,
        )
        filter_graph = (
            f"[0:v]setpts=PTS-STARTPTS,scale={VIEW_SIZE}:{VIEW_SIZE}:"
            "flags=lanczos[ego];"
            f"[1:v]setpts=PTS-STARTPTS,scale={VIEW_SIZE}:{VIEW_SIZE}:"
            "flags=lanczos[exo];"
            "[ego][exo]hstack=inputs=2[views];"
            f"[views]pad={OUTPUT_WIDTH}:{output_height}:0:0:black[base];"
            f"[base]subtitles=filename='{_escape_filter_path(ass_path)}'[out]"
        )
        command = [
            _executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source.ego_video),
            "-i",
            str(source.exo_video),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-frames:v",
            str(source.video_frames),
            "-r",
            f"{source.fps:g}",
            "-c:v",
            "libx264",
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-threads",
            str(int(threads)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_video),
        ]
        try:
            subprocess.run(command, check=True)
            rendered = _probe_video(temporary_video)
            if int(rendered["frames"]) != source.video_frames:
                raise RuntimeError(
                    "Composite frame validation failed: "
                    f"{rendered['frames']} != {source.video_frames}"
                )
            temporary_video.replace(output)
        except BaseException:
            temporary_video.unlink(missing_ok=True)
            raise

    return {
        "status": "complete",
        "rollout_dir": str(source.rollout_dir),
        "output": str(output),
        "frames": source.video_frames,
        "fps": source.fps,
        "duration_s": source.video_frames / source.fps,
        "resolution": [OUTPUT_WIDTH, output_height],
        "motion_steps": len(source.steps),
        "elapsed_s": time.perf_counter() - started,
    }


def discover_composite_rollouts(input_root: str | Path) -> list[Path]:
    """Recursively find completed rollout directories with both source videos."""

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input root not found: {root}")
    return [
        path.parent
        for path in sorted(root.rglob("trajectory_before.npz"))
        if (path.parent / "ego.mp4").is_file()
        and (path.parent / "exo.mp4").is_file()
    ]


def _compose_job(
    rollout: Path,
    output: Path,
    *,
    preset: str,
    crf: int,
    threads: int,
    force: bool,
) -> dict[str, Any]:
    """Run one batch item and convert exceptions to a compact failure record."""

    try:
        return compose_ego_exo_reasoning(
            rollout,
            output,
            preset=preset,
            crf=crf,
            threads=threads,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001 - keep independent jobs running
        return {
            "status": "failed",
            "rollout_dir": str(rollout),
            "output": str(output),
            "error": f"{type(exc).__name__}: {exc}",
        }


def compose_ego_exo_reasoning_batch(
    input_root: str | Path,
    output_root: str | Path,
    *,
    max_parallel: int = 4,
    preset: str = "veryfast",
    crf: int = 20,
    threads: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """Compose a complete output tree in parallel while preserving its layout."""

    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    source_root = Path(input_root).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    rollouts = discover_composite_rollouts(source_root)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {}
        for rollout in rollouts:
            relative = rollout.relative_to(source_root)
            output = destination_root / relative / "full_ego_exo_reasoning.mp4"
            future = executor.submit(
                _compose_job,
                rollout,
                output,
                preset=preset,
                crf=crf,
                threads=threads,
                force=force,
            )
            future_map[future] = relative.as_posix()
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"[{len(results)}/{len(rollouts)}] {future_map[future]} "
                f"{result['status']}",
                flush=True,
            )

    failed = [row for row in results if row["status"] == "failed"]
    return {
        "selected": len(rollouts),
        "completed": sum(
            row["status"] in {"complete", "already_complete"} for row in results
        ),
        "failed": len(failed),
        "failed_jobs": [
            {
                "rollout_dir": row["rollout_dir"],
                "error": row["error"],
            }
            for row in failed
        ],
        "elapsed_s": time.perf_counter() - started,
    }


__all__ = [
    "CompositeSource",
    "CompositeStep",
    "compose_ego_exo_reasoning",
    "compose_ego_exo_reasoning_batch",
    "discover_composite_rollouts",
    "load_composite_source",
]

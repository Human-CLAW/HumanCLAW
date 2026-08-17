"""Streaming ego/exo MP4 writer used only by ``--video``.

Frames are piped directly to ffmpeg.  There is no PNG frame directory and no
second encoding pass, so enabling video adds only the two final MP4 files plus
the encoder's bounded pipe buffer.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _ffmpeg_executable() -> str:
    """Use an installed ffmpeg, or the binary supplied by imageio-ffmpeg."""

    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001 - produce one actionable error
        raise RuntimeError(
            "--video requires ffmpeg. Install ffmpeg or "
            "`python -m pip install imageio-ffmpeg`."
        ) from exc


class _MP4Pipe:
    """One lazily opened raw-RGB pipe into an H.264 MP4 file."""

    def __init__(
        self,
        path: Path,
        fps: float,
        *,
        preset: str = "medium",
        crf: int = 18,
    ) -> None:
        """Start one ffmpeg stdin pipeline for fixed-size RGB video frames."""

        self.path = Path(path)
        self.fps = float(fps)
        self.preset = str(preset)
        self.crf = int(crf)
        if not 0 <= self.crf <= 51:
            raise ValueError(f"H.264 CRF must be in [0, 51], got {self.crf}")
        self._process: subprocess.Popen[bytes] | None = None
        self._shape: tuple[int, int] | None = None
        self.frame_count = 0

    def append(self, frame: Any) -> None:
        """Validate and stream one RGB frame to the ffmpeg encoder."""

        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Video frame must have shape (H, W, 3+), got {rgb.shape}")
        rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if width % 2 or height % 2:
            raise ValueError("H.264 video dimensions must both be even")
        if self._shape is None:
            self._shape = (height, width)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                _ffmpeg_executable(),
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                f"{self.fps:g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-crf",
                str(self.crf),
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ]
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
        elif self._shape != (height, width):
            raise ValueError(
                f"Video resolution changed from {self._shape} to {(height, width)}"
            )

        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(rgb.tobytes())
        self.frame_count += 1

    def close(self) -> None:
        """Release open files, processes, clients, and runtime resources."""

        if self._process is None:
            return
        assert self._process.stdin is not None
        self._process.stdin.close()
        return_code = self._process.wait()
        self._process = None
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} (exit {return_code})")


class RolloutVideoWriter:
    """Write synchronized first-person and third-person videos."""

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        *,
        preset: str = "medium",
        crf: int = 18,
    ) -> None:
        # Online rollout keeps the historical medium/18 defaults.  Delayed
        # saved-trajectory rendering can explicitly choose a faster encoder
        # preset without changing the benchmark rollout contract.
        """Create lazy ego/exo encoders and reasoning-overlay state for one rollout."""

        self.ego = _MP4Pipe(
            Path(output_dir) / "ego.mp4",
            fps,
            preset=preset,
            crf=crf,
        )
        self.exo = _MP4Pipe(
            Path(output_dir) / "exo.mp4",
            fps,
            preset=preset,
            crf=crf,
        )

    def append(self, ego_frame: Any, exo_frame: Any) -> None:
        """Compose and stream synchronized ego/exo frames with current reasoning text."""

        self.ego.append(ego_frame)
        self.exo.append(exo_frame)

    def close(self) -> None:
        """Release open files, processes, clients, and runtime resources."""

        errors: list[Exception] = []
        for stream in (self.ego, self.exo):
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001 - close both encoder pipes
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> "RolloutVideoWriter":
        """Enter the managed runtime context and return this object."""

        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        """Release resources when leaving the managed runtime context."""

        self.close()


__all__ = ["RolloutVideoWriter"]

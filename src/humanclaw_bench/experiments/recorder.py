"""Append-only, opt-in traces for experiment adapters and their real calls.

The active recorder and nesting are context-local. Async tasks inherit them;
thread-pool callers should use a separate ``contextvars.copy_context()`` for
each submitted job. Writes to a shared recorder are serialized. Every event
is flushed immediately, so an abruptly killed process retains its unfinished
starts rather than being presented as a completed run.
"""

from __future__ import annotations

import base64
import contextvars
import dataclasses
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import traceback
from typing import Any
from urllib.parse import unquote_to_bytes
import uuid


TRACE_SCHEMA = "humanclaw_trace_v1"
RUN_SCHEMA = "humanclaw_experiment_run_v1"
_ACTIVE: contextvars.ContextVar[TraceRecorder | None] = contextvars.ContextVar(
    "humanclaw_experiment_recorder", default=None
)
_SPAN: contextvars.ContextVar[_Span | None] = contextvars.ContextVar(
    "humanclaw_experiment_span", default=None
)
_SECRET_KEYS = {
    "authorization", "proxyauthorization", "password", "passwd", "token",
    "accesstoken", "refreshtoken", "authtoken", "bearertoken", "cookie",
    "setcookie", "credential", "credentials", "privatekey", "secret",
}
_EXTENSIONS = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
    "image/webp": ".webp", "application/pdf": ".pdf", "audio/wav": ".wav",
    "audio/mpeg": ".mp3", "video/mp4": ".mp4", "text/plain": ".txt",
    "application/json": ".json", "application/x-npy": ".npy",
}


def _utc() -> str:
    """Return an explicit UTC timestamp for a persisted event."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _id(prefix: str) -> str:
    """Create a globally unique identifier with a readable category prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _credential_key(key: str) -> bool:
    """Recognize credential keys without matching token accounting fields."""
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return (
        normalized in _SECRET_KEYS
        or normalized.endswith((
            "apikey", "clientsecret", "accesskey", "secretkey",
            "apitoken", "accesstoken", "refreshtoken", "authtoken",
        ))
        or normalized.endswith("password")
    )


def _error(error: Any) -> Any:
    """Preserve exception type, message, and traceback as structured data."""
    if isinstance(error, BaseException):
        return {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
    return error


def _exception_status(error: BaseException) -> str:
    """Distinguish ordinary failures from process interruption exceptions."""
    return "failed" if isinstance(error, Exception) else "interrupted"


def _binary_type(value: bytes) -> str:
    """Identify common media signatures without requiring decoding libraries."""
    if value.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if value.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if value.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if value.startswith(b"%PDF-"):
        return "application/pdf"
    if value.startswith(b"RIFF") and value[8:12] == b"WAVE":
        return "audio/wav"
    if value.startswith(b"RIFF") and value[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


class TraceRecorder:
    """Own one fresh run directory and a flushed JSONL event stream.

    Creating a recorder starts its run record. Entering its context enables the
    global helpers; exit finalizes the run and restores the enclosing context.
    No earlier directory is reused, including when metadata names the same
    benchmark episode. ``pricing`` is a recorded price specification, never an
    inferred or fetched price.
    """

    def __init__(
        self, output_dir: str | Path, metadata: dict[str, Any], pricing: Any = None
    ) -> None:
        """Initialize this recorder or operation without replacing prior traces."""
        if not isinstance(metadata, dict):
            raise TypeError("Experiment metadata must be a dictionary")
        self._lock = threading.RLock()
        self._sequence = 0
        self._finished = False
        self._context_token: contextvars.Token | None = None
        self._open_spans: dict[str, _Span] = {}
        self._started_ns = time.monotonic_ns()
        self.run_id = _id("run")
        self.session_id = _id("session")
        self.episode_id = _id("episode")
        self.directory = Path(output_dir) / "experiment_runs" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._events = (self.directory / "events.jsonl").open(
            "x", encoding="utf-8"
        )
        self.manifest: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "status": "running",
            "started_at_utc": _utc(),
            "finished_at_utc": None,
            "duration_ms": None,
            "trace_file": "events.jsonl",
            "metadata": self.artifact(metadata),
            "pricing": self.artifact(pricing),
            "outcome": {},
            "error": None,
        }
        try:
            self.write_json("run.json", self.manifest)
            self._emit(
                "run.start", kind="run", name="run",
                data={"metadata": self.manifest["metadata"], "pricing": self.manifest["pricing"]},
            )
        except BaseException:
            self._events.close()
            raise

    def __enter__(self) -> TraceRecorder:
        """Activate this context while preserving its enclosing context."""
        with self._lock:
            if self._finished:
                raise RuntimeError("Cannot activate a finished trace")
            if self._context_token is not None:
                raise RuntimeError("Recorder context is already active; copy its context for workers")
            self._context_token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        """Restore the previous context and propagate any raised exception."""
        try:
            if not self._finished:
                self.finish(
                    status=_exception_status(exc) if exc is not None else "completed",
                    error=exc,
                )
        finally:
            if self._context_token is not None:
                _ACTIVE.reset(self._context_token)
                self._context_token = None
        return False

    def _store(self, value: bytes, media_type: str) -> dict[str, Any]:
        """Persist one content-addressed binary artifact and return its reference."""
        digest = hashlib.sha256(value).hexdigest()
        suffix = _EXTENSIONS.get(media_type.split(";", 1)[0].lower(), ".bin")
        relative = f"artifacts/{digest}{suffix}"
        destination = self.directory / relative
        with self._lock:
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                self._atomic_bytes(destination, value)
        return {
            "artifact": relative, "sha256": digest,
            "media_type": media_type, "size_bytes": len(value),
        }

    @staticmethod
    def _atomic_bytes(destination: Path, value: bytes) -> None:
        """Flush a temporary file before atomically replacing its destination."""
        fd, temporary = tempfile.mkstemp(prefix=".trace-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def artifact(self, value: Any) -> Any:
        """Make a JSON value with content-addressed binary/image/array references.

        Credential-shaped dictionary keys are redacted, including nested
        provider configuration. Plain strings (including prompt text) are not
        searched or truncated. NumPy/Pillow are imported only for their values.
        Unsupported objects have an explicit type/repr record, not a fabricated
        JSON representation.
        """
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"non_finite": str(value)}
        if isinstance(value, str):
            if value.startswith("data:") and "," in value:
                header, payload = value[5:].split(",", 1)
                parts = header.split(";")
                media_type = parts[0] or "text/plain"
                parameters = [part for part in parts[1:] if part.lower() != "base64"]
                if parameters:
                    media_type += ";" + ";".join(parameters)
                try:
                    data = (
                        base64.b64decode(payload, validate=True)
                        if any(part.lower() == "base64" for part in parts[1:])
                        else unquote_to_bytes(payload)
                    )
                except (ValueError, base64.binascii.Error):
                    # Preserve malformed transport input without placing its
                    # potentially large base64 payload back in the event.
                    return self._store(value.encode("utf-8"), "text/plain")
                return self._store(data, media_type)
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            return self._store(data, _binary_type(data))
        if isinstance(value, dict):
            return {
                str(key): "[REDACTED]" if _credential_key(str(key)) else self.artifact(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.artifact(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return {"set_items": [self.artifact(item) for item in sorted(value, key=repr)]}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, BaseException):
            return self.artifact(_error(value))
        module = type(value).__module__.split(".", 1)[0]
        if module == "numpy":
            import numpy as np

            if isinstance(value, np.ndarray):
                output = io.BytesIO()
                np.save(output, value, allow_pickle=False)
                reference = self._store(output.getvalue(), "application/x-npy")
                reference.update(shape=list(value.shape), dtype=str(value.dtype))
                return reference
            if isinstance(value, np.generic):
                return self.artifact(value.item())
        if module == "PIL":
            from PIL import Image

            if isinstance(value, Image.Image):
                output = io.BytesIO()
                value.save(output, format="PNG")
                reference = self._store(output.getvalue(), "image/png")
                reference.update(mode=value.mode, size=list(value.size))
                return reference
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self.artifact(dataclasses.asdict(value))
        return {
            "unserialized_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }

    def write_json(self, name: str, data: Any) -> Path:
        """Atomically write one JSON artifact within this run directory."""
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
            raise ValueError("JSON artifact name must be a relative .json path within the run")
        destination = self.directory / relative
        with self._lock:
            if not destination.resolve().is_relative_to(self.directory.resolve()):
                raise ValueError("JSON artifact path escapes the run directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                self.artifact(data), ensure_ascii=False, indent=2, allow_nan=False
            ).encode("utf-8") + b"\n"
            self._atomic_bytes(destination, payload)
        return destination

    def _emit(
        self, event_name: str, *, kind: str, name: str, data: Any = None,
        step: int | None = None, span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Serialize and flush one event while holding the sequence lock."""
        with self._lock:
            if self._finished:
                return None
            prepared_data = self.artifact({} if data is None else data)
            sequence = self._sequence + 1
            row = {
                "schema": TRACE_SCHEMA,
                "event": event_name, "event_id": _id("event"),
                "sequence": sequence, "run_id": self.run_id,
                "session_id": self.session_id, "episode_id": self.episode_id,
                "step": step, "timestamp_utc": _utc(),
                "monotonic_ns": time.monotonic_ns(),
                "span_id": span_id, "parent_span_id": parent_span_id,
                "kind": kind, "name": name, "data": prepared_data,
            }
            self._events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self._events.flush()
            self._sequence = sequence
            return row

    def finish(
        self, status: str = "completed", outcome: dict[str, Any] | None = None,
        error: Any = None,
    ) -> None:
        """Finish once; never overwrite a prior terminal outcome.

        Finishing with open spans explicitly interrupts them and marks an
        otherwise completed run interrupted. This is distinguishable from a
        process crash, which leaves unmatched starts and a running manifest.
        """
        with self._lock:
            if self._finished:
                return
            open_ids = list(self._open_spans)
            if open_ids:
                for active in reversed(list(self._open_spans.values())):
                    active._end(
                        status="interrupted",
                        error={"type": "UnfinishedSpan", "message": "Run finished before span exited"},
                    )
                if status == "completed":
                    status = "interrupted"
                if error is None:
                    error = {"type": "UnfinishedSpans", "span_ids": open_ids}
            duration = (time.monotonic_ns() - self._started_ns) / 1_000_000
            final = {
                "status": status, "outcome": self.artifact(outcome or {}),
                "error": self.artifact(_error(error)),
                "finished_at_utc": _utc(), "duration_ms": duration,
            }
            self._emit("run.end", kind="run", name="run", data=final)
            self.manifest.update(final)
            if open_ids:
                self.manifest["unfinished_span_ids"] = open_ids
            self.write_json("run.json", self.manifest)
            self._finished = True
            self._events.close()


class _NoopSpan:
    """Provide the span interface without inspecting or retaining values."""
    span_id = None

    def __enter__(self) -> _NoopSpan:
        """Activate this context while preserving its enclosing context."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Restore the previous context and propagate any raised exception."""
        return False

    def set_output(self, value: Any) -> None:
        """Capture the operation output without retaining mutable binary inputs."""
        pass

    def set_usage(self, value: dict[str, Any]) -> None:
        """Retain provider usage fields without deriving missing quantities."""
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        """Update one structured operation attribute with credential redaction."""
        pass

    def fail(self, error: Any) -> None:
        """Record an explicit failure even when the caller handles its exception."""
        pass


class _Span:
    """Record one operation and restore its enclosing context on exit."""
    def __init__(
        self, recorder: TraceRecorder, name: str, *, kind: str,
        step: int | None, input: Any, attributes: dict[str, Any] | None,
    ) -> None:
        """Initialize this recorder or operation without replacing prior traces."""
        self.recorder = recorder
        self.name = name
        self.kind = kind
        self.step = step
        self.input = input
        self.attributes = dict(attributes or {})
        self.span_id = _id("span")
        self.parent_span_id: str | None = None
        self.output: Any = None
        self.usage: dict[str, Any] = {}
        self.status = "completed"
        self.error: Any = None
        self._started_ns: int | None = None
        self._token: contextvars.Token | None = None
        self._ended = False

    def __enter__(self) -> _Span:
        """Activate this context while preserving its enclosing context."""
        with self.recorder._lock:
            if self._started_ns is not None:
                raise RuntimeError("A trace span can only be entered once")
            if self.recorder._finished:
                raise RuntimeError("Cannot start a span on a finished trace")
            parent = _SPAN.get()
            if parent is not None and parent.recorder is self.recorder and not parent._ended:
                self.parent_span_id = parent.span_id
                if self.step is None:
                    self.step = parent.step
            self._started_ns = time.monotonic_ns()
            self.recorder._emit(
                "span.start", kind=self.kind, name=self.name,
                step=self.step, span_id=self.span_id, parent_span_id=self.parent_span_id,
                data={"input": self.input, "attributes": self.attributes},
            )
            self.recorder._open_spans[self.span_id] = self
            self._token = _SPAN.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        """Restore the previous context and propagate any raised exception."""
        try:
            if exc is not None:
                self._end(status=_exception_status(exc), error=exc)
            else:
                self._end(status=self.status, error=self.error)
        finally:
            if self._token is not None:
                _SPAN.reset(self._token)
                self._token = None
        return False

    def _end(self, *, status: str, error: Any) -> None:
        """Close a started operation once and remove it from outstanding spans."""
        with self.recorder._lock:
            if self._ended:
                return
            duration = (
                (time.monotonic_ns() - self._started_ns) / 1_000_000
                if self._started_ns is not None else None
            )
            self.recorder._emit(
                "span.end", kind=self.kind, name=self.name,
                step=self.step, span_id=self.span_id, parent_span_id=self.parent_span_id,
                data={
                    "output": self.output, "usage": self.usage,
                    "attributes": self.attributes, "status": status,
                    "error": _error(error), "duration_ms": duration,
                },
            )
            self._ended = True
            self.recorder._open_spans.pop(self.span_id, None)

    def set_output(self, value: Any) -> None:
        """Capture the operation output without retaining mutable binary inputs."""
        with self.recorder._lock:
            self.output = self.recorder.artifact(value)

    def set_usage(self, value: dict[str, Any]) -> None:
        """Retain provider usage fields without deriving missing quantities."""
        with self.recorder._lock:
            self.usage = self.recorder.artifact(value)

    def set_attribute(self, key: str, value: Any) -> None:
        """Update one structured operation attribute with credential redaction."""
        with self.recorder._lock:
            self.attributes[key] = self.recorder.artifact({key: value})[key]

    def fail(self, error: Any) -> None:
        """Record an explicit failure even when the caller handles its exception."""
        with self.recorder._lock:
            self.status = _exception_status(error) if isinstance(error, BaseException) else "failed"
            self.error = self.recorder.artifact(_error(error))


def get_recorder() -> TraceRecorder | None:
    """Return this context's unfinished recorder, or None when disabled."""
    recorder = _ACTIVE.get()
    return recorder if recorder is not None and not recorder._finished else None


def span(
    name: str, *, kind: str = "module", step: int | None = None,
    input: Any = None, attributes: dict[str, Any] | None = None,
) -> _Span | _NoopSpan:
    """Record a real operation, with a no-op context when tracing is disabled."""
    recorder = get_recorder()
    if recorder is None:
        return _NoopSpan()
    return _Span(recorder, name, kind=kind, step=step, input=input, attributes=attributes)


def event(
    event_name: str, *, kind: str = "module", name: str = "",
    data: Any = None, step: int | None = None,
) -> dict[str, Any] | None:
    """Append a real lifecycle event, inheriting the enclosing span and step."""
    recorder = get_recorder()
    if recorder is None:
        return None
    parent = _SPAN.get()
    active = parent if parent is not None and parent.recorder is recorder and not parent._ended else None
    return recorder._emit(
        event_name, kind=kind, name=name, data=data,
        step=step if step is not None else active.step if active is not None else None,
        span_id=active.span_id if active is not None else None,
        parent_span_id=active.parent_span_id if active is not None else None,
    )


__all__ = ["TRACE_SCHEMA", "RUN_SCHEMA", "TraceRecorder", "get_recorder", "span", "event"]

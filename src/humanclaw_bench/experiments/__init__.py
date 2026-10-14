"""Opt-in experiment traces; importing this package performs no recording."""

from .recorder import TraceRecorder, event, get_recorder, span

__all__ = ["TraceRecorder", "event", "get_recorder", "span"]

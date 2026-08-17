"""Paper-aligned HumanClawBench metrics, enabled explicitly at rollout time.

Exports are loaded lazily so importing the normal environment does not import
the end-of-episode collision and aggregation stack.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily expose metric implementations to avoid environment import cycles."""

    if name in {"PaperMetricRecorder", "aggregate_metric_files"}:
        from .episode import PaperMetricRecorder, aggregate_metric_files

        return {
            "PaperMetricRecorder": PaperMetricRecorder,
            "aggregate_metric_files": aggregate_metric_files,
        }[name]
    if name == "format_metric_summary":
        from .report import format_metric_summary

        return format_metric_summary
    raise AttributeError(name)


__all__ = [
    "PaperMetricRecorder",
    "aggregate_metric_files",
    "format_metric_summary",
]

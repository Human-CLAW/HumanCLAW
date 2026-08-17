"""Lazy public export for HumanClawBench rollout orchestration.

Metric helpers live below this package and are also imported by the runtime
environment.  Importing the evaluator eagerly here would therefore create the
cycle ``environment -> metric helper -> evaluation package -> evaluator ->
environment``.  A lazy package export preserves the convenient public import
without making package initialization execute the full rollout stack.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily expose evaluator classes without creating environment import cycles."""

    if name == "HCFindNavInteractEvaluator":
        from .evaluator import HCFindNavInteractEvaluator

        return HCFindNavInteractEvaluator
    raise AttributeError(name)


__all__ = ["HCFindNavInteractEvaluator"]

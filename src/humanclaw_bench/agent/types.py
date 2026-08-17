"""Planner data containers for HumanClawBench ego agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from humanclaw_bench.agent.skills import SkillCall


@dataclass
class PlannerResult:
    """Bundle planner/verifier records with the final executable action."""

    raw_plan: dict[str, Any]
    action: SkillCall
    planner_skill: dict[str, Any]
    verifier: dict[str, Any]
    stage_outputs: list[PSVStageOutput] = field(default_factory=list)


@dataclass
class PSVStageOutput:
    """One VLM call and its parsed result.

    ``usage`` stays in memory for optional cost metrics.  The user-facing step
    JSON remains the deliberately small ``prompt`` + ``response`` record.
    """

    stage: str
    raw: dict[str, Any]
    raw_output: str
    prompt: str
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


__all__ = ["PSVStageOutput", "PlannerResult"]

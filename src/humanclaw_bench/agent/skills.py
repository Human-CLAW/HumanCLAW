"""Self-contained skill-call schema for HumanClawBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillCall:
    """Represent one executable motion skill plus its numeric/string condition."""

    skill: str
    cond: Any
    source_index: int = 0
    action_id: int | None = None
    action_name: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the skill, condition, ID, and display name to JSON fields."""

        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "skill": self.skill,
            "cond": self.cond,
            "source_index": self.source_index,
        }


def skill_to_text(action: SkillCall) -> str:
    """Serialize a SkillCall into the action text used in prompts and logs."""

    if action.action_name:
        if action.action_id is None:
            return action.action_name
        return f"action id {action.action_id}: {action.action_name}"
    if action.cond is None:
        return action.skill
    return f"{action.skill} cond={action.cond}"


STAND_SKILL_CALL = SkillCall(skill="stand", cond=None, action_name="Stop/Stand")

__all__ = [
    "STAND_SKILL_CALL",
    "SkillCall",
    "skill_to_text",
]

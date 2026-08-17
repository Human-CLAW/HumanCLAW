"""Inference-only motion runtime (training modules are intentionally absent)."""

USE_SKILLS = (
    "walk_forward",
    "side_walk",
    "step_back",
    "turn",
    "step_climb_up",
    "step_climb_down",
    "sit",
    "stand",
)


def __getattr__(name):
    """Lazily import motion runtime classes so lightweight commands do not import Torch."""

    if name == "MotionSkillRunner":
        from .runner import MotionSkillRunner

        return MotionSkillRunner
    raise AttributeError(name)

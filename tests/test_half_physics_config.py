"""Release-contract tests for Half-Physics defaults."""

import pytest

from humanclaw_bench.envs.half_physics_env import HalfPhysicsEnv


def test_release_half_physics_defaults_match_validated_replay() -> None:
    """Keep the selected penetration/stair settings visible at the env boundary."""

    env = HalfPhysicsEnv(build_runtime=False)
    assert env.half_physics_backend == "hp"
    assert env.physics_config.name == "humanclaw.physics_config.json"
    assert env.pjsc_substeps == 4
    assert env.root_linear_xz_command_substeps == (0, 2)
    assert env.friction == 0.4
    assert env.pjsc_lambda_by_link["left_shoulder"] == 0.03
    assert env.pjsc_lambda_by_link["right_shoulder"] == 0.03
    assert env.pjsc_lambda_by_link["left_wrist"] == 0.1
    assert env.pjsc_lambda_by_link["right_wrist"] == 0.1

@pytest.mark.parametrize("schedule", [(), (1, 2), (0, 4), (0, -1)])
def test_invalid_root_command_schedules_fail_before_habitat_starts(schedule) -> None:
    """Reject schedules that cannot describe the four configured substeps."""

    with pytest.raises(ValueError, match="root_linear_xz_command_substeps"):
        HalfPhysicsEnv(
            root_linear_xz_command_substeps=schedule,
            build_runtime=False,
        )

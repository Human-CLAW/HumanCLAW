#!/usr/bin/env python3
"""Exercise the production Half-Physics controller without loading an HSSD scene.

This manual integration check imports the real Habitat/Magnum bindings but
uses a tiny fake articulated agent and simulator.  That makes the four
substeps directly observable and locks down the release behavior that ordinary
configuration tests cannot see: x/z command timing, root-angular timing, the
30-degree drive cap, and PJSC's use of the pre-limit velocity.
"""

from __future__ import annotations

import json
from typing import Any

import magnum as mn
import numpy as np

from humanclaw_bench.envs.half_physics import hp

_JOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
)


class _EmptyRigidObjectManager:
    """Expose the rigid-manager method used by movable-object gravity."""

    @staticmethod
    def get_object_handles() -> list[str]:
        """Return no rigid objects for this controller-only test."""

        return []


class _FakeAgent:
    """Implement the small Habitat articulated-object surface used by hp_step."""

    def __init__(self) -> None:
        """Initialize an identity root, four identity joints, and empty motors."""

        self.num_links = len(_JOINT_NAMES)
        self.transformation = mn.Matrix4.identity_init()
        self.joint_positions = np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            self.num_links,
        ).tolist()
        self.root_linear_velocity = mn.Vector3([0.0, 0.0, 0.0])
        self.root_angular_velocity = mn.Vector3([0.0, 0.0, 0.0])
        self.joint_velocities: list[float] = []
        self.motion_type: Any = None
        self.existing_joint_motor_ids: dict[int, int] = {}
        self.motor_gain_history: dict[int, list[float]] = {
            index: [] for index in range(self.num_links)
        }

    def get_link_name(self, link_id: int) -> str:
        """Return the release joint name for one fake link."""

        return _JOINT_NAMES[int(link_id)]

    def get_link_joint_name(self, link_id: int) -> str:
        """Return the same name through Habitat's joint-name accessor."""

        return _JOINT_NAMES[int(link_id)]

    def create_joint_motor(self, joint_id: int, settings: Any) -> int:
        """Create a stable fake motor ID and retain its configured gain."""

        motor_id = int(joint_id) + 100
        self.existing_joint_motor_ids[motor_id] = int(joint_id)
        self.motor_gain_history[int(joint_id)].append(float(settings.position_gain))
        return motor_id

    def update_joint_motor(self, motor_id: int, settings: Any) -> None:
        """Record every PJSC gain update for later release-contract assertions."""

        joint_id = self.existing_joint_motor_ids[int(motor_id)]
        self.motor_gain_history[joint_id].append(float(settings.position_gain))

    def remove_joint_motor(self, motor_id: int) -> None:
        """Remove a fake motor when the backend disables its joint gain."""

        self.existing_joint_motor_ids.pop(int(motor_id), None)


class _FakeSimulator:
    """Record pre-step velocities and emulate contact-induced velocity changes."""

    def __init__(self, agent: _FakeAgent) -> None:
        """Bind the fake agent and initialize per-substep audit arrays."""

        self.agent = agent
        self.gravity: list[float] | None = None
        self.root_linear_before_step: list[np.ndarray] = []
        self.root_angular_before_step: list[np.ndarray] = []
        self.step_dt: list[float] = []
        self._objects = _EmptyRigidObjectManager()

    def set_gravity(self, value: list[float]) -> None:
        """Record the world-gravity value selected by the HP controller."""

        self.gravity = list(value)

    def get_rigid_object_manager(self) -> _EmptyRigidObjectManager:
        """Return the empty rigid-object manager used by this test."""

        return self._objects

    def step_physics(self, dt: float) -> None:
        """Capture commands, then emulate a contact changing root velocities."""

        linear = np.asarray(self.agent.root_linear_velocity, dtype=np.float64)
        angular = np.asarray(self.agent.root_angular_velocity, dtype=np.float64)
        self.root_linear_before_step.append(linear.copy())
        self.root_angular_before_step.append(angular.copy())
        self.step_dt.append(float(dt))

        # A deterministic disturbance makes command re-injection observable:
        # substeps without a write retain these changes, whereas substep 2
        # must restore only the generated x/z command.
        linear[0] += 5.0
        angular[0] += 2.0
        self.agent.root_linear_velocity = mn.Vector3(linear)
        self.agent.root_angular_velocity = mn.Vector3(angular)


def main() -> None:
    """Run one frame and assert every selected release-control invariant."""

    agent = _FakeAgent()
    simulator = _FakeSimulator(agent)
    prelimit_omegas: list[np.ndarray] = []
    integrate_expected = hp.approximate_next_expected_joint_position

    def capture_expected_velocity(initial: Any, omega: Any, dt: float) -> np.ndarray:
        """Capture PJSC's omega before delegating to the real integrator."""

        prelimit_omegas.append(np.asarray(omega, dtype=np.float64).reshape(-1, 3))
        return integrate_expected(initial, omega, dt)

    hp.approximate_next_expected_joint_position = capture_expected_velocity
    try:
        hp.hp_step(
            simulator,
            agent,
            mn.Matrix4.identity_init(),
            np.tile(np.asarray([np.pi, 0.0, 0.0]), (agent.num_links, 1)),
            np.asarray([np.pi, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            1.0 / 30.0,
            root_gravity_scale=1.0,
            root_gravity_mode="midpoint",
            pjsc_lambda=1.0,
            pjsc_lambda_by_link={},
            pjsc_substeps=4,
            root_linear_xz_command_substeps=(0, 2),
        )
    finally:
        hp.approximate_next_expected_joint_position = integrate_expected

    linear_x = [float(value[0]) for value in simulator.root_linear_before_step]
    np.testing.assert_allclose(linear_x, [30.0, 35.0, 30.0, 35.0], atol=1.0e-8)

    angular_x = [float(value[0]) for value in simulator.root_angular_before_step]
    angular_limit = float(np.deg2rad(30.0) * 30.0)
    assert np.isclose(abs(angular_x[0]), angular_limit)
    np.testing.assert_allclose(
        angular_x,
        [angular_x[0] + 2.0 * index for index in range(4)],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(simulator.step_dt, [1.0 / 120.0] * 4)

    joint_velocity = np.asarray(agent.joint_velocities).reshape(-1, 3)
    assert np.all(np.linalg.norm(joint_velocity, axis=1) <= angular_limit + 1.0e-8)
    assert prelimit_omegas
    assert float(np.max(np.linalg.norm(prelimit_omegas[0], axis=1))) > angular_limit

    expected_gains = {
        "left_shoulder": 0.03,
        "right_shoulder": 0.03,
        "left_wrist": 0.1,
        "right_wrist": 0.1,
    }
    for joint_id, name in enumerate(_JOINT_NAMES):
        assert np.isclose(max(agent.motor_gain_history[joint_id]), expected_gains[name])

    print(
        json.dumps(
            {
                "status": "ok",
                "root_linear_x_by_substep": linear_x,
                "root_angular_x_by_substep": angular_x,
                "joint_speed_limit_rad_s": angular_limit,
                "pjsc_prelimit_speed_observed": float(
                    np.max(np.linalg.norm(prelimit_omegas[0], axis=1))
                ),
                "fixed_pjsc_gains": expected_gains,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

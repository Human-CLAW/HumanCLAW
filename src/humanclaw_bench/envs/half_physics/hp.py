"""Production Half-Physics controller used by HumanClawBench.

One 30 Hz motion-generator frame is converted into four 120 Hz Bullet steps.
The controller drives the articulated human toward the generated SMPL-X pose
while leaving gravity and contacts free to alter the realized trajectory.

The constants in this module are the validated release settings:

* all joint drives and the root angular drive are limited to 30 degrees per
  motion frame;
* PJSC position gains are 0.03 for shoulders and 0.1 for wrists;
* the root x/z command is injected on zero-based physics substeps 0 and 2;
* the root angular command is injected only once, before substep 0; and
* gravity is applied to movable rigid objects before every physics substep.

The PJSC target trajectory intentionally uses the *pre-limit* joint velocity.
The 30-degree cap therefore limits the direct velocity drive without slowing
the position-correction target.  This distinction is important for matching
the replay experiments that selected the release settings.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import magnum as mn
import numpy as np
from habitat_sim.physics import JointMotorSettings, MotionType
from scipy.spatial.transform import Rotation as R

ANGULAR_MOTION_THRESHOLD = 0.5 * np.pi
ANGULAR_LIMIT_DEGREES_PER_FRAME = 30.0
SHOULDER_PJSC_POSITION_GAIN = 0.03
WRIST_PJSC_POSITION_GAIN = 0.1
FIXED_PJSC_GAINS = {
    "left_shoulder": SHOULDER_PJSC_POSITION_GAIN,
    "right_shoulder": SHOULDER_PJSC_POSITION_GAIN,
    "left_wrist": WRIST_PJSC_POSITION_GAIN,
    "right_wrist": WRIST_PJSC_POSITION_GAIN,
}
DEFAULT_ROOT_LINEAR_XZ_COMMAND_SUBSTEPS = (0, 2)
MOVABLE_OBJECT_GRAVITY_MPS2 = 9.8
_CONFIGURATION_REPORTED = False

SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]


def make_smplx_and_urdf_mappings(agent):
    """Return mapping arrays between SMPL-X joint order and URDF link order."""
    smplx2urdf = []
    urdf2smplx = []
    for joint_name in SMPLX_JOINT_NAMES:
        urdf2smplx.append(agent.get_link_id_from_name(joint_name))
    for link_id in range(agent.num_links):
        joint_name = agent.get_link_joint_name(link_id)
        smplx2urdf.append(SMPLX_JOINT_NAMES[1:].index(joint_name))
    return smplx2urdf, np.array(urdf2smplx) + 1


def suspend_all_movable_items(sim):
    """Retained for the original hp_step contract; gravity is per substep."""
    del sim
    return {}


def _apply_movable_object_gravity(sim):
    """Apply one substep's persistent gravity force to every dynamic object."""
    rom = sim.get_rigid_object_manager()
    for handle in rom.get_object_handles():
        obj = rom.get_object_by_handle(handle)
        if obj.motion_type == MotionType.DYNAMIC:
            obj.apply_force(
                mn.Vector3([0.0, -MOVABLE_OBJECT_GRAVITY_MPS2, 0.0]) * obj.mass,
                [0.0, 0.0, 0.0],
            )


def _step_physics_with_movable_object_gravity(sim, dt):
    """Apply persistent gravity to movable objects, then advance one Bullet substep."""

    _apply_movable_object_gravity(sim)
    sim.step_physics(dt)


def resume_all_movable_items(sim, rigid_states):
    """Complete the legacy movable-item guard; objects remain dynamic in this backend."""

    del sim, rigid_states


def smplx2habitat(world_transformation, poses, global_orient, translation):
    """Convert SMPL-X pose params to Habitat root transform and joint quats."""
    axis_angle_root_rotation_vec = mn.Vector3(global_orient)
    root_trans = mn.Vector3(translation)
    if axis_angle_root_rotation_vec.length() > 0:
        axis_angle_root_rotation_angle = mn.Rad(axis_angle_root_rotation_vec.length())
        root_rot = mn.Quaternion.rotation(
            axis_angle_root_rotation_angle,
            axis_angle_root_rotation_vec.normalized(),
        ).to_matrix()
    else:
        root_rot = mn.Quaternion(((0.0, 0.0, 0.0), 1.0)).to_matrix()

    transformation = world_transformation @ mn.Matrix4.from_(root_rot, root_trans)
    rotations = R.from_rotvec(np.asarray(poses).reshape(-1, 3))
    joint_positions = rotations.as_quat().reshape(-1).tolist()
    return transformation, joint_positions


def quat_multiply_batch(q1, q2):
    """Multiply batches of xyzw quaternions with Hamilton products."""

    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_inverse_batch(q):
    """Return multiplicative inverses for a batch of xyzw quaternions."""

    q_conj = q * np.array([-1.0, -1.0, -1.0, 1.0])
    norm_sq = np.sum(q**2, axis=1, keepdims=True)
    return q_conj / norm_sq


def quat_rotate_batch(q, v):
    """Rotate a batch of three-vectors by matching xyzw quaternions."""

    qvec = q[:, :3]
    w = q[:, 3:4]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def quat_normalize_batch(q):
    """Normalize each xyzw quaternion to unit length."""

    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quat_to_omega_batch(quat, updated_quat, base_body, dt):
    """Convert initial/final quaternion batches into angular velocities."""

    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    updated_quat = updated_quat / np.linalg.norm(updated_quat, axis=1, keepdims=True)
    delta_q = quat_multiply_batch(updated_quat, quat_inverse_batch(quat))
    delta_q = delta_q / np.linalg.norm(delta_q, axis=1, keepdims=True)

    w = np.clip(delta_q[:, 3], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    sin_half_theta = np.sin(theta / 2.0)

    axis = np.zeros_like(delta_q[:, :3])
    small_angle = sin_half_theta < 1e-6
    axis[~small_angle] = delta_q[~small_angle, :3] / sin_half_theta[~small_angle, None]
    axis[small_angle] = delta_q[small_angle, :3]

    angvel = axis * (theta[:, None] / dt)
    if not base_body:
        return quat_rotate_batch(quat_inverse_batch(quat), angvel)
    return angvel


def approximate_joint_velocity(initial_joint_positions, final_joint_positions, dt):
    """Estimate local angular velocity for every URDF joint over one frame."""

    initial = np.asarray(initial_joint_positions, dtype=np.float64).reshape(-1, 4)
    final = np.asarray(final_joint_positions, dtype=np.float64).reshape(-1, 4)
    return quat_to_omega_batch(initial, final, False, dt).reshape(-1)


def _shortest_arc_joint_velocity(
    initial_joint_positions,
    final_joint_positions,
    dt,
):
    """Return local joint angular velocity after enforcing delta quaternion w>=0."""
    initial = np.asarray(initial_joint_positions, dtype=np.float64).reshape(-1, 4)
    final = np.asarray(final_joint_positions, dtype=np.float64).reshape(-1, 4)
    initial = quat_normalize_batch(initial)
    final = quat_normalize_batch(final)
    delta = quat_multiply_batch(final, quat_inverse_batch(initial))
    delta = quat_normalize_batch(delta)
    delta = np.where(delta[:, 3:4] < 0.0, -delta, delta)

    w = np.clip(delta[:, 3], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    sin_half_theta = np.sin(theta / 2.0)
    axis = np.zeros_like(delta[:, :3])
    small_angle = sin_half_theta < 1.0e-6
    axis[~small_angle] = delta[~small_angle, :3] / sin_half_theta[~small_angle, None]
    axis[small_angle] = delta[small_angle, :3]
    shortest_world = axis * (theta[:, None] / float(dt))
    return quat_rotate_batch(
        quat_inverse_batch(initial),
        shortest_world,
    ).reshape(-1)


def _normalize_name(value: Any) -> str:
    """Normalize Habitat link names for stable configuration lookup."""

    return str(value).strip().lower().replace(" ", "_")


def _joint_mask(agent, expected_names: set[str] | frozenset[str]) -> np.ndarray:
    """Resolve a required set of joints and return its URDF-order mask.

    Habitat exposes both a link name and a joint name.  Different URDF import
    versions may populate only one of them, so both are checked.  Failing
    loudly here prevents a spelling mismatch from silently disabling a PJSC
    gain on a release run.
    """

    expected = set(expected_names)
    mask = np.zeros(int(agent.num_links), dtype=bool)
    found: set[str] = set()
    for joint_id in range(int(agent.num_links)):
        candidates: set[str] = set()
        for getter in (agent.get_link_name, agent.get_link_joint_name):
            try:
                candidates.add(_normalize_name(getter(joint_id)))
            except Exception:
                continue
        matched = candidates & expected
        if matched:
            mask[joint_id] = True
            found.update(matched)
    if found != expected or int(np.count_nonzero(mask)) != len(expected):
        raise RuntimeError(
            f"Expected joints={sorted(expected)} in the articulated agent, "
            f"found names={sorted(found)} "
            f"indices={np.flatnonzero(mask).tolist()}"
        )
    return mask


def _clip_vector_norms(vectors, limits):
    """Clip each final-axis vector to its corresponding Euclidean limit."""

    values = np.asarray(vectors, dtype=np.float64)
    limit_values = np.asarray(limits, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1)
    scale = np.ones_like(norms)
    nonzero = norms > 1e-12
    scale[nonzero] = np.minimum(1.0, limit_values[nonzero] / norms[nonzero])
    return values * scale[..., None]


def integrate_angular_velocity_batch(omega, quat, base_body, dt):
    """Integrate batched angular velocities into updated unit quaternions."""

    if not base_body:
        angvel = quat_rotate_batch(quat, omega)
    else:
        angvel = omega

    angle = np.linalg.norm(angvel, axis=1)
    angle_clamped = np.where(
        angle * dt > ANGULAR_MOTION_THRESHOLD,
        ANGULAR_MOTION_THRESHOLD / dt,
        angle,
    )

    small_angle = angle_clamped < 0.001
    axis = np.zeros_like(angvel)
    sin_term = np.sin(0.5 * angle_clamped * dt)

    axis[small_angle] = (
        angvel[small_angle]
        * (0.5 * dt - (dt**3) * 0.020833333333 * angle_clamped[small_angle] ** 2)[
            :, None
        ]
    )
    axis[~small_angle] = (
        angvel[~small_angle]
        * (sin_term[~small_angle] / angle_clamped[~small_angle])[:, None]
    )

    w = np.cos(0.5 * angle_clamped * dt)
    delta_q = np.concatenate([axis, w[:, None]], axis=1)

    if not base_body:
        new_quat = quat_multiply_batch(delta_q, quat)
    else:
        delta_q[:, :3] *= -1.0
        new_quat = quat_multiply_batch(quat, delta_q)
    return quat_normalize_batch(new_quat)


def approximate_next_expected_joint_position(initial_joint_positions, omega, dt):
    """Advance PJSC's expected joint quaternions by one physics substep."""

    initial = np.asarray(initial_joint_positions, dtype=np.float64).reshape(-1, 4)
    omega_np = np.asarray(omega, dtype=np.float64).reshape(-1, 3)
    return integrate_angular_velocity_batch(omega_np, initial, False, dt).reshape(-1)


def _matrix3_to_numpy(matrix):
    """Convert a Magnum rotation matrix to a float64 3-by-3 array."""

    return np.array(matrix, dtype=np.float64).reshape(3, 3)


def approximate_root_velocities(initial_transformation, final_transformation, dt):
    """Estimate root linear and world angular velocity between two transforms."""

    initial_position = initial_transformation.translation
    final_position = final_transformation.translation
    linear_velocity = (final_position - initial_position) / dt

    initial_rotation_matrix = _matrix3_to_numpy(initial_transformation.rotation())
    final_rotation_matrix = _matrix3_to_numpy(final_transformation.rotation())
    relative_rotation_matrix = final_rotation_matrix @ initial_rotation_matrix.T
    angular_velocity = R.from_matrix(relative_rotation_matrix).as_rotvec() / dt
    return linear_velocity, angular_velocity


def _as_np3(value) -> np.ndarray:
    """Convert a Magnum or array-like value into a float64 three-vector."""

    return np.asarray(value, dtype=np.float64).reshape(3)


def _inherit_downward_root_velocity(
    linear_velocity,
    art_agent,
    inherit_downward_root_y_velocity: bool,
):
    """Carry prior downward speed into the next generated root command."""

    velocity = _as_np3(linear_velocity).copy()
    if inherit_downward_root_y_velocity:
        previous_velocity = _as_np3(art_agent.root_linear_velocity)
        velocity[1] += min(0.0, float(previous_velocity[1]))
    return velocity


def _apply_root_gravity(
    linear_velocity,
    art_agent,
    dt: float,
    root_gravity_scale: float,
    inherit_downward_root_y_velocity: bool,
):
    """Apply one full-frame root gravity bias for pre-step gravity mode."""

    velocity = _inherit_downward_root_velocity(
        linear_velocity,
        art_agent,
        inherit_downward_root_y_velocity,
    )
    if root_gravity_scale == 0.0:
        return velocity
    velocity[1] -= 9.8 * float(dt) * float(root_gravity_scale)
    return velocity


def _apply_substep_root_gravity(
    linear_velocity,
    sub_dt: float,
    root_gravity_scale: float,
):
    """Apply a root gravity velocity bias for one whole or half substep."""

    velocity = _as_np3(linear_velocity).copy()
    if root_gravity_scale != 0.0:
        velocity[1] -= 9.8 * float(sub_dt) * float(root_gravity_scale)
    return velocity


def _joint_motor_settings(target_quat, position_gain: float):
    """Build a spherical position motor with the requested PJSC gain."""

    quat = mn.Quaternion(
        (
            (float(target_quat[0]), float(target_quat[1]), float(target_quat[2])),
            float(target_quat[3]),
        )
    )
    return JointMotorSettings(
        spherical_position_target=quat,
        position_gain=float(position_gain),
        spherical_velocity_target=mn.Vector3([0.0, 0.0, 0.0]),
        velocity_gain=0.0,
        max_impulse=np.inf,
    )


def _motor_id_from_joint_id(art_agent, joint_id: int):
    """Find the existing Habitat motor that controls one articulated joint."""

    for motor_id, link_id in art_agent.existing_joint_motor_ids.items():
        if int(link_id) == int(joint_id):
            return motor_id
    return None


def _normalize_link_key(value: Any) -> str:
    """Normalize a link name or numeric ID for PJSC gain lookup."""

    return str(value).strip().lower().replace(" ", "_")


def _normalize_pjsc_lambda_by_link(
    pjsc_lambda_by_link: Mapping[Any, Any] | None,
) -> dict[str, float]:
    """Validate and normalize per-link PJSC gain overrides."""

    if not pjsc_lambda_by_link:
        return {}
    if not isinstance(pjsc_lambda_by_link, Mapping):
        raise TypeError(
            "pjsc_lambda_by_link must be a mapping from link/joint name or id to gain"
        )
    return {
        _normalize_link_key(key): float(value)
        for key, value in pjsc_lambda_by_link.items()
    }


def _force_fixed_pjsc_gains(
    pjsc_lambda_by_link: Mapping[str, float],
) -> dict[str, float]:
    """Make the validated shoulder and wrist gains backend invariants."""

    gains = dict(pjsc_lambda_by_link)
    gains.update(FIXED_PJSC_GAINS)
    return gains


def _link_key_candidates(art_agent, joint_id: int) -> list[str]:
    """Return every normalized name/ID that may identify one joint."""

    candidates = [str(int(joint_id))]
    for getter in (art_agent.get_link_name, art_agent.get_link_joint_name):
        try:
            candidates.append(str(getter(joint_id)))
        except Exception:
            continue
    return [_normalize_link_key(candidate) for candidate in candidates]


def _pjsc_gain_for_joint(
    art_agent,
    joint_id: int,
    default_gain: float,
    pjsc_lambda_by_link: Mapping[str, float],
) -> float:
    """Resolve a joint's PJSC gain, honoring the fixed release overrides."""

    candidates = _link_key_candidates(art_agent, joint_id)
    for candidate in candidates:
        if candidate in FIXED_PJSC_GAINS:
            return FIXED_PJSC_GAINS[candidate]
    if not pjsc_lambda_by_link:
        return float(default_gain)
    for key in candidates:
        if key in pjsc_lambda_by_link:
            return float(pjsc_lambda_by_link[key])
    return float(default_gain)


def _verify_and_report_configuration(
    art_agent,
    fixed_joint_mask: np.ndarray,
    pjsc_lambda: float,
    pjsc_lambda_by_link: Mapping[str, float],
):
    """Verify fixed gains against the loaded URDF and report settings once."""

    global _CONFIGURATION_REPORTED
    resolved: dict[str, dict[str, float | int]] = {}
    for joint_id in np.flatnonzero(fixed_joint_mask):
        names: set[str] = set()
        for getter in (art_agent.get_link_name, art_agent.get_link_joint_name):
            try:
                names.add(_normalize_name(getter(int(joint_id))))
            except Exception:
                continue
        matched = names & set(FIXED_PJSC_GAINS)
        if len(matched) != 1:
            raise RuntimeError(
                f"Could not uniquely identify fixed-gain joint {joint_id}: "
                f"{sorted(names)}"
            )
        name = next(iter(matched))
        gain = _pjsc_gain_for_joint(
            art_agent,
            int(joint_id),
            pjsc_lambda,
            pjsc_lambda_by_link,
        )
        if not np.isclose(
            gain,
            FIXED_PJSC_GAINS[name],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"{name} PJSC gain is {gain}, expected {FIXED_PJSC_GAINS[name]}"
            )
        resolved[name] = {
            "joint_id": int(joint_id),
            "position_gain": float(gain),
        }
    if set(resolved) != set(FIXED_PJSC_GAINS):
        raise RuntimeError(f"Did not resolve all fixed-gain joints: {sorted(resolved)}")
    if not _CONFIGURATION_REPORTED:
        print(
            "[humanclaw-hp-config] "
            + json.dumps(
                {
                    "angular_limit_degrees_per_motion_frame": (
                        ANGULAR_LIMIT_DEGREES_PER_FRAME
                    ),
                    "movable_object_gravity_each_physics_substep": True,
                    "pjsc_uses_prelimit_velocity": True,
                    "root_linear_xz_command_substeps": list(
                        DEFAULT_ROOT_LINEAR_XZ_COMMAND_SUBSTEPS
                    ),
                    "root_angular_command_substeps": [0],
                    "resolved_fixed_pjsc_joints": resolved,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _CONFIGURATION_REPORTED = True


def _pjsc_any_enabled(
    default_gain: float, pjsc_lambda_by_link: Mapping[str, float]
) -> bool:
    """Return whether any global or per-link PJSC position gain is positive."""

    if float(default_gain) > 0.0:
        return True
    return any(float(value) > 0.0 for value in pjsc_lambda_by_link.values())


def _remove_joint_motor_if_present(art_agent, joint_id: int):
    """Remove a joint's persistent Habitat motor when PJSC is disabled."""

    motor_id = _motor_id_from_joint_id(art_agent, joint_id)
    if motor_id is not None:
        art_agent.remove_joint_motor(motor_id)


def _ensure_pjsc_joint_motors(
    art_agent,
    pjsc_lambda: float,
    pjsc_lambda_by_link: Mapping[str, float],
):
    """Create missing motors for every joint with a positive PJSC gain."""

    current = np.asarray(art_agent.joint_positions, dtype=np.float64).reshape(-1, 4)
    for joint_id in range(min(art_agent.num_links, current.shape[0])):
        gain = _pjsc_gain_for_joint(
            art_agent,
            joint_id,
            pjsc_lambda,
            pjsc_lambda_by_link,
        )
        if gain <= 0.0:
            _remove_joint_motor_if_present(art_agent, joint_id)
            continue
        if _motor_id_from_joint_id(art_agent, joint_id) is None:
            art_agent.create_joint_motor(
                joint_id,
                _joint_motor_settings(current[joint_id], 0.0),
            )


def _update_pjsc_joint_motor_targets(
    art_agent,
    expected_joint_positions,
    pjsc_lambda: float,
    pjsc_lambda_by_link: Mapping[str, float] | None = None,
):
    """Update enabled PJSC motors to the current expected joint quaternions."""

    pjsc_lambda_by_link = pjsc_lambda_by_link or {}
    expected = np.asarray(expected_joint_positions, dtype=np.float64).reshape(-1, 4)
    for joint_id in range(min(art_agent.num_links, expected.shape[0])):
        motor_id = _motor_id_from_joint_id(art_agent, joint_id)
        gain = _pjsc_gain_for_joint(
            art_agent,
            joint_id,
            pjsc_lambda,
            pjsc_lambda_by_link,
        )
        if gain <= 0.0:
            _remove_joint_motor_if_present(art_agent, joint_id)
            continue
        if motor_id is None:
            motor_id = art_agent.create_joint_motor(
                joint_id,
                _joint_motor_settings(expected[joint_id], 0.0),
            )
        art_agent.update_joint_motor(
            motor_id,
            _joint_motor_settings(expected[joint_id], gain),
        )


def _zero_pjsc_joint_motors(art_agent):
    """Set all existing PJSC motor gains to zero after a motion frame."""

    current = np.asarray(art_agent.joint_positions, dtype=np.float64).reshape(-1, 4)
    for joint_id in range(min(art_agent.num_links, current.shape[0])):
        motor_id = _motor_id_from_joint_id(art_agent, joint_id)
        if motor_id is None:
            continue
        art_agent.update_joint_motor(
            motor_id, _joint_motor_settings(current[joint_id], 0.0)
        )


def _angular_speed_limit(dt: float) -> float:
    """Convert the 30-degree-per-motion-frame limit to radians/second."""

    frame_dt = float(dt)
    if not np.isfinite(frame_dt) or frame_dt <= 0.0:
        raise ValueError(f"dt must be a positive finite value, got {dt!r}")
    return float(np.deg2rad(ANGULAR_LIMIT_DEGREES_PER_FRAME) / frame_dt)


def _normalize_root_linear_xz_command_substeps(
    value,
    substeps: int,
) -> tuple[int, ...]:
    """Validate the zero-based substeps that receive the root x/z command.

    The frame command is assigned once before Bullet advances, which is
    substep 0.  Requiring index 0 keeps the configuration honest about that
    unavoidable first assignment; additional indices request re-injection
    after contacts have had a chance to alter horizontal root velocity.
    """

    try:
        schedule = tuple(dict.fromkeys(int(index) for index in value))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "root_linear_xz_command_substeps must be an iterable of integers"
        ) from exc
    if not schedule or 0 not in schedule:
        raise ValueError("root_linear_xz_command_substeps must include substep 0")
    invalid = [index for index in schedule if index < 0 or index >= substeps]
    if invalid:
        raise ValueError(
            "root_linear_xz_command_substeps contains indices outside "
            f"[0, {substeps}): {invalid}"
        )
    return schedule


def hp_step(
    sim,
    agent,
    world_transformation,
    pose,
    global_orient,
    translation_speed,
    dt,
    *,
    root_gravity_scale=0.0,
    inherit_downward_root_y_velocity=True,
    pjsc_lambda=0.0,
    pjsc_lambda_by_link=None,
    pjsc_substeps=4,
    root_linear_xz_command_substeps=DEFAULT_ROOT_LINEAR_XZ_COMMAND_SUBSTEPS,
    root_gravity_mode="midpoint",
):
    """Advance the articulated human by one motion-generator frame.

    ``root_gravity_mode`` controls only the human root's vertical velocity:
    ``"pre"`` applies a full-frame bias before physics, ``"substep"`` applies
    one semi-implicit bias per Bullet step, and ``"midpoint"`` applies half a
    bias on each side of every step.  The midpoint form gives the expected
    ``0.5 * g * dt**2`` displacement from rest.

    Horizontal root velocity is a command rather than a hard pose.  It is
    written on the configured substeps (0 and 2 for the release), while its y
    component is always preserved from gravity/contact response.  Root angular
    velocity is written only once before the loop.
    """

    substeps = max(1, int(pjsc_substeps))
    command_substeps = _normalize_root_linear_xz_command_substeps(
        root_linear_xz_command_substeps,
        substeps,
    )
    pjsc_lambda_by_link = _normalize_pjsc_lambda_by_link(pjsc_lambda_by_link)
    pjsc_lambda_by_link = _force_fixed_pjsc_gains(pjsc_lambda_by_link)
    root_gravity_mode = str(root_gravity_mode).lower()
    if root_gravity_mode not in {"pre", "substep", "midpoint"}:
        raise ValueError(f"Unsupported root_gravity_mode: {root_gravity_mode}")
    use_substep_gravity = (
        root_gravity_mode in {"substep", "midpoint"} and root_gravity_scale != 0.0
    )
    use_midpoint_gravity = root_gravity_mode == "midpoint" and root_gravity_scale != 0.0

    art_agent = agent
    fixed_joint_mask = _joint_mask(art_agent, frozenset(FIXED_PJSC_GAINS))
    _verify_and_report_configuration(
        art_agent,
        fixed_joint_mask,
        pjsc_lambda,
        pjsc_lambda_by_link,
    )
    curr_transformation = mn.Matrix4(art_agent.transformation)
    curr_joint_positions = art_agent.joint_positions.copy()

    current_local_translation = np.array(
        (world_transformation.inverted() @ curr_transformation).translation,
        dtype=np.float64,
    )
    next_translation = current_local_translation + np.asarray(
        translation_speed, dtype=np.float64
    )
    next_transformation, next_joint_positions = smplx2habitat(
        world_transformation,
        pose,
        global_orient,
        next_translation,
    )

    sim.set_gravity([0.0, 0.0, 0.0])
    movable_states = suspend_all_movable_items(sim)
    art_agent.motion_type = MotionType.DYNAMIC

    linear_velocity, angular_velocity = approximate_root_velocities(
        curr_transformation,
        next_transformation,
        dt,
    )
    if use_substep_gravity:
        linear_velocity = _inherit_downward_root_velocity(
            linear_velocity,
            art_agent,
            inherit_downward_root_y_velocity,
        )
    else:
        linear_velocity = _apply_root_gravity(
            linear_velocity,
            art_agent,
            dt,
            root_gravity_scale,
            inherit_downward_root_y_velocity,
        )
    # Quaternions q and -q encode the same orientation.  The legacy velocity
    # calculation can therefore occasionally request an almost-full turn when
    # the nearby target crosses quaternion sign.  Preserve the legacy numeric
    # result when both calculations agree, and replace only true long-arc
    # mismatches.  The 1e-6 tolerance is many orders below a 2*pi/dt mismatch
    # and avoids injecting harmless floating-point differences into contacts.
    joint_velocity_matrix = np.asarray(
        approximate_joint_velocity(
            curr_joint_positions,
            next_joint_positions,
            dt,
        ),
        dtype=np.float64,
    ).reshape(-1, 3)
    shortest_joint_velocity_matrix = np.asarray(
        _shortest_arc_joint_velocity(
            curr_joint_positions,
            next_joint_positions,
            dt,
        ),
        dtype=np.float64,
    ).reshape(-1, 3)
    if joint_velocity_matrix.shape[0] != int(art_agent.num_links):
        raise RuntimeError(
            f"Agent has {art_agent.num_links} joints but velocity has "
            f"{joint_velocity_matrix.shape[0]}"
        )
    long_arc = (
        np.linalg.norm(
            joint_velocity_matrix - shortest_joint_velocity_matrix,
            axis=1,
        )
        > 1.0e-6
    )
    joint_velocity_matrix[long_arc] = shortest_joint_velocity_matrix[long_arc]

    # PJSC deliberately advances from the pre-limit velocity.  The direct
    # velocity drive is capped below, but its position-correction target still
    # follows the complete generated motion, exactly as in the selected replay.
    pjsc_joint_velocities = joint_velocity_matrix.reshape(-1).copy()

    angular_speed_limit = _angular_speed_limit(dt)
    joint_limits = np.full(
        int(art_agent.num_links),
        angular_speed_limit,
        dtype=np.float64,
    )
    joint_velocity_matrix = _clip_vector_norms(joint_velocity_matrix, joint_limits)
    joint_velocities = joint_velocity_matrix.reshape(-1)

    # The same per-frame angular cap applies to global/root orientation.  The
    # root's linear command is intentionally not magnitude-clipped here.
    angular_velocity_clipped = _clip_vector_norms(
        np.asarray(angular_velocity, dtype=np.float64).reshape(1, 3),
        np.asarray([angular_speed_limit], dtype=np.float64),
    )
    angular_velocity = angular_velocity_clipped.reshape(3)

    art_agent.root_linear_velocity = mn.Vector3(linear_velocity)
    art_agent.root_angular_velocity = mn.Vector3(angular_velocity)
    art_agent.joint_velocities = joint_velocities.tolist()

    try:
        sub_dt = float(dt) / substeps
        expected = np.asarray(curr_joint_positions, dtype=np.float64).reshape(-1)
        pjsc_enabled = _pjsc_any_enabled(pjsc_lambda, pjsc_lambda_by_link)
        if pjsc_enabled:
            _ensure_pjsc_joint_motors(art_agent, pjsc_lambda, pjsc_lambda_by_link)
        try:
            for substep_index in range(substeps):
                # Read the realized velocity at the start of every substep so
                # y and any contact-induced x/z changes are retained.  Only
                # configured command points overwrite horizontal components.
                current_root_velocity = _as_np3(art_agent.root_linear_velocity)
                if substep_index in command_substeps:
                    current_root_velocity[0] = linear_velocity[0]
                    current_root_velocity[2] = linear_velocity[2]
                if use_midpoint_gravity:
                    current_root_velocity = _apply_substep_root_gravity(
                        current_root_velocity,
                        0.5 * sub_dt,
                        root_gravity_scale,
                    )
                elif use_substep_gravity:
                    current_root_velocity = _apply_substep_root_gravity(
                        current_root_velocity,
                        sub_dt,
                        root_gravity_scale,
                    )
                art_agent.root_linear_velocity = mn.Vector3(current_root_velocity)
                if pjsc_enabled:
                    _update_pjsc_joint_motor_targets(
                        art_agent,
                        expected,
                        pjsc_lambda,
                        pjsc_lambda_by_link,
                    )
                _step_physics_with_movable_object_gravity(sim, sub_dt)
                if use_midpoint_gravity:
                    current_root_velocity = _apply_substep_root_gravity(
                        art_agent.root_linear_velocity,
                        0.5 * sub_dt,
                        root_gravity_scale,
                    )
                    art_agent.root_linear_velocity = mn.Vector3(current_root_velocity)
                if pjsc_enabled:
                    expected = approximate_next_expected_joint_position(
                        expected,
                        pjsc_joint_velocities,
                        sub_dt,
                    )
        finally:
            # Motors persist on Habitat articulated objects.  Zero their gain
            # even if a physics step fails so no stale target leaks into the
            # next action.
            if pjsc_enabled:
                _zero_pjsc_joint_motors(art_agent)
    finally:
        resume_all_movable_items(sim, movable_states)

    return art_agent.joint_positions.copy()

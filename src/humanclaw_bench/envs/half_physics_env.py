"""Minimal HalfPhysics environment for HumanClawBench rollout.

The environment owns Habitat, the SMPL-X articulated agent, and the HP backend
step. Evaluators orchestrate it with an ego agent:

    obs = env.reset(episode)
    action, reasoning = ego_agent.act(obs, episode.instruction)
    obs, reward, done, info = env.step(action, reasoning=reasoning)

The action format is raw SMPL-X motion:

    {
        "transl": np.ndarray[T, 3],
        "global_orient": np.ndarray[T, 3],
        "body_pose": np.ndarray[T, 54, 3],
        "fps": 30,  # optional
    }
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from humanclaw_bench.paths import repository_root

REPO_ROOT = repository_root()
HALF_PHYSICS_ASSET_DIR = Path(__file__).resolve().parent / "half_physics"

DEFAULT_SCENE_DATASET_CONFIG = (
    REPO_ROOT / "resources" / "scenes" / "hssd-hab.scene_dataset_config.json"
)
DEFAULT_AGENT_DIR = REPO_ROOT / "resources" / "agent" / "neutral_beta0_handmerged"
DEFAULT_AGENT_URDF = DEFAULT_AGENT_DIR / "neutral_beta0_handmerged.urdf"
DEFAULT_AGENT_SHIFT = DEFAULT_AGENT_DIR / "shift.npy"
DEFAULT_PHYSICS_CONFIG = HALF_PHYSICS_ASSET_DIR / "humanclaw.physics_config.json"
DEFAULT_ROOT_LINEAR_XZ_COMMAND_SUBSTEPS = (0, 2)
DEFAULT_PJSC_LAMBDA_BY_LINK = {
    "left_ankle": 0.1,
    "right_ankle": 0.1,
    "left_shoulder": 0.03,
    "right_shoulder": 0.03,
    "left_wrist": 0.1,
    "right_wrist": 0.1,
}

# Match humanclaw.agent.world_interface: motion models emit Y-up 75D xb,
# while HalfPhysicsWorld expects the converted Z-up pose.
_RX90_YUP_TO_ZUP = R.from_matrix(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
_NECK_HEAD_BODY_POSE_JOINTS = (11, 14)
_MOTION_XB_DIM = 75
_MOTION_BODY_POSE_DIM = 69


@dataclass(frozen=True)
class EgoCameraConfig:
    """Head-mounted camera resolution and offsets in the head-link frame."""

    resolution: tuple[int, int] = (448, 448)
    forward_offset: float = 0.02
    pitch_down: float = 0.4


@dataclass(frozen=True)
class ThirdPersonCameraConfig:
    """Camera that follows behind the humanoid when video is requested."""

    resolution: tuple[int, int] = (512, 512)
    distance_behind: float = 1.5
    height_above: float = 1.8
    look_at_height: float = 1.0


@dataclass(frozen=True)
class RuntimeModules:
    """Hold lazily imported Habitat, Magnum, rotation, and Half-Physics modules."""

    habitat_sim: Any
    mn: Any
    rotation_cls: Any
    motion_type: Any
    hp: Any


@dataclass(frozen=True)
class HalfPhysicsObservation:
    """Observation returned by HalfPhysicsEnv."""

    head_rgb: np.ndarray


@dataclass(frozen=True)
class MotionAction:
    """Represent one raw SMPL-X motion chunk accepted by HalfPhysicsEnv."""

    transl: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    fps: Optional[float] = None


def _as_rgb_array(value: Any) -> np.ndarray:
    """Normalize supported observation/image values to an RGB uint8 array."""

    if isinstance(value, HalfPhysicsObservation):
        return value.head_rgb
    if isinstance(value, dict):
        for key in ("head_rgb", "ego_rgb", "rgb"):
            if key in value:
                return _as_rgb_array(value[key])
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB/RGBA image array, got shape={arr.shape}")
    return arr[:, :, :3].astype(np.uint8, copy=False)


def _require_file(path: Path, name: str) -> str:
    """Validate an asset path and return the string required by Habitat."""

    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return str(path)


def _load_module_from_path(module_name: str, path: Path) -> Any:
    """Import a Half-Physics backend directly from its release file."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_modules(backend: str) -> RuntimeModules:
    """Load Habitat, Magnum, SciPy Rotation, and the selected HP backend lazily."""

    half_physics_core_dir = str(HALF_PHYSICS_ASSET_DIR)
    if half_physics_core_dir not in sys.path:
        sys.path.insert(0, half_physics_core_dir)

    import habitat_sim
    import magnum as mn
    from habitat_sim.physics import MotionType
    from scipy.spatial.transform import Rotation as R

    backend_path = HALF_PHYSICS_ASSET_DIR / f"{backend}.py"
    if backend_path.is_file():
        hp_backend = _load_module_from_path(
            f"humanclawbench_habitat_hp_{backend}",
            backend_path,
        )
    else:
        hp_backend = importlib.import_module(backend)

    return RuntimeModules(
        habitat_sim=habitat_sim,
        mn=mn,
        rotation_cls=R,
        motion_type=MotionType,
        hp=hp_backend,
    )


def _episode_get(episode: Any, names: tuple[str, ...], default: Any = None) -> Any:
    """Read an episode field from a mapping or attribute object with aliases."""

    if episode is None:
        return default
    if isinstance(episode, dict):
        for name in names:
            if name in episode:
                return episode[name]
        return default
    for name in names:
        if hasattr(episode, name):
            return getattr(episode, name)
    return default


def _coerce_vector3(value: Any, name: str) -> np.ndarray:
    """Validate and convert an input to one float32 three-vector."""

    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def _coerce_body_pose(value: Any, name: str) -> np.ndarray:
    """Validate and reshape SMPL-X body pose data to 54 axis-angle joints."""

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 162:
        arr = arr.reshape(54, 3)
    if arr.shape != (54, 3):
        raise ValueError(f"{name} must have shape (54, 3), got {arr.shape}")
    return arr


def _zero_neck_head_body_pose(body_pose: np.ndarray) -> np.ndarray:
    """Zero neck/head motion to match the checkpoints' release preprocessing."""

    if body_pose.shape[-1] != _MOTION_BODY_POSE_DIM:
        raise ValueError(
            f"Expected body_pose last dim {_MOTION_BODY_POSE_DIM}, got {body_pose.shape[-1]}"
        )
    out = np.array(body_pose, copy=True)
    for joint_idx in _NECK_HEAD_BODY_POSE_JOINTS:
        start = joint_idx * 3
        out[..., start : start + 3] = 0
    return out


def xb75_yup_to_half_physics_pose(
    xb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert agent6-compatible Y-up 75D xb into HalfPhysics pose arrays."""

    arr = np.asarray(xb, dtype=np.float32)
    single_frame = arr.ndim == 1
    if single_frame:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != _MOTION_XB_DIM:
        raise ValueError(f"Expected xb shape (75,) or (T, 75), got {arr.shape}")

    transl_y = arr[:, :3]
    orient_y = arr[:, 3:6]
    bpose_y = _zero_neck_head_body_pose(arr[:, 6:75])

    transl_z = np.stack(
        [transl_y[:, 0], -transl_y[:, 2], transl_y[:, 1]], axis=1
    ).astype(np.float32)
    orient_z = (
        (_RX90_YUP_TO_ZUP * R.from_rotvec(orient_y)).as_rotvec().astype(np.float32)
    )

    bpose_23 = bpose_y.reshape(arr.shape[0], 23, 3)
    body_pose_54 = np.zeros((arr.shape[0], 54, 3), dtype=np.float32)
    body_pose_54[:, :21, :] = bpose_23[:, :21, :]

    if single_frame:
        return transl_z[0], orient_z[0], body_pose_54[0]
    return transl_z, orient_z, body_pose_54


def _parse_motion_action(
    action: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Return transl, global_orient, body_pose, fps from a raw motion action."""

    if isinstance(action, MotionAction):
        return action.transl, action.global_orient, action.body_pose, action.fps

    if isinstance(action, dict):
        if "body_state" in action and isinstance(action["body_state"], dict):
            action = action["body_state"]
        transl = np.asarray(action["transl"], dtype=np.float32)
        global_orient = np.asarray(action["global_orient"], dtype=np.float32)
        body_pose = np.asarray(action["body_pose"], dtype=np.float32)
        fps = action.get("fps")
    else:
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] not in (168, 75):
            raise ValueError(
                "Motion action must be a dict or an array with shape (T, 168) "
                "for SMPL-X or (T, 75) for the motion-model body pose."
            )
        if arr.shape[1] == 168:
            transl = arr[:, :3]
            global_orient = arr[:, 3:6]
            body_pose = arr[:, 6:].reshape(arr.shape[0], 54, 3)
        else:
            transl, global_orient, body_pose = xb75_yup_to_half_physics_pose(arr)
        fps = None

    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"transl must have shape (T, 3), got {transl.shape}")
    if global_orient.shape != transl.shape:
        raise ValueError(
            f"global_orient must have shape {transl.shape}, got {global_orient.shape}"
        )
    if body_pose.ndim == 2 and body_pose.shape[1] == 162:
        body_pose = body_pose.reshape(body_pose.shape[0], 54, 3)
    if body_pose.ndim != 3 or body_pose.shape[1:] != (54, 3):
        raise ValueError(f"body_pose must have shape (T, 54, 3), got {body_pose.shape}")
    if body_pose.shape[0] != transl.shape[0]:
        raise ValueError("transl/global_orient/body_pose must have the same T")

    return transl, global_orient, body_pose, None if fps is None else float(fps)


def _parse_named_float_map(value: Any, name: str) -> dict[str, float]:
    """Parse a JSON or comma-separated name->float mapping."""

    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            parsed: dict[str, str] = {}
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    key, raw = item.split("=", 1)
                elif ":" in item:
                    key, raw = item.split(":", 1)
                else:
                    raise ValueError(
                        f"{name} items must be key=value or key:value, got {item!r}"
                    )
                parsed[key.strip()] = raw.strip()
            value = parsed
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict or string mapping, got {type(value)}")

    out: dict[str, float] = {}
    for key, raw in value.items():
        key_text = str(key).strip().lower().replace(" ", "_")
        if not key_text:
            raise ValueError(f"{name} contains an empty key")
        out[key_text] = float(raw)
    return out


class HalfPhysicsEnv:
    """Benchmark env around Habitat-sim plus the HalfPhysics backend."""

    def __init__(
        self,
        *,
        scene_id: str = "apt_0",
        scene_dataset_config: str | Path = DEFAULT_SCENE_DATASET_CONFIG,
        physics_config: str | Path = DEFAULT_PHYSICS_CONFIG,
        agent_urdf: str | Path = DEFAULT_AGENT_URDF,
        agent_shift_npy: str | Path = DEFAULT_AGENT_SHIFT,
        half_physics_backend: str = "hp",
        max_episode_steps: int = 100,
        fps: float = 30.0,
        ego_resolution: tuple[int, int] = (448, 448),
        ego_semantic_enabled: bool = False,
        collect_metric_contacts: bool = False,
        video_enabled: bool = False,
        third_person_resolution: tuple[int, int] = (512, 512),
        root_gravity_scale: float = 1.0,
        root_gravity_mode: str = "midpoint",
        inherit_downward_root_y_velocity: bool = True,
        pjsc_lambda: float = 1.0,
        pjsc_lambda_by_link: dict[str, float] | str | None = None,
        pjsc_substeps: int = 4,
        root_linear_xz_command_substeps: tuple[int, ...] | list[int] = (
            DEFAULT_ROOT_LINEAR_XZ_COMMAND_SUBSTEPS
        ),
        friction: float = 0.4,
        build_runtime: bool = True,
    ) -> None:
        """Validate assets and physics settings, then optionally construct the Habitat runtime."""

        self.scene_id = str(scene_id)
        self.scene_dataset_config = Path(scene_dataset_config)
        self.physics_config = Path(physics_config)
        self.agent_urdf = Path(agent_urdf)
        self.agent_shift_npy = Path(agent_shift_npy)
        self.half_physics_backend = str(half_physics_backend)
        self.max_episode_steps = int(max_episode_steps)
        self.fps = float(fps)
        self.ego_camera = EgoCameraConfig(resolution=tuple(ego_resolution))
        # These switches are set only by explicit CLI flags.  Keeping them on
        # the environment makes it impossible for a nominal rollout to
        # accidentally pay semantic-render, contact-query, or exo-render cost.
        self.ego_semantic_enabled = bool(ego_semantic_enabled)
        self.collect_metric_contacts = bool(collect_metric_contacts)
        self.video_enabled = bool(video_enabled)
        self.third_person_camera = ThirdPersonCameraConfig(
            resolution=tuple(third_person_resolution)
        )
        self.friction = float(friction)
        self.root_gravity_scale = float(root_gravity_scale)
        self.root_gravity_mode = str(root_gravity_mode)
        self.inherit_downward_root_y_velocity = bool(inherit_downward_root_y_velocity)
        self.pjsc_lambda = float(pjsc_lambda)
        self.pjsc_lambda_by_link = {
            **DEFAULT_PJSC_LAMBDA_BY_LINK,
            **_parse_named_float_map(
                pjsc_lambda_by_link,
                "pjsc_lambda_by_link",
            ),
        }
        self.pjsc_substeps = int(pjsc_substeps)
        if self.pjsc_substeps < 1:
            raise ValueError("pjsc_substeps must be at least 1")
        self.root_linear_xz_command_substeps = tuple(
            dict.fromkeys(int(index) for index in root_linear_xz_command_substeps)
        )
        if 0 not in self.root_linear_xz_command_substeps:
            raise ValueError("root_linear_xz_command_substeps must include substep 0")
        invalid_substeps = [
            index
            for index in self.root_linear_xz_command_substeps
            if index < 0 or index >= self.pjsc_substeps
        ]
        if invalid_substeps:
            raise ValueError(
                "root_linear_xz_command_substeps contains indices outside "
                f"[0, {self.pjsc_substeps}): {invalid_substeps}"
            )
        self._runtime: Optional[RuntimeModules] = None
        self.sim: Any = None
        self.agent: Any = None
        self.world_transformation: Any = None
        self.world_transformation_mn: Any = None
        self.original_root_shift: Optional[np.ndarray] = None
        self.smplx2urdf: Any = None
        self.urdf2smplx: Any = None
        self._head_link_id: Optional[int] = None
        self._left_eye_link_id: Optional[int] = None
        self._right_eye_link_id: Optional[int] = None
        self._link_id_to_name: dict[int, str] = {}

        self._current_step = 0
        self._reset = False
        self._last_obs: Optional[HalfPhysicsObservation] = None
        self._last_semantic: np.ndarray | None = None
        self._last_third_person_rgb: np.ndarray | None = None
        self._video_frame_sink: Any = None

        if build_runtime:
            self._build_runtime()

    def _build_runtime(self) -> None:
        """Load runtime modules, validate assets, create Habitat, and load the human agent once."""

        if self.sim is not None:
            return

        self._runtime = _load_runtime_modules(self.half_physics_backend)
        self.scene_dataset_config = Path(
            _require_file(self.scene_dataset_config, "scene_dataset_config")
        )
        self.physics_config = Path(_require_file(self.physics_config, "physics_config"))
        self.agent_urdf = Path(_require_file(self.agent_urdf, "agent_urdf"))
        self.agent_shift_npy = Path(
            _require_file(self.agent_shift_npy, "agent_shift_npy")
        )

        self._build_simulator()
        self._load_agent()

    def _require_runtime(self) -> RuntimeModules:
        """Lazily build and return a complete Half-Physics runtime."""

        if self.sim is None or self.agent is None:
            self._build_runtime()
        if self._runtime is None:
            raise RuntimeError("HalfPhysics runtime failed to initialize.")
        return self._runtime

    def _build_simulator(self) -> None:
        """Configure Habitat sensors, physics, and scene loading for one environment."""

        runtime = self._require_loaded_modules()
        habitat_sim = runtime.habitat_sim
        mn = runtime.mn

        import os as _os

        _headless_only = _os.environ.get("HCB_DISABLE_RENDERER", "") in (
            "1",
            "true",
            "True",
        )

        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.enable_physics = True
        sim_config.scene_id = self.scene_id
        sim_config.scene_dataset_config_file = str(self.scene_dataset_config)
        sim_config.physics_config_file = str(self.physics_config)
        # Diagnostic replay can disable all rendering.  Otherwise the ego RGB
        # sensor is the only mandatory sensor; semantic and exo sensors are
        # allocated strictly by their metric/video flags below.
        if _headless_only and hasattr(sim_config, "create_renderer"):
            sim_config.create_renderer = False

        agent_config = habitat_sim.AgentConfiguration()
        sensors = []

        if not _headless_only:
            ego_spec = habitat_sim.CameraSensorSpec()
            ego_spec.uuid = "ego_rgb"
            ego_spec.resolution = list(self.ego_camera.resolution)
            ego_spec.position = mn.Vector3(0, 0, 0)
            ego_spec.orientation = mn.Vector3(0, 0, 0)
            sensors.append(ego_spec)

            if self.ego_semantic_enabled:
                semantic_spec = habitat_sim.CameraSensorSpec()
                semantic_spec.uuid = "ego_semantic"
                semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
                semantic_spec.resolution = list(self.ego_camera.resolution)
                semantic_spec.position = mn.Vector3(0, 0, 0)
                semantic_spec.orientation = mn.Vector3(0, 0, 0)
                sensors.append(semantic_spec)

            if self.video_enabled:
                third_spec = habitat_sim.CameraSensorSpec()
                third_spec.uuid = "third_person_rgb"
                third_spec.resolution = list(self.third_person_camera.resolution)
                third_spec.position = mn.Vector3(0, 2, -2)
                third_spec.orientation = mn.Vector3(0, 0, 0)
                sensors.append(third_spec)

        agent_config.sensor_specifications = sensors
        self.sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_config, [agent_config])
        )
        self.sim.reset()

        self.world_transformation = runtime.rotation_cls.from_matrix(
            np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        )
        self.world_transformation_mn = mn.Matrix4.from_(
            mn.Matrix3(self.world_transformation.as_matrix()),
            mn.Vector3(0, 0, 0),
        )

    def _require_loaded_modules(self) -> RuntimeModules:
        """Return imported runtime modules without recursively constructing the simulator."""

        if self._runtime is None:
            raise RuntimeError("Runtime modules are not loaded.")
        return self._runtime

    def _load_agent(self) -> None:
        """Load the SMPL-X URDF, mappings, collision settings, friction, and root shift."""

        runtime = self._require_loaded_modules()
        mn = runtime.mn

        self.original_root_shift = np.load(self.agent_shift_npy)[0]
        articulated_obj_mgr = self.sim.get_articulated_object_manager()
        self.agent = articulated_obj_mgr.add_articulated_object_from_urdf(
            str(self.agent_urdf)
        )
        self.agent.auto_clamp_joint_limits = False
        self.smplx2urdf, self.urdf2smplx = runtime.hp.make_smplx_and_urdf_mappings(
            self.agent
        )

        self._head_link_id = self.agent.get_link_id_from_name("head")
        self._left_eye_link_id = self.agent.get_link_id_from_name("left_eye_smplhf")
        self._right_eye_link_id = self.agent.get_link_id_from_name("right_eye_smplhf")
        if self.collect_metric_contacts:
            # Contact records use readable link names.  Build this map only in
            # metric mode; default rollout and video-only mode never use it.
            try:
                self._link_id_to_name[-1] = self.agent.get_link_name(-1)
            except Exception:
                self._link_id_to_name[-1] = "pelvis"
            for link_id in range(self.agent.num_links):
                try:
                    self._link_id_to_name[link_id] = self.agent.get_link_name(link_id)
                except Exception:
                    self._link_id_to_name[link_id] = f"link_{link_id}"
        # Habitat exposes per-child-link friction for articulated objects.
        # This covers the feet, ankles, legs, torso, head, arms, and hands;
        # the pelvis/base has no Python per-link setter.  The 1.0 value in
        # humanclaw.physics_config.json remains the scene/object default; this
        # explicit loop is the validated 0.4 human-child-link override.
        for link_id in range(self.agent.num_links):
            self.agent.set_link_friction(link_id, self.friction)

        from habitat_sim.gfx import LightInfo, LightPositionModel

        agent_lights = [
            LightInfo(
                vector=mn.Vector4(0.0, -1.0, -0.5, 0.0),
                color=mn.Vector3(0.4, 0.4, 0.4),
                model=LightPositionModel.Global,
            ),
            LightInfo(
                vector=mn.Vector4(0.0, -0.5, 0.5, 0.0),
                color=mn.Vector3(0.2, 0.2, 0.2),
                model=LightPositionModel.Global,
            ),
        ]
        self.sim.set_light_setup(agent_lights, "agent_lights")
        self.agent.set_light_setup("agent_lights")

    def reset(
        self,
        episode: Any = None,
        *,
        initial_transl: Any = None,
        initial_global_orient: Any = None,
        initial_body_pose: Any = None,
    ) -> HalfPhysicsObservation:
        """Reset the humanoid pose and return the first ego observation."""

        if episode is not None:
            if initial_transl is None:
                initial_transl = _episode_get(
                    episode,
                    ("initial_transl", "init_transl"),
                    None,
                )
            if initial_global_orient is None:
                initial_global_orient = _episode_get(
                    episode,
                    ("initial_global_orient", "init_global_orient"),
                    None,
                )
            if initial_body_pose is None:
                initial_body_pose = _episode_get(
                    episode,
                    ("initial_body_pose", "init_body_pose"),
                    None,
                )
            episode_max_steps = _episode_get(
                episode,
                ("max_steps", "max_episode_steps"),
                None,
            )
            if episode_max_steps is not None:
                self.max_episode_steps = int(episode_max_steps)

        transl = _coerce_vector3(
            [0.0, 0.0, 0.0] if initial_transl is None else initial_transl,
            "initial_transl",
        )
        global_orient = _coerce_vector3(
            [0.0, 0.0, 0.0] if initial_global_orient is None else initial_global_orient,
            "initial_global_orient",
        )
        body_pose = _coerce_body_pose(
            np.zeros((54, 3), dtype=np.float32)
            if initial_body_pose is None
            else initial_body_pose,
            "initial_body_pose",
        )

        obs = self._reset_runtime_pose(
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
        )
        self._current_step = 0
        self._reset = True
        self._last_obs = HalfPhysicsObservation(head_rgb=_as_rgb_array(obs["ego_rgb"]))
        self._last_semantic = (
            np.asarray(obs["ego_semantic"]) if "ego_semantic" in obs else None
        )
        self._last_third_person_rgb = (
            _as_rgb_array(obs["third_person_rgb"])
            if "third_person_rgb" in obs
            else None
        )
        return self._last_obs

    def _reset_runtime_pose(
        self,
        *,
        transl: np.ndarray,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
    ) -> dict[str, Any]:
        """Place the articulated human at the episode's initial SMPL-X pose."""

        runtime = self._require_runtime()
        root_shift_world = self.world_transformation.apply(self.original_root_shift)
        self.agent.translation = root_shift_world + self.world_transformation.apply(
            transl
        )

        rot_agent = self.world_transformation * runtime.rotation_cls.from_rotvec(
            global_orient
        )
        q = rot_agent.as_quat()
        self.agent.rotation = runtime.mn.Quaternion(((q[0], q[1], q[2]), q[3]))

        body_pose_reordered = body_pose[self.smplx2urdf]
        joint_quats = runtime.rotation_cls.from_rotvec(body_pose_reordered).as_quat()
        self.agent.joint_positions = joint_quats.reshape(-1).tolist()
        self.agent.motion_type = runtime.motion_type.DYNAMIC

        self._update_cameras()
        obs = self.sim.get_sensor_observations()
        return dict(obs)

    def step(
        self,
        action: Any,
        reasoning: Any = None,
        i_flag: int | None = None,
    ) -> tuple[HalfPhysicsObservation, float, bool, Dict[str, Any]]:
        """Execute one raw SMPL-X motion action."""

        del i_flag
        if not self._reset:
            raise RuntimeError("Reset env before stepping.")

        transl, global_orient, body_pose, action_fps = _parse_motion_action(action)
        self._current_step += 1
        result = self._step_runtime_motion(
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            fps=self.fps if action_fps is None else action_fps,
        )

        ego_rgb = _as_rgb_array(result["ego_rgb"][-1])
        obs = HalfPhysicsObservation(head_rgb=ego_rgb)
        self._last_obs = obs
        self._last_semantic = result.get("ego_semantic")
        self._last_third_person_rgb = result.get("third_person_rgb")

        reward = 0.0
        done = self._current_step >= self.max_episode_steps
        info = {
            "body_state": result["body_state"],
            "object_states": result["object_states"],
        }
        if "metric_frames" in result:
            info["metric_frames"] = result["metric_frames"]
        return obs, reward, done, info

    def _is_dynamic_motion_object(self, obj: Any) -> bool:
        """Return whether Habitat currently treats an object as dynamic."""

        runtime = self._require_runtime()
        try:
            if obj.motion_type == runtime.motion_type.DYNAMIC:
                return True
        except Exception:
            pass
        return "DYNAMIC" in str(getattr(obj, "motion_type", "")).upper()

    def _tracked_dynamic_objects(self) -> dict[str, Any]:
        """Return movable rigid objects whose realized poses must be recorded."""

        tracked: dict[str, Any] = {}
        try:
            mgr = self.sim.get_rigid_object_manager()
            handles = list(mgr.get_object_handles())
        except Exception:
            handles = []
            mgr = None
        if mgr is not None:
            for handle in handles:
                try:
                    obj = mgr.get_object_by_handle(handle)
                except Exception:
                    continue
                if self._is_dynamic_motion_object(obj):
                    tracked[str(handle)] = obj
        return tracked

    @staticmethod
    def _vector3_array(value: Any) -> np.ndarray:
        """Convert a Magnum/vector-like value to three float32 components."""

        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            array = np.asarray(
                [getattr(value, axis) for axis in ("x", "y", "z")],
                dtype=np.float32,
            )
        if array.shape != (3,):
            raise ValueError(f"Expected a 3-vector, got shape={array.shape}")
        return array

    def _record_object_state(
        self,
        obj: Any,
        position_out: np.ndarray,
        rotation_out: np.ndarray,
        frame: int,
    ) -> None:
        """Record one dynamic object in the trajectory coordinate frame."""

        try:
            position_out[frame] = self.world_transformation.inv().apply(
                self._vector3_array(obj.translation)
            )
            quaternion = obj.rotation
            habitat_quaternion = np.asarray(
                [
                    quaternion.vector.x,
                    quaternion.vector.y,
                    quaternion.vector.z,
                    quaternion.scalar,
                ],
                dtype=np.float32,
            )
            rotation_out[frame] = (
                self.world_transformation.inv()
                * self._require_loaded_modules().rotation_cls.from_quat(
                    habitat_quaternion
                )
            ).as_quat()
        except Exception:  # noqa: BLE001 - preserve the frame with explicit NaNs
            position_out[frame] = np.nan
            rotation_out[frame] = np.nan

    def _read_agent_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the simulated humanoid pose in trajectory coordinates."""

        runtime = self._require_loaded_modules()
        transl = self.world_transformation.inv().apply(
            self._vector3_array(self.agent.translation)
            - self.world_transformation.apply(self.original_root_shift)
        )

        quaternion = self.agent.rotation
        habitat_quaternion = np.asarray(
            [
                quaternion.vector.x,
                quaternion.vector.y,
                quaternion.vector.z,
                quaternion.scalar,
            ],
            dtype=np.float32,
        )
        global_orient = (
            self.world_transformation.inv()
            * runtime.rotation_cls.from_quat(habitat_quaternion)
        ).as_rotvec()

        joint_quaternions = np.asarray(
            self.agent.joint_positions, dtype=np.float32
        ).reshape(54, 4)
        simulator_joints = np.zeros((55, 3), dtype=np.float32)
        simulator_joints[0] = global_orient
        simulator_joints[1:] = runtime.rotation_cls.from_quat(
            joint_quaternions
        ).as_rotvec()
        smplx_joints = simulator_joints[self.urdf2smplx]
        return (
            np.asarray(transl, dtype=np.float32),
            np.asarray(global_orient, dtype=np.float32),
            np.asarray(smplx_joints[1:], dtype=np.float32),
        )

    def replay_initial_state(self) -> dict[str, Any]:
        """Capture the exact simulator state needed to begin a forward replay."""

        transl, global_orient, body_pose = self._read_agent_state()
        try:
            root_linear_velocity = self._vector3_array(self.agent.root_linear_velocity)
            root_angular_velocity = self._vector3_array(
                self.agent.root_angular_velocity
            )
            joint_velocities = np.asarray(
                self.agent.joint_velocities, dtype=np.float32
            ).reshape(-1)
        except Exception:  # noqa: BLE001 - older Habitat builds may omit velocities
            root_linear_velocity = np.full(3, np.nan, dtype=np.float32)
            root_angular_velocity = np.full(3, np.nan, dtype=np.float32)
            joint_velocities = np.zeros(0, dtype=np.float32)
        human = {
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "root_linear_velocity": root_linear_velocity,
            "root_angular_velocity": root_angular_velocity,
            "joint_velocities": joint_velocities,
        }

        objects: dict[str, dict[str, Any]] = {}
        for handle, obj in self._tracked_dynamic_objects().items():
            position = np.full(3, np.nan, dtype=np.float32)
            rotation = np.full(4, np.nan, dtype=np.float32)
            self._record_object_state(obj, position[None, :], rotation[None, :], 0)
            try:
                linear_velocity = self._vector3_array(obj.linear_velocity)
                angular_velocity = self._vector3_array(obj.angular_velocity)
            except Exception:  # noqa: BLE001 - retain object identity and pose
                linear_velocity = np.full(3, np.nan, dtype=np.float32)
                angular_velocity = np.full(3, np.nan, dtype=np.float32)
            objects[str(handle)] = {
                "object_id": int(getattr(obj, "object_id", -1)),
                "motion_type": str(getattr(obj, "motion_type", "")),
                "position": position,
                "rotation": rotation,
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity,
            }
        return {"human": human, "objects": objects}

    def _step_runtime_motion(
        self,
        *,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        fps: float,
    ) -> dict[str, Any]:
        """Drive every frame in one generated motion chunk and collect enabled outputs."""

        runtime = self._require_runtime()
        T = body_pose.shape[0]
        dt = 1.0 / float(fps)
        # Dynamic-object identity is captured once per action.  Their poses are
        # still sampled every frame because trajectory_after must support exact
        # no-physics rendering and disturbance metrics.
        tracked_objects = self._tracked_dynamic_objects()
        out_transl = np.empty((T, 3), dtype=np.float32)
        out_global_orient = np.empty((T, 3), dtype=np.float32)
        out_body_pose = np.empty((T, 54, 3), dtype=np.float32)
        object_positions = {
            name: np.empty((T, 3), dtype=np.float32) for name in tracked_objects
        }
        object_rotations = {
            name: np.empty((T, 4), dtype=np.float32) for name in tracked_objects
        }
        agent_contact_frames: list[list[dict[str, Any]]] = []
        dynamic_contact_frames: list[list[dict[str, Any]]] = []
        final_observation: dict[str, Any] | None = None

        for t in range(T):
            pose_urdf_order = body_pose[t][self.smplx2urdf]
            orient = global_orient[t]

            for obj in tracked_objects.values():
                try:
                    obj.awake = True
                except Exception:
                    pass

            # Generated chunks include their own first pose but no preceding
            # within-chunk displacement.  Giving frame zero a zero translation
            # command avoids inventing motion across the action boundary; the
            # current simulated root state still supplies physical continuity.
            speed = transl[t] - transl[max(t - 1, 0)]
            runtime.hp.hp_step(
                self.sim,
                self.agent,
                self.world_transformation_mn,
                pose_urdf_order,
                orient,
                speed,
                dt,
                root_gravity_scale=self.root_gravity_scale,
                root_gravity_mode=self.root_gravity_mode,
                inherit_downward_root_y_velocity=self.inherit_downward_root_y_velocity,
                pjsc_lambda=self.pjsc_lambda,
                pjsc_lambda_by_link=self.pjsc_lambda_by_link,
                pjsc_substeps=self.pjsc_substeps,
                root_linear_xz_command_substeps=(self.root_linear_xz_command_substeps),
            )

            if self.collect_metric_contacts:
                # One shared discrete query at each realized 30 Hz pose feeds
                # collision, interaction, and disturbance. No metric performs
                # a second contact pass at episode end.
                from humanclaw_bench.envs.runtime_records import (
                    collect_metric_contacts,
                )

                agent_rows, dynamic_rows = collect_metric_contacts(self, t)
                agent_contact_frames.append(agent_rows)
                dynamic_contact_frames.append(dynamic_rows)

            actual_transl, actual_orient, actual_pose = self._read_agent_state()
            out_transl[t] = actual_transl
            out_global_orient[t] = actual_orient
            out_body_pose[t] = actual_pose
            for name, obj in tracked_objects.items():
                self._record_object_state(
                    obj,
                    object_positions[name],
                    object_rotations[name],
                    t,
                )

            # Closed-loop planning needs only the final ego frame.  Video mode
            # opts into per-frame ego/exo rendering and streams it immediately,
            # avoiding an image directory or an episode-length frame buffer.
            if self.video_enabled or t == T - 1:
                self._update_cameras()
                final_observation = dict(self.sim.get_sensor_observations())
                if self.video_enabled:
                    if "third_person_rgb" not in final_observation:
                        raise RuntimeError(
                            "Video mode requires the third-person RGB sensor"
                        )
                    self._emit_video_pair(
                        final_observation["ego_rgb"],
                        final_observation["third_person_rgb"],
                    )

        if final_observation is None:
            raise RuntimeError("Motion action contained no renderable frames")

        result: dict[str, Any] = {
            "ego_rgb": np.expand_dims(final_observation["ego_rgb"], axis=0),
            "body_state": {
                "transl": out_transl,
                "global_orient": out_global_orient,
                "body_pose": out_body_pose,
            },
            "object_states": {
                name: {
                    "position": object_positions[name],
                    "rotation": object_rotations[name],
                }
                for name in tracked_objects
            },
        }
        if "ego_semantic" in final_observation:
            result["ego_semantic"] = np.asarray(final_observation["ego_semantic"])
        if "third_person_rgb" in final_observation:
            result["third_person_rgb"] = _as_rgb_array(
                final_observation["third_person_rgb"]
            )
        if self.collect_metric_contacts:
            result["metric_frames"] = {
                "agent_contacts": agent_contact_frames,
                "dynamic_contacts": dynamic_contact_frames,
            }
        return result

    def _update_cameras(self) -> None:
        """Attach ego and optional exo cameras to the human's realized pose."""

        runtime = self._require_loaded_modules()
        mn = runtime.mn
        hab_agent = self.sim.get_agent(0)
        del hab_agent

        left_eye_transform = self.agent.get_link_scene_node(
            self._left_eye_link_id
        ).transformation
        right_eye_transform = self.agent.get_link_scene_node(
            self._right_eye_link_id
        ).transformation
        head_transform = self.agent.get_link_scene_node(
            self._head_link_id
        ).transformation
        eye_center = (
            left_eye_transform.translation + right_eye_transform.translation
        ) * 0.5
        head_forward = mn.Vector3(
            head_transform[2][0],
            head_transform[2][1],
            head_transform[2][2],
        )
        cam_pos = eye_center + head_forward * self.ego_camera.forward_offset
        head_down = mn.Vector3(
            -head_transform[1][0],
            -head_transform[1][1],
            -head_transform[1][2],
        )
        look_dir = head_forward * (1.0 - self.ego_camera.pitch_down) + (
            head_down * self.ego_camera.pitch_down
        )
        look_at_pos = cam_pos + look_dir
        head_up = mn.Vector3(
            head_transform[1][0],
            head_transform[1][1],
            head_transform[1][2],
        )

        ego_sensor_node = self.sim._sensors["ego_rgb"]._sensor_object.node
        ego_sensor_node.transformation = mn.Matrix4.look_at(
            cam_pos, look_at_pos, head_up
        )

        if self.ego_semantic_enabled and "ego_semantic" in self.sim._sensors:
            self.sim._sensors[
                "ego_semantic"
            ]._sensor_object.node.transformation = ego_sensor_node.transformation

        if self.video_enabled and "third_person_rgb" in self.sim._sensors:
            config = self.third_person_camera
            root = self.agent.translation
            root_transform = mn.Matrix4(self.agent.transformation)
            forward = mn.Vector3(
                -root_transform[2][0],
                -root_transform[2][1],
                -root_transform[2][2],
            )
            third_position = mn.Vector3(root) + forward * config.distance_behind
            third_position = mn.Vector3(
                third_position.x,
                root.y + config.height_above,
                third_position.z,
            )
            third_look_at = mn.Vector3(
                root.x,
                root.y + config.look_at_height,
                root.z,
            )
            third_node = self.sim._sensors["third_person_rgb"]._sensor_object.node
            third_node.transformation = mn.Matrix4.look_at(
                third_position,
                third_look_at,
                mn.Vector3(0, 1, 0),
            )

    def set_video_frame_sink(self, sink: Any | None) -> None:
        """Attach a streaming writer; no frames are retained by the env."""

        self._video_frame_sink = sink

    def _emit_video_pair(self, ego_frame: Any, third_frame: Any) -> None:
        """Send the latest ego/exo frames to the configured streaming video sink."""

        if self._video_frame_sink is not None:
            self._video_frame_sink(
                _as_rgb_array(ego_frame),
                _as_rgb_array(third_frame),
            )

    def emit_initial_video_frame(self) -> None:
        """Write the post-reset, post-lighting frame once."""

        if not self.video_enabled:
            return
        if self._last_obs is None or self._last_third_person_rgb is None:
            raise RuntimeError("Reset the video-enabled environment before recording")
        self._emit_video_pair(
            self._last_obs.head_rgb,
            self._last_third_person_rgb,
        )

    def render(self, mode: str = "rgb_array") -> np.ndarray | Image.Image:
        """Render the latest ego observation."""

        if self._last_obs is None:
            raise RuntimeError("Reset env before rendering.")
        rgb = self._last_obs.head_rgb
        if mode in {"rgb", "rgb_array"}:
            return rgb
        if mode in {"pil", "image"}:
            return Image.fromarray(rgb)
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        """Release open files, processes, clients, and runtime resources."""

        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.agent = None
        self._last_semantic = None
        self._last_third_person_rgb = None
        self._video_frame_sink = None


__all__ = [
    "DEFAULT_AGENT_SHIFT",
    "DEFAULT_AGENT_URDF",
    "DEFAULT_PHYSICS_CONFIG",
    "DEFAULT_SCENE_DATASET_CONFIG",
    "EgoCameraConfig",
    "ThirdPersonCameraConfig",
    "HalfPhysicsEnv",
    "HalfPhysicsObservation",
    "MotionAction",
]

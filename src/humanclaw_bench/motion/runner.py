"""Motion skill runner for HumanClawBench Find/Nav/Interact rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from . import USE_SKILLS
from .checkpoints import load_skill_models
from .conditioning import (
    condition_args,
    load_seed_state,
    zero_neck_head_body_pose,
    zero_neck_head_xb_state,
)
from .solver import denoise
from .state import (
    VEL_DIM,
    VEL_START,
    canonicalize_from_world,
    decanonicalize_joints,
    decanonicalize_xb,
    state_to_joints,
    state_to_xb,
)


@dataclass(frozen=True)
class GeneratedMotion:
    """One motion chunk before HalfPhysics modifies it."""

    xb_world_75: np.ndarray


class MotionSkillRunner:
    """Generate world-frame SMPL body chunks from skill calls."""

    def __init__(
        self,
        skills: Sequence[str] = USE_SKILLS,
        device: str = "cuda",
        weights_root: str = "weights/paper_fullval_v1",
        weights_manifest: str = "resources/weights/paper_fullval_v1.json",
        verify_weights: bool = True,
    ) -> None:
        """Configure checkpoint paths, device, solver settings, and deterministic seed state."""

        self.skills_requested = tuple(skills)
        self.device = device
        from humanclaw_bench.paths import resolve_release_path

        self.weights_root = resolve_release_path(weights_root)
        self.weights_manifest = resolve_release_path(weights_manifest)
        self.verify_weights = bool(verify_weights)
        self.skills: dict[str, tuple[Any, str]] = {}
        self.motion_state: Optional[torch.Tensor] = None
        self.T_ref_accum: Optional[torch.Tensor] = None
        self.floor_y_accum: float = 0.0
        self.initial_pelvis_y: float = 0.0
        self.current_pelvis_y: float = 0.0
        self.step_index: int = 0

    def load_skills(self) -> None:
        """Load requested skill checkpoints once and cache them by canonical skill name."""

        if self.skills:
            return
        self.skills = load_skill_models(
            self.skills_requested,
            weights_root=self.weights_root,
            manifest_path=self.weights_manifest,
            device=self.device,
            verify=self.verify_weights,
        )

    def unload(self) -> None:
        """Release inference tensors after an episode has generated all motion.

        Paper metrics can spend appreciable time replaying saved CPU poses for
        no-step contact queries.  None of those queries uses the motion model,
        so retaining eight control branches on the GPU during finalization
        only reduces safe batch concurrency.  ``reset`` remains reusable: its
        normal ``load_skills`` call reconstructs the same pinned models if a
        caller requests another rollout from this runner.
        """

        self.skills.clear()
        self.motion_state = None
        self.T_ref_accum = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(
        self,
        seed_mode: str,
        seed_pkl: str | None,
        seed_pt: str | None,
        init_offset: Sequence[float],
        init_yaw: float,
    ) -> np.ndarray:
        """Place the deterministic seed in episode coordinates and clear motion history."""

        self.load_skills()
        seed_state = load_seed_state(seed_mode, seed_pkl=seed_pkl, seed_pt=seed_pt)
        seed_state = zero_neck_head_xb_state(seed_state)
        self.motion_state = seed_state.unsqueeze(0).to(self.device)

        # T_ref_accum maps the canonical motion-model frame into the episode's
        # Y-up world frame.  Episode offsets retain the historical (x,z,y)
        # storage convention, hence the explicit axis reorder below.
        self.T_ref_accum = torch.eye(4)
        offset = np.array(init_offset, dtype=np.float32)
        if float(init_yaw) != 0.0:
            yaw_rad = np.radians(float(init_yaw))
            R_yaw = Rot.from_rotvec([0, yaw_rad, 0]).as_matrix()
            self.T_ref_accum[:3, :3] = torch.tensor(R_yaw, dtype=torch.float32)
        self.T_ref_accum[0, 3] = float(offset[0])
        self.T_ref_accum[1, 3] = float(offset[2])
        self.T_ref_accum[2, 3] = float(-offset[1])
        self.T_ref_accum = self.T_ref_accum.to(self.device)
        self.floor_y_accum = 0.0
        self.step_index = 0

        init_xb_canon = seed_state[0, :75].numpy()
        R_ref = self.T_ref_accum[:3, :3].cpu().numpy()
        t_ref = self.T_ref_accum[:3, 3].cpu().numpy()
        transl_world = R_ref @ init_xb_canon[:3] + t_ref
        R_canon = Rot.from_rotvec(init_xb_canon[3:6]).as_matrix()
        orient_world = Rot.from_matrix(R_ref @ R_canon).as_rotvec().astype(np.float32)
        init_xb_world = np.concatenate(
            [transl_world.astype(np.float32), orient_world, init_xb_canon[6:75]]
        )
        self.initial_pelvis_y = float(transl_world[1])
        self.current_pelvis_y = float(transl_world[1])
        return init_xb_world

    @torch.no_grad()
    def generate(self, skill: str, cond: Any) -> GeneratedMotion:
        """Generate one conditioned motion chunk and update the rolling canonical state."""

        if self.motion_state is None or self.T_ref_accum is None:
            raise RuntimeError("Call reset before generate.")
        used_skill, used_cond = skill, cond
        if used_skill not in self.skills:
            # Unknown/disabled skills degrade to the loaded stand checkpoint;
            # they never select an arbitrary model by filename or "latest".
            used_skill, used_cond = "stand", None

        model, kind = self.skills[used_skill]
        cond_args = condition_args(used_skill, kind, used_cond, self.device)
        # The solver predicts the next 15 canonical frames from the retained
        # five-frame history.  Network-specific condition packing is confined
        # to condition_args so the rollout path is uniform across skills.
        pred_future = denoise(model, self.motion_state, cond_args)

        # Convert model features to SMPL-X, then place both body parameters and
        # joints in the accumulated episode/world reference frame.
        pred_transl_canon, pred_orient_canon, pred_body_pose = state_to_xb(pred_future)
        pred_body_pose = zero_neck_head_body_pose(pred_body_pose)
        pred_joints_canon = state_to_joints(pred_future)
        transl_world, orient_world = decanonicalize_xb(
            pred_transl_canon,
            pred_orient_canon,
            self.T_ref_accum,
            self.floor_y_accum,
        )
        xb_world = torch.cat([transl_world, orient_world, pred_body_pose], dim=-1)

        joints_world = decanonicalize_joints(
            pred_joints_canon, self.T_ref_accum, self.floor_y_accum
        )
        last5_vel_canon = pred_future[-5:, VEL_START : VEL_START + VEL_DIM]
        R_old = self.T_ref_accum[:3, :3]
        # Recanonicalize the final five frames to become the next call's
        # history.  Updating T_ref and floor_y here keeps sequential actions
        # continuous without growing an episode-length tensor on the GPU.
        new_state, T_ref_new, floor_y_new = canonicalize_from_world(
            transl_world[-5:],
            orient_world[-5:],
            pred_body_pose[-5:],
            joints_world[-5:],
            last5_vel_canon,
            R_old,
            ref_frame=4,
        )
        self.motion_state = new_state.unsqueeze(0).to(self.device)
        self.T_ref_accum = T_ref_new
        self.floor_y_accum = floor_y_new
        self.current_pelvis_y = float(transl_world[-1, 1].item())
        self.step_index += 1

        return GeneratedMotion(
            xb_world_75=xb_world.detach().cpu().numpy().astype(np.float32, copy=False),
        )


__all__ = [
    "GeneratedMotion",
    "MotionSkillRunner",
    "USE_SKILLS",
]

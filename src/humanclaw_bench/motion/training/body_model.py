"""Neutral-SMPL forward kinematics required only while building AMASS chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SMPLForwardKinematics(nn.Module):
    """Expose the 24 SMPL joints used by HumanClaw's 219-D motion state."""

    def __init__(self, model_path: str | Path, num_betas: int = 10) -> None:
        """Load a neutral SMPL NPZ through the optional ``smplx`` package."""

        super().__init__()
        import smplx as smplx_library
        from smplx.utils import Struct

        archive = np.load(Path(model_path), allow_pickle=True)
        values = {key: archive[key] for key in archive.files}
        if "kintree_table" in values:
            values["kintree_table"] = values["kintree_table"].astype(np.int64)
        self.body_model = smplx_library.SMPL(
            model_path="",
            data_struct=Struct(**values),
            num_betas=num_betas,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        )
        self.body_model.eval()
        self.num_betas = int(num_betas)

    @torch.no_grad()
    def forward(
        self,
        translation: torch.Tensor,
        global_orientation: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run neutral-body FK and return ``[batch, 24, 3]`` joints."""

        batch_size = translation.shape[0]
        if betas is None:
            betas = torch.zeros(
                batch_size,
                self.num_betas,
                device=translation.device,
                dtype=translation.dtype,
            )
        if body_pose.shape[-1] == 63:
            body_pose = torch.cat(
                [
                    body_pose,
                    torch.zeros(
                        batch_size,
                        6,
                        device=body_pose.device,
                        dtype=body_pose.dtype,
                    ),
                ],
                dim=-1,
            )
        output = self.body_model(
            transl=translation,
            global_orient=global_orientation,
            body_pose=body_pose,
            betas=betas,
        )
        return output.joints[:, :24]


__all__ = ["SMPLForwardKinematics"]

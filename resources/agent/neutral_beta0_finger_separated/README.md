<div align="right"><a href="README_zh.md">中文</a></div>

# Optional finger-separated humanoid

This directory contains an optional neutral SMPL-X beta=0 humanoid. Each of
the 30 finger links has its own convex-hull STL for both rendering and
collision. The URDF, `shift.npy`, and all 55 referenced meshes are bundled, so
the runtime does not need the source SMPL-X model.

The reported HumanClawBench results use
`resources/agent/neutral_beta0_handmerged`, not this asset. Independent finger
collisions change contact physics and collision metrics. Select this variant
only for a new experiment and label the resulting metrics accordingly.

Use it with any rollout command:

```bash
humanclaw-bench rollout \
  --profile paper_fullval_v1 \
  --agent-asset finger-separated \
  --model-config configs/models/my_model.json \
  --scene-id <scene-id> \
  --episode-index 0
```

Omitting `--agent-asset` preserves the complete paper profile unchanged.

The original runtime asset contains 57 files totaling 492,436 bytes. Its
URDF SHA256 is
`caf3be74ee250469b20a5e7fcb18a2f0959854a6af1e337ba00f8d85f559059d`;
the full identity is recorded in `metadata.json`.

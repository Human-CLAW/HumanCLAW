<div align="right"><a href="README.md">English</a></div>

# 可选分指 Humanoid

本目录提供一个可选的 neutral SMPL-X beta=0 humanoid。左右手共 30 个 finger
link 均有独立的 convex-hull STL，同时用于 rendering 与 collision。URDF、
`shift.npy` 和全部 55 个引用 mesh 已完整包含，runtime 不需要源 SMPL-X model。

HumanClawBench 论文报告结果使用的是
`resources/agent/neutral_beta0_handmerged`，不是本资产。独立手指 collision
会改变 contact physics 和 collision metric；请只在新实验中显式选择本版本，
并相应标注产生的指标。

任何 rollout 命令都可以选择它：

```bash
humanclaw-bench rollout \
  --profile paper_fullval_v1 \
  --agent-asset finger-separated \
  --model-config configs/models/my_model.json \
  --scene-id <scene-id> \
  --episode-index 0
```

省略 `--agent-asset` 时，完整 paper profile 保持不变。

原始 runtime asset 共 57 个文件、492,436 bytes。URDF SHA256 为
`caf3be74ee250469b20a5e7fcb18a2f0959854a6af1e337ba00f8d85f559059d`；完整
identity 记录在 `metadata.json`。

<div align="right"><a href="ASSETS.md">English</a></div>

# Assets 与 checkpoints

## 仓库内包含的内容

仓库包含一个 Habitat split header 和 41 个 scene shard。Header 定义六个
category 映射；41 个 shard 包含全部 1,218 个 Find/Nav/Interact episode。
仓库还包含 beta-zero、hand-merged 的论文默认运行时人体、一个可选分指人体、
natural seed state、稳定后的 environment metadata 和 provenance 记录。可选
人体为 30 个 finger link 分别提供 visual/collision mesh，必须通过
`--agent-asset finger-separated` 显式选择；新增 contact 会改变 physics 和
collision metric。Motion Jerk 使用的 66 个
pelvis-relative neutral-joint 常数位于
`resources/metrics/smpl_neutral_body22.json`；运行评测既不需要也不会重新
分发完整 SMPL archive。版本化 planner/verifier prompt 是 `agent/` 下普通、
可直接检查的 Python module。仓库不包含预计算的模型结果。

`resources/benchmark/val100.json` 是历史五个 scene、100 个 episode 开发
子集的索引。它不重复 scene 或 episode payload；每一行都指向同一份 canonical
full-validation shard。该列表从归档的 `val_s5_e20_seed20260530.json`
manifest 恢复，来源 SHA-256 为
`6182cc3a224353e34a54bd0946fd7a9639eb9bc4e0d6278813bde84cfc5a43b7`；
全部 100 个 scene/episode/category 条目以及 seed/category 统计均与来源一致。
使用 `humanclaw-bench run --episodes val100` 选择它。

每个 episode 都直接保存 `start_position`、`start_rotation`、`init_offset`
和 `init_yaw`。Provenance 文件
`spawn_repair_history_20260806_v2.csv` 列出 308 个修正 episode 各自最终人体
坐标来自哪一轮验证。它不是运行时输入。合并规则见
[`SPAWN_REPAIRS_zh.md`](SPAWN_REPAIRS_zh.md)。

运行 `humanclaw-bench assets` 可验证 deterministic file/tree hash。对于目录
asset，verifier 除 SHA-256 外还检查文件数和总大小。

## 外部 HSSD 数据

官方 HSSD stage 和 object mesh 受其自身访问和再分发条款约束，因此保持为
外部数据。Full-val benchmark 引用 41 个 scene。使用以下命令适配一份有
权限的 HSSD val 安装：

```bash
humanclaw-bench prepare-hssd --hssd-root /path/to/hssd-hab
```

`resources/hssd/humanclaw-hssd-val41/` 包含 41 个 scene 描述、14,537 个可
直接检查的 per-instance physics config，以及官方数据中没有的 1,693 个
per-instance baked mesh 的不可变 manifest。mesh 通过 gated Hugging Face
asset 分发，压缩大小 79.8 MiB，解压逻辑大小 176 MiB。
`asset_requirements.json` 按大小和 SHA-256 固定全部官方及补充 asset。生成的
`data/humanclaw-hssd-val41/` 通过 symlink 复用官方目录与校验后的 HF cache；
补充 mesh 用于 baked scale 修正和精确的 render-mesh collision geometry。

默认 `prepare-hssd` 命令会自动下载固定归档。离线机器可通过 `--supplement`
传入已经下载的归档。

如需准备到其他目录，传入 `--output`，随后在 `run` 命令中通过
`--scene-dataset-config` 指定该目录下的
`hssd-hab.scene_dataset_config.json`。

## 外部 motion data

仓库不再分发 AMASS、BABEL 或 neutral SMPL model，只在
`resources/motion/training/manifests/` 提供八份透明的训练 CSV list（合计
1.67 MB）。每行仅标识相对 AMASS chunk pickle 及其 source index，不包含
motion array。`resources/motion/training/segments/` 另提供人工 reviewed
BABEL interval，以及最终实际使用 chunk 的首尾秒/毫秒时间；每个 segment 的
chunk 数之和与机器训练 list 完全一致。用户应在本地构建 20-frame corpus，再按
[motion training 说明](../src/humanclaw_bench/motion/training/README_zh.md)使用这些
list。

## 外部 motion weights

`resources/weights/paper_fullval_v1.json` 是唯一权威 manifest。它固定一个
canonical MotionDiT state、两个可确定重建的 numerical base variant，以及
八个从准确 step 1,500,000 选出的 control-only skill state，不会解析
`latest`。

九个 inference 文件通过 `HumanCLAW/HumanCLAW` Hugging Face model repository
中的 `HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz` 发布；归档
SHA-256 为
`3b3c0c1b232af4c462301655de909a4bc54fd4756bcd42c22e7965fccb650667`。

| Skill | Network | Conditioning |
|---|---|---|
| walk_forward | `WalkForwardCtrlDiTFourierXZYaw` | Fourier x/z/yaw，6 frequencies |
| side_walk | `SideWalkCtrlDiTFourier` | Fourier，6 frequencies |
| step_back | `WalkForwardCtrlDiTFourier` | Fourier，6 frequencies |
| turn | `TurnCtrlDiT` | MLP |
| step_climb_up | `StepClimbUpCtrlDiT` | MLP |
| step_climb_down | `StepClimbDownCtrlDiT` | MLP |
| stop | `NonCondCtrlDiT` | 无 |
| sit | `SitCtrlDiT` | MLP |

Base architecture 的输入/输出维度为 219，hidden size 512，10 层，8 个 head，
MLP ratio 2，5 个 history frame，15 个 future frame，以及 30 个 flow
evaluation step。`walk_forward`、`turn`、`stop`、`sit` 使用其归档
checkpoint 内嵌的准确 BF16-roundtrip variant；另外四个使用 canonical FP32
tensor。前者是后者确定性舍入后的副本，并不是单独训练的 MotionDiT。Weight
目录中的 README 记录了审计过程，以及为什么保留两种 numerical variant 才能
复现论文评测。

原始九个文件占 3,227,420,338 bytes，因为每个 ControlNet 都重复 frozen base，
而 base Lightning checkpoint 还包含 training state。Inference-only 集合为
1,135,639,268 bytes。Loader 会逐 tensor 重建八个归档 `model_state` 映射，
并在内存中让四个 skill 共享各自的 base variant。

任何 checkpoint 在 unpickle 前都会先完成验证。这能保证可复现性，并减少误载
非预期 artifact 的风险，但不能让不可信的 pickle 文件变安全。Weight 只能从
maintainer 批准的分发渠道获取。

## 再分发检查

不要因为 scene data、body asset 或 weight 出现在本地工作目录中，就直接上传。
发布前必须确认第三方条款并补充 notice；release checklist 将此项视为阻塞条件。

<div align="right"><a href="README.md">English</a></div>

# Motion training

本目录包含 HumanClaw motion generator 的可选训练部分。Evaluation 不会 import
它。Trainer 直接复用 rollout 使用的 `humanclaw_bench.motion.networks`，因此
不会出现第二套 model implementation 与 runtime 逐渐不一致的问题。

Release 包含代码和透明的数据索引，但不分发 AMASS、BABEL 或 neutral SMPL
model；请遵循各自 license 获取这些资产。

## 包含内容

```text
motion/training/
├── build_chunks.py             AMASS -> per-sequence [N,20,219] pickle tree
├── body_model.py               neutral-SMPL forward kinematics
├── datasets.py                 base corpus 与 manifest-selected skill data
├── flow.py                     linear flow-matching target
├── runtime.py                  fast kernel 与 resumable checkpoint
├── train_base.py               concat-history MotionDiT trainer
├── train_control.py            八个 skill 共用的唯一 trainer
└── curation/
    ├── export_skill_manifest.py  legacy reviewed subset -> 小型 CSV list
    └── filter_manifest.py        不复制数据的显式数值筛选

resources/motion/training/
├── paper_training_v1.json      八个 skill 共用的训练配置
└── manifests/*.csv             每个 skill 的精确 source chunk
```

八个 ControlNet 使用同一个 trainer 和同一组 optimization setting，只替换
skill network、condition 与 manifest。

## 安装

在仓库根目录运行：

```bash
python -m pip install -e '.[training]'
```

论文规模训练需要 CUDA PyTorch。构建 chunk 还需要 `smplx` 和 neutral SMPL
NPZ；小规模检查也可在 CPU 上预处理。

## 构建 20-frame corpus

AMASS sequence 的原始帧率不同，并使用 Z-up world。Builder 执行与已发布模型
一致的转换：

1. 以约 30 Hz 选择 21 个 source frame；
2. 在 sequence 原始帧率上以五帧为 stride 滑动 source window；
3. 从 Z-up 转为 Y-up，并运行 neutral-SMPL forward kinematics；
4. 通过 floor-projected hip frame，以 history frame 4 为参考做 canonicalization；
5. 拼接 75 维 body parameter、72 维 joint coordinate 和 72 维单帧 joint
   velocity；
6. 最后一个 source frame 没有 forward velocity，因此将其丢弃。

结果为 `[20, 219]`：五个 clean history frame 和十五个 future frame。

```bash
python -m humanclaw_bench.motion.training.build_chunks \
  --amass-root /path/to/amass \
  --smpl-model /path/to/SMPL_NEUTRAL.npz \
  --output-root /path/to/amass_chunks_yup_v3 \
  --gpus 8 --workers 32
```

输出按 AMASS 目录结构镜像，每个 source sequence 对应一个 pickle。命令可续跑：
已有 output pickle 不会被重写。

## Curated skill list

Skill corpus 是指向 base chunk tree 的 CSV 索引，不重复保存 motion array。
每行包含 `rel_pkl` 与 `source_chunk_idx`；Side Walk 与 Sit 还保存 scalar
condition，避免重建时的数值偏差。八份 list 合计仅 1.67 MB；仓库不复制任何
受 license 约束的 AMASS motion array。

| Skill | 行数 | Training condition |
|---|---:|---|
| Walk Forward | 16,023 | 最后一帧 canonical pelvis x/z 与 body yaw |
| Side Walk | 2,012 | 最后一帧 canonical lateral displacement |
| Step Back | 2,294 | 最后一帧 canonical pelvis x/z |
| Turn | 6,773 | 有符号 final body yaw |
| Step Climb Up | 2,086 | chunk 内最大 toe height/horizontal separation |
| Step Climb Down | 2,094 | chunk 内最大 toe height/horizontal separation |
| Stop | 4,011 | 无数值 condition |
| Sit | 5,912 | accepted segment 的最低 pelvis height |

这些 list 已编码完成后的人工 review。训练 release skill 时直接使用它们，不需要
review video 或复制后的 subset tree。`curation/filter_manifest.py` 可为新实验
显式设置 yaw、translation、direction 和 lateral-motion threshold；它只写新
list，不修改或复制 base chunk data。

## 训练 base MotionDiT

```bash
python -m humanclaw_bench.motion.training.train_base \
  --chunk-root /path/to/amass_chunks_yup_v3 \
  --output outputs/motion_base
```

Base 的 architecture 为 219 维输入/输出、hidden size 512、十个 DiT
block、八个 head、五个 history frame 和十五个 predicted frame。使用 batch
size 512、AdamW、learning rate `1e-4`，训练 135 epochs（109,620 optimizer
iterations）。

重构后的 trainer 按指定 interval 写 resumable checkpoint，并导出
`motion_dit.pt`。每个 numbered checkpoint 只写一次；`checkpoints/last.pt` 是
relative link，从而避免旧版重复写数百 MB 的 checkpoint I/O。

## 训练一个 ControlNet

只需要五个路径/选择；architecture 与论文 hyperparameter 来自 profile：

```bash
python -m humanclaw_bench.motion.training.train_control \
  --skill walk_forward \
  --base-checkpoint outputs/motion_base/motion_dit.pt \
  --chunk-root /path/to/amass_chunks_yup_v3 \
  --output outputs/walk_forward
```

八个 ControlNet 均使用 frozen base MotionDiT、batch size 2,048、AdamW 和
learning rate `3e-4`，每个训练 1.5M iterations。这些共用参数只在
`paper_training_v1.json` 中记录一次。

每个 run 输出：

```text
config.json                 实际 portable configuration
train_log.jsonl             稀疏 progress record
checkpoints/stepXXXXXXXX.pt resumable full training state
checkpoints/last.pt         relative link，不是第二份 tensor copy
control.pt                  inference-only control branch
```

## 重建或扩展 list

将 reviewed mirrored subset 转为 list，而不再次复制其 array：

```bash
python -m humanclaw_bench.motion.training.curation.export_skill_manifest \
  --skill sit \
  --subset-root /path/to/reviewed_sit_subset \
  --expected-samples 5912 \
  --output /tmp/sit.csv
```

使用 pickle 是因为它是历史 chunk 格式。只能把 AMASS chunk 与 legacy subset
pickle 当作可信本地 artifact；pickle 不适合加载不可信下载。

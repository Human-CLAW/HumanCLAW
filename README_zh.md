<div align="right"><a href="README.md">English</a></div>

# HumanClawBench

HumanClawBench 在 41 个 HSSD scene、1,218 个 find–navigate–interact episode
中，将 vision-language model 作为 full-body agent 进行评测。

Release assets：

- Code：<https://github.com/Human-CLAW/HumanCLAW>
- Motion weights（`paper_fullval_v1`）：
  <https://huggingface.co/HumanCLAW/HumanCLAW>
- HSSD supplement（gated dataset）：
  <https://huggingface.co/datasets/HumanCLAW/HumanCLAW-HSSD>

## 快速开始

完成[安装](#安装)、[准备-HSSD](#准备-hssd)和
[安装 motion weights](#安装-motion-weights)后，复制模型接口模板，并填写
served model name 和 endpoint：

```bash
cp configs/models/vllm_openai_compatible.json my_model.json
```

先运行一个 episode 验证安装，再选择固定 100-episode 子集或完整 validation
split：

```bash
# 一个确定性的 smoke episode。
humanclaw-bench run --episodes one --model-config my_model.json \
  --gpus auto --output outputs/smoke

# 固定的小型 validation 子集。
humanclaw-bench run --episodes val100 --model-config my_model.json \
  --gpus auto --workers-per-gpu 1 --metrics --output outputs/val100

# 完整的 1,218-episode 评测。
humanclaw-bench run --episodes fullval --model-config my_model.json \
  --gpus auto --workers-per-gpu 1 --metrics --output outputs/fullval
```

`--gpus auto` 严格使用 `CUDA_VISIBLE_DEVICES` 暴露的设备；若未设置该变量，
则使用检测到的全部 GPU。若 VLM server 运行在本机，请在启动 server 时为它
保留 GPU，并在这里仅传入剩余的 evaluation GPU。添加 `--video` 会保存同步的
ego/exo MP4；video 与 metrics 相互独立。不开启 `--metrics` 时，不会进行
semantic rendering、contact query 或任何 metric accumulation。

详细信息见[评测流程](docs/ARCHITECTURE_zh.md)、
[指标定义](docs/METRICS_zh.md)、[视频工具](docs/VIDEOS_zh.md)和
[模型接口约定](docs/MODELS_zh.md)。

## 仓库内容

```text
configs/                     rollout 与 model-adapter 配置
patches/habitat-sim/         必需的 Habitat-Sim patch
resources/benchmark/         固定 1,218-episode split 和透明的 val100 索引
resources/hssd/              HumanClaw scene 与 object 配置
resources/agent/             humanoid runtime assets
resources/seeds/             确定性的初始 humanoid state
resources/weights/           外部 checkpoint manifest
src/humanclaw_bench/
  agent/                     planner、verifier、prompts、action schema
  benchmark/                 episode 加载
  envs/                      Habitat task/runtime 集成
    half_physics/
      hp.py                  唯一的 production Half-Physics controller
      humanclaw.physics_config.json  Bullet scene/default 设置
  evaluation/evaluator.py    唯一的 rollout loop
  evaluation/trajectory.py   紧凑 replay bundle writer
  evaluation/metrics/        论文指标定义与聚合
  evaluation/video.py        直接写入 MP4 的 streaming
  rendering/                 延迟渲染与 ego/exo/reasoning 合成
  motion/                    motion generation 与可选 training 工具
  vlm/                       模型 transport
```

仓库不包含官方 HSSD mesh、motion weight、受许可约束的 motion dataset、
provider credential 或模型 rollout 结果。仓库包含 motion training 实现与
每个 skill 的精确 source-chunk list，详见
[`src/humanclaw_bench/motion/training/README_zh.md`](src/humanclaw_bench/motion/training/README_zh.md)。
1,693 个 HSSD per-instance baked mesh 作为单独版本化的 gated Hugging Face
asset 发布；仓库只保留其精确文件名、大小和 SHA256 manifest。仓库包含
inference、replay recording 和 metric implementation。

## 安装

完整 rollout 需要 Linux、Python 3.10+、支持 CUDA 的 PyTorch、带 Bullet 且
应用过补丁的 Habitat-Sim、有权限访问的 HSSD-Hab val 数据、HumanClaw motion
checkpoint，以及一个 VLM endpoint 或 queue worker。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rollout,test]'
```

Video 优先使用系统 `ffmpeg`。如需安装打包的 fallback：

```bash
python -m pip install -e '.[video]'
```

Motion training 是可选功能，并与 evaluation 依赖隔离：

```bash
python -m pip install -e '.[training]'
```

在固定 revision 构建 Habitat-Sim，并应用仓库内 patch：

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout acbe6f4922e68145e401e55c30f9dfea460a3f24
git submodule update --init --recursive
git apply --check /absolute/path/to/HumanCLAW/patches/habitat-sim/humanclaw_halfphysics.patch
git apply /absolute/path/to/HumanCLAW/patches/habitat-sim/humanclaw_halfphysics.patch
python -m pip install -r requirements.txt
python setup.py build_ext --inplace --headless --with-cuda --bullet
python -m pip install -e .
python -m pip install -e build/deps/magnum-bindings/src/python
```

经过干净机器完整验证的构建流程，以及精简 CUDA toolkit、损坏的 `ccache` 和
`libgomp.so.1` 问题的处理方式，见
[`patches/habitat-sim/README_zh.md`](patches/habitat-sim/README_zh.md)。

## 准备 HSSD

原始 HSSD val 下载应包含：

```text
/path/to/hssd-hab/
├── hssd-hab.scene_dataset_config.json
├── objects/
├── stages/
└── semantics/
```

将它适配为 HumanClawBench 数据：

```bash
hf auth login  # gated supplement 只需登录一次
humanclaw-bench prepare-hssd --hssd-root /path/to/hssd-hab
```

该命令按大小和 SHA-256 验证每个 mesh，将官方 HSSD asset 与固定版本的 1,693
个 HumanClaw 补充 mesh 合并，复制 HumanClaw scene 和 per-instance object
JSON，并 symlink mesh、stage 和 semantic 文件。首次运行会从 gated
`HumanCLAW/HumanCLAW-HSSD` dataset 下载 79.8 MiB 压缩包，并将校验后的解压
内容保存在 `~/.cache/humanclaw-bench/assets/`。这样会保留 baked scale 修正，
以及以精细 render mesh 作为 collision mesh 的配置；不会静默退回官方的粗略
collider。原始 HSSD 安装不会被修改。默认输出为
`data/humanclaw-hssd-val41/`。

如需准备到其他位置：

```bash
humanclaw-bench prepare-hssd \
  --hssd-root /path/to/hssd-hab \
  --output /path/to/humanclaw-hssd-val41
```

对于离线机器，可传入已下载并转移过来的 HF 归档：

```bash
humanclaw-bench prepare-hssd \
  --hssd-root /path/to/hssd-hab \
  --supplement /path/to/humanclaw-hssd-val41-supplement-v1.tar.gz
```

运行 episode 时，通过 `--scene-dataset-config` 传入该目录中的
`hssd-hab.scene_dataset_config.json`。

Benchmark episode 已是最终版本。每个 episode 都保存 Habitat start pose 和
HumanClaw `init_offset`/`init_yaw`；运行时不会应用 spawn-repair overlay。

快速开发可向 `humanclaw-bench run` 传入 `--episodes val100`。它选择
`resources/benchmark/val100.json` 中透明列出的固定五个 scene、100 个
episode。论文报告仍使用全部 1,218 个 episode。

## 安装 motion weights

从 HumanCLAW Hugging Face model repository 下载 inference-only 归档，并在
当前源码目录的父目录解压：

```bash
cd ..
hf download HumanCLAW/HumanCLAW \
  HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz \
  --local-dir .
tar -xzf HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz
cd HumanCLAW
```

将分发的 checkpoint 放到 `resources/weights/paper_fullval_v1.json` 固定的路径：

```text
weights/paper_fullval_v1/
├── README.md
├── base/motion_dit.pt
└── skills/
    ├── walk_forward.pt
    ├── side_walk.pt
    ├── step_back.pt
    ├── turn.pt
    ├── step_climb_up.pt
    ├── step_climb_down.pt
    ├── stand.pt
    └── sit.pt
```

这些是 inference-only state，不是原始 trainer checkpoint。Release 移除了
optimizer/trainer state 和重复 frozen base，同时准确重建每个被评测 model
tensor。FP32/BF16 base-variant 关系的审计见
[`weights/paper_fullval_v1/README_zh.md`](weights/paper_fullval_v1/README_zh.md)。

验证仓库 asset 和 weights：

```bash
humanclaw-bench assets
humanclaw-bench assets --weights-root weights/paper_fullval_v1
```

## 配置 VLM

对于 vLLM 或其他 OpenAI-compatible server，复制
`configs/models/vllm_openai_compatible.json`，设置 model、endpoint 和
model-specific `max_tokens`：

```json
{
  "backend": "openai_compatible",
  "model": "served-model-name",
  "base_url": "http://127.0.0.1:8100/v1",
  "api_key_env": "OPENAI_API_KEY",
  "max_tokens": 4096,
  "temperature": 0.0,
  "response_format": {"type": "json_object"},
  "extra_body": {}
}
```

本地 vLLM endpoint 不需要 key；adapter 会提供 OpenAI-compatible server
需要的 placeholder。远程 endpoint 请导出对应环境变量，不要把 credential
写入 JSON。

对于持有 credential 的外部 worker，使用
`configs/models/filesystem_queue.json`。Adapter 将 request 原子地放入
`<queue_dir>/pending/<call_id>/`；worker 返回
`<queue_dir>/done/<call_id>/response.json`，至少包含：

```json
{"content": "{...model JSON response...}"}
```

Request image 和 queue JSON 是 transport file，不是 rollout artifact；读取
response 后会被删除。

## 运行一个 episode

```bash
humanclaw-bench run \
  --episodes one \
  --profile paper_fullval_v1 \
  --model-config my_model.json \
  --scene-id 102343992 \
  --episode-id 0 \
  --object-category bed \
  --gpus 0 \
  --output outputs/example
```

Planner 使用 prompt v4，verifier 使用 verifier v3。最终执行的 motion action
始终是 verifier-final action。Stop 会结束 episode；否则 rollout 最多运行
100 个 environment step。

Planner 和 verifier 遇到 provider call 或 JSON parsing 失败时，都会在当前
simulator state 原地重试，最多 5 次。Planner 连续 5 次失败后执行一次
`Walk<forward><slow>`，再根据新 observation 重新规划；verifier 连续 5 次失败
则接受有效的 planner proposal。两种情况都不会从头重跑 episode。

默认 rollout 生成：

```text
outputs/example/
└── <scene_id>_ep<episode_id>_<category>/
    └── rollout_00/
        ├── step000_percept_mid_low.json
        ├── step000_verifier.json       # 仅在 verifier 确实被调用时存在
        ├── step001_percept_mid_low.json
        ├── ...
        ├── trajectory_before.npz
        ├── trajectory_after.npz
        └── replay_manifest.json
```

每个 step JSON 只包含该逻辑 stage 的 prompt 和最终 parsed response。仅当该
stage 的全部尝试都失败时，才添加 `error` 字段。

`trajectory_before.npz` 只保存真正传给 HalfPhysics 的 world-frame
`xb_world_75` chunk、action/step boundary、fps，以及启动 forward replay 所需
的准确初始 humanoid/dynamic-object pose 和 velocity；不会重复保存未使用的
219-D internal motion feature。`trajectory_after.npz` 保存相应的 simulated
humanoid pose，以及每个 physics frame 中所有 dynamic object 的 position 和
rotation。`replay_manifest.json` 固定 episode、physics parameter、关键 asset
hash 和两个 NPZ hash。不会再写一份重复的 `trajectory.npz`。

### 保存视频

```bash
humanclaw-bench run \
  --episodes one \
  --model-config my_model.json \
  --gpus 0 \
  --video \
  --output outputs/example_video
```

这会增加 `ego.mp4` 和 `exo.mp4`。二者都包含 post-reset frame，以及之后全部
30 fps simulated motion frame。Frame 直接送入 H.264，不创建临时图像目录，
也不执行第二次 encoding。Video 模式不会开启 semantic rendering、contact
query 或 metrics。

### Rollout 后单独渲染保存的 trajectory

完整 rollout 的 `trajectory_after.npz` 已保存每帧 post-physics humanoid pose
和所有 dynamic object pose，因此视频不需要再次 physics replay。Delayed
renderer 只加载一次 scene，逐帧恢复这些 pose，更新 ego/exo camera，并把 RGB
直接送入两个 MP4 encoder：

```bash
humanclaw-bench render \
  --rollout-dir outputs/example/102343992_ep0_bed/rollout_00 \
  --output-dir outputs/example_rendered
```

它只写 `ego.mp4`、`exo.mp4` 和 `render_report.json`，不会调用 VLM、motion
generator、physics step、contact query 或 semantic render。Habitat 仍需初始化
scene 和 articulated-object runtime，以便赋 pose 和 rasterization，但 simulation
time 不会前进。默认 `veryfast` H.264 preset 优先吞吐量；使用
`--preset medium --crf 18` 可匹配在线 encoder 设置。

用相互隔离的 Habitat/OpenGL process 渲染完整输出树：

```bash
humanclaw-bench render-batch \
  --input-root outputs/fullval \
  --output-root outputs/fullval_rendered \
  --max-parallel 8 \
  --devices 0,1
```

将已有 ego/exo stream 与原样保存的逐步 model text 合成：

```bash
humanclaw-bench compose-video \
  --rollout-dir outputs/example/EPISODE/rollout_00
```

完整输出树可使用 `compose-video-batch` 并行处理。该展示过程不加载 Habitat、
physics、motion model 或 VLM。详见 [docs/VIDEOS_zh.md](docs/VIDEOS_zh.md)。

输出会保留每个 rollout 的相对目录。Habitat/OpenGL context 无法在线程间共享，
所以使用 process isolation。对于修改后的 trajectory，可用 JSONL manifest
代替 `--input-root`；每一行包含 `episode_key`、`rollout_dir` 和可选的
`trajectory_path`。

### 计算论文指标

```bash
humanclaw-bench run \
  --episodes one \
  --model-config my_model.json \
  --gpus 0 \
  --metrics \
  --output outputs/example_metrics
```

这只增加一份 `metrics.json`。Rollout 中，每个 decision 的 semantic observation
复用于 FindSR；每个 physics frame 的一次 contact query 由 Collision、InteractSR
和 disturbance 共享。Motion Jerk 在 episode 结束时读取已经保存的 pre-physics
trajectory。不执行 contact replay，也不保存中间 metric artifact。

要汇总已经完成的任意输出树，包括包含多个分布式 shard 的上层目录，运行：

```bash
humanclaw-bench metrics outputs/fullval
```

该命令递归读取每个 episode 的 `metrics.json`，打印论文主表、success variants、
collision body groups 和各项 denominator。它不调用 VLM，不运行 simulation、
rendering 或 replay，默认也不写文件。使用 `--json` 可打印完整的机器可读汇总；
增加 `--write-json` 则同时保存 `outputs/fullval/metrics_summary.json`。

报告字段与论文一致：

- FindSR 要求发送给 VLM 的同一张 ego image 中至少有 100 个 target semantic
  pixel，并且该 decision 的 `visible_state` 对 target 作出非否定确认。
  GeoFindSR 只使用 pixel 条件。
- NavSR@20cm 要求 active Stop，并且任意 body joint 到 target AABB 的最终最小
  3D distance ≤ 0.2 m。GeoNavSR@20cm 不要求 stop；NavSR@1m 要求 active
  stop 且 distance < 1 m。
- InteractSR 只适用于 bed/couch/toilet episode，要求至少一次 Sit、active stop，
  且 stop 前最后一段 motion 的最终帧存在 pelvis-to-target mesh contact。
  GeoInteractSR 检查是否有任意一次 Sit 的最后一帧满足该 mesh contact。
- Collision 检查 30 Hz realized post-physics pose，并统计高于 episode spawn
  floor 0.0205 m 的 fixed-geometry contact。Score 是发生 collision 的 motion
  decision step 比例。
- Disturbance 统计 humanoid 直接影响，或通过有时间顺序的 dynamic-object contact
  chain 间接影响的 dynamic object。Distance 是所有成功映射 affected object
  的 pooled path length。
- Motion Jerk 使用 generated pre-physics root-rigid trajectory、neutral 22-joint
  body、宽度 3 的 centered moving average，以及 30 fps 下 stride 8。准确的
  pelvis-relative neutral joint 常数包含在
  `resources/metrics/smpl_neutral_body22.json`。
- Cost 报告 decision step 和 provider token usage per step。隐藏 reasoning 不计入
  visible output。Provider 不返回 usage 时，会明确把字符数 fallback 标成
  estimated。

初始人体穿模会在 `metrics.json` 中标记用于诊断，但不排除任何 episode；
collision、disturbance 和 jerk 均使用完整的 1,218-episode split。准确 data
flow 见 [docs/METRICS_zh.md](docs/METRICS_zh.md)。

`--video` 和 `--metrics` 可以同时开启。此时每个 physics frame 的 RGB 与
semantic data 在同一次 rollout 中产生，metric 路径仍只写 `metrics.json`。

## 选择 episodes 和 GPUs

```bash
humanclaw-bench run \
  --episodes fullval \
  --model-config my_model.json \
  --gpus 0,1,2,3 \
  --workers-per-gpu 2 \
  --metrics \
  --resume \
  --output outputs/fullval
```

上面的命令最多并行八个独立 episode process，并在四张 GPU 上保持负载均衡。
请先使用每卡一个 worker；只有在显存能容纳多个 motion runtime 时再提高。
`--episodes` 接受 `one`、`val100`、`fullval` 或自定义 JSON episode-list path。
Metric 模式会在全部 episode 完成后写一份 `metrics_summary.json`。

## 验证 source release

```bash
humanclaw-bench config paper_fullval_v1
pytest -q
```

准备好 HSSD 后，可在安装了 Habitat-Sim 和 ffmpeg 的环境中运行真实 runtime
smoke：

```bash
PYTHONPATH=src python tests/runtime_habitat_smoke.py \
  --output /tmp/humanclaw_habitat_smoke
```

它使用一个固定 episode 和 25 个 stationary requested frame，不需要 VLM
credential 或 motion weight。它验证 scene/target 加载、HalfPhysics、contacts、
dynamic-object trajectory、两路 MP4 stream 和唯一的最终 metrics artifact。

更小的 controller-contract check 只需要 Habitat/Magnum binding，不需要 scene
data 或 renderer：

```bash
PYTHONPATH=src python tests/runtime_hp_contract.py
```

它直接验证四个 substep、substep 0/2 的 root x/z write、一次 root-angular
write、每帧 30-degree cap，以及 pre-limit PJSC target。

验证真实 saved motion 能否复现 post-physics trajectory：

```bash
PYTHONPATH=src python tests/runtime_forward_replay.py \
  outputs/<run>/<scene_id>_ep<episode_id>_<category>/rollout_00 \
  --max-steps 1
```

该检查恢复 saved human 和每个 dynamic object 的 pose/velocity，从
`trajectory_before.npz` 推进 Half-Physics，并逐帧对照
`trajectory_after.npz`。`--max-steps 0` 会检查完整 trajectory。

Asset、execution、metric 和 provider 约定分别见
`docs/ASSETS_zh.md`、`docs/ARCHITECTURE_zh.md`、`docs/METRICS_zh.md` 和
`docs/MODELS_zh.md`。

## License

本 release（代码、配置与随包 resources）以非商业许可
[CC BY-NC 4.0](LICENSE) 发布。Hugging Face 上分发的 motion weights 与 HSSD
supplement 采用相同许可。官方 HSSD 数据仍受其自身许可与访问条款约束。

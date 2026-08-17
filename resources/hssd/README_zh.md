<div align="right"><a href="README.md">English</a></div>

# HSSD val41 修改

`humanclaw-hssd-val41/` 包含 benchmark 专用的全部轻量配置与 manifest：

- `scenes/`：41 个 scene-instance JSON。其 `object_instances` 保存 template、
  transform、scale，以及 `STATIC`/`DYNAMIC` 状态。
- `objects/humanclaw/`：14,537 个 per-instance physics object config。
- `supplement.json`：官方 HSSD 中没有的 1,693 个 per-instance baked mesh 的
  不可变 HF 地址、归档 checksum 和逐文件记录。
- `hssd-hab.scene_dataset_config.json`：Habitat-Sim 数据集入口。
- `asset_requirements.json`：全部官方及补充 asset 的名称、大小和 hash。

`humanclaw-bench prepare-hssd` 在 cache 缺失时从 gated
`HumanCLAW/HumanCLAW-HSSD` dataset 下载固定 supplement，校验归档及每个 GLB，
复制明确列出的 JSON，并链接匹配的官方及补充 mesh、stage 和 semantic 文件。
补充 mesh 保留 baked instance scale 和精确 render-mesh collider；任何预期
文件或 hash 缺失时命令会失败，而不会静默替换为粗略的官方 collider。离线
机器可用 `--supplement` 传入本地归档或已解压目录。

场景 `104862384_172226319` 包含一项可通行性修正：无法打开的淋浴间
（object instance 166）被移到有效场景下方，避免从浴室起步的 agent 被困在
其中。该条目继续占用 index 166，只是为了保持其后所有 Habitat object ID
不变；在这个场景的全部 episode 中，该淋浴间均不参与渲染或碰撞。

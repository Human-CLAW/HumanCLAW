# HumanCLAW experiment pipeline

本阶段实现 HumanCLAW 的统一调用追踪、指标汇总和回放。Codex、Pi、
LIBERO-Pro 的适配，以及 feedback/state estimator、skill library 和自动裁剪，
留到后续实验。Klone 的环境准备与提交方式见 [Klone 指南](KLONE_EXPERIMENTS.md)。

## 运行一个实验

在项目根目录的 `.env` 中填写 `AZURE_OPENAI_API_KEY=你的密钥` 并保存；
新 checkout 可以从 `.env.example` 复制。项目会自动加载该文件，不再要求
先在同一个终端 export。已有环境变量优先于 `.env`，密钥文件不会提交到 Git。
更新依赖后即可先确认模型接口可用，无需 GPU、HSSD 或运动权重：

```bash
PYTHONPATH=src python -m humanclaw_bench api-check \
  --model-config configs/models/azure_gpt6_astra.json
```

该配置使用用户提供的 Azure 部署名称 `gpt-6-astra`、资源
`ai-thopo-9134`，从 `AZURE_OPENAI_API_KEY` 读取密钥。名称是部署标识，
不代表已经验证该资源上模型的可用性。
这里采用 Azure OpenAI v1 的 `openai_compatible` 路径，不需要 API version；
密钥仍属于 Azure。不要把 Foundry 的 `/models` 地址原样作为该接口的 base URL。
地址规则见 [微软 Azure v1 文档](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)。

`api-check` 只发送一个合成图像和 JSON 格式的测试请求，关闭 SDK 自动重试，
默认请求超时 60 秒（可用 `--timeout` 调整）。它不执行 benchmark，也不保存
轨迹或密钥；输出请求状态、JSON/图像检查结果、时延和 token 用量。预期颜色
只在本地校验，不在文本提示词中给出。`passed: true` 表示该次图文接口检查
通过，不能替代真实 rollout 验证；`usage: null` 表示用量未知。测试会产生真实
API 调用费用，但此项目不预设报价。SDK 重试与超时选项见
[OpenAI Python 文档](https://developers.openai.com/api/reference/python)。

完成仓库原有安装、HSSD 准备、运动权重准备和模型配置后：

```bash
humanclaw-bench run --episodes one --model-config my_model.json \
  --harness humanclaw_psv --trace --metrics --seed 42 \
  --gpus auto --output outputs/experiment_psv
```

`--trace` 记录实验；`--metrics` 开启原有 benchmark 评分。二者独立，未开启
metrics 时 task success 为未知。`--max-steps 2` 可用于有明确步数上限的 smoke
检查，但它改变任务预算，结果不能直接与完整 episode 比较。`--video` 可选。

| Harness | 输入与执行方式 |
| --- | --- |
| `humanclaw_psv`（默认） | 原有视觉 planner、history、条件触发 verifier，以及原有失败回退规则 |
| `bare_api` | 当前 ego 图像、任务说明和相同的原子动作空间；每步一个逻辑模型阶段，不使用 history 或 verifier |

Bare API 仍有图像编码、JSON 解析、动作转换和传输重试。JSON/传输错误最多
尝试 5 次，每次请求都独立计数；耗尽重试会结束运行。无效 action ID 会报错；
动作参数沿用公共转换器的默认值和限幅，转换前后记录在 `action.normalized`。
这是一个明确标记的实验条件，不能将原有 PSV 结果称为裸 API 结果。

同一组对比应固定模型配置、episode 清单、初始状态、随机种子、步数预算、
metrics/video 开关、GPU 分配和并发数。`--seed` 给 Python/NumPy/Torch 设置
随机种子；第 i 个 rollout 使用 seed+i。物理后端和远程模型仍可能存在非确定性。

## 数据组织与调用语义

每次运行创建新的 run ID，旧 trace 和其回放数据不会被下一次运行覆盖：

```text
outputs/experiment_psv/
  batch_attempts/batch_.../batch.json
  <scene>_ep<id>_<category>/rollout_00/
    metrics.json                       # 原有格式，启用 --metrics 时存在
    replay_manifest.json               # 原有最新一次 rollout 文件
    trajectory_before.npz
    trajectory_after.npz
    experiment_runs/run_.../
      run.json                         # 配置、来源、最终状态、benchmark 结果
      events.jsonl                     # 按事件追加并 flush
      summary.json                     # 可离线重新计算
      artifacts/                       # 图像、数组，按内容哈希去重
      replay/                          # 本次 attempt 独立的物理回放副本
        replay_manifest.json
        trajectory_before.npz
        trajectory_after.npz
  experiment_summary.json
```

每条事件包含 schema 版本、run/session/episode ID、step、时间戳、事件序号、
span ID 和 parent span ID。一次 decision 下可以关联：

```text
decision
  agent.act
    percept_mid_low / bare_api
      model.respond × 各次尝试
    verifier（实际执行时）
      model.respond × 各次尝试
  metrics.record_decision（启用时）
  walk_forward / turn / stand / ...
    motion.generate（非 Stop）
    environment.step
    metrics.record_motion（启用且非 Stop）
  history.update
```

- **模型**：完整消息、实际发送的图像、原始回复、供应商 usage、错误和时长。
- **动作工具**：注册、向模型展示、请求、拒绝/替换，以及执行 span 和结果。
  这些是 `structured_action_json` 环境技能，并非供应商原生 function calling。
- **内部模块**：planner、verifier、运动生成、history、metrics 和收尾的执行边界。
  未执行的可观测模块可显示零次；SDK 内部重试、供应商内部模块标为不可观测。
- **环境**：执行动作、前后 ego 画面、返回的身体/物体状态、奖励与终止信息。
  模型实际接收到的状态以模型消息为准；记录状态不会自动把它加入 planner prompt。

`run.json` 保存完整 profile、episode、模型配置、运行覆盖参数、软件版本、
Git revision 和源码内容哈希（包括新增文件），以及随机种子。凭据字段脱敏，
API key 不作为重放输入。图片和 NumPy 数组由 artifact 引用连接，不在 JSONL
反复嵌入大块二进制。现有物理回放还保存初始身体状态和物体状态及资产标识。

正常退出、异常和 KeyboardInterrupt 都会闭合记录。进程被强制终止时，已刷新的
事件仍可读取；未闭合的 span、缺少 run.end、损坏的末尾行会在校验和汇总中指出。
记录不是磁盘断电持久性的保证。每次 trace 的 replay 文件在收尾时归档；在此之前
被强制终止的 attempt 可能只有部分事件，不能承诺物理回放。

## 指标口径

| 维度 | 统计口径 |
| --- | --- |
| 任务表现 | 保留原有 Find/Nav/Interact 等 benchmark metrics；另记录明确派生的 task_success：交互 episode 用 interact_sr，其余用 nav_sr_20cm |
| 运行状态 | completed / failed / interrupted / running；completed 不代表任务成功 |
| Tokens | 仅从模型调用 span 的供应商 usage 累加，包含失败调用中已知的用量；缺失值为 null 并报告覆盖率 |
| 费用 | 依据显式提供的版本化费率估算；没有费率、用量不完整时总费用为 null，保留已知小计 |
| 时延 | run 墙钟时间；各类 span 的 inclusive 时间和区间并集，不能把嵌套 span 简单相加 |
| 工具/模块 | 按 kind + name 自动汇总注册、展示、请求、执行和终态次数，以及执行失败率、耗时分布、观测覆盖率 |

OpenAI 的 inclusive 输出 token 不再额外加 reasoning；Gemini 分开的可见输出和
reasoning 才合并。缓存 token 是输入的子集，单独费率只替换相应输入部分。
可选 `--pricing /path/to/rates.json`，对象必须包含你实际使用的费率：
`version`、`currency: "USD"`、`input_per_million`、
`cached_input_per_million`、`output_per_million`。不要把示例价格当作当前报价；
此工具不联网查价。如果有缓存折扣却没有缓存用量明细，该次输入费用也保持未知。

时延范围为 episode 执行及归档，包含追踪开销；模型/模拟器构造、排队启动和
离线汇总不包含在 run 时延中。该范围写入元数据。首 token 时延未测量。

单次运行自动写入的 `summary.json` 中，`capabilities.entries` 的每一行对应
一个工具或模块。现有调用计数保留，并增加以下字段；同名工具和模块分别统计：

| 字段 | 含义 |
| --- | --- |
| `failure_rate` | 执行失败次数 /（完成次数 + 执行失败次数），取值 0–1；无已知完成/失败的执行时为 null |
| `failure_rate_denominator` / `failure_rate_coverage` | 失败率的分母，以及该分母占已开始执行次数的比例 |
| `latency.inclusive_ms` / `latency.union_ms` | 有效闭合 span 的时长总和 / 同一次运行、同一名称的时间区间并集，单位毫秒 |
| `latency.mean_ms` / `p50_ms` / `p95_ms` / `max_ms` | 单次调用时长的平均值、线性插值分位数和最大值 |
| `latency.spans` / `closed_spans` / `incomplete_spans` / `invalid_spans` / `coverage` | 观测到的 span 数、具有有效起止时间的数量、缺起止事件的数量、起止时间无效的数量，以及有效计时覆盖率 |

请求拒绝与取消不算执行失败；中断、未闭合、未知终态单独计数，不并入失败率
分母。仅有结束事件但没有开始事件的 span 记为 `orphan_ends`，不虚增执行或
完成/失败次数。完成只表示该执行边界未报错，不表示动作取得进展或任务成功。
内部模块的曝光/请求若未埋点，应理解为不适用，不能把其零事件当作模型未选用。

耗时包含失败与中断调用的有效闭合区间；缺失或无效计时不会补成零时长样本。
没有有效样本时，平均值、分位数和最大值为 null，总和与并集为已观测小计 0；
需要结合计时覆盖率和 trace 校验结果解读。不同名称之间仍可能嵌套，不能将
各行的并集相加作为端到端耗时。

目录级 `experiment_summary.json` 在 `aggregate.capabilities.entries` 中
自动汇总所有已发现的 attempt。失败率按合并后的调用次数计算，耗时分位数
按所有有效调用样本重算，不平均各 run 的失败率或 P95。`union_ms` 是各 run
内部并集之和，不合并不同机器/进程的时钟，也不代表整个目录的墙钟时间。
`runs_with_entry`、`runs_with_execution` 和 `availability_runs` 保留观测范围；
某次运行缺少该能力的记录不能推断其零调用。样本仅在离线汇总时临时使用，
原始依据仍为 `events.jsonl`；不需要重新调用模型或执行 episode。

批量调度另保存本次请求的 episode 清单和子进程状态，因此初始化就失败、没有
生成 trace 的任务不会静默消失。对比 benchmark 应使用同一批清单，并同时报告
成功率、失败数和评分覆盖率。目录级汇总包含历史上的所有 attempt，适合审计，
不能把其中混合配置、重复尝试的平均数当作一个实验条件的 benchmark 成绩。

`--resume --trace` 需要匹配已完成且校验通过的实验 trace；仅存在旧轨迹/metrics
不够。配置无法确认匹配时重新执行，并保留旧 attempt。

## 离线查看与回放

以下 `RUN` 指向一个 `experiment_runs/run_...` 目录：

```bash
humanclaw-bench trace validate "$RUN"
humanclaw-bench trace inspect "$RUN"
humanclaw-bench trace summarize "$RUN" --write-json
humanclaw-bench trace summarize outputs/experiment_psv --write-json

# 读取时间线，不调用模型或模拟器。
humanclaw-bench replay "$RUN" --mode log

# 用保存的 motion 输入重跑物理过程，与保存的实际轨迹比较。
humanclaw-bench replay "$RUN" --mode physics --output-dir outputs/replay_check

# 使用已保存的 post-physics 位姿生成视频。
humanclaw-bench replay "$RUN" --mode render --output-dir outputs/replay_video

# 这是一次新的在线实验，会产生模型调用和新的费用。
humanclaw-bench rerun "$RUN" --model-config my_model.json --output outputs/rerun_01
```

log 回放和汇总在普通 CPU 环境运行。physics/render 需要原有 Habitat/Bullet、
场景和角色资产；physics 回放不调用 VLM 或运动生成网络。rerun 使用冻结的
episode/profile 和有效种子，显式重新提供模型配置，并写入新目录。凭据可重新
提供；改变模型/endpoint 等设置应作为新的对比条件运行。固定种子不能保证远程
API 返回相同答案；只有日志回放完整保留历史输出。

## 后续裁剪实验

首轮先收集完整的覆盖矩阵。零调用只能说明该批 episode 中没有观察到使用，
不能证明可安全删除。应先区分“未注册、未展示、不可观测、已展示但未选用”，
再形成裁剪候选；单独改动、重跑同一 episode/seed 清单，比较成功率、费用与
时延，并保留恢复工具等低频能力的专项任务。当前 pipeline 不自动删除工具。

<div align="right"><a href="METRICS.md">English</a></div>

# 论文指标

Metrics 是可选路径。`humanclaw-bench run --metrics` 为每个 episode 写一份
`metrics.json`，多 episode 运行还会写一份 `metrics_summary.json`。不开启
该 flag 时，不会创建 metric sensor，不会执行 contact query、target resolution
或 episode 结束后的指标分析。

## 打印已完成运行的指标

```bash
humanclaw-bench metrics outputs/fullval
```

该命令递归查找 `metrics.json`，因此输入既可以是普通 run，也可以是包含多个
分布式 shard 的上层目录。它无需加载 Habitat、motion model 或 VLM，直接打印
论文主表、success variants、collision body groups 和聚合计数。默认只读；
`--json` 打印完整 summary object，`--write-json` 还会额外写入
`<output-root>/metrics_summary.json`。

## 共享的 episode timeline

在 decision step *s*，evaluator 已经持有将发送给 VLM 的 ego RGB 图像。Metric
模式下，semantic sensor 使用同一个 camera transform 渲染，并在内存中保留
target pixel 数。随后将解析后的 planner response 与该计数配对，计算 FindSR。

非 stop action 会生成一段 30 Hz motion。每个 physics frame 在 post-physics
落地姿态上执行一次共享 contact query。Collision 读取 fixed-geometry contact；
Interact 读取 pelvis/target contact；disturbance 读取 human/dynamic 和
dynamic/dynamic contact。三个 recorder 更新紧凑状态后立即丢弃 contact rows。

Generated trajectory 和 post-physics trajectory 始终会为 replay 保存。Metric
模式在 episode 结束时只 materialize 一次这两组数组。Motion Jerk 读取 generated
array；collision 已经从共享 contact stream 在线累计完毕。随后把数组压缩成两份
trajectory NPZ。

## Find

每个解析到的目标实例都会获得一个临时 semantic ID。在 448×448 ego semantic
图像中，这些 ID 的并集达到至少 100 pixels 时，该 decision 具有客观可见性。

主观规则先按句子边界拆分 `visible_state`。当一句话提到 `target` 或 category
alias，且不包含以下否定词时，认为模型确认目标可见：`no`、`not`、`cannot`、
`can't`、`don't`、`isn't`、`aren't`、`n't`、`without`、`unable`、`none`。

- `GeoFindSR`：任意 decision 具有客观可见性。
- `FindSR`：同一个 decision 同时具有客观可见性和主观确认。

两个条件必须发生在同一个 decision step。

## Nav

终止时，evaluator 使用 pelvis、articulated root 和所有 humanoid link origin
组成 point set。它对每个 target instance 计算完整 3D point-to-world-AABB
distance，再在所有 point 和 target 中取最小值。

- `GeoNavSR@20cm`：最终距离 ≤ 0.2 m。
- `NavSR@20cm`：模型主动选择 Stop，且最终距离 ≤ 0.2 m。
- `NavSR@1m`：模型主动选择 Stop，且最终距离 < 1.0 m。

只有 planner/verifier 选择的 Stop 才是 active stop；达到 100-step 上限
不算。

## Interact

Interact 只在 bed、couch 和 toilet episode 上评测。Pelvis contact 必须匹配
准确解析出的 target object ID/handle/template；仅仅 AABB 接近不够。

- `GeoInteractSR`：至少一次 Sit action 的最后一个 physics frame 保持
  pelvis-to-target mesh contact。同一 motion chunk 中更早出现、但落地前消失的
  短暂 contact 不计入。
- `InteractSR`：至少执行过一次 Sit，agent 主动 stop，并且 stop 前最后一段
  motion 的最后一帧仍有 pelvis-to-target mesh contact。

## Fixed-geometry collision

Collision 检查 30 Hz 的 realized post-physics pose。它复用 Interact 和
disturbance 已经需要的同一次 discrete contact query；不会恢复 pose、插值
requested motion，也不会在 episode 结束时执行第二轮 contact pass。

在 spawn 处向下投射一条 ray，选择不高于站立人体最低点 5 cm 的最高表面，作为
该 episode 的 floor。若 fixed contact 的高度减去 floor 高度大于 0.0205 m，
则计为碰撞。Coll% 是以下 episode 内比例再跨 episode 取均值：

```text
发生过有效 fixed contact 的唯一 motion decision steps
-------------------------------------------------------
               唯一 motion decision steps
```

同样的 step set 分别统计 arm/hand、torso、leg/foot 和 head。

## Movable-object disturbance

当 dynamic rigid object 与 humanoid 接触时，它被标记为 directly affected。
从同一帧起以及之后的每一帧，affected 状态沿 dynamic-to-dynamic contact edge
传播。时间顺序确保未来的 contact 不会反向影响过去。

- `#Dtb/ep`：全部 episode 的 affected-object 数量均值。
- `dDtb(m)`：每个成功映射的 affected object 从首次受影响帧开始的 path
  length 总和，除以所有成功映射 object 的总数。

距离是 pooled object-level mean，不是 episode mean 的均值。

## Motion Jerk

Motion Jerk 读取生成的 pre-physics `xb_world_75`。它把 local body articulation
固定为 neutral 22-joint skeleton，只应用 root translation 和 global root
rotation。Joint position 先做宽度为 3 的 centered moving average，再以 30 fps、
stride 8 计算三阶 finite difference：

Neutral joint center 是固定 neutral SMPL source 中
`J_regressor @ v_template` 的前 22 行，并转为 pelvis-relative。得到的 66 个
meter-valued 常数公开保存在 `resources/metrics/smpl_neutral_body22.json`；
release 不从 simulation URDF 近似这些值，也不需要完整 body-model archive。

```text
P[t+3s] - 3P[t+2s] + 3P[t+s] - P[t]
------------------------------------- ,  s = 8
               (s/30)^3
```

一个 episode 的 score 是所有 joint 和有效 frame 的 vector magnitude 均值，
最终再对全部 episode 取均值。

## 初始穿模诊断

在第一帧 physics 运行前，evaluator 对准确的 reset-time human 和 dynamic-object
state 执行一次 discrete contact query。最大 penetration 定义为
`max(0, -contact_distance)`。是否超过 0.01 m 会保存在 `metrics.json` 中用于
诊断，但不会排除该 episode。Collision、disturbance、jerk、high-level success
和 cost 均使用完整的 1,218-episode full-validation split。

## Cost

Steps 指 VLM decision step，包括最后的 Stop decision。Input 和 visible
output token 总量优先使用 provider usage。隐藏 reasoning token 单独保存，不计入
visible output。聚合时用 token 总量除以 decision step 总数，与论文的 per-step
列一致。如果 provider 不返回 usage，该 episode 会明确标记为
`estimated_chars_div_4`；混合准确值和 fallback 的结果也会明确标记。
Provider error 或 JSON parse error 会在当前 state 原地重试，最多 5 次。
Planner 连续 5 次失败后执行一次 `Walk<forward><slow>` fallback；verifier
连续 5 次失败则接受 planner proposal。两种情况都会继续 episode，所有真实
attempt 仍计入 token accounting。

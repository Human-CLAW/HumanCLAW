<div align="right"><a href="SPAWN_REPAIRS.md">English</a></div>

# Humanoid spawn 修正

Habitat 标准字段 `start_position` 和 `start_rotation` 描述 episode 起始姿态。
HumanClaw 还需要 simulator-space 的 `init_offset` 和 `init_yaw` 来放置
articulated humanoid。原始起点中有一部分离 scene geometry 太近，或会产生
不稳定的初始状态，因此在发布前对这些人体坐标进行了验证和修正。

修正来自三轮有先后顺序的验证。保留 source revision 名称只是为了审计：

| 验证轮次 | Source revision | 检查的 episodes | 最终值发生变化 |
|---|---|---:|---:|
| 初始坐标修正 | `20260719-first-pass` | 308 | 308 |
| 1 cm clearance 精修 | `20260720-refined-1cm` | 308 | 38 |
| 最终定向修正 | `20260806-v2` | 38 | 1 |

后面的轮次只替换其明确包含的 episode key。最终 308 个修正 episode 中，
270 个值来自初始修正，37 个来自 clearance 精修，1 个来自最终定向修正。

评测路径中不存在 repair overlay。最终的 `start_position`、
`start_rotation`、`init_offset` 和 `init_yaw` 已经直接写入对应 episode
shard。用户直接加载 1,218-episode split，不需要再次应用上述三轮修正。

逐 episode、机器可读的审计列表位于
[`spawn_repair_history_20260806_v2.csv`](../resources/provenance/spawn_repair_history_20260806_v2.csv)。
它只记录 provenance；修改或删除该文件不会改变 rollout。

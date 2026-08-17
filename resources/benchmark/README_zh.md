<div align="right"><a href="README.md">English</a></div>

# Benchmark episodes

`episodes/val/content/` 是 1,218 个 HumanClawBench episode 唯一的运行时
来源。其中 41 个 gzip 文件是与 Habitat 兼容的 scene shard；每个文件包含
一个 `episodes` 数组和共享的 `goals_by_category` 记录。旁边的
`episodes/val/val.json.gz` 是 Habitat split header：它的 `episodes` 数组
有意为空，只提供六个 category-ID 映射，并不是 benchmark 的第二份副本。

每个 episode 都保存标准的 `start_position`、`start_rotation`，以及准确的
HumanClaw `init_offset`、`init_yaw`。后两个字段用于在 simulator 坐标系中
放置 articulated humanoid。所有验证过的 spawn 修正都已经写入这些 shard，
评测时不会再加载单独的 manifest 或 repair overlay。

一个 episode 由 `(scene_id, episode_id)` 唯一确定。重复的 scene 或 target
category 表示不同的起点，不是重复 episode。

`val100.json` 是透明的开发子集索引，不是另一份 episode 数据。它列出
canonical 数据中的 100 个 episode：按记录的随机种子，从五个 scene 各选
20 个。全部 100 行都已对照上面的 1,218-episode 数据验证。它适合快速比较
模型，但论文表格使用完整 validation split。

运行这个固定子集：

```bash
humanclaw-bench run \
  --episodes val100 \
  --model-config /path/to/model.json \
  --gpus auto \
  --output outputs/my-val100
```

<div align="right"><a href="README_zh.md">中文</a></div>

# Benchmark episodes

`episodes/val/content/` is the only runtime source for the 1,218
HumanClawBench episodes. Its 41 gzip files are Habitat-compatible scene shards;
each file contains an `episodes` array and shared `goals_by_category` records.
The sibling `episodes/val/val.json.gz` is the Habitat split header: its
`episodes` array is intentionally empty and it supplies the six category-ID
mappings. It is not a second copy of the benchmark.

Every episode stores its standard `start_position` and `start_rotation` plus
the exact HumanClaw `init_offset` and `init_yaw`. The latter two fields place
the articulated humanoid in simulator coordinates. All validated spawn
corrections are already materialized here, so evaluation does not load a
separate manifest or repair overlay.

An episode is uniquely identified by `(scene_id, episode_id)`. Repeated scene
and target-category values represent different starting poses, not duplicate
episodes.

`val100.json` is a transparent development subset index, not another episode
dataset. It lists 100 of the canonical episodes: 20 episodes from each of five
scenes, selected with the recorded seed. All 100 rows are validated against the
1,218-episode source above. It is useful for faster model comparisons, but the
paper tables use the complete validation split.

Run that exact subset with:

```bash
humanclaw-bench run \
  --episodes val100 \
  --model-config /path/to/model.json \
  --gpus auto \
  --output outputs/my-val100
```

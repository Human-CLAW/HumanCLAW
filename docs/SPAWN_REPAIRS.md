<div align="right"><a href="SPAWN_REPAIRS_zh.md">中文</a></div>

# Humanoid spawn corrections

Habitat's standard `start_position` and `start_rotation` describe the episode
start pose. HumanClaw additionally needs a simulator-space `init_offset` and
`init_yaw` for the articulated humanoid. A subset of the original starts put
that body too close to scene geometry or produced an unstable initial state,
so those humanoid coordinates were validated and corrected before release.

The corrections were delivered in three ordered validation passes. The source
revision names are retained only so the merge can be audited:

| Validation pass | Source revision | Episodes inspected | Final values changed |
|---|---|---:|---:|
| Initial coordinate repair | `20260719-first-pass` | 308 | 308 |
| 1 cm clearance refinement | `20260720-refined-1cm` | 308 | 38 |
| Final targeted correction | `20260806-v2` | 38 | 1 |

Later passes replace earlier values only for the episode keys they explicitly
contain. The resulting 308 corrected episodes comprise 270 values from the
initial repair, 37 from the clearance refinement, and one from the final
targeted correction.

There is no repair overlay in the evaluation path. Each final
`start_position`, `start_rotation`, `init_offset`, and `init_yaw` is already
materialized in its bundled episode shard. A user loads the 1,218-episode split
directly and does not need to apply these passes again.

The machine-readable per-episode audit list is
[`spawn_repair_history_20260806_v2.csv`](../resources/provenance/spawn_repair_history_20260806_v2.csv).
It records provenance only; changing or deleting it cannot change a rollout.

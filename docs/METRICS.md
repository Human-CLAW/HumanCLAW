<div align="right"><a href="METRICS_zh.md">中文</a></div>

# Paper metrics

Metrics are opt-in. `humanclaw-bench run --metrics` writes one `metrics.json`
per episode and one `metrics_summary.json` for a multi-episode run. Without
that flag, none of the metric sensors, contact queries, target resolution, or
end-of-episode analyses run.

## Print a completed run

```bash
humanclaw-bench metrics outputs/fullval
```

The command recursively finds `metrics.json`, so the input may be one normal
run or a parent containing distributed shards. It prints the paper main table,
success variants, collision body groups, and aggregation counts without
loading Habitat, a motion model, or a VLM. The default is read-only. Use
`--json` for the full summary object and `--write-json` to additionally write
`<output-root>/metrics_summary.json`.

## Shared episode timeline

At decision step *s*, the evaluator already has the ego RGB image that will be
sent to the VLM. In metric mode, the semantic sensor was rendered from the same
camera transform and its target pixel count is retained in memory. The parsed
planner response is then joined with that count for FindSR.

A non-stop action generates a 30 Hz motion chunk. Each physics frame produces
one shared post-physics contact query. Collision consumes fixed-geometry
contacts, Interact consumes pelvis/target contacts, and disturbance consumes
human/dynamic and dynamic/dynamic contacts. The rows are discarded after the
three recorders update their compact state.

The generated and post-physics trajectories are always recorded for replay.
Metric mode materializes their arrays once at episode end. Motion Jerk reads the
generated array; collision has already been accumulated online from the shared
contact stream. The materialized arrays are then compressed into the two
trajectory NPZ files.

## Find

Every resolved goal instance receives a temporary semantic ID. A decision has
objective visibility when the union of those IDs occupies at least 100 pixels
in the 448×448 ego semantic image.

The subjective rule splits `visible_state` at sentence boundaries. A sentence
acknowledges the target when it mentions `target` or a category alias and does
not contain any of these negations: `no`, `not`, `cannot`, `can't`, `don't`,
`isn't`, `aren't`, `n't`, `without`, `unable`, `none`.

- `GeoFindSR`: any decision has objective visibility.
- `FindSR`: any single decision has both objective visibility and subjective
  acknowledgement.

The two conditions must occur at the same decision step.

## Nav

At termination, the evaluator forms a point set from pelvis, articulated root,
and every humanoid link origin. For every target instance it computes the full
3D point-to-world-AABB distance, then takes the minimum over points and targets.

- `GeoNavSR@20cm`: final distance ≤ 0.2 m.
- `NavSR@20cm`: active Stop and final distance ≤ 0.2 m.
- `NavSR@1m`: active Stop and final distance < 1.0 m.

Only an action selected by the planner/verifier as Stop is an active stop;
reaching the 100-step limit is not.

## Interact

Interact metrics are evaluated only for bed, couch, and toilet episodes. A
pelvis contact matches an exact resolved target object ID/handle/template; AABB
proximity alone is insufficient.

- `GeoInteractSR`: the final physics frame of at least one Sit action has
  pelvis-to-target mesh contact. A transient earlier contact inside the same
  motion chunk does not count.
- `InteractSR`: at least one Sit was executed, the agent actively stops, and
  the final physics frame of the last motion before that stop has pelvis-to-
  target mesh contact.

## Fixed-geometry collision

The collision metric measures the realized post-physics pose at 30 Hz. The
same discrete contact query already required by Interact and disturbance is
reused; collision does not restore poses, interpolate requested motion, or run
a second end-of-episode contact pass.

One downward ray at the spawn selects the highest surface no more than 5 cm
above the standing humanoid's minimum height. This is the episode floor. A
fixed contact counts when its contact height minus that floor is greater than
0.0205 m. Coll% is the mean across episodes of:

```text
unique motion decision steps with a counted fixed contact
----------------------------------------------------------
              unique motion decision steps
```

The same step sets are grouped into arm/hand, torso, leg/foot, and head scores.

## Movable-object disturbance

A dynamic rigid object becomes directly affected when it contacts the humanoid.
At each same or later frame, affected status propagates across dynamic-to-
dynamic contact edges. Time ordering prevents a future contact from affecting
an object retroactively.

- `#Dtb/ep`: mean affected-object count across all episodes.
- `dDtb(m)`: sum of each mapped affected object's path length beginning at its
  first affected frame, divided by the total mapped affected-object count.

The distance is a pooled object-level mean, not a mean of episode means.

## Motion Jerk

Motion Jerk reads generated pre-physics `xb_world_75`. It freezes local body
articulation at the neutral 22-joint skeleton and applies only root translation
and global root rotation. Joint positions receive a centered moving average of
width 3. Jerk is the third finite difference with stride 8 at 30 fps:

The neutral joint centers are the exact first 22 rows of
`J_regressor @ v_template` from the pinned neutral SMPL source, made
pelvis-relative. The 66 resulting meter-valued constants are stored openly in
`resources/metrics/smpl_neutral_body22.json`; the release does not approximate
them from the simulation URDF and does not require the full body-model archive.

```text
P[t+3s] - 3P[t+2s] + 3P[t+s] - P[t]
------------------------------------- ,  s = 8
               (s/30)^3
```

The score is the mean vector magnitude over joints and valid frames for an
episode, followed by the mean over all episodes.

## Initial penetration diagnostic

Before physical metrics, the evaluator restores the exact reset-time human and
dynamic-object state, before any physics frame has run, then performs one
discrete contact query. Maximum penetration is
`max(0, -contact_distance)`. Whether the depth exceeds 0.01 m is stored in
`metrics.json` for diagnosis, but it does not exclude the episode. Collision,
disturbance, jerk, high-level success, and cost all use the complete 1,218-
episode full-validation split.

## Cost

Steps are VLM decision steps, including the final Stop decision. Input and
visible output token totals come from provider usage when available. Hidden
reasoning tokens are stored separately and excluded from visible output. The
aggregate divides total tokens by total decision steps, matching the paper's
per-step columns. If usage is unavailable, the episode is explicitly labelled
`estimated_chars_div_4`; mixed exact/fallback episodes are also labelled.
Provider or JSON-parse failures are retried at the current state up to five
times. Five planner failures execute one `Walk<forward><slow>` fallback; five
verifier failures accept the planner proposal. The episode continues in both
cases, and all real attempts remain included in token accounting.

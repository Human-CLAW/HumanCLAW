<div align="right"><a href="VIDEOS_zh.md">中文</a></div>

# Videos

HumanClawBench keeps video capture, delayed rendering, and presentation
composition separate. All three are optional and none changes benchmark
actions or metrics.

## Save ego and exo video during rollout

Add `--video` to `humanclaw-bench run`. Each rollout then contains synchronized
`ego.mp4` and `exo.mp4`. Without this flag, the exo sensor and video encoders
are never created.

## Render from a saved trajectory

If a rollout was recorded without video, render its saved post-physics poses
without rerunning physics, motion generation, or the VLM:

```bash
humanclaw-bench render \
  --rollout-dir outputs/fullval/EPISODE/rollout_00
```

Use `humanclaw-bench render-batch` for a complete output tree; see the README
for its parallel and GPU options.

## Compose ego, exo, and reasoning

The composition command reads existing `ego.mp4`, `exo.mp4`,
`trajectory_before.npz`, and `stepNNN_percept_mid_low.json` files. When a
verifier was called, it also reads `stepNNN_verifier.json`. It places the two
views side by side and shows the exact saved percept, mid-level reasoning,
goal, low-level reasoning, verifier decision, and executed action below them.
Trajectory step offsets determine the text timing exactly.

For one rollout:

```bash
humanclaw-bench compose-video \
  --rollout-dir outputs/fullval/EPISODE/rollout_00
```

The default output is
`outputs/fullval/EPISODE/rollout_00/full_ego_exo_reasoning.mp4`. Use `--output`
to choose another path.

For a complete output tree:

```bash
humanclaw-bench compose-video-batch \
  --input-root outputs/fullval \
  --output-root outputs/fullval_composite \
  --max-parallel 8
```

The batch command mirrors the input directory layout under the output root.
It is resumable: a video with the expected frame count is skipped unless
`--force` is supplied. Composition does not load Habitat, physics, a motion
model, or a VLM. It writes only the final MP4; its temporary subtitle file is
removed automatically.

Composition requires a system `ffmpeg`/`ffprobe` build with the libass
`subtitles` filter. Check it with:

```bash
ffmpeg -hide_banner -filters | grep subtitles
```


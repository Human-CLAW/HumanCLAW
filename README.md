<div align="right"><a href="README_zh.md">中文</a></div>

# HumanClawBench

HumanClawBench evaluates a vision-language model as a full-body agent in 1,218
find–navigate–interact episodes across 41 HSSD scenes.

Release assets:

- Code: <https://github.com/Human-CLAW/HumanClawBench>
- Motion weights (`paper_fullval_v1`):
  <https://huggingface.co/Human-CLAW/HumanCLAW>
- HSSD supplement (gated dataset):
  <https://huggingface.co/datasets/Human-CLAW/HumanCLAW-HSSD>

## Quick start

After [installation](#install), [HSSD preparation](#prepare-hssd), and
[motion-weight setup](#install-motion-weights), copy the model-interface
template and fill in the served model name and endpoint:

```bash
cp configs/models/vllm_openai_compatible.json my_model.json
```

Verify the setup with one episode, then choose the fixed 100-episode subset or
the complete validation split:

```bash
# One deterministic smoke episode.
humanclaw-bench run --episodes one --model-config my_model.json \
  --gpus auto --output outputs/smoke

# Fixed small validation subset.
humanclaw-bench run --episodes val100 --model-config my_model.json \
  --gpus auto --workers-per-gpu 1 --metrics --output outputs/val100

# Complete 1,218-episode evaluation.
humanclaw-bench run --episodes fullval --model-config my_model.json \
  --gpus auto --workers-per-gpu 1 --metrics --output outputs/fullval
```

`--gpus auto` uses exactly the devices exposed by `CUDA_VISIBLE_DEVICES` (or
all detected GPUs when it is unset). For a local VLM server, reserve its GPUs
when starting the server and pass only the remaining evaluation GPUs here.
Add `--video` for synchronized ego/exo MP4 files; video and metrics are
independent. Without `--metrics`, semantic rendering, contact queries, and all
metric accumulation are disabled.

See [the evaluation flow](docs/ARCHITECTURE.md), [metric definitions](docs/METRICS.md),
[video tools](docs/VIDEOS.md), and [model-interface contract](docs/MODELS.md).

## Repository contents

```text
configs/                     rollout and model-adapter configuration
patches/habitat-sim/         required Habitat-Sim patch
resources/benchmark/         fixed 1,218-episode split and transparent val100 index
resources/hssd/              HumanClaw scene and object configuration
resources/agent/             humanoid runtime assets
resources/seeds/             deterministic initial humanoid state
resources/weights/           external-checkpoint manifest
src/humanclaw_bench/
  agent/                     planner, verifier, prompts, action schema
  benchmark/                 episode loading
  envs/                      Habitat task/runtime integration
    half_physics/
      hp.py                  single production Half-Physics controller
      humanclaw.physics_config.json  Bullet scene/default settings
  evaluation/evaluator.py    single rollout loop
  evaluation/trajectory.py   compact replay bundle writer
  evaluation/metrics/        paper metric definitions and aggregation
  evaluation/video.py        direct-to-MP4 streaming
  rendering/                 delayed rendering and ego/exo/reasoning composition
  motion/                    motion generation and optional training tools
  vlm/                       model transports
```

Official HSSD meshes, motion weights, licensed motion datasets, provider
credentials, and model rollout results are not included. The repository does
include the motion-training implementation and exact per-skill source-chunk
lists; see
[`src/humanclaw_bench/motion/training/README.md`](src/humanclaw_bench/motion/training/README.md).
The 1,693 instance-specific HSSD baked meshes are a separately versioned,
gated Hugging Face asset; the repository contains their exact filename, size,
and SHA-256 manifest. Inference, replay recording, and metric implementations
are included.

## Install

Full rollouts require Linux, Python 3.10+, CUDA-compatible PyTorch, patched
Habitat-Sim with Bullet, authorized HSSD-Hab val data, HumanClaw motion
checkpoints, and a VLM endpoint or queue worker.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rollout,test]'
```

Video uses a system `ffmpeg` when available. To install a packaged fallback:

```bash
python -m pip install -e '.[video]'
```

Motion training is optional and isolated from evaluation dependencies:

```bash
python -m pip install -e '.[training]'
```

Build Habitat-Sim at the pinned revision and apply the bundled patch:

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout acbe6f4922e68145e401e55c30f9dfea460a3f24
git submodule update --init --recursive
git apply --check /absolute/path/to/HumanClawBench/patches/habitat-sim/humanclaw_halfphysics.patch
git apply /absolute/path/to/HumanClawBench/patches/habitat-sim/humanclaw_halfphysics.patch
python -m pip install -r requirements.txt
python setup.py build_ext --inplace --headless --with-cuda --bullet
python -m pip install -e .
python -m pip install -e build/deps/magnum-bindings/src/python
```

The fully validated clean-machine recipe and fixes for minimal CUDA toolkits,
broken `ccache`, and `libgomp.so.1` are documented in
[`patches/habitat-sim/README.md`](patches/habitat-sim/README.md).

## Prepare HSSD

The original HSSD val download should contain:

```text
/path/to/hssd-hab/
├── hssd-hab.scene_dataset_config.json
├── objects/
├── stages/
└── semantics/
```

Prepare it for HumanClawBench:

```bash
hf auth login  # required once for the gated supplement
humanclaw-bench prepare-hssd --hssd-root /path/to/hssd-hab
```

The command validates every mesh by size and SHA-256, combines the official
HSSD assets with the pinned 1,693-mesh HumanClaw supplement, copies the
HumanClaw scene and per-instance object JSON files, and symlinks the meshes,
stages, and semantics. On first use it downloads the 79.8 MiB compressed
supplement from the gated `Human-CLAW/HumanCLAW-HSSD` dataset and stores its
verified extraction under `~/.cache/humanclaw-bench/assets/`. This preserves
baked scale fixes and cases that use the render mesh for collision; it does
not silently substitute an official coarse collider. The original HSSD
installation is not modified. The default output is
`data/humanclaw-hssd-val41/`.

To prepare elsewhere:

```bash
humanclaw-bench prepare-hssd \
  --hssd-root /path/to/hssd-hab \
  --output /path/to/humanclaw-hssd-val41
```

For an offline machine, transfer the HF archive and pass it directly:

```bash
humanclaw-bench prepare-hssd \
  --hssd-root /path/to/hssd-hab \
  --supplement /path/to/humanclaw-hssd-val41-supplement-v1.tar.gz
```

Then pass its `hssd-hab.scene_dataset_config.json` with
`--scene-dataset-config` when running an episode.

The benchmark episodes are already final. Each stores the Habitat start pose
and HumanClaw `init_offset`/`init_yaw`; no spawn-repair overlay is applied at
runtime.

For a faster development run, pass `--episodes val100` to
`humanclaw-bench run`. This selects the transparent list at
`resources/benchmark/val100.json`: a fixed 100-episode, five-scene subset.
Reported paper results still use all 1,218 episodes.

## Install motion weights

Download the inference-only archive from the HumanCLAW Hugging Face model
repository and extract it from the parent directory of this source tree:

```bash
cd ..
hf download Human-CLAW/HumanCLAW \
  HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz \
  --local-dir .
tar -xzf HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz
cd HumanCLAW
```

Place the distributed checkpoints at the paths pinned by
`resources/weights/paper_fullval_v1.json`:

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

These are inference-only states, not the original trainer checkpoints. The
release removes optimizer/trainer state and repeated frozen bases while
reconstructing every evaluated model tensor exactly. See the weight README for
the audited FP32/BF16 base-variant relationship:
[`weights/paper_fullval_v1/README.md`](weights/paper_fullval_v1/README.md).

Verify bundled assets and weights:

```bash
humanclaw-bench assets
humanclaw-bench assets --weights-root weights/paper_fullval_v1
```

## Configure a VLM

For vLLM or another OpenAI-compatible server, copy
`configs/models/vllm_openai_compatible.json` and set the model, endpoint, and
model-specific `max_tokens`:

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

For a local vLLM endpoint no key is required; the adapter supplies the
placeholder expected by OpenAI-compatible servers. For a remote endpoint,
export the named environment variable instead of writing a credential into
the JSON file.

For a credential-owning external worker, use
`configs/models/filesystem_queue.json`. The adapter atomically places a request
under `<queue_dir>/pending/<call_id>/`; the worker returns
`<queue_dir>/done/<call_id>/response.json` containing at least:

```json
{"content": "{...model JSON response...}"}
```

Request images and queue JSON are transport files, not rollout artifacts. They
are removed after a response is consumed.

## Run one episode

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

The planner uses prompt v4 and the verifier uses verifier v3. The selected
motion action is always the verifier-final action. Stop ends the episode;
otherwise the rollout ends at 100 environment steps.

Each planner or verifier stage retries provider and JSON-parsing failures at
the current simulator state, up to five attempts. After five failed planner
attempts, the agent executes one `Walk<forward><slow>` and replans from the new
observation. Five failed verifier attempts accept the valid planner proposal.
Neither case restarts the episode.

The files created by a default rollout are:

```text
outputs/example/
└── <scene_id>_ep<episode_id>_<category>/
    └── rollout_00/
        ├── step000_percept_mid_low.json
        ├── step000_verifier.json       # only when verifier was called
        ├── step001_percept_mid_low.json
        ├── ...
        ├── trajectory_before.npz
        ├── trajectory_after.npz
        └── replay_manifest.json
```

Each file contains exactly the VLM prompt and final parsed response for that
logical stage. An `error` field appears only when all attempts for that stage
failed.

`trajectory_before.npz` contains only the world-frame `xb_world_75` chunks
actually passed to HalfPhysics, action/step boundaries, fps, and the exact
initial humanoid and dynamic-object poses and velocities needed to start a
forward replay. The unused 219-D internal motion feature is not duplicated.
`trajectory_after.npz` contains the corresponding simulated humanoid pose and
every dynamic object's position and rotation at every physics frame.
`replay_manifest.json` pins the episode, physics parameters, critical asset
hashes, and both NPZ hashes. No duplicate `trajectory.npz` is written.

### Save videos

```bash
humanclaw-bench run \
  --episodes one \
  --model-config my_model.json \
  --gpus 0 \
  --video \
  --output outputs/example_video
```

This adds `ego.mp4` and `exo.mp4`. Both contain the post-reset frame followed
by every simulated motion frame at 30 fps. Frames are piped directly to H.264;
there is no temporary image directory or second encoding pass. Video mode does
not enable semantic rendering, contact queries, or metrics.

### Render a saved trajectory after rollout

A completed rollout already records the post-physics humanoid pose and every
dynamic object's pose in `trajectory_after.npz`, so video does not require a
second physics replay. The delayed renderer loads the scene once, restores
those poses frame by frame, updates the ego/exo cameras, and streams RGB
directly to two MP4 encoders:

```bash
humanclaw-bench render \
  --rollout-dir outputs/example/102343992_ep0_bed/rollout_00 \
  --output-dir outputs/example_rendered
```

This writes only `ego.mp4`, `exo.mp4`, and `render_report.json`. It makes zero
VLM calls, motion-generation calls, physics steps, contact queries, and
semantic renders. Habitat initializes its scene and articulated-object runtime
because those objects are needed for pose assignment and rasterization, but
simulation time is never advanced. The default `veryfast` H.264 preset favors
throughput; use `--preset medium --crf 18` to match the online encoder settings.

Render a complete output tree with isolated Habitat/OpenGL processes:

```bash
humanclaw-bench render-batch \
  --input-root outputs/fullval \
  --output-root outputs/fullval_rendered \
  --max-parallel 8 \
  --devices 0,1
```

To combine existing ego/exo streams with the exact saved per-step model text:

```bash
humanclaw-bench compose-video \
  --rollout-dir outputs/example/EPISODE/rollout_00
```

Use `compose-video-batch` for a parallel output tree. This presentation pass
does not load Habitat, physics, a motion model, or a VLM. See
[docs/VIDEOS.md](docs/VIDEOS.md).

The output preserves each rollout's relative directory. Process isolation is
intentional because Habitat/OpenGL contexts are not shared between threads.
For a modified trajectory, pass a JSONL manifest instead of `--input-root`;
each row contains `episode_key`, `rollout_dir`, and optional `trajectory_path`.

### Compute the paper metrics

```bash
humanclaw-bench run \
  --episodes one \
  --model-config my_model.json \
  --gpus 0 \
  --metrics \
  --output outputs/example_metrics
```

This adds exactly one `metrics.json`. During rollout, one semantic observation
is reused for FindSR and one contact query per physics frame is shared by
Collision, InteractSR, and disturbance. Motion Jerk reads the already-recorded
pre-physics trajectory at episode end. No contact replay is performed and no
intermediate metric artifact is saved.

To summarize any completed output tree, including a parent directory that
contains multiple distributed-run shards, run:

```bash
humanclaw-bench metrics outputs/fullval
```

This recursively reads the per-episode `metrics.json` files and prints the
paper main table, success variants, collision body groups, and denominators.
It performs no VLM call, simulation, rendering, or replay, and is read-only by
default. Add `--json` for the complete machine-readable summary. Add
`--write-json` to also save `outputs/fullval/metrics_summary.json`.

The reported fields match the paper:

- FindSR requires at least 100 target semantic pixels in the same ego image
  used for a VLM decision and a non-negated target acknowledgement in that
  decision's `visible_state`. GeoFindSR uses pixels only.
- NavSR@20cm requires an active Stop and final minimum 3D distance from
  any body joint to a target AABB of at most 0.2 m. GeoNavSR@20cm omits the
  stop requirement. NavSR@1m uses active stop and distance below 1 m.
- InteractSR applies to bed/couch/toilet episodes. It requires at least one Sit,
  an active stop, and pelvis-to-target mesh contact on the final frame of the
  last motion before the stop. GeoInteractSR asks whether the final frame of
  any Sit decision made that mesh contact.
- Collisions inspect the realized post-physics pose at 30 Hz and count
  fixed-geometry contacts more than 0.0205 m above the episode's spawn floor.
  The score is the fraction of motion decision steps with a collision.
- Disturbance counts dynamic objects affected directly by the humanoid or
  indirectly through a time-ordered dynamic-object contact chain. Distance is
  affected-object path length, pooled over mapped affected objects.
- Motion Jerk uses the generated pre-physics root-rigid trajectory, neutral
  22-joint body, centered moving average 3, and stride 8 at 30 fps. The exact
  pelvis-relative neutral joint constants are included in
  `resources/metrics/smpl_neutral_body22.json`.
- Cost reports decision steps and provider token usage per step. Hidden
  reasoning is excluded from visible output tokens. If a provider omits usage,
  the file explicitly labels the character-based fallback as estimated.

Initial human penetration is marked in `metrics.json` for diagnosis, but no
episode is excluded: collision, disturbance, and jerk all use the complete
1,218-episode split. See [docs/METRICS.md](docs/METRICS.md) for exact data flow.

The two flags can be combined. In that case the same sensor render at each
physics frame supplies video RGB and semantic data while the metric path still
writes only `metrics.json`.

## Select episodes and GPUs

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

The command above runs at most eight independent episode processes and balances
them over four GPUs. Use one worker per GPU first; increase it only when the
available memory can hold multiple motion runtimes. `--episodes` accepts
`one`, `val100`, `fullval`, or a custom JSON episode-list path. Metric mode
writes one `metrics_summary.json` after all episodes finish.

## Validate the source release

```bash
humanclaw-bench config paper_fullval_v1
pytest -q
```

For a real runtime check after preparing HSSD, run the manual integration
smoke below in an environment with Habitat-Sim and ffmpeg:

```bash
PYTHONPATH=src python tests/runtime_habitat_smoke.py \
  --output /tmp/humanclaw_habitat_smoke
```

It uses one fixed episode and 25 stationary requested frames, so it needs no
VLM credentials or motion weights. It validates scene and target loading,
HalfPhysics, contacts, dynamic-object trajectories, both MP4 streams, and the
single final metrics artifact.

The smaller controller-contract check needs Habitat/Magnum bindings but no
scene data or renderer:

```bash
PYTHONPATH=src python tests/runtime_hp_contract.py
```

It directly verifies the four substeps, root x/z writes on substeps 0 and 2,
one-time root-angular write, 30-degree-per-frame cap, and pre-limit PJSC target.

To verify that a real saved motion can reproduce its post-physics trajectory,
run one or all recorded action chunks through the same environment:

```bash
PYTHONPATH=src python tests/runtime_forward_replay.py \
  outputs/<run>/<scene_id>_ep<episode_id>_<category>/rollout_00 \
  --max-steps 1
```

The check restores the saved human and every dynamic-object pose/velocity,
advances Half-Physics from `trajectory_before.npz`, and compares every frame
with `trajectory_after.npz`. Use `--max-steps 0` for the complete trajectory.

See `docs/ASSETS.md`, `docs/ARCHITECTURE.md`, `docs/METRICS.md`, and
`docs/MODELS.md` for the asset, execution, metric, and provider contracts.

## License

This release — code, configuration, and bundled resources — is licensed for
non-commercial use under [CC BY-NC 4.0](LICENSE). The motion weights and the
HSSD supplement distributed on Hugging Face carry the same license. Official
HSSD data remains subject to its own license and access terms.

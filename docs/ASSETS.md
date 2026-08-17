<div align="right"><a href="ASSETS_zh.md">中文</a></div>

# Assets and checkpoints

## Bundled

The repository includes one Habitat split header plus 41 scene shards. The
header defines the six category mappings; the 41 shards contain all 1,218
Find/Nav/Interact episodes. It also includes the beta-zero hand-merged runtime
body, natural seed state, stabilized environment metadata, and provenance
records. It also includes the 66 pelvis-relative neutral-joint constants used
by Motion Jerk in `resources/metrics/smpl_neutral_body22.json`; the full SMPL
archive is neither needed nor redistributed. Versioned planner and verifier
prompts are ordinary, inspectable Python modules under `agent/`. Precomputed
model results are not bundled.

`resources/benchmark/val100.json` is an episode index for the historical
five-scene, 100-episode development subset. It contains no duplicate scene or
episode payloads; every row points into the same canonical full-validation
shards. It was recovered from the archived
`val_s5_e20_seed20260530.json` manifest (source SHA-256
`6182cc3a224353e34a54bd0946fd7a9639eb9bc4e0d6278813bde84cfc5a43b7`);
all 100 scene/episode/category rows and the seed/category counts match that
source exactly. Select it with `humanclaw-bench run --episodes val100`.

Every episode stores `start_position`, `start_rotation`, `init_offset`, and
`init_yaw` directly. The provenance file
`spawn_repair_history_20260806_v2.csv` lists which validation pass supplied the
final humanoid coordinates for each of the 308 corrected episodes. It is not a
runtime input. See
[`SPAWN_REPAIRS.md`](SPAWN_REPAIRS.md) for the merge rules.

Run `humanclaw-bench assets` to verify their deterministic file or tree hashes.
The verifier checks file count and total size as well as SHA-256 for directory
assets.

## External HSSD data

Official HSSD stage and object meshes remain external because they have their
own access and redistribution terms. The full-val benchmark references 41
scenes. Prepare an authorized HSSD val installation with:

```bash
humanclaw-bench prepare-hssd --hssd-root /path/to/hssd-hab
```

`resources/hssd/humanclaw-hssd-val41/` contains the 41 scene descriptions,
14,537 directly inspectable per-instance physics configs, and the immutable
manifest for 1,693 instance-specific baked meshes absent from the official
download. The meshes are distributed as a gated 79.8 MiB Hugging Face asset;
their extracted logical size is 176 MiB. `asset_requirements.json` pins every
official and supplemental asset by size and SHA-256. The generated
`data/humanclaw-hssd-val41/` reuses the official tree and verified HF cache
through symlinks. The supplement is required for baked scale fixes and exact
render-mesh collision geometry.

The default `prepare-hssd` command downloads the pinned archive automatically.
For an offline host, pass the downloaded archive with `--supplement`.

Pass `--output` to prepare elsewhere, then pass its
`hssd-hab.scene_dataset_config.json` to `run` with
`--scene-dataset-config`.

## External motion data

AMASS, BABEL, and the neutral SMPL model are not redistributed. The repository
contains only eight transparent CSV training lists under
`resources/motion/training/manifests/` (1.67 MB total). Each row identifies a
relative AMASS chunk pickle and its source index; it does not contain motion
arrays. Build the 20-frame corpus locally and use these lists as described in
the [motion-training guide](../src/humanclaw_bench/motion/training/README.md).

## External motion weights

`resources/weights/paper_fullval_v1.json` is authoritative. It pins one
canonical MotionDiT state, two deterministic numerical base variants, and
eight control-only skill states selected from exact step 1,500,000. There is no
`latest` resolution.

The nine inference files are distributed as
`HumanCLAW_pretrained_weights_paper_fullval_v1_20260816.tar.gz` in the
`Human-CLAW/HumanCLAW` Hugging Face model repository. The archive SHA-256 is
`3b3c0c1b232af4c462301655de909a4bc54fd4756bcd42c22e7965fccb650667`.

| Skill | Network | Conditioning |
|---|---|---|
| walk_forward | `WalkForwardCtrlDiTFourierXZYaw` | Fourier x/z/yaw, 6 frequencies |
| side_walk | `SideWalkCtrlDiTFourier` | Fourier, 6 frequencies |
| step_back | `WalkForwardCtrlDiTFourier` | Fourier, 6 frequencies |
| turn | `TurnCtrlDiT` | MLP |
| step_climb_up | `StepClimbUpCtrlDiT` | MLP |
| step_climb_down | `StepClimbDownCtrlDiT` | MLP |
| stop | `NonCondCtrlDiT` | none |
| sit | `SitCtrlDiT` | MLP |

The base architecture is 219 input/output dimensions, hidden size 512, 10
layers, 8 heads, MLP ratio 2, five history frames, 15 future frames, and 30
flow-evaluation steps. `walk_forward`, `turn`, `stop`, and `sit` use the exact
BF16-roundtrip variant embedded in their archived checkpoints; the other four
use the canonical FP32 tensors. The former is a deterministic rounded copy of
the latter, not a separately trained MotionDiT. The weight-directory README
documents the audit and why retaining both numerical variants matches the
reported evaluation.

The original nine files occupied 3,227,420,338 bytes because every ControlNet
repeated its frozen base and the base Lightning checkpoint retained training
state. The inference-only set is 1,135,639,268 bytes. The loader recreates all
eight archived `model_state` mappings tensor-for-tensor and shares each base
variant among its four skills in memory.

Checkpoint validation occurs before any checkpoint is unpickled. This protects
reproducibility and reduces accidental loading of an unexpected artifact; it
does not make untrusted pickle files safe. Obtain weights only from the
maintainer-approved distribution.

## Redistribution gate

Do not upload scene data, body assets, or weights merely because they appear in
a local working copy. Confirm third-party terms and add notices first. The
release checklist treats this as a publication blocker.

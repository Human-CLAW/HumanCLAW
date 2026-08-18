<div align="right"><a href="README_zh.md">中文</a></div>

# Motion training

This directory contains the optional training side of HumanClaw's motion
generator. Evaluation does not import it. The trainers reuse the exact
`humanclaw_bench.motion.networks` classes used at rollout time, so there is no
second model implementation to drift out of sync.

The release contains code and transparent data indices, but not AMASS, BABEL,
or the neutral SMPL model. Obtain those assets under their own licenses.

## What is included

```text
motion/training/
├── build_chunks.py             AMASS -> per-sequence [N,20,219] pickle tree
├── body_model.py               neutral-SMPL forward kinematics
├── datasets.py                 base corpus and manifest-selected skill data
├── flow.py                     linear flow-matching target
├── runtime.py                  fast kernels and resumable checkpoints
├── train_base.py               concat-history MotionDiT trainer
├── train_control.py            one shared trainer for all eight skills
└── curation/
    ├── export_skill_manifest.py  legacy reviewed subset -> small CSV list
    └── filter_manifest.py        explicit numeric filters without data copies

resources/motion/training/
├── paper_training_v1.json      one shared training configuration
├── manifests/*.csv             exact source chunks used by each skill
└── segments/*.csv              reviewed source and final-used time ranges
```

All eight ControlNets use the same trainer and shared optimization settings.
Only the skill network, condition, and manifest change.

## Install

From the repository root:

```bash
python -m pip install -e '.[training]'
```

Training requires CUDA PyTorch for the paper-scale jobs. Chunk construction
also requires `smplx` and a neutral SMPL NPZ. CPU preprocessing is supported
for small checks.

## Build the 20-frame corpus

AMASS sequences have different native frame rates and use a Z-up world. The
builder performs the same transformation used for the released models:

1. choose 21 source frames at approximately 30 Hz;
2. slide the source window by five frames at the sequence's original rate;
3. convert Z-up to Y-up and run neutral-SMPL forward kinematics;
4. canonicalize to history frame 4 using the floor-projected hip frame;
5. concatenate 75 body parameters, 72 joint coordinates, and 72 one-frame
   joint velocities;
6. drop the final source frame, whose forward velocity is unavailable.

The result is `[20, 219]`: five clean history frames followed by fifteen
future frames.

```bash
python -m humanclaw_bench.motion.training.build_chunks \
  --amass-root /path/to/amass \
  --smpl-model /path/to/SMPL_NEUTRAL.npz \
  --output-root /path/to/amass_chunks_yup_v3 \
  --gpus 8 --workers 32
```

Output mirrors the AMASS directory tree with one pickle per source sequence.
The command is resumable: an existing output pickle is not rewritten.

## Curated skill lists

Skill corpora are CSV indices into the base chunk tree. They do not duplicate
motion arrays. Every row has `rel_pkl` and `source_chunk_idx`; Side Walk and
Sit rows also carry their stored scalar condition to avoid reconstruction
drift. All eight lists total 1.67 MB; no licensed AMASS motion array is copied
into the repository.

| Skill | Rows | Training condition |
|---|---:|---|
| Walk Forward | 16,023 | final canonical pelvis x/z and body yaw |
| Side Walk | 2,012 | final canonical lateral displacement |
| Step Back | 2,294 | final canonical pelvis x/z |
| Turn | 6,773 | signed final body yaw |
| Step Climb Up | 2,086 | maximum toe height/horizontal separation |
| Step Climb Down | 2,094 | maximum toe height/horizontal separation |
| Stop | 4,011 | no numeric condition |
| Sit | 5,912 | accepted segment's minimum pelvis height |

The lists encode the completed manual review. Users training the released
skills consume them directly; they do not need the review videos or copied
subset trees. `curation/filter_manifest.py` supports new experiments with
explicit yaw, translation, direction, and lateral-motion thresholds. It writes
a new list and never mutates or copies the base chunk data.

## Reviewed segment times

`resources/motion/training/segments/` makes the data selection readable without
shipping motion arrays. Each row names the source AMASS sequence and BABEL ID,
then records two distinct ranges in both seconds and milliseconds:

- `segment_start/end`: the BABEL or human-reviewed source interval;
- `first/last_used_chunk`: the actual coverage of final selected chunks owned
  by that interval.

The per-skill tables contain only intervals that contribute at least one final
training chunk. Their `used_chunk_count` columns sum exactly to the row counts
in `manifests/*.csv`:

| Skill | Final used segments | Final chunks |
|---|---:|---:|
| Walk Forward | 493 | 16,023 |
| Side Walk | 67 | 2,012 |
| Step Back | 129 | 2,294 |
| Turn | 436 | 6,773 |
| Step Climb Up | 96 | 2,086 |
| Step Climb Down | 125 | 2,094 |
| Stop | 111 | 4,011 |
| Sit | 116 | 5,912 |

These two ranges are intentionally not forced to be identical. Selection can
apply yaw, displacement, or future/action-window filters after a segment is
accepted; Side Walk in particular uses future-window containment, so a full
20-frame chunk may begin slightly before its reviewed interval.

Seven skills retain their accepted BABEL segment manifests. Step Climb Up is
the exception: its surviving final human-reviewed 2,086-row chunk manifest
encodes the owner interval in each reviewed clip filename, while an older
intermediate segment CSV is no longer available. Every row states this
provenance explicitly. `segments/index.json` pins source and normalized file
hashes.

## Train the base MotionDiT

```bash
python -m humanclaw_bench.motion.training.train_base \
  --chunk-root /path/to/amass_chunks_yup_v3 \
  --output outputs/motion_base
```

The base uses 219 input/output dimensions, hidden size 512, ten DiT blocks,
eight heads, five history frames, and fifteen predicted frames. Train it for
135 epochs (109,620 optimizer iterations) with batch size 512, AdamW, and
learning rate `1e-4`.

The refactored trainer writes a resumable checkpoint at the requested interval
and exports `motion_dit.pt`. A numbered checkpoint is written once;
`checkpoints/last.pt` is a relative link, avoiding the old duplicate hundreds
of megabytes of checkpoint I/O.

## Train one ControlNet

Only five paths/choices are needed; architecture and paper hyperparameters
come from the profile:

```bash
python -m humanclaw_bench.motion.training.train_control \
  --skill walk_forward \
  --base-checkpoint outputs/motion_base/motion_dit.pt \
  --chunk-root /path/to/amass_chunks_yup_v3 \
  --output outputs/walk_forward
```

Train each of the eight ControlNets for 1.5M iterations with batch size 2,048,
AdamW, learning rate `3e-4`, and the frozen base MotionDiT. These settings are
shared by every skill and are stored once in `paper_training_v1.json`.

Each run emits:

```text
config.json                 effective portable configuration
train_log.jsonl             sparse progress records
checkpoints/stepXXXXXXXX.pt resumable full training state
checkpoints/last.pt         relative link, not a second tensor copy
control.pt                  inference-only control branch
```

## Reconstructing or extending the lists

To convert a reviewed mirrored subset without copying its arrays again:

```bash
python -m humanclaw_bench.motion.training.curation.export_skill_manifest \
  --skill sit \
  --subset-root /path/to/reviewed_sit_subset \
  --expected-samples 5912 \
  --output /tmp/sit.csv
```

Pickle is used because that is the historical chunk format. Treat AMASS chunk
and legacy subset pickles as trusted local artifacts; pickle is not a safe
format for untrusted downloads.

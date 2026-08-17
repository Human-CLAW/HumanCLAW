<div align="right"><a href="README_zh.md">中文</a></div>

# Evaluation weights

This directory is the complete inference-only weight set for the paper
evaluation.  It contains one canonical MotionDiT state and eight
skill-specific ControlNet states.  Optimizer state, trainer state, duplicate
base tensors, and checkpoint-selection machinery are intentionally absent.
Here, *canonical* means the one full-precision MotionDiT state used as the
reference for exact reconstruction; it is not an additional model.

The exact files, sizes, SHA-256 digests, architectures, and source-checkpoint
provenance are pinned in
`resources/weights/paper_fullval_v1.json`.  The runtime never scans for a
newest checkpoint.

## Why the manifest has two base variants

The archived bundle loaded every skill independently:

1. load the base checkpoint;
2. construct that skill's ControlNet around the base;
3. strictly load the skill checkpoint's full `model_state`.

The third step also loaded the embedded `base_dit.*` tensors, so the embedded
copy—not merely the base file opened in step 1—determined the evaluated model.
A tensor-level audit found two groups:

| Base variant | Skills |
|---|---|
| `bf16_roundtrip` | `walk_forward`, `turn`, `stop`, `sit` |
| `fp32` | `side_walk`, `step_back`, `step_climb_up`, `step_climb_down` |

These are not two independently trained MotionDiTs.  All 112 tensors
(41,114,331 elements) in the first variant are exactly equal to applying
`FP32 -> BF16 -> FP32` to the canonical base in the second variant.  The
root cause is the 500k-to-1500k resume path.  The 1500k jobs were configured
with `param_dtype=fp32` and first constructed the same canonical FP32 base, but
their strict resume call then loaded the *entire* earlier `model_state`,
including the otherwise frozen `base_dit.*` branch.  The actual 500k resume
states for `walk_forward`, `turn`, `stop`, and `sit` store that branch as
`torch.bfloat16`; loading it into FP32 parameters produced the exact rounded
variant above.  The `step_climb_up` 500k resume state stores an FP32 branch,
and the other three FP32-group skills also retain the canonical FP32 values.

In other words, there was a training-time precision/configuration oversight:
four skill runs used BF16-valued tensors in their 500k resume checkpoints,
and the continuation code unintentionally allowed that BF16 copy to overwrite
the freshly loaded FP32 frozen base.  This was not an intentional design in
which different skills should use different base precision.  Replacing those
values now with the full-precision base would change the evaluated motion and
is therefore not done here.

This historical precision mismatch does not change the reported paper result:
the reported benchmark used the archived full checkpoints and therefore used
these exact tensors.  The compact loader reconstructs both variants exactly;
it does not retrain, average, or otherwise alter them.

## Lossless compaction

The release stores the canonical FP32 base once.  It creates the rounded
variant deterministically at load time and shares each immutable variant among
its four skills.  Each skill file stores only keys outside `base_dit.*`.
Control tensors that already lie exactly on the BF16 grid may be serialized as
BF16; PyTorch restores their original FP32 values exactly when loading the FP32
module.  Tensor equality against all eight archived `model_state` mappings is
part of the release validation.

### Equivalence validation

The compact files were accepted only after comparison with all eight pinned
1.5M trainer checkpoints from the evaluation bundle:

1. Every archived source file matched the SHA-256 recorded in the manifest.
2. The compact loader reconstructed each complete model and compared all 1,888
   tensors (645,163,754 elements, including each skill's base) with the archived
   `model_state`.  Keys, shapes, dtypes, and values were identical; every value
   comparison passed `torch.equal`.
3. Each archived model implementation and full checkpoint was then run against
   the release implementation and compact checkpoint with the same history,
   condition, timestep, and initial Gaussian noise.  For all eight skills, both
   a direct network forward and the complete 30-step midpoint integration were
   bit-identical, with maximum absolute error 0.

The size reduction therefore changes checkpoint serialization only.  It does
not change the reconstructed inference tensors or generated motion.

### Why four skill files are about 84 MB and four are about 158 MB

BF16 appears in the release only because the release reproduces that
training-time oversight exactly; it is not a new inference-time approximation.
The final 1.5M trainer checkpoints stored their `model_state` tensors in FP32
containers, but that storage dtype does not imply that every stored number
uses all FP32 precision.  For `walk_forward`, `turn`, `stop`, and `sit`, the
36,766,720 values in `ctrl_blocks.*` lie *exactly* on the BF16 numeric grid as
a consequence of their BF16 500k resume states.  For every such source tensor
`value`, the following equality holds element by element:

```python
torch.equal(value, value.to(torch.bfloat16).to(torch.float32))
```

The release therefore stores those values as BF16 (two bytes per element).
When `load_state_dict` copies them into the FP32 runtime module, it recovers the
exact FP32 values found in the archived checkpoint; this is not approximate
quantization.  The condition encoders and zero projections that require FP32
remain FP32.  This gives approximately 73.5 MB of BF16 control-block tensors
plus 10.5--11.6 MB of FP32 tensors per small skill file.

The other four skills contain control tensors that do not satisfy that exact
round-trip equality.  They remain FP32 (four bytes per element), making each
file approximately 158 MB.  Converting those tensors to BF16 would be lossy
and would no longer reproduce the evaluated model.

| Runtime file | Stored control precision | Size (decimal MB) |
|---|---|---:|
| `base/motion_dit.pt` | FP32 | 164.5 |
| `skills/walk_forward.pt` | BF16-grid tensors in BF16; remainder FP32 | 84.2 |
| `skills/side_walk.pt` | FP32 | 157.6 |
| `skills/step_back.pt` | FP32 | 157.7 |
| `skills/turn.pt` | BF16-grid tensors in BF16; remainder FP32 | 85.1 |
| `skills/step_climb_up.pt` | FP32 | 158.7 |
| `skills/step_climb_down.pt` | FP32 | 158.7 |
| Stop (`skills/stand.pt`) | BF16-grid tensors in BF16; remainder FP32 | 84.1 |
| `skills/sit.pt` | BF16-grid tensors in BF16; remainder FP32 | 85.1 |

The nine files total 1,135,639,268 bytes (1.136 GB, or 1.058 GiB).

The nine archived trainer checkpoints occupy 3,227,420,338 bytes.  The nine
pinned inference tensor files below occupy 1,135,639,268 bytes, a 64.8% lossless
reduction.

Only these files are required for evaluation:

```text
base/motion_dit.pt
skills/walk_forward.pt
skills/side_walk.pt
skills/step_back.pt
skills/turn.pt
skills/step_climb_up.pt
skills/step_climb_down.pt
skills/stand.pt
skills/sit.pt
```

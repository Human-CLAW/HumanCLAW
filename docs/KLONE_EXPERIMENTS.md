# HumanCLAW experiment runs on Klone

This guide covers the HumanCLAW stage of the experiment pipeline. Codex, Pi
and LIBERO-Pro are later integrations. The batch template performs one real
HumanCLAW episode; it does not install software, download assets, start a VLM
server, or provision resources beyond its explicit Slurm request.

## Verified preflight, 2026-10-13

A read-only check used the existing SSH control connection for
`hengyil6@klone.hyak.uw.edu`. No files were uploaded, packages installed, assets
downloaded or jobs submitted during this check.

| Item | Observed state |
|---|---|
| Login host | `klone-login03` |
| Running/queued jobs for the user | None returned by `squeue` |
| Project directories | `/gscratch/stf/hengyil6/{HumanCLAW,envs,data,outputs}` exist and are empty |
| Access permissions | User-private directories (`drwx--S---`, owned by `hengyil6`, group `stf`) |
| Default Python | `/usr/bin/python3`, version 3.6.8; below the project's Python 3.10 requirement |
| Other default runtime commands | `/usr/local/bin/uv`; no `python` or `conda` on the default PATH |
| Home storage | 10 GB quota |
| STF shared storage | `hyakstorage` reported 60,414 / 61,440 GB used and 53,160,692 / 60,000,000 files; this is a shared allocation, not a personal reservation |
| GPU account associations | `gpu-l40-stf`, `gpu-l40s-stf`, both with `normal` QoS |
| Corresponding partitions | `gpu-l40`, `gpu-l40s`, both up |
| Checkpoint association | `ckpt-stf`, with `ckpt`, `ckpt-gpu`, `ckpt-scav` QoS available; exact job policy must be checked before using it |

These are a dated readiness snapshot. A visible partition does not guarantee
immediate allocation, and no GPU application or driver compatibility was tested.
STF storage is nearly full; recheck `hyakstorage` before staging large assets.

## Remaining prerequisites

The server currently cannot run the real episode because its project directories
contain no checkout, Python environment, motion weights or scene data. Setup
requires the following, prepared separately:

1. A checkout containing this pipeline under `/gscratch/stf/hengyil6/HumanCLAW`.
2. A Python 3.10+ environment under `/gscratch/stf/hengyil6/envs/`, with the
   project's rollout dependencies and the pinned, patched Habitat-Sim build.
   Follow the [installation instructions](../README.md#install) and
   [Habitat-Sim patch guide](../patches/habitat-sim/README.md). Choose the PyTorch
   CUDA build after checking the driver within an allocated GPU job; the
   release's A100/CUDA 12.4 validation is not a Klone L40 validation.
3. Authorized HSSD-Hab inputs and the authorized HumanCLAW HSSD supplement,
   prepared under `/gscratch/stf/hengyil6/data/humanclaw-hssd-val41/`.
4. Motion checkpoints under the checkout's `weights/paper_fullval_v1/`, matching
   `resources/weights/paper_fullval_v1.json`. The run verifies checksums.
5. An explicit, working model configuration and a reachable model endpoint or
   queue worker. The model/deployment is not selected by this template. If using
   an external API, supply credentials through the configured environment variable,
   never as a command-line argument or in the tracked JSON file.

The prior instruction to defer uploads remains in effect. This guide and the
local template do not initiate setup. Licensed dataset terms and gated access
must be handled by the authorized account owner; do not auto-accept terms.

Do not change `HOME`. During future preparation, direct package/model caches to
the scratch storage paths used in the template. The existing `prepare-hssd`
archive extraction code uses `~/.cache/humanclaw-bench/assets` directly and does
not honor `XDG_CACHE_HOME`; use an already extracted, authorized supplement
directory on scratch via `--supplement /absolute/path/to/extracted-supplement`
to avoid that home-cache extraction path. Keep all required scene/mesh inputs
for later replay, including the targets of scene-data symlinks.

## Single-episode submission after setup

Run the following only after setup is complete and job submission is intended.
Use the actual prepared Python and model-config paths; the example names do not
imply those files exist. Environment variables already used by the adapter for
credentials may be inherited by Slurm; do not print or place their values in the
submission command. The script creates private cache/output files with `umask 077`.

```bash
cd /gscratch/stf/hengyil6/HumanCLAW
export HC_PYTHON=/gscratch/stf/hengyil6/envs/humanclaw/bin/python
export HC_MODEL_CONFIG=/gscratch/stf/hengyil6/data/configs/model.json
export HC_HARNESS=humanclaw_psv
export HC_SEED=0
sbatch tools/klone_experiment.sbatch
```

The verified default account/partition is `gpu-l40-stf` / `gpu-l40`, requesting
one L40, eight CPU cores, 32 GB host memory and one hour. This is an initial smoke
allocation, not a measured resource requirement. For the verified L40S account:

```bash
sbatch --account=gpu-l40s-stf --partition=gpu-l40s --gres=gpu:l40s:1 \
  tools/klone_experiment.sbatch
```

Slurm creates logs under the existing `/gscratch/stf/hengyil6/outputs/` directory.
The script writes a unique run directory containing the harness name and job ID,
refuses to reuse an existing output directory, and disables automatic requeue.
If a job is interrupted, retain its partial logs and submit a new run directory.
This template does not claim resume support.

Optional environment overrides are `HC_ROOT`, `HC_REPO`, `HC_SCENE_CONFIG`,
`HC_OUTPUT`, `HC_HARNESS` and `HC_SEED`. Paths must be absolute. If changing the
root, also override Slurm's `--output` and `--error` paths at submission because
`#SBATCH` directives do not expand shell variables. Keep `HC_HARNESS` explicit
when switching between `humanclaw_psv` and `bare_api`; a run with the existing
planner/verifier is not the bare-API condition.

An endpoint at `127.0.0.1` is local to the allocated compute node, not the login
host or laptop. The template assumes an independently provisioned, reachable
VLM service. It allocates one GPU for evaluation and leaves `CUDA_VISIBLE_DEVICES`
as assigned by Slurm; model serving requires its own resource plan.

## Readiness and verification boundaries

The batch script checks Python version, CUDA availability, imports and asset
checksums inside the allocated GPU job before starting the episode. A successful
import alone does not prove the Habitat-Sim patch was applied correctly; the
first real episode remains the integration check. No rollout, GPU probe or
rendering workload should run on the login node.

After a run, inspect its recorded configuration, trace completeness, final status,
token accounting and benchmark metrics before treating it as an experiment
result. Missing endpoint usage must remain unknown or explicitly estimated,
not a zero-cost call. Keep an interrupted run separate from a completed
benchmark failure.

Trace inspection and metric aggregation can be performed without a GPU. Habitat
re-rendering and action re-execution require the prepared scene inputs, compatible
environment and an allocated GPU. Re-rendering the saved trajectory is a visual
inspection, not evidence that rerunning the model or physics will reproduce the
same result. See the [experiment pipeline guide](EXPERIMENTS.md) for the
implemented replay commands and their current guarantees.

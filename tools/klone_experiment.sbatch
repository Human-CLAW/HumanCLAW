#!/usr/bin/env bash
# Submit only after preparing the environment, licensed assets and model config.
# Account/partition verified for hengyil6 on 2026-10-13; override if necessary.
#SBATCH --job-name=humanclaw-one
#SBATCH --account=gpu-l40-stf
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/stf/hengyil6/outputs/slurm-%j.out
#SBATCH --error=/gscratch/stf/hengyil6/outputs/slurm-%j.err

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run a rollout on a login node.}"
: "${HC_PYTHON:?Set HC_PYTHON to the absolute Python path in a prepared Python 3.10+ environment.}"
: "${HC_MODEL_CONFIG:?Set HC_MODEL_CONFIG to an absolute model-config JSON path.}"

HC_ROOT="${HC_ROOT:-/gscratch/stf/hengyil6}"
HC_REPO="${HC_REPO:-${HC_ROOT}/HumanCLAW}"
HC_SCENE_CONFIG="${HC_SCENE_CONFIG:-${HC_ROOT}/data/humanclaw-hssd-val41/hssd-hab.scene_dataset_config.json}"
HC_HARNESS="${HC_HARNESS:-humanclaw_psv}"
HC_SEED="${HC_SEED:-0}"
HC_OUTPUT="${HC_OUTPUT:-${HC_ROOT}/outputs/one-${HC_HARNESS}-${SLURM_JOB_ID}}"

case "${HC_HARNESS}" in
  humanclaw_psv|bare_api) ;;
  *) printf 'Unsupported HC_HARNESS: %s\n' "${HC_HARNESS}" >&2; exit 2 ;;
esac
for task_path in "${HC_PYTHON}" "${HC_MODEL_CONFIG}" "${HC_REPO}" "${HC_SCENE_CONFIG}" "${HC_OUTPUT}"; do
  if [[ "${task_path}" != /* ]]; then
    printf 'An absolute path is required: %s\n' "${task_path}" >&2
    exit 2
  fi
done
[[ -x "${HC_PYTHON}" ]] || { printf 'Prepared Python executable is missing.\n' >&2; exit 2; }
[[ -f "${HC_REPO}/pyproject.toml" ]] || { printf 'HumanCLAW checkout is missing.\n' >&2; exit 2; }
[[ -f "${HC_MODEL_CONFIG}" ]] || { printf 'Model config is missing.\n' >&2; exit 2; }
[[ -f "${HC_SCENE_CONFIG}" ]] || { printf 'Prepared HSSD scene config is missing.\n' >&2; exit 2; }
[[ ! -e "${HC_OUTPUT}" ]] || { printf 'Choose a new output directory; this one already exists.\n' >&2; exit 2; }

# Keep bulk caches and temporary files away from the 10 GB home quota.
# Credentials are supplied through the model adapter's named environment variable.
# Do not enable shell tracing or print the environment/model config.
export XDG_CACHE_HOME="${HC_ROOT}/data/cache"
export HF_HOME="${HC_ROOT}/data/cache/huggingface"
export TORCH_HOME="${HC_ROOT}/data/cache/torch"
export PIP_CACHE_DIR="${HC_ROOT}/data/cache/pip"
export UV_CACHE_DIR="${HC_ROOT}/data/cache/uv"
export CUDA_CACHE_PATH="${HC_ROOT}/data/cache/cuda"
export MPLCONFIGDIR="${HC_ROOT}/data/cache/matplotlib"
export TMPDIR="${HC_ROOT}/outputs/tmp-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export HUMANCLAW_BENCH_HOME="${HC_REPO}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${TMPDIR}"

cd "${HC_REPO}"

# CUDA and Habitat checks run inside this allocated GPU job, never on login.
"${HC_PYTHON}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("HumanCLAW requires Python 3.10 or newer.")
import torch
import habitat_sim
import humanclaw_bench

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the allocated job.")
print(f"Python: {sys.version.split()[0]}; torch: {torch.__version__}; CUDA: {torch.version.cuda}")
print(f"Visible GPUs: {torch.cuda.device_count()}; device 0: {torch.cuda.get_device_name(0)}")
PY

"${HC_PYTHON}" -m humanclaw_bench.main assets \
  --weights-root "${HC_REPO}/weights/paper_fullval_v1"

# Respect the devices Slurm exposes. This job allocates one evaluation GPU and
# assumes the VLM endpoint is already available; it does not start a model server.
exec srun --ntasks=1 "${HC_PYTHON}" -m humanclaw_bench.main run \
  --episodes one \
  --profile paper_fullval_v1 \
  --harness "${HC_HARNESS}" \
  --seed "${HC_SEED}" \
  --model-config "${HC_MODEL_CONFIG}" \
  --scene-dataset-config "${HC_SCENE_CONFIG}" \
  --gpus auto \
  --workers-per-gpu 1 \
  --trace --metrics \
  --output "${HC_OUTPUT}"

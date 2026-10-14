#!/usr/bin/env bash
# Build only in a Slurm allocation; keep native source and caches on scratch.
#SBATCH --job-name=humanclaw-setup
#SBATCH --account=cpu-g2-stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/stf/hengyil6/outputs/setup-%j.out
#SBATCH --error=/gscratch/stf/hengyil6/outputs/setup-%j.err

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through sbatch.}"
source /gscratch/stf/hengyil6/HumanCLAW/tools/klone_env.sh
mkdir -p "${TMPDIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" \
  "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${HC_ROOT}/data/src"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-8}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
cd "${HC_REPO}"
printf 'phase=python_dependencies host=%s job=%s\n' "$(hostname)" "${SLURM_JOB_ID}"
uv pip install --python "${HC_PYTHON}" pip 'setuptools==75.8.0' wheel ninja
uv pip install --python "${HC_PYTHON}" 'torch==2.6.0' \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${HC_PYTHON}" -c constraints/eval-cu124.txt -e '.[rollout,test,video]'

printf 'phase=habitat_source\n'
HC_HABITAT="${HC_ROOT}/data/src/habitat-sim"
HC_HABITAT_REV=acbe6f4922e68145e401e55c30f9dfea460a3f24
if [[ ! -d "${HC_HABITAT}/.git" ]]; then
  git clone --filter=blob:none https://github.com/facebookresearch/habitat-sim.git "${HC_HABITAT}"
  git -C "${HC_HABITAT}" checkout "${HC_HABITAT_REV}"
fi
[[ "$(git -C "${HC_HABITAT}" rev-parse HEAD)" == "${HC_HABITAT_REV}" ]] || {
  printf 'Habitat source is at an unexpected revision; preserving it.\n' >&2; exit 2;
}
git -C "${HC_HABITAT}" submodule update --init --recursive --jobs 4
if git -C "${HC_HABITAT}" apply --reverse --check "${HC_REPO}/patches/habitat-sim/humanclaw_halfphysics.patch"; then
  printf 'HumanCLAW patch is already applied.\n'
else
  git -C "${HC_HABITAT}" apply --check "${HC_REPO}/patches/habitat-sim/humanclaw_halfphysics.patch"
  git -C "${HC_HABITAT}" apply "${HC_REPO}/patches/habitat-sim/humanclaw_halfphysics.patch"
fi
[[ "$(git -C "${HC_HABITAT}/src/deps/bullet3" rev-parse HEAD)" == "2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5" ]]
uv pip install --python "${HC_PYTHON}" -c "${HC_REPO}/constraints/eval-cu124.txt" -r "${HC_HABITAT}/requirements.txt"
cd "${HC_HABITAT}"
printf 'phase=habitat_build\n'
"${HC_PYTHON}" setup.py build_ext --inplace --headless --with-cuda --bullet --cache-args
HEADLESS=True WITH_CUDA=True WITH_BULLET=True "${HC_PYTHON}" -m pip install --no-build-isolation -e .
"${HC_PYTHON}" -m pip install --no-build-isolation -e build/deps/magnum-bindings/src/python

printf 'phase=import_and_tests\n'
cd "${HC_REPO}"
"${HC_PYTHON}" - <<'PY'
import json
import habitat_sim
import torch
from habitat_sim._ext import habitat_sim_bindings
assert habitat_sim.built_with_bullet
print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "habitat": habitat_sim.__file__, "bindings": habitat_sim_bindings.__file__,
                  "built_with_bullet": habitat_sim.built_with_bullet}))
PY
"${HC_PYTHON}" -m pytest -q
"${HC_PYTHON}" -m pip freeze > "${HC_ROOT}/outputs/environment-${SLURM_JOB_ID}.txt"
printf 'phase=complete\n'

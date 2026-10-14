#!/usr/bin/env bash
# Assemble and verify the authorized 41-scene dataset on a CPU compute node.
#SBATCH --job-name=humanclaw-scenes
#SBATCH --account=cpu-g2-stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/stf/hengyil6/outputs/scenes-%j.out
#SBATCH --error=/gscratch/stf/hengyil6/outputs/scenes-%j.err

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through sbatch.}"
source /gscratch/stf/hengyil6/HumanCLAW/tools/klone_env.sh
cd "${HC_REPO}"
HC_HSSD="${HC_ROOT}/data/datasets/hssd-hab-4369cb987621"
HC_SUPPLEMENT="${HC_ROOT}/data/datasets/supplements/humanclaw-hssd-val41-supplement-v1-fd3422b302fc"
HC_PREPARED="${HC_ROOT}/data/humanclaw-hssd-val41"
[[ -f "${HC_HSSD}/humanclaw-download.json" ]] || {
  printf 'Complete the public asset download and transfer first.\n' >&2; exit 2;
}
[[ -d "${HC_SUPPLEMENT}/blobs" ]] || {
  printf 'The verified, extracted supplement is missing.\n' >&2; exit 2;
}
printf 'phase=scene_assembly_and_hashes\n'
"${HC_PYTHON}" -m humanclaw_bench.main prepare-hssd \
  --hssd-root "${HC_HSSD}" --supplement "${HC_SUPPLEMENT}" --output "${HC_PREPARED}" \
  --workers 16

"${HC_PYTHON}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["HC_ROOT"])
repo = Path(os.environ["HC_REPO"])
target = root / "data/humanclaw-hssd-val41"
link = repo / "data/humanclaw-hssd-val41"
link.parent.mkdir(parents=True, exist_ok=True)
if link.is_symlink() and link.resolve() == target.resolve():
    pass
elif link.exists() or link.is_symlink():
    raise FileExistsError(f"Preserving existing data path: {link}")
else:
    link.symlink_to(target, target_is_directory=True)
PY
printf 'phase=bundled_assets_and_weights\n'
"${HC_PYTHON}" -m humanclaw_bench.main assets \
  --weights-root "${HC_REPO}/weights/paper_fullval_v1" \
  > "${HC_ROOT}/outputs/assets-${SLURM_JOB_ID}.json"
printf 'phase=tests\n'
"${HC_PYTHON}" -m pytest -q
printf 'phase=complete\n'

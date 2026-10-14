#!/usr/bin/env bash
# Validate the prepared simulator, motion models and replay without VLM calls.
# These diagnostic artifacts are not benchmark results.
#SBATCH --job-name=humanclaw-runtime-check
#SBATCH --account=gpu-l40-stf
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/stf/hengyil6/outputs/runtime-check-%j.out
#SBATCH --error=/gscratch/stf/hengyil6/outputs/runtime-check-%j.err

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through sbatch.}"
source /gscratch/stf/hengyil6/HumanCLAW/tools/klone_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export HC_SMOKE_OUTPUT="${HC_ROOT}/outputs/runtime-smoke-${SLURM_JOB_ID}"
cd "${HC_REPO}"
[[ -f data/humanclaw-hssd-val41/prepared.json ]] || {
  printf 'The prepared scene-data link is missing.\n' >&2; exit 2;
}
[[ ! -e "${HC_SMOKE_OUTPUT}" ]] || {
  printf 'Preserving an existing runtime-check output.\n' >&2; exit 2;
}

printf 'phase=cuda_and_motion\n'
"${HC_PYTHON}" - <<'PY'
import json
import numpy as np
import torch
import habitat_sim
from humanclaw_bench.motion.runner import MotionSkillRunner
from humanclaw_bench.paths import resolve_release_path

assert torch.cuda.is_available(), "CUDA is unavailable"
assert habitat_sim.built_with_bullet, "Habitat was built without Bullet"
torch.manual_seed(0)
runner = MotionSkillRunner()
try:
    runner.reset("pt", None, str(resolve_release_path("resources/seeds/natural_beta0_fk_seed.pt")),
                 [0.0, 0.0, 0.0], 0.0)
    motion = runner.generate("walk_forward", [0.0, 0.2, 0.0]).xb_world_75
    assert motion.ndim == 2 and motion.shape[0] > 0 and motion.shape[1] == 75
    assert np.isfinite(motion).all(), "Generated motion contains non-finite values"
    print(json.dumps({"cuda_available": True, "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "loaded_skills": sorted(runner.skills), "motion_shape": list(motion.shape)}))
finally:
    runner.unload()
PY

printf 'phase=controller_contract\n'
"${HC_PYTHON}" tests/runtime_hp_contract.py
printf 'phase=scene_physics_and_video\n'
"${HC_PYTHON}" tests/runtime_habitat_smoke.py --output "${HC_SMOKE_OUTPUT}"
printf 'phase=physics_and_render_replay\n'
"${HC_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
from humanclaw_bench.experiments.replay import replay_run

source = Path(os.environ["HC_SMOKE_OUTPUT"])
reports = {}
for mode in ("physics", "render"):
    output = source.with_name(source.name + "-" + mode)
    reports[mode] = replay_run(source, mode=mode, output_dir=output,
                               max_steps=1 if mode == "physics" else None)
    if mode == "physics":
        assert reports[mode].get("passed") is True, "Physics replay diverged; inspect its report"
result = {"status": "passed", "purpose": "environment_validation",
          "model_api_calls": 0, "benchmark_result": False,
          "source": str(source), "replay": reports}
source.with_name(source.name + "-readiness.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
printf 'phase=complete\n'

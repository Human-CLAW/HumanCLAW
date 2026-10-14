#!/usr/bin/env bash
# A short driver/toolchain probe; no model calls or scene rollout.
#SBATCH --job-name=humanclaw-gpu-check
#SBATCH --account=gpu-l40-stf
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/stf/hengyil6/outputs/gpu-check-%j.out
#SBATCH --error=/gscratch/stf/hengyil6/outputs/gpu-check-%j.err
set -euo pipefail
umask 077
source /gscratch/stf/hengyil6/HumanCLAW/tools/klone_env.sh
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version
"${HC_PYTHON}" - <<'PY'
import ctypes
import json
import os
cuda = ctypes.CDLL('libcuda.so.1')
result = cuda.cuInit(0)
if result:
    raise SystemExit('CUDA driver initialization failed: {}'.format(result))
print(json.dumps({'cuda_driver_initialization': True,
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}))
PY

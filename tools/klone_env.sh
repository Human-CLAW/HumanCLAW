#!/usr/bin/env bash
# Source this file before using the prepared Klone environment.
export HC_ROOT="${HC_ROOT:-/gscratch/stf/hengyil6}"
export HC_REPO="${HC_REPO:-${HC_ROOT}/HumanCLAW}"
export HC_PYTHON="${HC_PYTHON:-${HC_ROOT}/envs/humanclaw/bin/python}"
export HUMANCLAW_BENCH_HOME="${HC_REPO}"
export XDG_CACHE_HOME="${HC_ROOT}/data/cache"
export HF_HOME="${HC_ROOT}/data/cache/huggingface"
export UV_CACHE_DIR="${HC_ROOT}/data/cache/uv"
export UV_PYTHON_INSTALL_DIR="${HC_ROOT}/envs/python"
export PIP_CACHE_DIR="${HC_ROOT}/data/cache/pip"
export TORCH_HOME="${HC_ROOT}/data/cache/torch"
export CUDA_CACHE_PATH="${HC_ROOT}/data/cache/cuda"
export MPLCONFIGDIR="${HC_ROOT}/data/cache/matplotlib"
export TMPDIR="${HC_ROOT}/outputs/tmp"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
export PATH="${HC_ROOT}/envs/humanclaw/bin:${PATH}"
module load gcc/11.2.0 cmake/3.25.1 cuda/12.4.1 realitylab/glvnd/1.7.0
export LIBRARY_PATH="${GLVND_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CMAKE_PREFIX_PATH="${GLVND_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
# This GLVND module was built with /usr/local as its prefix, while Klone's
# NVIDIA vendor JSON is installed under /usr/share. Keep explicit overrides.
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES+x}" && -z "${__EGL_VENDOR_LIBRARY_DIRS+x}" ]]; then
  export __EGL_VENDOR_LIBRARY_DIRS="/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:/usr/local/etc/glvnd/egl_vendor.d:/usr/local/share/glvnd/egl_vendor.d"
fi

<div align="right"><a href="README_zh.md">中文</a></div>

# Patched Habitat-Sim

Apply `humanclaw_halfphysics.patch` at the root of a clean Habitat-Sim checkout
at commit `acbe6f4922e68145e401e55c30f9dfea460a3f24`, after initializing
submodules. The expected Bullet commit is
`2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5`.

Before applying:

```bash
git apply --check /path/to/humanclaw_halfphysics.patch
git apply /path/to/humanclaw_halfphysics.patch
```

The patch has SHA-256
`6f57ec8130b4ccca7d208a0754ba691d52e44b469cd6dcee220fa193fcad6766`.
It changes the human URDF collision margin from 1 mm to 1 cm, suppresses
free-motion articulated-body bias acceleration in the HumanClaw control path,
and passes desired velocity through the spherical joint motor.

See `resources/provenance/paper_fullval_v1.json` for patched source hashes and
the audited AVA binding hash.

## Reproducible headless build

The evaluated runtime uses all three build options: **headless**, **CUDA**, and
**Bullet**. CUDA is therefore not an optional release-build flag even though
Bullet performs the rigid-body simulation on the CPU.

Before building, make sure the active host provides Git, CMake, a C++ compiler,
`nvcc`, and OpenGL/EGL/GLX development headers and linker stubs. These graphics
development files are still required for a headless EGL build. On Ubuntu, the
validated package set is:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build \
  libglvnd-dev libgl-dev libegl-dev libglx-dev libopengl-dev \
  libgles-dev libglu1-mesa-dev
```

On a host without root access, the same headers and libraries may be extracted
into a local sysroot and supplied through `CPATH`, `LIBRARY_PATH`, and
`LD_LIBRARY_PATH`. Ninja is optional but substantially shortens a clean build.
Also verify that the PyTorch installed in the HumanClawBench environment can
see a GPU:

```bash
git --version
cmake --version
gcc --version
nvcc --version
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

From the HumanClawBench virtual environment:

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout acbe6f4922e68145e401e55c30f9dfea460a3f24
git submodule update --init --recursive
git apply --check /absolute/path/to/HumanClawBench/patches/habitat-sim/humanclaw_halfphysics.patch
git apply /absolute/path/to/HumanClawBench/patches/habitat-sim/humanclaw_halfphysics.patch

python -m pip install -r requirements.txt
python setup.py build_ext --inplace --headless --with-cuda --bullet --cache-args
HEADLESS=True WITH_CUDA=True WITH_BULLET=True python -m pip install -e .
python -m pip install -e build/deps/magnum-bindings/src/python
```

The explicit environment variables on the editable-install command are
intentional: `pip install -e .` can invoke another native build, and Habitat's
pip defaults do not enable CUDA. The cached setup arguments preserve any
additional CMake options from the first build.

Verify the native extension before preparing HSSD or starting a rollout:

```bash
python - <<'PY'
import habitat_sim
from habitat_sim._ext import habitat_sim_bindings

assert habitat_sim.built_with_bullet
print(habitat_sim.__file__)
print(habitat_sim_bindings.__file__)
PY
```

## Minimal CUDA installations

Some compute hosts provide `nvcc` but omit CUDA runtime, cuRAND, or CCCL
headers. A complete CUDA toolkit needs no special handling. If CMake reports
`CUDART_LIBRARY=NOTFOUND`, `curand_kernel.h` is missing, or `nv/target` is
missing, install the corresponding NVIDIA CUDA 12 packages into the active
environment:

```bash
python -m pip install \
  nvidia-cuda-runtime-cu12 \
  nvidia-curand-cu12 \
  nvidia-cuda-cccl-cu12

NVIDIA_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia"
export CPATH="${NVIDIA_SITE}/cuda_runtime/include:${NVIDIA_SITE}/curand/include:${NVIDIA_SITE}/cuda_cccl/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/curand/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/curand/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python setup.py build_ext --inplace --headless --with-cuda --bullet \
  --cmake-args="-DCUDART_LIBRARY=${NVIDIA_SITE}/cuda_runtime/lib/libcudart.so.12"
```

Recent CMake releases can reject policy versions used by Habitat-Sim's pinned
third-party dependencies (for example zstd). If configuration stops with a
compatibility-policy error, retain the same build flags and add:

```bash
python setup.py build_ext --inplace --headless --with-cuda --bullet \
  --cache-args --cmake \
  --cmake-args="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

This selects CMake's compatibility floor; it does not change the pinned
Habitat-Sim, Bullet, or HumanCLAW source revisions.

Habitat-Sim automatically uses `ccache` when it is discoverable. If the
installed `ccache` executable itself fails, disable only that optional wrapper
while preserving the same build:

```bash
python setup.py build_ext --inplace --headless --with-cuda --bullet --cmake \
  --cmake-args="-DCUDART_LIBRARY=${NVIDIA_SITE}/cuda_runtime/lib/libcudart.so.12 -DCCACHE_FOUND:FILEPATH="
```

If importing the completed extension reports `libgomp.so.1` missing, use the
OpenMP runtime distributed with the installed PyTorch rather than adding a
whole system library directory to `LD_LIBRARY_PATH`:

```bash
TORCH_LIB="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Adding `/usr/lib64` wholesale can mix an incompatible system glibc into a
Python environment with its own runtime loader and should be avoided.

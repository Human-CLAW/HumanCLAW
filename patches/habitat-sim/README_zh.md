<div align="right"><a href="README.md">English</a></div>

# 打过补丁的 Habitat-Sim

请在一个干净的 Habitat-Sim checkout 根目录中应用
`humanclaw_halfphysics.patch`。目标 commit 是
`acbe6f4922e68145e401e55c30f9dfea460a3f24`，应用前需先初始化
submodule。预期的 Bullet commit 为
`2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5`。

应用前先检查：

```bash
git apply --check /path/to/humanclaw_halfphysics.patch
git apply /path/to/humanclaw_halfphysics.patch
```

该补丁的 SHA-256 为
`6f57ec8130b4ccca7d208a0754ba691d52e44b469cd6dcee220fa193fcad6766`。
它将人体 URDF 的碰撞 margin 从 1 mm 改为 1 cm，在 HumanClaw 控制路径中
抑制自由运动 articulated body 的 bias acceleration，并将期望速度传递给
spherical joint motor。

补丁源文件的 hash 和经过审计的 AVA binding hash 见
`resources/provenance/paper_fullval_v1.json`。

## 可复现的 headless 构建

论文评测运行时同时使用 **headless**、**CUDA** 和 **Bullet** 三个构建选项。
因此，即使 Bullet 的刚体模拟在 CPU 上运行，CUDA 也不是 release 构建中可省略
的选项。

构建前请确认当前机器提供 Git、CMake、C++ compiler 和 `nvcc`。Ninja 不是
必需项，但能明显缩短干净构建时间。同时确认 HumanClawBench 环境中的 PyTorch
能识别 GPU：

```bash
git --version
cmake --version
gcc --version
nvcc --version
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

在 HumanClawBench 的虚拟环境中执行：

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout acbe6f4922e68145e401e55c30f9dfea460a3f24
git submodule update --init --recursive
git apply --check /absolute/path/to/HumanClawBench/patches/habitat-sim/humanclaw_halfphysics.patch
git apply /absolute/path/to/HumanClawBench/patches/habitat-sim/humanclaw_halfphysics.patch

python -m pip install -r requirements.txt
python setup.py build_ext --inplace --headless --with-cuda --bullet
python -m pip install -e .
python -m pip install -e build/deps/magnum-bindings/src/python
```

准备 HSSD 或开始 rollout 前，先验证 native extension：

```bash
python - <<'PY'
import habitat_sim
from habitat_sim._ext import habitat_sim_bindings

assert habitat_sim.built_with_bullet
print(habitat_sim.__file__)
print(habitat_sim_bindings.__file__)
PY
```

## 精简 CUDA 安装

部分计算节点提供 `nvcc`，却没有 CUDA runtime、cuRAND 或 CCCL header。完整
CUDA toolkit 不需要额外配置。若 CMake 报告 `CUDART_LIBRARY=NOTFOUND`、缺少
`curand_kernel.h` 或缺少 `nv/target`，在当前环境中安装对应的 CUDA 12 包：

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

Habitat-Sim 会自动使用可发现的 `ccache`。若失败的是 `ccache` 程序本身，只禁用
这个可选 wrapper，保持其他构建选项不变：

```bash
python setup.py build_ext --inplace --headless --with-cuda --bullet --cmake \
  --cmake-args="-DCUDART_LIBRARY=${NVIDIA_SITE}/cuda_runtime/lib/libcudart.so.12 -DCCACHE_FOUND:FILEPATH="
```

若完成后的 extension 在导入时报告缺少 `libgomp.so.1`，应使用已安装 PyTorch
附带的 OpenMP runtime，而不是把整个系统库目录加入 `LD_LIBRARY_PATH`：

```bash
TORCH_LIB="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

不要直接加入整个 `/usr/lib64`；对于带独立 runtime loader 的 Python 环境，这样
可能混入不兼容的系统 glibc。

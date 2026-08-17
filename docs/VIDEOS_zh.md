<div align="right"><a href="VIDEOS.md">English</a></div>

# 视频

HumanClawBench 将 rollout 时的视频采集、trajectory 延迟渲染和展示视频合成
彼此分开。三者均为可选功能，不会改变 benchmark action 或 metric。

## Rollout 时保存 ego 和 exo 视频

为 `humanclaw-bench run` 添加 `--video`。每个 rollout 会得到同步的 `ego.mp4`
和 `exo.mp4`。不开启该 flag 时，不创建 exo sensor 和视频 encoder。

## 从保存的 trajectory 渲染

如果 rollout 时没有开启视频，可以直接渲染保存的 post-physics pose，无需重新
运行 physics、motion generation 或 VLM：

```bash
humanclaw-bench render \
  --rollout-dir outputs/fullval/EPISODE/rollout_00
```

完整输出树使用 `humanclaw-bench render-batch`；并行和 GPU 参数见 README。

## 合成 ego、exo 和 reasoning

合成命令读取已有的 `ego.mp4`、`exo.mp4`、`trajectory_before.npz` 和
`stepNNN_percept_mid_low.json`。若实际调用过 verifier，还会读取对应的
`stepNNN_verifier.json`。输出将两个视角左右排列，并在下方显示原样保存的
percept、mid-level reasoning、goal、low-level reasoning、verifier decision 和
最终执行 action。文本时间严格来自 trajectory 的 step frame offset。

合成单个 rollout：

```bash
humanclaw-bench compose-video \
  --rollout-dir outputs/fullval/EPISODE/rollout_00
```

默认输出为
`outputs/fullval/EPISODE/rollout_00/full_ego_exo_reasoning.mp4`；可用 `--output`
指定其他路径。

并行合成完整输出树：

```bash
humanclaw-bench compose-video-batch \
  --input-root outputs/fullval \
  --output-root outputs/fullval_composite \
  --max-parallel 8
```

批处理会在 output root 下保留输入目录结构，并支持续跑：帧数正确的已有视频会
被跳过，除非使用 `--force`。合成过程不加载 Habitat、physics、motion model 或
VLM，只写最终 MP4；临时 subtitle 文件会自动删除。

该功能需要带有 libass `subtitles` filter 的系统 `ffmpeg` 和 `ffprobe`：

```bash
ffmpeg -hide_banner -filters | grep subtitles
```


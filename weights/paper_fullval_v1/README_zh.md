<div align="right"><a href="README.md">English</a></div>

# 评测 weights

该目录是论文评测所需的完整 inference-only weight 集合，包含一个 canonical
MotionDiT state 和八个 skill-specific ControlNet state。Optimizer state、trainer
state、重复的 base tensor 和 checkpoint-selection 逻辑均已移除。这里的
*canonical* 指用于准确重建的唯一 full-precision MotionDiT reference state，
并不是额外的一套模型。

准确的文件、大小、SHA-256、architecture 和 source-checkpoint provenance 固定在
`resources/weights/paper_fullval_v1.json`。Runtime 不会扫描 newest checkpoint。

## 为什么 manifest 中有两个 base variant

归档 bundle 会独立加载每个 skill：

1. 加载 base checkpoint；
2. 围绕该 base 构造对应 skill 的 ControlNet；
3. 严格加载 skill checkpoint 的完整 `model_state`。

第三步也会加载内嵌的 `base_dit.*` tensor，因此真正决定评测模型的是这份内嵌
副本，而不只是第一步打开的 base 文件。逐 tensor 审计发现两个分组：

| Base variant | Skills |
|---|---|
| `bf16_roundtrip` | `walk_forward`、`turn`、`stop`、`sit` |
| `fp32` | `side_walk`、`step_back`、`step_climb_up`、`step_climb_down` |

它们不是两套独立训练的 MotionDiT。第一个 variant 的全部 112 个 tensor
（41,114,331 elements），都逐元素等于对第二个 variant 的 canonical base 执行
`FP32 -> BF16 -> FP32`。根因出现在 500k-to-1500k resume 路径。1500k job
配置为 `param_dtype=fp32`，并先构造同一个 canonical FP32 base，但严格 resume
随后加载了更早的完整 `model_state`，其中包括原本 frozen 的 `base_dit.*`
branch。`walk_forward`、`turn`、`stop`、`sit` 的实际 500k resume state 把
该 branch 保存为 `torch.bfloat16`；加载进 FP32 parameter 后，得到上面的准确
舍入 variant。`step_climb_up` 的 500k resume state 保存 FP32 branch，其余
三个 FP32 组 skill 也保留 canonical FP32 值。

换句话说，这里存在一次训练时的 precision/configuration 疏忽：四个 skill run
在 500k resume checkpoint 中使用了 BF16-valued tensor，而 continuation code
意外允许该 BF16 副本覆盖刚加载的 FP32 frozen base。这不是“不同 skill 应使用
不同 base precision”的有意设计。现在若替换为 full-precision base，会改变当时
被评测的 motion，因此 release 不做这种替换。

这一历史 precision mismatch 不改变论文结果：论文 benchmark 使用归档 full
checkpoint，因此实际使用的就是这些 tensor。Compact loader 会准确重建两种
variant，不会 retrain、average 或以其他方式修改它们。

## 无损压缩

Release 只保存一次 canonical FP32 base，在加载时确定性生成 rounded variant，
并让四个 skill 共享每个 immutable variant。每个 skill 文件只保存
`base_dit.*` 之外的 key。已经准确落在 BF16 grid 上的 control tensor 可以用
BF16 序列化；PyTorch 将它们加载进 FP32 module 时会准确恢复原始 FP32 值。
Release validation 会将重建结果与全部八个归档 `model_state` 做 tensor equality。

### 等价性验证

Compact 文件只有在对照 evaluation bundle 中八个固定的 1.5M trainer
checkpoint 后才被接受：

1. 每个归档 source file 都匹配 manifest 中记录的 SHA-256。
2. Compact loader 重建每个完整模型，并将全部 1,888 个 tensor
   （645,163,754 elements，包括每个 skill 的 base）与归档 `model_state`
   比较。Key、shape、dtype 和 value 均一致；所有 value comparison 都通过
   `torch.equal`。
3. 随后使用相同 history、condition、timestep 和 initial Gaussian noise，分别
   运行归档 model implementation/full checkpoint 与 release implementation/
   compact checkpoint。八个 skill 的 direct network forward 和完整 30-step
   midpoint integration 都 bit-identical，最大 absolute error 为 0。

因此，体积缩减只改变 checkpoint serialization，不改变重建出的 inference
tensor 或生成 motion。

### 为什么四个 skill 文件约 84 MB，另四个约 158 MB

Release 中出现 BF16，只是为了准确复现上述训练疏忽，并不是新的 inference-time
近似。最终 1.5M trainer checkpoint 在 FP32 container 中保存 `model_state`
tensor，但 storage dtype 并不意味着每个数都使用了全部 FP32 precision。
对于 `walk_forward`、`turn`、`stop`、`sit`，`ctrl_blocks.*` 中的
36,766,720 个值因其 BF16 500k resume state 而准确落在 BF16 numeric grid 上。
对每一个 source tensor `value`，以下等式逐元素成立：

```python
torch.equal(value, value.to(torch.bfloat16).to(torch.float32))
```

因此 release 用 BF16（每个元素两 bytes）保存这些值。当 `load_state_dict`
将其复制进 FP32 runtime module 时，会准确恢复归档 checkpoint 中的 FP32 值；
这不是近似 quantization。仍需 FP32 的 condition encoder 和 zero projection 保持
FP32。于是每个小 skill 文件大约由 73.5 MB BF16 control-block tensor，加上
10.5--11.6 MB FP32 tensor 组成。

另外四个 skill 的 control tensor 不满足准确 round-trip equality，因此保持
FP32（每个元素四 bytes），每个文件约 158 MB。把它们转成 BF16 会有损，无法
继续复现被评测模型。

| Runtime file | 保存的 control precision | 大小（decimal MB） |
|---|---|---:|
| `base/motion_dit.pt` | FP32 | 164.5 |
| `skills/walk_forward.pt` | BF16-grid tensor 用 BF16；其余 FP32 | 84.2 |
| `skills/side_walk.pt` | FP32 | 157.6 |
| `skills/step_back.pt` | FP32 | 157.7 |
| `skills/turn.pt` | BF16-grid tensor 用 BF16；其余 FP32 | 85.1 |
| `skills/step_climb_up.pt` | FP32 | 158.7 |
| `skills/step_climb_down.pt` | FP32 | 158.7 |
| Stop（`skills/stand.pt`） | BF16-grid tensor 用 BF16；其余 FP32 | 84.1 |
| `skills/sit.pt` | BF16-grid tensor 用 BF16；其余 FP32 | 85.1 |

九个文件合计 1,135,639,268 bytes（1.136 GB，或 1.058 GiB）。

九个归档 trainer checkpoint 占 3,227,420,338 bytes；下面九个固定 inference
tensor 文件占 1,135,639,268 bytes，实现 64.8% 无损缩减。

评测只需要以下文件：

```text
base/motion_dit.pt
skills/walk_forward.pt
skills/side_walk.pt
skills/step_back.pt
skills/turn.pt
skills/step_climb_up.pt
skills/step_climb_down.pt
skills/stand.pt
skills/sit.pt
```

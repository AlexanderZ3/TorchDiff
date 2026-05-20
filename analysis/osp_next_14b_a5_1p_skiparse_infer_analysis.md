# OSPNext 14B A5 1P 非 HiF8 推理中的 Skiparse 分析

本文针对 `configs/infer/npu/osp_hif8_14b_A51P.yaml` 这条单卡 A5 1P 推理路径，分析在不使用 HiF8 量化时，`skiparse` 在哪里生效、优化了什么，以及理论效率收益边界。结论只覆盖 `TorchDiff` 中 NPU 推理相关代码，不讨论 GPU 分支和训练链路。

## 1. 当前 YAML 对 skiparse 的含义

配置中与单卡和 skiparse 直接相关的字段如下：

- 单卡并行关闭：`fsdp_size: 1`、`cp_size: 1`、`skiparse_cp_size: 1`、`use_context_parallel: False`、`use_skiparse_context_parallel: False`。
- 模型内部 skiparse 仍开启：`skiparse_model_type: "dual_end"`、`sparse_ratio: 2`、`num_full_blocks: 8`、`skiparse_2d: True`。
- 若要分析“不要 HiF8 量化”的推理，应把 `model_config.quant` 和 `model_config.quant_attn` 改成 `null` 或删除，让 `_make_linear` 和 attention 走普通 bf16 路径，而不是 HiF8 linear / HiF8 attention。

这意味着：单卡 A5 1P 下，FSDP、普通 CP、skiparse CP 都不提供收益；但模型结构里的 skiparse attention 仍然会改变 self-attention 的 token 排布和每次 attention 的序列长度。

## 2. 推理入口和主链路

入口是 `eval/eval_osp.py`：

- 读取 `model_config` 中的 `sparse_ratio`、`skiparse_1d`、`skiparse_2d`、`num_full_blocks`。
- 单卡配置下 `use_skiparse_context_parallel` 会因为 `skiparse_cp_size == 1` 和配置为 `False` 而保持关闭。
- 模型实例化时仍把 `model_config` 传给 `OSPNextModel(**model_config)`。
- 推理时 `T2VInferencePipeline.__call__` 构造 latent，然后调用 `scheduler.sample(model=self.predictor, latents=latents, ...)`，扩散模型每一步都会进入 `OSPNextModel.forward`。

代码位置：

- `eval/eval_osp.py:50-54`：读取 skiparse 配置。
- `eval/eval_osp.py:101-129`：skiparse CP 只在多卡 `skiparse_cp_size > 1` 时初始化。
- `eval/eval_osp.py:192`：仍会调用 FSDP wrapper，但单卡 `fsdp_size=1` 时不构成跨卡优化。
- `torchdiff/pipelines/t2v_pipeline.py:201-218`：构造 latent 并进入 scheduler 采样。

## 3. Skiparse 在模型里的实际作用点

核心文件是 `torchdiff/modules/osp_next.py` 和 `torchdiff/modules/skiparse_func.py`。

### 3.1 block 类型分配

`OSPNextModel.__init__` 根据 `skiparse_model_type` 和 `num_full_blocks` 决定哪些 block 是 full attention，哪些是 sparse attention。

对于当前 14B 配置：

- `num_layers = 40`
- `num_full_blocks = 8`
- `skiparse_model_type = dual_end`

`dual_end` 的逻辑是：前 `num_full_blocks // 2 = 4` 个 block 和后 4 个 block 用 full attention，中间 32 个 block 用 skiparse。实际 full block index 是：

```text
0, 1, 2, 3, 36, 37, 38, 39
```

中间 block `4-35` 交替使用：

- 偶数 block：`SkiparseBlockType.Single`
- 奇数 block：`SkiparseBlockType.Group`

代码位置：

- `torchdiff/modules/osp_next.py:1351-1357`：`dual_end` full block index 计算。
- `torchdiff/modules/osp_next.py:1372-1377`：full / single / group block 类型分配。
- `torchdiff/modules/osp_next.py:1572-1578`：forward 中按 block 类型选择 mask 和 shard 信息。

### 3.2 每个 block 内部的作用点

每个 `OSPNextAttentionBlock.forward` 进入 block 后，先对输入 `x` 做 `rearrange_input`，再做 self-attention、cross-attention、FFN，最后做 `rearrange_output`。

代码位置：

- `torchdiff/modules/osp_next.py:1080-1128`：按 Full / Single / Group 构造不同的 `SkiparseRearrange`。
- `torchdiff/modules/osp_next.py:1185-1188`：进入 block 时重排 `x`、`text`、时间调制 `e`、register tokens。
- `torchdiff/modules/osp_next.py:1139-1163`：block 内先 self-attention，再 cross-attention，再 FFN。

对当前 `skiparse_2d=True, sparse_ratio=2` 而言：

- Full block：`rearrange_input = identity`，self-attention 看完整序列。
- 第一个 sparse block，也就是 block 4：从 full 序列进入 `skiparse_2d_single`。
- 后续 sparse block：在 `single` 和 `group` 两种稀疏视角之间用 `skiparse_2d_single_to_group` / `skiparse_2d_group_to_single` 互相转换。
- 最后一个 sparse block，也就是 block 35：通过 `skiparse_2d_group_reverse` 回到 full 序列，交给 block 36-39。

### 3.3 具体重排函数

`skiparse_func.py` 里没有真正稀疏矩阵格式，核心是通过 `einops.rearrange` 把空间 token 分组到 batch 维，使每次 attention 只在局部子序列中做全注意力。

关键函数：

- `skiparse_2d_single`：把 `(H, W)` 按 `p, q` 间隔拆成 `p*q` 个子序列，batch 变为 `P^2 * B`，每个子序列长度约为原来的 `1 / P^2`。
- `skiparse_2d_group`：另一种 2D 分组视角，用于和 single 交替覆盖不同 token 邻接关系。
- `skiparse_2d_single_to_group` / `skiparse_2d_group_to_single`：single / group 之间转换。

代码位置：

- `torchdiff/modules/skiparse_func.py:40-46`：`skiparse_2d_single` 和 reverse。
- `torchdiff/modules/skiparse_func.py:48-59`：`skiparse_2d_group` 和 reverse。
- `torchdiff/modules/skiparse_func.py:63-77`：single / group 转换。
- `torchdiff/modules/osp_next.py:190-309`：`SkiparseRearrange` 统一调度这些重排函数。

## 4. 单卡 A5 1P 下优化的具体内容

### 4.1 真正被优化的是 self-attention 的二次复杂度

skiparse 只直接优化 diffusion transformer block 里的 self-attention。具体是把一层 full self-attention 的序列长度从 `N` 改成 `P^2` 个长度约 `N / P^2` 的 attention。

当 `P = sparse_ratio = 2` 时：

```text
full self-attn 计算量       ~ N^2
skiparse self-attn 总计算量 ~ P^2 * (N / P^2)^2 = N^2 / P^2 = N^2 / 4
```

所以每个 sparse self-attention block 的 attention score / softmax / AV 主体理论上约为 full attention 的 25%，即 self-attention 主体约 4x 降低。

代码位置：

- `torchdiff/modules/osp_next.py:800-991`：`OSPNextSelfAttention` 中 q/k/v 投影、RoPE、attention、输出投影。
- `torchdiff/modules/attention.py:481-510`：非 HiF8 时 `attention_with_mask` 在 NPU 上走 `scaled_dot_product_attention_with_mask`。

### 4.2 当前分辨率下的 token 数

pipeline 对 81 帧、720x1280 的 latent shape 是：

```text
latent T = (81 - 1) / 4 + 1 = 21
latent H = 720 / 8 = 90
latent W = 1280 / 8 = 160
```

模型 `patch_size=(1,2,2)`，进入 transformer 后：

```text
grid = 21 x 45 x 80
N = 75,600 tokens
```

`skiparse_2d` 的 padding 以 `sparse_ratio^2 = 4` 为粒度。H=45 会补到 48，W=80 不需要补。因此 sparse block 实际处理：

```text
padded grid = 21 x 48 x 80
N_pad = 80,640
single/group 子序列长度 = N_pad / 4 = 20,160
batch 维变为 4 * B
```

注意 padding 只影响 sparse block mask 和序列长度；full block 仍是原始 `75,600` token。

### 4.3 40 层整体 self-attention 理论收益

当前 40 层中：

- 8 层 full attention：每层 self-attn 主体为 `1.0x`
- 32 层 skiparse attention：每层 self-attn 主体约为 `0.25x`

只看 self-attention 的二次复杂度主体，整体比例是：

```text
(8 * 1.0 + 32 * 0.25) / 40 = 0.40
```

也就是说，整网 self-attention 主体理论计算量约降到 full attention 的 40%，对应约 `2.5x` 的 self-attention 主体理论加速。

这个数字不是端到端速度。端到端收益会小于 2.5x，因为以下部分没有被 skiparse 直接减少：

- q/k/v/o linear 投影
- cross-attention
- FFN 两层大矩阵乘
- layer norm、RoPE、time/text embedding
- scheduler、VAE decode、文本编码
- `einops.rearrange`、padding、mask、single/group 转换开销

在 14B 模型里 FFN 和 linear 投影占比很高，因此“整条推理链路”的加速通常会明显低于 self-attention 主体的理论 2.5x。没有实测 profiling 的情况下，不应声称一个固定端到端百分比。

## 5. 单卡时哪些 skiparse 能力不再生效

单卡配置关闭了两类并行收益：

### 5.1 skiparse context parallel 不生效

`use_skiparse_context_parallel=False` 且 `skiparse_cp_size=1`，因此 `SkiparseRearrange._skiparse_cp_scatter/_gather` 都会直接返回原张量，不做跨卡 scatter/gather。

相关代码：

- `eval/eval_osp.py:101-129`：只有 `skiparse_cp_size > 1` 才会建 skiparse CP group。
- `torchdiff/modules/osp_next.py:233-246`：未开启 skiparse CP 时 scatter/gather 是 no-op。
- `torchdiff/modules/skiparse_func.py:80-170`：分布式 2D single/group 转换包含 `all_to_all_single`，单卡路径不会走到。

### 5.2 Ulysses context parallel / full-block CP 不生效

`use_context_parallel=False` 且 `cp_size=1`，所以 `ContextParallelPreprocessor.preprocess/postprocess` 也直接返回，不切分序列。

相关代码：

- `torchdiff/modules/osp_next.py:444-480`：只有 `use_context_parallel()` 或 `use_full_blocks_context_parallel()` 且 `cp_size > 1` 时才切序列。
- `torchdiff/modules/osp_next.py:512-568`：同理，只有多卡 CP 时才 all-gather 还原。

所以，单卡 A5 1P 上 skiparse 的收益来源不是通信减少，也不是显存分片，而是单卡内每个 sparse self-attention 的序列长度变短。

## 6. 不使用 HiF8 时的差异

YAML 文件名和当前字段仍是 HiF8：

```yaml
quant: "hif8"
quant_attn: "hif8"
```

若要分析或运行“不要 HiF8 量化”的路径，需要改为：

```yaml
quant: null
quant_attn: null
weight_dtype: "bf16"
```

这样：

- Linear 层不再使用 HiF8 quant linear，回到普通 linear。
- Attention 不再走 `hif8_attention_with_mask`，而是走 `attention_with_mask`。
- 在 NPU 上，普通 attention 路径会进入 `scaled_dot_product_attention_with_mask`，也就是 PyTorch SDPA 路径。

skiparse 与 HiF8 是两套相对独立的优化：

- skiparse：减少 sparse self-attention 的序列二次复杂度。
- HiF8：降低 linear/attention 的数值精度和带宽/算力压力。

因此关掉 HiF8 后，skiparse 仍然有效；但 linear、FFN、attention kernel 的单算子吞吐可能下降，端到端性能会比 HiF8 配置差。

## 7. 最终回答

### skiparse 作用在哪里？

在 `OSPNextModel` 的 40 个 transformer block 中作用于 self-attention 前后的 token 排布。当前 14B A5 1P 配置下，block `0-3` 和 `36-39` 是 full attention；block `4-35` 是 skiparse attention，按 single/group 交替。具体重排由 `SkiparseRearrange` 调用 `skiparse_2d_single`、`skiparse_2d_group`、`skiparse_2d_single_to_group`、`skiparse_2d_group_to_single` 完成。

### 提升效率是多少？

只看 self-attention 的二次复杂度主体：

- sparse block 单层 attention 主体理论约 `4x` 降低计算量。
- 40 层里 32 层 sparse、8 层 full，所以整网 self-attention 主体理论约降到 full attention 的 `40%`，即约 `2.5x` self-attention 主体理论加速。

端到端推理速度不能直接等于 2.5x。因为 FFN、linear、cross-attention、VAE、T5、scheduler、数据保存和重排开销不按这个比例下降。需要用 NPU profiling 实测才能给出端到端收益。

### 具体优化哪些地方？

优化的是 diffusion transformer 中 self-attention 的 `QK^T`、softmax 和 `Attn*V` 这类随序列长度平方增长的部分。它不直接减少 FFN、cross-attention、q/k/v/o 投影、VAE、T5 文本编码。单卡下也不提供 FSDP、CP、skiparse CP 的并行/通信收益。


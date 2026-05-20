# SkiParse 单卡 A5 推理分析

**分析路径**: `osp_next` 14B · NPU A5 单卡 (1P) · 无 HIF8 量化  
**主配置文件**: `configs/infer/npu/osp_hif8_14b_A51P.yaml`  
**核心代码**: `torchdiff/modules/osp_next.py`, `torchdiff/modules/skiparse_func.py`

---

## 1. 关键配置（单卡）

```yaml
fsdp_size: 1
cp_size: 1
skiparse_cp_size: 1
use_context_parallel: False
use_skiparse_context_parallel: False   # 单卡关键开关

skiparse_model_type: "dual_end"
sparse_ratio: 2
num_full_blocks: 8
skiparse_1d: False
skiparse_2d: True                      # 使用 2D 稀疏模式
num_layers: 40
```

无 HIF8 时将 `quant`/`quant_attn` 改为 `null`，`nn.Linear` 替代 `HIF8Linear`，`attention_with_mask` 替代 `hif8_attention_with_mask`。

---

## 2. SkiParse 是什么

SkiParse 是一种**稀疏注意力机制**，通过对视频 token 序列做特定的重排（rearrange），把原本需要做 O(N²) 全局自注意力的序列，拆解成 P² 组各长 N/P² 的子序列，在每组内部做局部/稀疏注意力，从而显著降低计算复杂度。

核心思想：**不是通过 mask 跳过计算，而是通过物理重排让序列变短，然后对短序列做完整注意力。**

---

## 3. Token 数量推算（720p 81 帧）

| 阶段 | 形状 | Token 数 |
|------|------|---------|
| 原始视频 | `[B,3,81,720,1280]` | — |
| VAE 压缩后 latent（T 4×，空间 8×） | `[B,16,21,90,160]` | — |
| Patch Embedding `(1,2,2)` 后 | `[B, T×H×W, C]` → `[B,75600,5120]` | 75,600 |
| SkiParse 2D padding（H: 45→48） | `[B, 80640, 5120]` | 80,640 |
| SkiParse rearrange 后（P=2） | `[4B, 20160, 5120]` | 20,160/组 × 4 组 |

padding 原因：`block_size = sparse_ratio² = 4`；H=45 不整除 4，需补 3 行到 48。

---

## 4. Block 布局（dual_end 模式）

`dual_end` 把 `num_full_blocks=8` 个 Full Block 对称放在首尾：

```
skiparse_start_index = num_full_blocks // 2 = 4
skiparse_end_index   = num_layers - 4 - 1  = 35

Block 索引:
  0,1,2,3           → Full Block（前 4）
  4                 → Single Block（is_full2skiparse_block=True）
  5                 → Group Block
  6                 → Single Block
  7                 → Group Block
  ...（交替）...
  34                → Single Block
  35                → Group Block（is_skiparse2full_block=True）
  36,37,38,39       → Full Block（后 4）
```

共 8 个 Full + 16 个 Single + 16 个 Group = 40 blocks。

---

## 5. SkiParse 2D 两种稀疏模式

### 5.1 Single 模式（间隔采样）

```python
# skiparse_func.py L40-46
def skiparse_2d_single(x, grid_sizes, sparse_ratio):
    T, H, W = grid_sizes
    return rearrange(x, 'b (t h p w q) c -> (p q b) (t h w) c',
                     p=sparse_ratio, q=sparse_ratio,
                     h=H//sparse_ratio, w=W//sparse_ratio)
```

效果：在 H 和 W 方向以 `sparse_ratio=2` 为步长间隔采样。  
每组 token 覆盖空间上**均匀分布的稀疏采样点**，保留全局远程依赖。

### 5.2 Group 模式（相邻聚合）

```python
# skiparse_func.py L48-59
def skiparse_2d_group(x, grid_sizes, sparse_ratio):
    T, H, W = grid_sizes
    return rearrange(x, 'b (txh p1 p2 w q1 q2) c -> (p1 q1 b) (txh p2 w q2) c',
                     p1=sr, q1=sr, p2=sr, q2=sr, w=W//(sr**2))
```

效果：将相邻 2×2 的 token 聚合成一个子组。  
每组 token 覆盖一个**连续的局部空间块**，保留局部细节。

### 5.3 两者配合

| 属性 | Single | Group |
|------|--------|-------|
| 空间覆盖 | 全局稀疏（步长 2） | 局部聚合（2×2 块） |
| 保留信息 | 远程结构、低频信息 | 局部纹理、高频细节 |
| 交替目的 | 使每层既有长程又有短程注意力 | 同左 |

两者**互为补充**：Single 让 token 看到远处，Group 让 token 看清周围。交替排列保证每两层后信息完整流通。

---

## 6. 单卡下 SkiParse 的执行路径

### 6.1 CP 相关代码全部短路

由于 `use_skiparse_context_parallel()=False`，以下均退化为无操作：

```python
# osp_next.py L233-247
def _skiparse_cp_scatter(self, x, dim=0):
    if not use_skiparse_context_parallel():
        return x                            # 直接返回，无通信

def _skiparse_cp_gather(self, x, dim=0):
    if not use_skiparse_context_parallel():
        return x                            # 直接返回，无通信
```

`_dispatch_rearrange` 中的 2D single⇄group 也走本地路径：

```python
# osp_next.py L270-277
if rt in (Skiparse2DSingle2Group, Skiparse2DGroup2Single):
    if use_skiparse_context_parallel():     # False，跳过
        return parallel_fn(...)
    return plain_fn(x, grid_sizes, sr)      # 走这里：纯 einops
```

### 6.2 单卡实际执行流程

```
forward() 入口
│
├── context_preprocessor.preprocess()  → cp_size=1，直接返回 x
│
├── mask_preprocessor.preprocess()
│     为 padding token 生成 bool mask（H=45 时有 3 个 padding token 行）
│     local_single_mask / local_group_mask 传给对应类型的 block
│
└── for block in self.blocks:
      │
      ├── [Full Block, indices 0-3, 36-39]
      │     rearrange_input = Identity       → x 不变
      │     context_rearrange_input = Identity → text/e 不变
      │     self-attention: [B, 75600, 5120]，全序列，attn_mask=None
      │
      ├── [Single Block, indices 4,6,8,...,34]
      │     rearrange_input = Skiparse2DSingle
      │       [B, 75600, C] → [4B, 20160, C]  (einops rearrange，无通信)
      │     context_rearrange_input = Repeat
      │       text [B, 512, C] → [4B, 512, C]
      │     self-attention: [4B, 20160, 5120]，序列长度缩短至 1/4
      │     rearrange_output = Identity
      │
      └── [Group Block, indices 5,7,9,...,35]
            rearrange_input = Skiparse2DGroup2Single（若非 skiparse2full）
              or Skiparse2DSingle2Group（若 is_skiparse2full_block）
            self-attention: [4B, 20160, 5120]，同 Single
            rearrange_output 按 block 位置决定是否 reverse
```

### 6.3 NPU 上的注意力实现

```python
# attention.py L499-507
if is_npu_available() or (not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE):
    output = scaled_dot_product_attention_with_mask(
        q=q, k=k, v=v,
        attn_mask=attn_mask_kv,  # bool mask [B, 1, Nq, Nkv]，NPU 需要
    )
```

A5 NPU 无法使用 FlashAttention，统一走 `torch.nn.functional.scaled_dot_product_attention`。  
SkiParse 的 mask 负责屏蔽 padding token。

---

## 7. 计算量分析

### 7.1 自注意力 FLOPs

以 720p 81 帧为基准（N=75,600，num_heads=40，head_dim=128）：

| 场景 | 注意力 FLOPs（每 block） |
|------|------------------------|
| Full Block | 2 × N² × head_dim × heads = **5.83 × 10¹³** |
| Sparse Block（P=2） | 1/P² × Full = **1.46 × 10¹³** |

**4 倍** 注意力 FLOPs 减少（每个 Sparse Block）。

### 7.2 全模型 FLOPs 分解

每 block 的主要开销（B=1）：

| 计算 | FLOPs | Full Block | Sparse Block |
|------|-------|-----------|--------------|
| 自注意力 QK^T | ~5.83×10¹³ | ✗ 全量 | ✓ 缩减 4× |
| FFN（2层线性） | ~2.14×10¹³ | 全量 | **全量（不变）** |
| Q/K/V/O 投影 | ~1.59×10¹² | 全量 | **全量（不变）** |
| Cross-Attention | ~3.96×10¹¹ | 全量 | **全量（不变）** |
| **合计** | | ~8.1×10¹³ | ~3.76×10¹³ |

> **重要结论**：SkiParse 只减少**自注意力**的计算，FFN、投影、Cross-Attention 的计算量不变。  
> 原因：P² 组 × N/P² tokens = 总 N tokens，线性运算总量不变；只有 QK^T 是 O(N²) 才能因缩短序列获益。

### 7.3 整体 FLOPs 对比

| | 计算量 |
|-|--------|
| 全 Full Block（无 SkiParse） | 40 × 8.1×10¹³ = **3.24×10¹⁵** |
| 实际（8 Full + 32 Sparse） | 8×8.1×10¹³ + 32×3.76×10¹³ = **1.85×10¹⁵** |
| **理论加速比** | **约 1.75×** |

换算：SkiParse 约减少 **43%** 的总 FLOPs。

其中自注意力的加速更显著：

```
自注意力总 FLOPs 减少：
  原始: 40 × 5.83e13 = 2.33e15
  优化: 8×5.83e13 + 32×1.46e13 = 4.664e14 + 4.672e14 = 9.34e14
  加速比: 2.33/0.934 ≈ 2.5×
```

### 7.4 内存收益

在注意力计算期间，QK^T 矩阵的内存占用从 O(N²) → O(N²/4)：

| | 每 Sparse Block 注意力矩阵大小 |
|-|-------------------------------|
| 无 SkiParse | 75,600² × 40 heads ≈ 217 GB（理论峰值） |
| 有 SkiParse | 20,160² × 40 × 4 组 ≈ 65 GB（理论峰值） |

实际 SDPA 会用流式方式避免物化全矩阵，但峰值 HBM 需求仍按此比例缩减（约 3.5×）。

---

## 8. 优化效果总结

| 优化维度 | 效果 | 说明 |
|---------|------|------|
| 自注意力 FLOPs | **−75%**（4×） | 针对 32/40 个 Sparse Block |
| 整体模型 FLOPs | **−43%**（1.75×） | FFN 不受影响，全 Full Block 不受影响 |
| 注意力内存峰值 | **约 −71%** | QK^T 矩阵缩减 ~3.5× |
| 多卡通信 | **无影响** | 单卡无 all_to_all/gather/scatter |
| 生成质量 | **保持** | Single+Group 交替保证全局+局部覆盖 |

---

## 9. 单卡 vs 多卡的差异

| 特性 | 单卡（A5 1P，当前情况） | 多卡（skiparse_cp_size > 1） |
|-----|------------------------|------------------------------|
| Token 重排 | 纯本地 einops rearrange | 重排前后有 all_to_all 通信 |
| Batch 扩展 | B → P²B，全在同一卡 | B/cp_size 分布在各卡 |
| Scatter/Gather | 无（直接返回） | all_gather + split |
| `_parallel_skiparse_2d_single_to_group` | 不调用（走 plain_fn） | 调用（含 dist.all_to_all_single） |
| 效率来源 | 短序列注意力本身 | 短序列 + 序列并行 |

---

## 10. 相关代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| SkiParse 元函数（rearrange 实现） | `modules/skiparse_func.py` | L7-78 |
| SkiparseRearrange 模块 | `modules/osp_next.py` | L190-372 |
| Block 类型分配逻辑 | `modules/osp_next.py` | L1339-1402 |
| Block forward（rearrange 入/出口） | `modules/osp_next.py` | L1183-1247 |
| NPU 注意力路径（SDPA） | `modules/attention.py` | L499-507 |
| CP 短路逻辑 | `modules/osp_next.py` | L233-247 |
| 推理入口 | `infer/infer_osp.py` | L103-388 |
| 单卡配置 YAML | `configs/infer/npu/osp_hif8_14b_A51P.yaml` | — |

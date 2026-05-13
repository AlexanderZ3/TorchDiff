# HiF8 技术特性与 Current Scaling 量化方案介绍

## 1. HiF8 数据格式技术特性

HiFloat8（HiF8）是华为昇腾针对深度学习训练与推理场景自研的新型 8-bit 浮点数据格式，已于 2025 年捐献给全球计算联盟（Global Computing Consortium, GCC）并开源。相比 NVIDIA 主导的 FP8-E4M3 / FP8-E5M2 格式，HiF8 在格式设计层面进行了系统性创新，在 8-bit 位宽约束下兼顾了更大的动态范围与更优的数值精度分布。

### 1.1 字段结构与 Dot 域创新

HiF8 由四个字段构成：1-bit 符号域（Sign）、可变长前缀编码的点位域（Dot）、符号-幅度编码的阶码域（Exponent）以及尾数域（Mantissa）。其核心创新在于引入 Dot 域——采用即时可译的变长前缀编码动态指示当前阶码的位宽，使得同一个 8-bit 编码空间能够根据数值大小自适应地在精度与动态范围之间进行权衡，而非 FP8 格式中阶码与尾数宽度固定不变的割裂设计。

### 1.2 锥形精度（Tapered Precision）

HiF8 最重要的特性是锥形精度分布。在靠近零的小幅值区间（指数 E ∈ [−3, 3]），HiF8 提供 3-bit 尾数精度；随着指数绝对值增大，尾数位宽平滑收缩至 1-bit，最终在极端幅值处退化为 0-bit 尾数的 Denormal 编码。这一特性使 HiF8 的精度资源集中在神经网络参数与激活值所呈现的类高斯分布的高频数值区间，与深度学习数据分布天然契合，有效抑制量化误差。

### 1.3 大动态范围与 Denormal 扩展

HiF8 Normal 模式支持的指数范围为 [−15, 15]（共 31 个指数值）。通过引入非标准的 Denormal 编码机制，HiF8 将指数范围向负方向额外扩展 7 个量级，最终覆盖 [−22, 15] 共 38 个指数值（binades），已接近 FP16 的 40 个 binades。相比之下，FP8-E4M3 仅覆盖约 16 个指数值，FP8-E5M2 约覆盖 32 个指数值但尾数精度更低。HiF8 的大动态范围显著降低了量化过程中的上溢（overflow）风险，是其在训练场景中表现稳健的根本原因之一。

### 1.4 无冗余编码

HiF8 阶码域采用符号-幅度（Sign-Magnitude）编码，并将幅值最高位设为隐含位（不占存储），确保不同宽度阶码域的指数表达范围严格互不重叠，实现 256 个编码状态的无冗余利用。此外，HiF8 对零值采用单一编码（不区分正零负零），并完整支持 NaN 和 ±Inf 等特殊值。

### 1.5 与主流格式的对比总结

| 格式 | 动态范围（binades） | 精度特性 | 特殊值支持 |
|------|------------|----------|------------|
| FP16 | ~40 | 固定 10-bit 尾数 | Zero / NaN / ±Inf |
| FP8-E4M3 | ~16 | 固定 3-bit 尾数 | Zero / NaN |
| FP8-E5M2 | ~32 | 固定 2-bit 尾数 | Zero / NaN / ±Inf |
| **HiF8** | **~38** | **锥形精度（0~3-bit 渐变尾数）** | **Zero / NaN / ±Inf** |

---

## 2. 基于 Current Scaling 的 HiF8 量化方案

量化的核心任务是为高精度张量（BF16/FP32）确定缩放因子（Scaling Factor），使张量中的数值范围安全映射至 8-bit 浮点格式可表示的动态范围内。对于给定张量 **X**，缩放因子 $s$ 的计算方式为：

$$s = \frac{\text{F8max}}{\text{Amax}(X)}$$

其中 $\text{F8max}$ 为 HiF8 可表示的最大值，$\text{Amax}(X) = \max|X_i|$ 为张量的最大绝对值。量化后的张量 $\hat{X}$ 满足：

$$\hat{X} = \text{round\_to\_hif8}(X \cdot s)$$

### 2.1 Current Scaling 工作流程

本项目采用的 **Current Scaling**（即时缩放）策略在每次训练迭代中遍历当前输入张量 **X**，实时统计 $\text{Amax}$，并据此计算本次迭代的缩放因子。其工作流程如下：

```
每次前向/反向传播：
  1. 遍历张量 X，计算 Amax = max|X_i|
  2. 计算 Scale = F8max / Amax
  3. 量化：X_q = round_to_hif8(X * Scale)
  4. 下游计算（如 MatMul）使用 X_q
  5. 反量化时乘以 1/Scale 恢复数值量级
```

Current Scaling 的主要优势在于**精度有保障**——缩放因子始终与当前输入分布精确匹配，没有 Delayed Scaling 中历史统计值滞后导致的上溢或下溢风险。这一特性在训练早期、学习率大幅变动或激活值分布骤变的阶段尤为重要，能够保证量化数值的稳定性。

### 2.2 HiF8 对 Per-Tensor 粗粒度缩放的支持

Current Scaling 通常与量化粒度（Quantization Granularity）相结合使用。量化粒度决定了为多大范围的数据共享同一个缩放因子：

- **Per-Tensor**：整个张量共享一个缩放因子，开销最小，但对分布不均匀的张量误差较大；
- **Per-Token / Per-Channel**：为每个 Token 或每个输出通道独立计算缩放因子，精度更高但开销增加；
- **Per-Block**：将张量划分为固定大小的块（如 128×128），每块独立缩放，精度与开销折中。

FP8-E4M3 由于动态范围仅约 16 个指数值，在训练中通常需要 Per-Token 或 Per-Block 的细粒度缩放才能维持精度。**HiF8 凭借其约 38 个指数值的大动态范围以及锥形精度对高频数值的保护，在训练场景下通常仅需 Per-Tensor 粒度的 Current Scaling 即可取得与细粒度 FP8 方案相当的精度表现。** 这一特性使量化逻辑更简洁，也降低了缩放因子本身带来的内存带宽与控制开销。

### 2.3 Current Scaling 与 Delayed Scaling 的比较

| 对比维度 | Current Scaling | Delayed Scaling |
|----------|----------------|-----------------|
| Amax 计算时机 | 每次迭代实时计算 | 使用历史统计值（每 K 次迭代更新） |
| 精度风险 | 无滞后，精度有保证 | 存在历史 Amax 估计偏差导致上溢 |
| 计算开销 | 与主计算串行，略有额外延迟 | Amax 计算与主计算可解耦，吞吐更高 |
| 实现复杂度 | 简单 | 需维护历史 Amax 缓存及 N_guard 保护参数 |
| 适用场景 | 训练稳定性优先；HiF8 Per-Tensor 量化 | 追求极致吞吐；配合 HiF8 大动态范围可适当延长更新周期 |

本项目基于 Current Scaling 方案实现 HiF8 量化，在保证每步训练量化精度的前提下，充分利用 HiF8 大动态范围带来的 Per-Tensor 粗粒度缩放优势，以较低的工程复杂度实现高效的 8-bit 混合精度训练。

---

## 参考资料

- Ascend HiFloat8 Format for Deep Learning. arXiv: [2409.16626](https://arxiv.org/abs/2409.16626)
- Unleashing Low-Bit Inference on Ascend NPUs: A Comprehensive Evaluation of HiFloat Formats. arXiv: [2602.12635](https://arxiv.org/abs/2602.12635)
- HiFloat8 开源仓库: [global-computing-consortium/HiFloat8](https://github.com/global-computing-consortium/HiFloat8)

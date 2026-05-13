# OSP-Next-14B HiFloat8 Inference on NPU

## 概述

OSP-Next 是基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1) 深度改造的新一代文生视频扩散模型，核心创新为 **Skiparse（稀疏跳跃注意力）** 机制：将标准全量 Attention 替换为交替的 Single / Group 稀疏注意力块，在保持视频质量的前提下大幅降低序列计算量。

**OSP-Next-14B-HiF8** 是业内首次在昇腾 NPU 上完成 8 比特量化训练和推理的 AIGC 文生视频模型。基于 Ascend 950 A5 原生支持的 **HiFloat8（HiF8）数据格式**进行量化感知训练（QAT），并完成了基于 GRPO 方法的 HiF8 强化学习后训练。由于 Atlas A3（Ascend 910C）不原生支持 HiF8，训练阶段借助 `quant_cy_npu` 仿真算子在 BF16 框架下对所有 Linear 投影层及 Attention Q/K/V 输入施加 HiF8 精度仿真，使权重收敛至 HiF8 精度范围。所得权重可在 Ascend 950 A5 上直接进行 HiF8 量化推理，无需格式转换。

本样例的并行策略与性能优化详情可参见 [TODO: 填写技术文档或论文链接]()。

## 硬件要求

- 产品型号：Ascend 950 A5（推理单卡即可，训练至少需要 8 卡服务器， 推荐4*8卡训练集群）
- 操作系统：Linux ARM
- 镜像版本：TODO（如 `cann8.0_pt2.8.0_aarch_image:v0.x`）
- 驱动版本：Ascend HDK TODO（运行 `npu-smi info` 确认）

> 如固件或驱动版本不符，请下载[固件和驱动](TODO: 填写驱动下载链接)后自行安装。

## 快速启动

### 下载源码

```bash
mkdir -p /home/code; cd /home/code/
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

### 下载权重

从 [TODO: 填写权重下载地址（ModelScope）]() 下载 OSP-Next-14B-HiF8 完整权重包，上传到服务器固定路径，例如 `/data/models/osp-next-14b-hif8`。

下载完成后目录结构参考如下：

```
/data/models/osp-next-14b-hif8/
├── ema_model_state_dict.pt            # DiT 主模型权重（HiF8 QAT 训练）
├── Wan2.1_VAE.pth                     # VAE 权重
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 文本编码器权重
└── google/
    └── umt5-xxl/                      # T5 tokenizer
```

### 获取 docker 镜像

从 [ARM 镜像地址](TODO: 填写镜像下载链接) 下载镜像，导入到 A5 服务器：

```bash
docker load -i TODO_镜像文件名.tar
```

### 拉起 docker 容器

OSP-Next-14B-HiF8 单张 Ascend 950 A5 即可完成推理，无需多卡。

```bash
docker run -u root -itd \
    --name cann_recipes_infer \
    --ulimit nproc=65535:65535 \
    --ipc=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /data/models:/data/models \
    -v /home/code/cann-recipes-infer:/home/code/cann-recipes-infer \
    --shm-size=128g \
    --privileged \
    TODO_镜像名称 /bin/bash
```

进入容器：

```bash
docker attach cann_recipes_infer
cd /home/code/cann-recipes-infer/models/osp-next-14b-hif8
```

### 修改代码

修改 `config/osp_next_14b_hif8.yaml` 中的权重路径：

```yaml
# 单卡推理，所有并行度设为 1
fsdp_size: 1
cp_size: 1
skiparse_cp_size: 1
use_context_parallel: False
use_skiparse_context_parallel: False

model_config:
  pretrained_model_dir_or_checkpoint: "/data/models/osp-next-14b-hif8/ema_model_state_dict.pt"

vae_config:
  vae_path: "/data/models/osp-next-14b-hif8/Wan2.1_VAE.pth"

text_encoder_config:
  checkpoint_path: "/data/models/osp-next-14b-hif8/models_t5_umt5-xxl-enc-bf16.pth"
  text_tokenizer_path: "/data/models/osp-next-14b-hif8/google/umt5-xxl"
```

修改推理提示词文件 `prompts.txt`（每行一条描述）：

```
A majestic eagle soaring over snow-capped mountains at sunrise, cinematic 4K.
Close-up of ocean waves crashing against rocky cliffs, slow motion, golden hour.
```

### 拉起推理

```bash
bash infer.sh
```

## 附录

### YAML 配置参数说明

| 参数                    | 说明                                           | 默认值                        |
| ----------------------- | ---------------------------------------------- | ----------------------------- |
| `prompt_txt`          | 提示词文件路径，每行一条描述                   | `prompts.txt`               |
| `output_dir`          | 生成视频保存目录                               | `samples/osp_next_14b_hif8` |
| `num_frames`          | 生成帧数                                       | `81`（约 5 秒 @16fps）      |
| `height` / `width`  | 输出分辨率                                     | `720` / `1280`（720p）    |
| `num_inference_steps` | 扩散采样步数                                   | `50`                        |
| `guidance_scale`      | 分类器自由引导强度                             | `5.0`                       |
| `quant`               | Linear 层量化类型，`"hif8"` 启用 HiF8        | `"hif8"`                    |
| `quant_attn`          | Attention Q/K/V 量化类型，`"hif8"` 启用 HiF8 | `"hif8"`                    |
| `scale_max_forward`   | HiF8 前向激活/权重缩放上界                     | `15.0`                      |

### HiF8 量化覆盖范围

每个 OSPNextAttentionBlock（共 40 层）中，以下位置被 HiF8 精度约束覆盖：

| 子模块                                 | 量化类型                   | 数量/层 |
| -------------------------------------- | -------------------------- | ------- |
| Self-Attention Q/K/V/O 投影（Linear）  | 激活 × 权重双量化         | 4       |
| Cross-Attention Q/K/V/O 投影（Linear） | 激活 × 权重双量化         | 4       |
| FFN Linear1 / Linear2                  | 激活 × 权重双量化         | 2       |
| Attention Q/K/V → SDPA 输入           | per-tensor Current Scaling | 每层    |

> HiF8 采用 Current Scaling 策略（`scale_max_forward=15`），每次前向实时计算 per-tensor 缩放因子，通过 `quant_cy_npu` 算子实现。

### FAQ

- **`ImportError: quant_cy_npu`**：HiF8 NPU 算子包未编译安装，请参考 [TODO: quant_cy_npu 编译说明]() 执行 `bash build.sh`。若只需验证视频生成效果，可将 YAML 中 `quant` 和 `quant_attn` 均设为 `null`，退回标准 BF16 推理。
- **`HCCL_BUFFSIZE` 不足**：报错含 `"HCCL_BUFFSIZE is too SMALL"`，通过 `export HCCL_BUFFSIZE=实际所需大小` 解决，详见[昇腾资料](TODO: 填写链接)。
- **显存不足（OOM）**：OSP-Next-14B-HiF8 推理仅需单张 Ascend 950 A5，**无需多卡**。若遇 OOM，请确认硬件型号是否为 A5，并检查 `fsdp_size`、`cp_size`、`skiparse_cp_size` 是否均为 `1`，`use_skiparse_context_parallel` 是否为 `False`。

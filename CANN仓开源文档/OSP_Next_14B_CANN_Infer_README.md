# OSP-Next-14B BF16 / HiF8 推理样例

## 概述

本样例面向 `cann-recipes-infer` 仓库，提供 OSP-Next-14B 在昇腾 NPU 上的文生视频推理说明。OSP-Next 基于 Wan2.1-T2V-14B 改造，引入 **Skiparse（稀疏跳跃注意力）** 结构，在 720p、81 帧视频生成场景下减少长序列 Attention 的计算压力。

本次拟开源两个训练好的权重：

| 模型权重 | 精度/训练方式 | 适用硬件 | 下载地址 |
|----------|----------------|----------|----------|
| OSP-Next-14B-BF16 | BF16 正常训练权重，支持 NPU 推理 | Atlas A2/A3、Ascend 950 A5 | TODO：填写 ModelScope / GitCode 下载地址 |
| OSP-Next-14B-HiF8 | 已训练完成的 HiF8 权重，面向 Ascend 950 A5 开箱即用 | Ascend 950 A5 | TODO：填写 ModelScope / GitCode 下载地址 |

> 说明：本 README 是入仓前文档框架。当前任务暂不修改 `cann-recipes-infer` 仓库代码，后续需按照 `CONTRIBUTION.md` 将样例代码、配置、依赖和优化文档整理到目标仓库。

## 支持特性

- 支持 OSP-Next-14B BF16 权重推理。
- 支持 OSP-Next-14B HiF8 权重推理。
- 支持 2D Skiparse 稀疏注意力结构。
- 支持 FSDP、标准 Context Parallel 和 Skiparse Context Parallel 多卡推理。
- 支持 Ascend 950 A5 HiF8 推理配置。
- 支持 T5 文本编码器、Wan2.1 VAE 与 OSP-Next DiT 权重分路径加载。

## 硬件与软件要求

| 场景 | 推荐硬件 | 卡数 | 说明 |
|------|----------|------|------|
| BF16 多卡推理 | Atlas A2/A3 或 Ascend 950 A5 | 8 卡或 16 卡 | 使用 `infer/infer_osp.py` 与 `configs/infer/npu/osp_14b.yaml` |
| HiF8 多卡推理 | Ascend 950 A5 | 8 卡 | 使用已训练好的 HiF8 权重 |
| HiF8 单卡推理 | Ascend 950 A5 | 1 卡 | 使用 `infer/infer_osp_A51P.py` 与 `osp_hif8_14b_A51P.yaml` |

软件环境参考：

| 软件 | 版本 |
|------|------|
| OS | Linux ARM，具体发行版 TODO |
| CANN | TODO：建议与 torch_npu 版本匹配 |
| Python | 3.10 |
| PyTorch | `torch==2.8.0` |
| torch_npu | `torch_npu==2.8.0.post2` |
| 其他依赖 | `opencv-python`、`diffusers`、`transformers`、`imageio-ffmpeg`、`pyyaml` 等 |

## 目录结构建议

后续合入 `cann-recipes-infer` 时，建议按模型目录组织：

```text
models/osp-next-14b
├── README.md
├── requirements.txt
├── infer.sh
├── generate.py 或 infer_osp.py
├── config
│   ├── osp_next_14b_bf16.yaml
│   ├── osp_next_14b_hif8.yaml
│   └── osp_next_14b_hif8_single_a5.yaml
├── assets
│   └── prompts.txt
└── torchdiff
    ├── modules
    ├── pipelines
    └── distributed
```

## 环境准备

### 1. 下载源码

```bash
mkdir -p /home/code
cd /home/code

git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/models/osp-next-14b
```

（TODO:正式入仓前，需将当前 TorchDiff 中的推理入口、模型实现、NPU 分布式工具和配置整理到该目录。)

### 2. 安装依赖

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

conda create -n osp-next-infer python=3.10 -y
conda activate osp-next-infer

pip install torch==2.8.0 torch_npu==2.8.0.post2 torchvision==0.23.0
pip install -r requirements.txt
pip install -e .
```

TODO：

- 填写推荐 Docker 镜像名称。
- 填写 CANN 下载与安装链接。
- 填写 torch_npu 安装链接。

## 权重准备

### BF16 权重

从 TODO 下载 OSP-Next-14B-BF16 权重包，建议放到 `/data/models/osp-next-14b-bf16`：

```text
/data/models/osp-next-14b-bf16/
├── ema_model_state_dict.pt            # OSP-Next DiT 主模型权重
├── Wan2.1_VAE.pth                     # Wan2.1 VAE 权重
├── models_t5_umt5-xxl-enc-bf16.pth    # T5 文本编码器权重
└── google/
    └── umt5-xxl/                      # T5 tokenizer
```

### HiF8 权重

从 TODO 下载 OSP-Next-14B-HiF8 权重包，建议放到 `/data/models/osp-next-14b-hif8`：

```text
/data/models/osp-next-14b-hif8/
├── ema_model_state_dict.pt            # OSP-Next-14B-HiF8 DiT 主模型权重
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── google/
    └── umt5-xxl/
```

OSP-Next-14B-HiF8 权重面向 Ascend 950 A5 推理场景提供，用户下载权重包并配置路径后即可运行，无需额外转换权重格式。

## 快速启动：BF16 推理

### 1. 修改配置

打开 `config/osp_next_14b_bf16.yaml`，修改权重路径、输出路径和提示词路径：

```yaml
model_name: "osp_next"
pipeline_name: "t2v"
seed: 1024

prompt_txt: "assets/prompts.txt"
output_dir: "samples/osp_next_14b_bf16"

num_frames: 81
height: 720
width: 1280
save_fps: 16
batch_size: 1

fsdp_size: 8
cp_size: 2
skiparse_cp_size: 4
use_context_parallel: true
use_skiparse_context_parallel: true
weight_dtype: "bf16"
save_with_dcp_api: false

model_config:
  dim: 5120
  ffn_dim: 13824
  freq_dim: 256
  in_dim: 16
  num_heads: 40
  num_layers: 40
  out_dim: 16
  text_len: 512
  skiparse_model_type: "dual_end"
  sparse_ratio: 2
  num_full_blocks: 8
  num_register_tokens: 0
  skiparse_1d: false
  skiparse_2d: true
  pretrained_model_dir_or_checkpoint: "/data/models/osp-next-14b-bf16/ema_model_state_dict.pt"

scheduler_config:
  scheduler_name: "flow_matching"
  num_inference_steps: 50
  shift: 7.0
  guidance_scale: 5.0

vae_config:
  vae_path: "/data/models/osp-next-14b-bf16/Wan2.1_VAE.pth"
  dtype: "fp16"

text_encoder_config:
  checkpoint_path: "/data/models/osp-next-14b-bf16/models_t5_umt5-xxl-enc-bf16.pth"
  text_tokenizer_path: "/data/models/osp-next-14b-bf16/google/umt5-xxl"
  use_fsdp: true
```

### 2. 准备提示词

`assets/prompts.txt` 每行一条文生视频提示词：

```text
A majestic eagle soaring over snow-capped mountains at sunrise, cinematic 4K.
Close-up of ocean waves crashing against rocky cliffs, slow motion, golden hour.
```

### 3. 启动推理

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600

torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29505 \
  infer/infer_osp.py \
  --config config/osp_next_14b_bf16.yaml
```

## 快速启动：HiF8 推理

### 1. 多卡 HiF8 推理

打开 `config/osp_next_14b_hif8.yaml`，将权重路径指向已下载的 OSP-Next-14B-HiF8 权重目录：

```yaml
model_config:
  skiparse_model_type: "dual_end"
  sparse_ratio: 2
  num_full_blocks: 8
  skiparse_1d: false
  skiparse_2d: true
  pretrained_model_dir_or_checkpoint: "/data/models/osp-next-14b-hif8/ema_model_state_dict.pt"

vae_config:
  vae_path: "/data/models/osp-next-14b-hif8/Wan2.1_VAE.pth"

text_encoder_config:
  checkpoint_path: "/data/models/osp-next-14b-hif8/models_t5_umt5-xxl-enc-bf16.pth"
  text_tokenizer_path: "/data/models/osp-next-14b-hif8/google/umt5-xxl"
```

启动：

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29505 \
  infer/infer_osp.py \
  --config config/osp_next_14b_hif8.yaml
```

### 2. Ascend 950 A5 单卡 HiF8 推理

如使用 Ascend 950 A5 单卡，可关闭所有并行：

```yaml
fsdp_size: 1
cp_size: 1
skiparse_cp_size: 1
use_context_parallel: false
use_skiparse_context_parallel: false
text_encoder_config:
  use_fsdp: false
```

启动：

```bash
export ASCEND_VISIBLE_DEVICES=0

torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29505 \
  infer/infer_osp_A51P.py \
  --config config/osp_next_14b_hif8_single_a5.yaml
```

## 配置参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `prompt_txt` | 提示词文件，每行一条 prompt | `assets/prompts.txt` |
| `output_dir` | 视频输出目录 | `samples/osp_next_14b_*` |
| `num_frames` | 生成帧数 | `81` |
| `height` / `width` | 输出分辨率 | `720` / `1280` |
| `num_inference_steps` | Flow Matching 采样步数 | `50` |
| `guidance_scale` | CFG 强度 | `5.0` |
| `fsdp_size` | FSDP 并行度 | 多卡按 world size 调整，单卡为 `1` |
| `cp_size` | 标准 Context Parallel 并行度 | 多卡可设 `2`，单卡为 `1` |
| `skiparse_cp_size` | Skiparse Context Parallel 并行度 | 多卡可设 `4`，单卡为 `1` |
| `sparse_ratio` | Skiparse 稀疏比例 | `2` |
| `num_full_blocks` | 保留全量 Attention Block 数 | `8` |

并行度约束：

- `world_size = nnodes * nproc_per_node`。
- 多卡 Skiparse 推理时，`world_size` 需能被 `cp_size * skiparse_cp_size` 整除。
- 单卡推理必须将 `fsdp_size`、`cp_size`、`skiparse_cp_size` 设为 `1`，并关闭 `use_context_parallel`、`use_skiparse_context_parallel`。

## 输出结果

推理完成后，视频文件保存到 `output_dir`，建议按模型与时间组织：

```text
samples/
├── osp_next_14b_bf16/
│   └── *.mp4
└── osp_next_14b_hif8/
    └── *.mp4
```

## 性能数据

TODO：补充实测性能表。

| 权重 | 硬件 | 卡数 | 规格 | 采样步数 | 端到端耗时 |
|------|------|------|------|----------|------------|
| OSP-Next-14B-BF16 | TODO | TODO | 1280 x 720 x 81 | 50 | TODO |
| OSP-Next-14B-HiF8 | TODO | TODO | 1280 x 720 x 81 | 50 | TODO |

## FAQ

**1. HiF8 权重支持哪些设备？**

当前 OSP-Next-14B-HiF8 权重面向 Ascend 950 A5 推理场景提供。Atlas A2/A3 请使用 OSP-Next-14B-BF16 权重。

**2. 单卡推理 OOM 怎么处理？**

先确认使用单卡配置：`fsdp_size=1`、`cp_size=1`、`skiparse_cp_size=1`，并关闭所有 CP。仍 OOM 时可降低分辨率、帧数或采样 batch size。



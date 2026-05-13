# OSP-Next-14B Skiparse 训练样例

## 概述

本样例面向 `cann-recipes-train` 仓库，提供 OSP-Next-14B 在昇腾 NPU 上进行 Skiparse 稀疏注意力训练的参考流程。OSP-Next 基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1) T2V-14B 基座模型改造，核心变化是将部分全量 Attention Block 替换为 **Skiparse（稀疏跳跃注意力）** Block，通过 Single / Group 稀疏重排交替建模，在保持生成质量的同时降低长视频序列上的注意力计算压力。

本次拟开源内容包括：

| 内容 | 说明 | 拟入仓位置 |
|------|------|------------|
| 训练 recipe | OSP-Next-14B Skiparse BF16 训练流程、配置与启动脚本 | `multimodal_rl/osp-next-14b/` 或社区评审后指定目录 |
| 训练配置 | NPU 多机多卡 FSDP、Context Parallel、Skiparse Context Parallel 配置 | `configs/train/npu/osp_14b.yaml` |
| 训练入口 | 文生视频扩散模型训练入口 | `train/train_osp.py` |
| 参考权重 | 训练从 Wan2.1-T2V-14B 权重初始化 | TODO：填写 Wan2.1-T2V-14B 权重下载地址 |

> 说明：本 README 是入仓前说明文档框架。当前任务暂不修改 `cann-recipes-train` 仓库代码，后续实际入仓时需按照 `CONTRIBUTION.md` 补齐样例代码、依赖文件、优化文档和 PR 说明。

## 模型与训练方案

OSP-Next-14B 继承 Wan2.1-T2V-14B 的 DiT、T5 文本编码器和 VAE 组件，并在 DiT 主干中引入 Skiparse 结构。

核心配置如下：

| 配置项 | 取值 |
|--------|------|
| 基座模型 | `Wan2.1-T2V-14B` |
| 模型类型 | `osp_next` |
| 参数规模 | 14B 级别 |
| 训练任务 | Text-to-Video |
| 输出规格 | `1280 x 720 x 81 frames`，16 FPS |
| 主干层数 | 40 |
| Hidden dim | 5120 |
| Attention heads | 40 |
| FFN dim | 13824 |
| Skiparse 类型 | `dual_end` |
| 稀疏方式 | 2D Skiparse，`skiparse_2d: true` |
| 保留全量 Attention Block | `num_full_blocks: 8` |
| 训练精度 | BF16 |
| 优化器 | AdamW，学习率默认 `2e-5`，权重衰减默认 `1e-3` |

Skiparse 训练关注以下能力：

- 在视频 token 的空间维度进行 2D 稀疏重排，降低长序列 Attention 的显存和计算压力。
- 在模型两端保留全量 Attention Block，中间层使用 Single / Group 稀疏注意力交替建模。
- 结合 FSDP、Context Parallel 和 Skiparse Context Parallel 支持 14B 模型在 NPU 集群上训练。
- 支持 EMA 权重保存，可直接作为后续推理开源权重的来源。

## 硬件要求

推荐使用昇腾多卡服务器或多机集群训练。

| 场景 | 推荐硬件 | 卡数建议 | 说明 |
|------|----------|----------|------|
| 功能验证 | Atlas A2/A3 | 至少 16 卡 | 当前 `osp_14b.yaml` 中 `cp_size * skiparse_cp_size = 16`，建议 world size 不小于 16 |
| 常规训练 | Atlas A2/A3 集群 | 32 卡及以上 | 可使用 FSDP + Skiparse CP 承载 720p 81 帧训练 |
| 大规模长稳训练 | Atlas A2/A3 多机集群 | 4 机 x 8 卡或更高 | 推荐用于完整收敛和大数据集训练 |

软件环境参考：

| 软件 | 版本 |
|------|------|
| OS | Linux ARM，具体发行版 TODO |
| CANN | TODO：建议与 `torch_npu==2.8.0.post2` 匹配 |
| Python | 3.10 或 3.11 |
| PyTorch | `torch==2.8.0` |
| torch_npu | `torch_npu==2.8.0.post2` |
| 其他依赖 | 见 `requirements_npu.txt` |

## 目录结构建议

后续合入 `cann-recipes-train` 时，建议按模型样例组织目录：

```text
osp-next-14b
├── README.md                         # 本训练说明
├── requirements.txt                  # NPU 训练依赖
├── configs
│   └── train
│       └── npu
│           └── osp_14b.yaml          # OSP-Next-14B Skiparse BF16 训练配置
├── scripts
│   └── train
│       └── npu
│           └── train_osp_14b_multinode.sh
├── train
│   └── train_osp.py                  # 训练入口
├── torchdiff                         # 模型、数据、分布式与调度器实现
└── docs
    └── osp_next_14b_skiparse_train_optimization.md  # TODO：训练优化文档
```

## 环境准备

### 1. 获取代码

```bash
mkdir -p /home/code
cd /home/code

git clone https://gitcode.com/cann/cann-recipes-train.git
cd cann-recipes-train
```

后续实际入仓后，进入样例目录：

```bash
cd multimodal_rl/osp-next-14b
```

> (TODO: 当前本地参考实现来自 `D:\Doc\OSP\TorchDiff`。正式开源时需要按 `cann-recipes-train/CONTRIBUTION.md` 将样例代码整理进目标仓库，不应直接提交数据集或大模型权重。)

### 2. 安装 CANN 与 torch_npu

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

conda create -n osp-next-train python=3.10 -y
conda activate osp-next-train

pip install torch==2.8.0 torch_npu==2.8.0.post2 torchvision==0.23.0
pip install -r requirements.txt
pip install -e .
```

如使用官方容器镜像，可将镜像下载、`docker load`、`docker run` 命令补充到本节。

TODO：

- 填写推荐镜像名称。
- 填写 CANN 软件包下载链接。
- 填写 torch_npu 安装链接。

## 权重准备

训练从 Wan2.1-T2V-14B 权重初始化。请下载以下文件并放到统一目录，例如 `/data/models/Wan2.1-T2V-14B`：

```text
/data/models/Wan2.1-T2V-14B/
├── want2v_14b.pt                     # DiT 初始化权重，文件名以实际转换产物为准
├── Wan2.1_VAE.pth                    # VAE 权重
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 文本编码器权重
└── google/
    └── umt5-xxl/                     # T5 tokenizer
```

若基座权重以 HuggingFace / ModelScope diffusers 格式发布，需要先转换为 TorchDiff 训练入口可加载的 `.pt` 权重。

TODO：

- 填写 Wan2.1-T2V-14B 权重下载地址。
- 填写权重转换脚本说明或转换后的权重包地址。

## 数据集准备

当前训练配置使用 `wan_t2v` 数据集格式，并通过 LMDB 元数据读取视频样本。数据集目录只在文档中说明，不随样例一起提交。

推荐目录结构：

```text
/data/datasets/osp-next/
├── filtered_samples.lmdb
└── videos/
```

配置字段示例：

```yaml
data_config:
  batch_size: 1
  num_workers: 16
  dataset_name: "wan_t2v"
  dataset_config:
    metafile_or_dir_path: "/data/datasets/osp-next/filtered_samples.lmdb"
    text_tokenizer_path: "/data/models/Wan2.1-T2V-14B/google/umt5-xxl"
    text_drop_ratio: 0.1
    sample_height: 720
    sample_width: 1280
    sample_num_frames: 81
    train_fps: 16
    tokenizer_max_length: 512
    return_prompt_mask: true
  sampler_name: "stateful_distributed"
  collator_name: "wan_t2v"
```

TODO：

- 填写开源可用示例数据集或数据准备脚本。
- 说明 LMDB 构建方式。
- 说明数据 License 与使用限制。

## 修改训练配置

打开 `configs/train/npu/osp_14b.yaml`，按实际环境修改路径。

```yaml
output_dir: "output/osp_next_14b_81f720p_sparse2d2_ssp4"

fsdp_size: 16
cp_size: 4
skiparse_cp_size: 4
use_context_parallel: false
use_skiparse_context_parallel: true
gradient_checkpointing: true
weight_dtype: "bf16"
ema_decay: 0.999
save_with_dcp_api: true

model_config:
  dim: 5120
  ffn_dim: 13824
  num_heads: 40
  num_layers: 40
  skiparse_model_type: "dual_end"
  sparse_ratio: 2          # 或按实验使用 4
  num_full_blocks: 8
  skiparse_1d: false
  skiparse_2d: true
  pretrained_model_dir_or_checkpoint: "/data/models/Wan2.1-T2V-14B/want2v_14b.pt"

vae_config:
  vae_path: "/data/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
  dtype: "fp16"

text_encoder_config:
  checkpoint_path: "/data/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
  use_fsdp: true
```

并行度约束：

- `world_size = nnodes * nproc_per_node`。
- `world_size` 需能被 `cp_size * skiparse_cp_size` 整除。
- 2D Skiparse 下，`skiparse_cp_size <= sparse_ratio ** 2` 且 `(sparse_ratio ** 2) % skiparse_cp_size == 0`。
- `fsdp_size <= world_size`，且推荐 `world_size % fsdp_size == 0`。

## 启动训练

### 单机多卡

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
  --nproc_per_node=16 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  train/train_osp.py \
  --config configs/train/npu/osp_14b.yaml
```

### 多机多卡

```bash
export NPROC_PER_NODE=8
export NNODES=4
export NODE_RANK=${NODE_RANK}
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=29501

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_osp.py \
  --config configs/train/npu/osp_14b.yaml
```

训练产物默认保存在 `output_dir`：

```text
output/osp_next_14b_81f720p_sparse2d2_ssp4/
├── iter_000000500/
│   ├── ema_model_state_dict.pt
│   └── ...
├── iter_000001000/
└── wandb/
```

其中 `ema_model_state_dict.pt` 可作为推理侧开源权重的 DiT 主模型权重。

## 配置参数说明

| 参数 | 说明 |
|------|------|
| `fsdp_size` | FSDP 分组大小 |
| `cp_size` | 标准 Context Parallel 并行度 |
| `skiparse_cp_size` | Skiparse 重排后的 Context Parallel 并行度 |
| `use_skiparse_context_parallel` | 是否启用 Skiparse CP |
| `sparse_ratio` | Skiparse 稀疏比例，2D 模式下约束与 `sparse_ratio ** 2` 相关 |
| `num_full_blocks` | 保留全量 Attention 的 Block 数 |
| `save_with_dcp_api` | 是否使用分布式 checkpoint API 保存 |
| `model_cpu_offload` | 是否启用模型 CPU offload |
| `text_encoder_config.use_fsdp` | T5 文本编码器是否使用 FSDP |
| `ema_decay` | EMA 衰减系数 |

## FAQ

**1. `world_size` 小于 16 能否直接跑 14B Skiparse 配置？**

不建议。当前 14B Skiparse NPU 配置默认 `cp_size=4`、`skiparse_cp_size=4`，`cp_size * skiparse_cp_size = 16`，world size 至少需要满足该并行切分约束。若只做功能冒烟，需要同步调小配置中的并行度和分辨率。

**2. 权重路径应该填目录还是 `.pt` 文件？**

`model_config.pretrained_model_dir_or_checkpoint` 填 DiT 主模型权重路径；当前 TorchDiff 训练入口可加载 `.pt` checkpoint。VAE、T5 encoder 和 tokenizer 分别在 `vae_config`、`text_encoder_config` 中指定。



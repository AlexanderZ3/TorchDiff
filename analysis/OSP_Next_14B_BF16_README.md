# OSP-Next-14B BF16 Inference on NPU

## 概述

OSP-Next 是基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1) 深度改造的新一代文生视频扩散模型，核心创新为 **Skiparse（稀疏跳跃注意力）** 机制：将标准全量 Attention 替换为交替的 Single / Group 稀疏注意力块，在保持视频质量的前提下大幅降低序列计算量。

**OSP-Next-14B-BF16** 在 BF16 精度下完成训练，并通过基于 GRPO 方法的强化学习后训练进一步提升视频生成质量，原生支持昇腾 NPU 推理。

本样例的并行策略与性能优化详情可参见 [TODO: 填写技术文档或论文链接]()。

## 硬件要求

支持以下两类昇腾硬件：

| 硬件 | 推理 | 训练（最低） | 训练（推荐） |
|------|------|------------|------------|
| Ascend 950 A5 | 单卡 | 单机 8 卡 | 4 机 × 8 卡集群 |
| Ascend 910 A2 / A3 | 单卡 | 单机 8 卡 | 8 机 × 16 卡集群 |

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

从 [TODO: 填写权重下载地址（ModelScope）]() 下载 OSP-Next-14B-BF16 完整权重包，上传到服务器固定路径，例如 `/data/models/osp-next-14b-bf16`。

下载完成后目录结构参考如下：

```
/data/models/osp-next-14b-bf16/
├── ema_model_state_dict.pt            # DiT 主模型权重
├── Wan2.1_VAE.pth                     # VAE 权重
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 文本编码器权重
└── google/
    └── umt5-xxl/                      # T5 tokenizer
```

### 获取 docker 镜像

从 [ARM 镜像地址](TODO: 填写镜像下载链接) 下载镜像，导入到服务器：

```bash
docker load -i TODO_镜像文件名.tar
```

### 拉起 docker 容器

OSP-Next-14B-BF16 单张昇腾 NPU 即可完成推理，无需多卡。

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
cd /home/code/cann-recipes-infer/models/osp-next-14b-bf16
```

### 修改代码

修改 `config/osp_next_14b_bf16.yaml` 中的权重路径：

```yaml
# 单卡推理，所有并行度设为 1
fsdp_size: 1
cp_size: 1
skiparse_cp_size: 1
use_context_parallel: False
use_skiparse_context_parallel: False

model_config:
  pretrained_model_dir_or_checkpoint: "/data/models/osp-next-14b-bf16/ema_model_state_dict.pt"

vae_config:
  vae_path: "/data/models/osp-next-14b-bf16/Wan2.1_VAE.pth"

text_encoder_config:
  checkpoint_path: "/data/models/osp-next-14b-bf16/models_t5_umt5-xxl-enc-bf16.pth"
  text_tokenizer_path: "/data/models/osp-next-14b-bf16/google/umt5-xxl"
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

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `prompt_txt` | 提示词文件路径，每行一条描述 | `prompts.txt` |
| `output_dir` | 生成视频保存目录 | `samples/osp_next_14b_bf16` |
| `num_frames` | 生成帧数 | `81`（约 5 秒 @16fps） |
| `height` / `width` | 输出分辨率 | `720` / `1280`（720p） |
| `num_inference_steps` | 扩散采样步数 | `50` |
| `guidance_scale` | 分类器自由引导强度 | `5.0` |
| `weight_dtype` | 推理精度 | `"bf16"` |

### FAQ

- **`HCCL_BUFFSIZE` 不足**：报错含 `"HCCL_BUFFSIZE is too SMALL"`，通过 `export HCCL_BUFFSIZE=实际所需大小` 解决，详见[昇腾资料](TODO: 填写链接)。

- **显存不足（OOM）**：推理仅需单张昇腾 NPU，**无需多卡**。若遇 OOM，请检查 `fsdp_size`、`cp_size`、`skiparse_cp_size` 是否均为 `1`，`use_skiparse_context_parallel` 是否为 `False`。

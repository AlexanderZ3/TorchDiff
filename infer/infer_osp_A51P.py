import os
import math
import yaml
from argparse import ArgumentParser
import torch.distributed as dist

import torch
from torchdiff.utils.utils import check_and_import_npu
check_and_import_npu()

from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

from torchdiff.distributed.utils import (
    setup_distributed_env,
    cleanup_distributed_env,
    gather_tensor_list_to_one,
    set_modules_to_forward_prefetch,
)
from torchdiff.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
from torchdiff.distributed.cp_state import cp_state
from torchdiff.modules import (
    WanVAE,
    T5EncoderModel,
    models,
    models_main_block,
    models_blocks_to_float,
    models_blocks_to_output_float,
)
from torchdiff.schedulers import schedulers
from torchdiff.distributed.checkpoint import Checkpointer
from torchdiff.utils.utils import str_to_precision, get_memory_allocated
from torchdiff.utils.log_utils import get_logger, log_on_main_process
from torchdiff.pipelines import pipelines
from torchdiff.utils.infer_utils import load_prompts, load_images, save_videos, save_video_grid
from torchdiff.utils.random_utils import set_seed

def load_lora_and_merge(
    model,
    lora_path,
    lora_rank=32,
    lora_alpha=64,
    lora_target_modules=None,
    logger=None,
    rank=0,
):
    """
    Load LoRA weights from a manually saved adapter_model.bin, then merge into base model.
    """
    from peft import LoraConfig, get_peft_model
    if lora_target_modules is None:
        lora_target_modules = [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        ]

    if not os.path.isfile(lora_path):
        raise ValueError(f"LoRA file not found: {lora_path}")

    if logger is not None:
        from torchdiff.utils.log_utils import log_on_main_process
        log_on_main_process(logger, f"Loading LoRA from {lora_path}")
        log_on_main_process(logger, f"LoRA rank={lora_rank}, alpha={lora_alpha}")
        log_on_main_process(logger, f"LoRA target_modules={lora_target_modules}")

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.set_adapter("default")

    lora_sd = torch.load(lora_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)

    if rank == 0:
        missing_lora = [k for k in missing if "lora_" in k]
        if missing_lora:
            print(f"[LoRA] missing lora keys: {len(missing_lora)}, example: {missing_lora[:5]}")
        print(f"[LoRA] unexpected keys: {len(unexpected)}")

    missing_lora = [k for k in missing if "lora_" in k]
    if len(unexpected) > 0:
        raise RuntimeError(f"LoRA load has unexpected keys, example: {unexpected[:20]}")
    if len(missing_lora) > 0:
        raise RuntimeError(f"LoRA load missing LoRA keys, example: {missing_lora[:20]}")

    if logger is not None:
        log_on_main_process(logger, "LoRA weights loaded successfully, merging into base model...")

    model = model.merge_and_unload()

    if logger is not None:
        log_on_main_process(logger, "LoRA merged into base model successfully.")

    return model

def main(config):
    logger = get_logger()

    # config analysis
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model_name", "osp_next")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})
    sparse_ratio = model_config.get("sparse_ratio", 1)
    skiparse_1d = model_config.get("skiparse_1d", False)
    skiparse_2d = model_config.get("skiparse_2d", False)
    num_full_blocks = model_config.get("num_full_blocks", 0)

    # inference config
    pipeline_name = config.get("pipeline_name", "t2v")
    weight_dtype = config.get("weight_dtype", "bfloat16")
    prompt_txt = config.get("prompt_txt", None)
    batch_size = config.get("batch_size", 1)
    num_frames = config.get("num_frames", 49)
    height = config.get("height", 480)
    width = config.get("width", 832)
    save_fps = config.get("save_fps", 16)
    use_context_parallel = config.get("use_context_parallel", False)
    use_skiparse_context_parallel = config.get("use_skiparse_context_parallel", False)
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    explicit_prefetching_num_blocks = config.get("explicit_prefetching_num_blocks", 0)

    # LoRA config
    lora_path = config.get("lora_path", None)
    lora_rank = config.get("lora_rank", 32)
    lora_alpha = config.get("lora_alpha", 64)
    lora_target_modules = config.get(
        "lora_target_modules",
        [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        ]
    )

    # save config
    output_dir = config.get("output_dir", "./output")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    # distributed setup
    setup_distributed_env()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # [A51P] NPU 通过 torch_npu 将 torch.cuda API 重定向到 NPU，此处写法与原版相同
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    # [A51P] 单卡安全检查：防止误用多卡配置导致 CP mesh 崩溃
    if world_size == 1:
        if use_context_parallel or use_skiparse_context_parallel:
            raise ValueError(
                "[A51P] world_size=1 但 use_context_parallel 或 use_skiparse_context_parallel "
                "为 True，单卡下不支持 CP，请在 yaml 中将其设为 False。"
            )
        log_on_main_process(logger, "[A51P] 单卡推理模式，跳过所有 Context Parallel 初始化。")
    else:
        raise RuntimeError(
            f"[A51P] 此脚本仅用于单卡推理（world_size=1），当前 world_size={world_size}。"
            "多卡请使用原始 infer_osp.py。"
        )

    # [A51P] 单卡下 fsdp_size/ddp_size 均为1，init_device_mesh((1,1)) 合法，fully_shard 退化为 no-op
    fsdp_size = config.get("fsdp_size", 1)
    if fsdp_size > world_size:
        fsdp_size = world_size
        log_on_main_process(logger, f"[A51P] fsdp_size 自动 clamp 为 {fsdp_size}")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))

    dp_group = dist.group.WORLD

    # [A51P] 单卡下完全跳过 CP mesh 初始化（原代码此块在 world_size>1 且 CP 开启时执行）
    # cp_state 保持默认值：global_cp_size=1, global_cp_rank=0, is_initialized=False

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, indent=4)

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_config.get("text_tokenizer_path", None))
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device,
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        # [A51P] 单卡下 use_fsdp=False，T5 直接驻留在单卡上，无需 FSDP 分片
        use_fsdp=text_encoder_config.get("use_fsdp", False),
        device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    )
    log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing diffusion model and scheduler...")

    scheduler = schedulers[scheduler_config.pop("scheduler_name", "flow_matching")](**scheduler_config)

    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)

    # [A51P] 检查路径是否仍为占位符
    if pretrained_model_dir_or_checkpoint and pretrained_model_dir_or_checkpoint.startswith("**"):
        raise ValueError(
            "[A51P] pretrained_model_dir_or_checkpoint 尚未填写，"
            "请在 osp_hif8_14b_A51P.yaml 中替换 ** 占位符为实际路径。"
        )

    has_loaded_pretrained_model = False
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Init model from scratch")
        with torch.device("meta"):
            model = models[model_name](**model_config)
    else:
        raise ValueError(f"In inference mode, pretrained_model_dir_or_checkpoint {pretrained_model_dir_or_checkpoint} must be specified!")

    # [A51P] 单卡下无 CP，跳过 CP 相关 num_heads 整除性检查
    model.eval()

    if not has_loaded_pretrained_model:
        model.to_empty(device=device)
        set_seed(seed, device_specific=False)
        model.reset_parameters()

    if pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_checkpoint {pretrained_model_dir_or_checkpoint}")
        if pretrained_model_dir_or_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load
            full_sd = safe_load(pretrained_model_dir_or_checkpoint, device="cpu")
        else:
            full_sd = torch.load(pretrained_model_dir_or_checkpoint, mmap=True, weights_only=True, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(full_sd, strict=False)
        if rank == 0:
            if missing_keys:
                print(f"[Base model checkpoint] missing_keys: {missing_keys[:20]}...")
            if unexpected_keys:
                print(f"[Base model checkpoint] unexpected_keys: {unexpected_keys[:20]}...")
        del full_sd
        has_loaded_pretrained_model = True

    if lora_path is not None:
        model = load_lora_and_merge(
            model=model,
            lora_path=lora_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            logger=logger,
            rank=rank,
        )

    # [A51P] 单卡下 FSDP2 mesh=(1,1)，fully_shard 退化为 no-op，不做实际分片
    FSDP2_mix_wrapper(
        model,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=weight_dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        blocks_to_output_float=models_blocks_to_output_float[model_name],
        reshard_after_forward=reshard_after_forward,
        cpu_offload=model_cpu_offload,
    )

    if explicit_prefetching_num_blocks > 0:
        set_modules_to_forward_prefetch(model.blocks, num_to_forward_prefetch=explicit_prefetching_num_blocks)

    log_on_main_process(logger, f"Diffusion model initialized, memory allocated: {get_memory_allocated()} GiB")

    pipeline = pipelines[pipeline_name](
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        predictor=model,
        scheduler=scheduler
    )

    prompts = load_prompts(prompt_txt)

    set_seed(seed, device_specific=True, process_group=dp_group)

    # [A51P] 单卡下 dp_rank=0, dp_size=1, cp_rank=0, cp_size=1（cp_state 未初始化时的默认值）
    dp_rank = torch.distributed.get_rank(dp_group)
    dp_size = torch.distributed.get_world_size(dp_group)
    cp_rank = cp_state.global_cp_rank   # 默认 0
    cp_size = cp_state.global_cp_size   # 默认 1
    cp_group = cp_state.global_cp_group # 默认 None

    # [A51P] dp_size=1 时 len(prompts) % 1 == 0 恒成立，无需 padding
    video_grid = []
    for index in range(dp_rank * batch_size, len(prompts), batch_size * dp_size):
        batch_prompts = prompts[index: index + batch_size]
        videos = pipeline(
            prompt=batch_prompts,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            max_sequence_length=512,
            device=device
        )
        # [A51P] cp_rank 恒为 0，始终保存
        save_videos(videos, index, output_dir, save_fps)
        video_grid.append(videos)

    if len(video_grid) > 0:
        video_grid = torch.cat(video_grid, dim=0).to(device)

    # [A51P] 单卡下无需跨 rank gather，直接保存
    video_grid = video_grid.cpu()
    save_video_grid(video_grid, output_dir, fps=save_fps)
    print("Inference finished.")
    print(f"Saved {video_grid.shape[0]} samples to {output_dir}")

    cleanup_distributed_env()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/infer/npu/osp_hif8_14b_A51P.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f"Config file not found: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)

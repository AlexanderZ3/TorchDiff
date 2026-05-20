import os
import yaml
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from torchdiff.utils.utils import check_and_import_npu
check_and_import_npu()

from torchdiff.modules import (
    WanVAE,
    T5EncoderModel,
    models,
    models_blocks_to_float,
)
from torchdiff.schedulers import schedulers
from torchdiff.distributed.utils import setup_distributed_env, cleanup_distributed_env
from torchdiff.utils.utils import str_to_precision, get_memory_allocated
from torchdiff.utils.log_utils import get_logger, log_on_main_process
from torchdiff.pipelines import pipelines
from torchdiff.utils.infer_utils import load_prompts, save_videos, save_video_grid
from torchdiff.utils.random_utils import set_seed


def _load_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load
        return safe_load(path, device="cpu")
    return torch.load(path, mmap=True, weights_only=True, map_location="cpu")


def _move_wan_model_to_single_device(model, model_name, device, weight_dtype):
    model.to(device=device, dtype=weight_dtype)
    for module in model.modules():
        for block_type in models_blocks_to_float.get(model_name, []) or []:
            if isinstance(module, block_type):
                module.to(dtype=torch.float32)
    return model


def _assert_single_card_config(config, world_size):
    if world_size != 1:
        raise RuntimeError(
            f"This A51P entry is single-card only, but world_size={world_size}. "
            "Use infer_t2v.py for multi-card inference."
        )

    forbidden_true_keys = [
        "use_context_parallel",
    ]
    for key in forbidden_true_keys:
        if config.get(key, False):
            raise ValueError(f"{key} must be False for single-card A51P inference.")

    if int(config.get("fsdp_size", 1)) != 1:
        raise ValueError("fsdp_size must be 1 for this single-card A51P entry.")
    if int(config.get("cp_size", 1)) != 1:
        raise ValueError("cp_size must be 1 for this single-card A51P entry.")

    text_encoder_config = config.get("text_encoder_config", {})
    if text_encoder_config.get("use_fsdp", False):
        raise ValueError("text_encoder_config.use_fsdp must be False for this single-card A51P entry.")


def main(config):
    logger = get_logger()

    seed = config.get("seed", 42)
    model_name = config.get("model_name", "wan_t2v")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})

    pipeline_name = config.get("pipeline_name", "t2v")
    weight_dtype = str_to_precision(config.get("weight_dtype", "bf16"))
    prompt_txt = config.get("prompt_txt", None)
    batch_size = config.get("batch_size", 1)
    num_frames = config.get("num_frames", 49)
    height = config.get("height", 480)
    width = config.get("width", 832)
    save_fps = config.get("save_fps", 16)
    output_dir = config.get("output_dir", "./output")

    if batch_size != 1:
        raise ValueError("This A51P single-card script expects batch_size=1.")
    if model_name != "wan_t2v":
        raise ValueError(f"This entry is for Wan2.1 T2V only, got model_name={model_name}.")

    setup_distributed_env()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")

    _assert_single_card_config(config, world_size)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, indent=4)

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device,
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_config.get("text_tokenizer_path", None))
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device,
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=False,
        device_mesh=None,
    )
    log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing Wan2.1 T2V model and scheduler...")
    scheduler = schedulers[scheduler_config.pop("scheduler_name", "flow_matching")](**scheduler_config)

    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    if pretrained_model_dir_or_checkpoint is None:
        raise ValueError("model_config.pretrained_model_dir_or_checkpoint must be specified.")
    if isinstance(pretrained_model_dir_or_checkpoint, str) and pretrained_model_dir_or_checkpoint.startswith("**"):
        raise ValueError("Please replace the placeholder pretrained_model_dir_or_checkpoint in the A51P yaml.")

    if os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
    elif os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, "Init model from meta device and load checkpoint")
        with torch.device("meta"):
            model = models[model_name](**model_config)
        model.to_empty(device=device)
        set_seed(seed, device_specific=False)
        model.reset_parameters()
        state_dict = _load_state_dict(pretrained_model_dir_or_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            if missing_keys:
                print(f"[Wan2.1 checkpoint] missing_keys: {missing_keys[:20]}...")
            if unexpected_keys:
                print(f"[Wan2.1 checkpoint] unexpected_keys: {unexpected_keys[:20]}...")
        del state_dict
    else:
        raise ValueError(f"Invalid pretrained model path: {pretrained_model_dir_or_checkpoint}")

    model.eval()
    model = _move_wan_model_to_single_device(model, model_name, device, weight_dtype)
    log_on_main_process(logger, f"Wan2.1 T2V model initialized, memory allocated: {get_memory_allocated()} GiB")

    pipeline = pipelines[pipeline_name](
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        predictor=model,
        scheduler=scheduler,
    )

    prompts = load_prompts(prompt_txt)
    set_seed(seed, device_specific=True, process_group=dist.group.WORLD)

    video_grid = []
    for index in range(0, len(prompts), batch_size):
        batch_prompts = prompts[index: index + batch_size]
        videos = pipeline(
            prompt=batch_prompts,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            max_sequence_length=text_encoder_config.get("text_len", 512),
            device=device,
        )
        save_videos(videos, index, output_dir, save_fps)
        video_grid.append(videos)

    if len(video_grid) > 0:
        video_grid = torch.cat(video_grid, dim=0)
        save_video_grid(video_grid, output_dir, fps=save_fps)
        print("Inference finished.")
        print(f"Saved {video_grid.shape[0]} samples to {output_dir}")
    else:
        print("No prompts found; nothing to infer.")

    cleanup_distributed_env()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/infer/npu/t2v_14b_A51P.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f"Config file not found: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)

import argparse
import math
import os
from typing import Optional

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate import init_empty_weights
from tqdm import tqdm
from library.clip_tokenizer import CLIPTokenizer  # transformers.CLIPTokenizer with the legacy (ftfy) text normalization
from library import model_util, sdxl_model_util, checkpoint_io, sampling, sdxl_original_unet
import library.model_io as model_io
from library.clip_text_model import wrap_clip_text_model
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# DEFAULT_NOISE_OFFSET = 0.0357


def load_target_model(args, accelerator, model_version: str, weight_dtype):
    model_dtype = match_mixed_precision(args, weight_dtype)  # prepare fp16/bf16
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(
                args.pretrained_model_name_or_path,
                args.vae,
                model_version,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                model_dtype,
                args.disable_mmap_load_safetensors,
            )

            # Expand 4-channel conv_in to 9 channels when training inpainting from a
            # standard (non-inpainting) checkpoint.
            if getattr(args, "train_inpainting", False) and getattr(unet, "in_channels", 4) == 4:
                logger.info(
                    "train_inpainting: expanding UNet conv_in from 4 to 9 channels "
                    "(standard checkpoint → inpainting training from scratch)"
                )
                model_util.expand_unet_to_inpainting(unet)

            # work on low-ram device
            if args.lowram:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(
    name_or_path: str, vae_path: Optional[str], model_version: str, weight_dtype, device="cpu", model_dtype=None, disable_mmap=False
):
    # model_dtype only work with full fp16/bf16
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    from library.sdxl_tiny_checkpoint import is_tiny_sdxl_checkpoint, load_tiny_sdxl_checkpoint

    if load_stable_diffusion_format and is_tiny_sdxl_checkpoint(name_or_path):
        logger.info("load tiny SDXL-shaped test checkpoint (ss_sdxl_arch=tiny-test-v1): %s", name_or_path)
        text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info = load_tiny_sdxl_checkpoint(
            name_or_path, device=device, dtype=model_dtype
        )
        return True, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info

    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(model_version, name_or_path, device, model_dtype, disable_mmap)
    else:
        # Diffusers model is loaded to CPU
        from diffusers import StableDiffusionXLPipeline

        variant = "fp16" if weight_dtype == torch.float16 else None
        logger.info(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    name_or_path, torch_dtype=model_dtype, variant=variant, tokenizer=None
                )
            except EnvironmentError as ex:
                if variant is not None:
                    logger.info("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = wrap_clip_text_model(pipe.text_encoder)
        text_encoder2 = pipe.text_encoder_2

        # convert to fp32 for cache text_encoders outputs
        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        state_dict = sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        actual_in_channels = state_dict["input_blocks.0.0.weight"].shape[1]
        with init_empty_weights():
            unet = sdxl_original_unet.SdxlUNet2DConditionModel(in_channels=actual_in_channels)  # overwrite unet
        sdxl_model_util._load_state_dict_on_device(unet, state_dict, device=device, dtype=model_dtype)
        logger.info("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    # VAEを読み込む
    if vae_path is not None:
        vae = model_util.load_vae(vae_path, weight_dtype)
        logger.info("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def load_tokenizers(args: argparse.Namespace):
    logger.info("prepare tokenizers")

    original_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH]
    tokeniers = []
    for i, original_path in enumerate(original_paths):
        tokenizer: CLIPTokenizer = None
        if args.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokeniers.append(tokenizer)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        logger.info(f"update token length: {args.max_token_length}")

    return tokeniers


def match_mixed_precision(args, weight_dtype):
    if args.full_fp16:
        assert (
            weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        return weight_dtype
    elif args.full_bf16:
        assert (
            weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        return weight_dtype
    else:
        return None


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = model_io.get_sai_model_spec(None, args, True, False, False, is_stable_diffusion_ckpt=True)
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    checkpoint_io.save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = model_io.get_sai_model_spec(None, args, True, False, False, is_stable_diffusion_ckpt=True)
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    checkpoint_io.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


def add_sdxl_training_arguments(parser: argparse.ArgumentParser, support_text_encoder_caching: bool = True):
    if support_text_encoder_caching:
        parser.add_argument(
            "--cache_text_encoder_outputs",
            action="store_true",
            help="cache text encoder outputs / text encoderの出力をキャッシュする",
        )
        parser.add_argument(
            "--cache_text_encoder_outputs_to_disk",
            action="store_true",
            help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
        )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )
    parser.add_argument(
        "--playground_v25",
        action="store_true",
        help="train Playground v2.5 (playgroundai/playground-v2.5-1024px-aesthetic) with EDM noise, "
        "c_skip/c_out/c_in preconditioning and per-channel latent normalization. "
        "Implemented for SDXL LoRA (sdxl_train_network.py). Full fine-tuning rejects this flag. "
        "Latent caches are written to *_pgv25.npz and are not reused from SDXL *_sdxl.npz. "
        "Sample image generation during training is disabled. "
        "Validation uses a fixed Karras sigma index (the same integers the SDXL loop pins).",
    )
    parser.add_argument(
        "--pgv25_sigma_sampling",
        type=str,
        default="karras_uniform",
        choices=["karras_uniform", "lognormal"],
        help="Playground v2.5 sigma sampling. karras_uniform (default) draws an index on the "
        "EDMEulerScheduler Karras grid, matching the diffusers SDXL EDM training script. "
        "lognormal draws ln(sigma) ~ Normal(--pgv25_sigma_mean, --pgv25_sigma_std). "
        "Requires --playground_v25. The default may change after a real-GPU A/B.",
    )
    parser.add_argument(
        "--pgv25_sigma_mean",
        type=float,
        default=-1.2,
        help="P_mean for --pgv25_sigma_sampling=lognormal. EDM paper default -1.2. Ignored for karras_uniform.",
    )
    parser.add_argument(
        "--pgv25_sigma_std",
        type=float,
        default=1.2,
        help="P_std for --pgv25_sigma_sampling=lognormal. EDM paper default 1.2. Ignored for karras_uniform.",
    )
    parser.add_argument(
        "--pgv25_loss_weighting",
        type=str,
        default="none",
        choices=["none", "edm"],
        help="Playground v2.5 loss weight. none (default) is unweighted x0 MSE, matching the diffusers "
        "reference. edm multiplies by lambda(sigma)=(sigma^2+sigma_data^2)/(sigma*sigma_data)^2, "
        "which is the F-space loss. Requires --playground_v25.",
    )
    parser.add_argument(
        "--pgv25_cache_fp16",
        action="store_true",
        help="store Playground raw VAE latents as fp16 in *_pgv25.npz (about half the disk of fp32). "
        "Train-time normalization is still fp32. Off by default. Requires --playground_v25.",
    )


def verify_sdxl_training_args(
    args: argparse.Namespace, support_text_encoder_caching: bool = True, support_playground_v25: bool = False
):
    assert not args.v2, "v2 cannot be enabled in SDXL training / SDXL学習ではv2を有効にすることはできません"

    sampling = getattr(args, "pgv25_sigma_sampling", "karras_uniform")
    weighting = getattr(args, "pgv25_loss_weighting", "none")
    sigma_mean = getattr(args, "pgv25_sigma_mean", -1.2)
    sigma_std = getattr(args, "pgv25_sigma_std", 1.2)
    cache_fp16 = bool(getattr(args, "pgv25_cache_fp16", False))
    pg_options_changed = (
        sampling not in (None, "karras_uniform")
        or weighting not in (None, "none")
        or (sigma_mean is not None and float(sigma_mean) != -1.2)
        or (sigma_std is not None and float(sigma_std) != 1.2)
        or cache_fp16
    )
    if pg_options_changed and not getattr(args, "playground_v25", False):
        raise ValueError(
            "--pgv25_sigma_sampling, --pgv25_sigma_mean, --pgv25_sigma_std, --pgv25_loss_weighting, "
            "and --pgv25_cache_fp16 require --playground_v25."
        )

    if getattr(args, "playground_v25", False) and not support_playground_v25:
        raise ValueError(
            "--playground_v25 is only implemented for SDXL LoRA training (sdxl_train_network.py). "
            "Full fine-tuning and other SDXL scripts still use the DDPM epsilon objective, "
            "so this flag is rejected there instead of being ignored."
        )
    if getattr(args, "playground_v25", False):
        from library.edm_playground import validate_training_args

        validate_training_args(args)

    if args.clip_skip is not None:
        logger.warning("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    # if args.multires_noise_iterations:
    #     logger.info(
    #         f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
    #     )
    # else:
    #     if args.noise_offset is None:
    #         args.noise_offset = DEFAULT_NOISE_OFFSET
    #     elif args.noise_offset != DEFAULT_NOISE_OFFSET:
    #         logger.info(
    #             f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
    #         )
    #     logger.info(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    # assert (
    #     not hasattr(args, "weighted_captions") or not args.weighted_captions
    # ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if support_text_encoder_caching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )


def sample_images(*args, **kwargs):
    from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline

    return sampling.sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)

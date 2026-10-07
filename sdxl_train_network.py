import argparse
import os
import re
from typing import List, Optional, Union

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import edm_playground, sdxl_model_util, sdxl_train_util, strategy_base, strategy_sd, strategy_sdxl
import library.args as args_util
import library.model_io as model_io
from library.dataset import DatasetGroup, MinimalDataset
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        sdxl_train_util.verify_sdxl_training_args(args, support_playground_v25=True)
        if getattr(args, "playground_v25", False):
            logger.info(
                "Playground v2.5 LoRA: EDM sigma noise, preconditioned x0 MSE, "
                "latent norm (z-mean)*0.5/std. Caches use %s.",
                edm_playground.PGV25_NPZ_SUFFIX,
            )

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        # Reject an unmarked SDXL load of a Playground checkpoint before reading the weights.
        edm_playground.raise_if_playground_checkpoint_without_flag(
            args.pretrained_model_name_or_path, bool(getattr(args, "playground_v25", False))
        )
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)
        if getattr(args, "playground_v25", False):
            mean, std, scaling = edm_playground.read_latent_stats(args.pretrained_model_name_or_path)
            self.pg_latents_mean = mean
            self.pg_latents_std = std
            self.pg_scaling_factor = scaling
        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # モデルに xformers とか memory efficient attention を組み込む
        model_io.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]

    def get_latents_caching_strategy(self, args):
        if getattr(args, "playground_v25", False):
            latents_caching_strategy = strategy_sd.PlaygroundV25LatentsCachingStrategy(
                args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
            )
            latents_caching_strategy.cache_fp16 = bool(getattr(args, "pgv25_cache_fp16", False))
            return latents_caching_strategy
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        indices: Optional[List[int]] = None,
    ):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        if not getattr(args, "playground_v25", False):
            return super().shift_scale_latents(args, latents)
        return edm_playground.normalize_latents(
            latents,
            mean=getattr(self, "pg_latents_mean", None),
            std=getattr(self, "pg_latents_std", None),
            scaling_factor=getattr(self, "pg_scaling_factor", edm_playground.SCALING_FACTOR),
        )

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        if not getattr(args, "playground_v25", False):
            return super().get_noise_pred_and_target(
                args,
                accelerator,
                noise_scheduler,
                latents,
                batch,
                text_encoder_conds,
                unet,
                network,
                weight_dtype,
                train_unet,
                is_train=is_train,
            )

        # EDM: x = x0 + sigma * n, UNet sees c_in * x and c_noise.
        # Default sampling is the diffusers Karras grid (uniform index). Lognormal
        # draws a continuous sigma. Validation pins min_timestep == max_timestep
        # and that integer is a Karras index, so the sigma is not random.
        noise = torch.randn_like(latents, dtype=torch.float32)
        pinned = None if is_train else edm_playground.pinned_validation_index(
            getattr(args, "min_timestep", None), getattr(args, "max_timestep", None)
        )
        sampling = getattr(args, "pgv25_sigma_sampling", edm_playground.SIGMA_SAMPLING_KARRAS) or edm_playground.SIGMA_SAMPLING_KARRAS
        if pinned is not None or sampling == edm_playground.SIGMA_SAMPLING_KARRAS:
            if pinned is not None:
                indices = torch.full((latents.shape[0],), pinned, dtype=torch.long)
            else:
                indices = torch.randint(0, edm_playground.NUM_TRAIN_TIMESTEPS, (latents.shape[0],), device="cpu")
            noisy, scaled, c_noise, sigma = edm_playground.prepare_edm_inputs(latents, noise, indices)
        elif sampling == edm_playground.SIGMA_SAMPLING_LOGNORMAL:
            sigma = edm_playground.sample_lognormal_sigma(
                latents.shape[0],
                p_mean=float(getattr(args, "pgv25_sigma_mean", edm_playground.DEFAULT_P_MEAN)),
                p_std=float(getattr(args, "pgv25_sigma_std", edm_playground.DEFAULT_P_STD)),
            )
            noisy, scaled, c_noise, sigma = edm_playground.prepare_edm_inputs_from_sigma(latents, noise, sigma)
        else:
            raise ValueError(
                f"--pgv25_sigma_sampling={sampling} is not supported. "
                f"Use {edm_playground.SIGMA_SAMPLING_KARRAS} or {edm_playground.SIGMA_SAMPLING_LOGNORMAL}."
            )

        # Cast to the UNet dtype before requires_grad_. Doing requires_grad_(True)
        # on the fp32 tensor and then .to(bf16).requires_grad_(...) raises on the
        # non-leaf cast when gradient checkpointing is on, which is the
        # text-encoder-only + bf16 path (train_unet is False).
        model_input = scaled.detach().to(weight_dtype)
        if is_train and args.gradient_checkpointing and train_unet:
            model_input.requires_grad_(True)
        if is_train and args.gradient_checkpointing:
            for cond in text_encoder_conds:
                if torch.is_tensor(cond) and cond.is_floating_point() and cond.is_leaf:
                    cond.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_output = self.call_unet(
                args,
                accelerator,
                unet,
                model_input,
                c_noise,
                text_encoder_conds,
                batch,
                weight_dtype,
            )
        pred_x0 = edm_playground.x0_target_from_model_output(noisy, model_output, sigma)
        target = latents.float()

        # Differential output preservation compares x0 to the frozen UNet's x0, not to epsilon.
        if "custom_attributes" in batch and network is not None and hasattr(network, "set_multiplier"):
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)
            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    prior_output = self.call_unet(
                        args,
                        accelerator,
                        unet,
                        scaled.to(weight_dtype),
                        c_noise,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                network.set_multiplier(1.0)
                prior_noisy = noisy[diff_output_pr_indices]
                prior_sigma = sigma[diff_output_pr_indices]
                prior_x0 = edm_playground.x0_target_from_model_output(prior_noisy, prior_output, prior_sigma)
                target[diff_output_pr_indices] = prior_x0.to(target.dtype)

        weighting = None
        loss_weighting = getattr(args, "pgv25_loss_weighting", edm_playground.LOSS_WEIGHTING_NONE) or edm_playground.LOSS_WEIGHTING_NONE
        if loss_weighting == edm_playground.LOSS_WEIGHTING_EDM:
            # Per-sample λ(σ). process_batch multiplies the unreduced x0 MSE by this,
            # which is the F-space loss. Default remains unweighted (None).
            weighting = edm_playground.edm_loss_weight(sigma).to(device=latents.device, dtype=torch.float32).view(-1, 1, 1, 1)
        elif loss_weighting != edm_playground.LOSS_WEIGHTING_NONE:
            raise ValueError(
                f"--pgv25_loss_weighting={loss_weighting} is not supported. "
                f"Use {edm_playground.LOSS_WEIGHTING_NONE} or {edm_playground.LOSS_WEIGHTING_EDM}."
            )
        return pred_x0, target, c_noise, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        if getattr(args, "playground_v25", False):
            # Min-SNR and v-pred weightings are rejected in validate_training_args.
            # Optional EDM λ is applied earlier as the per-sample loss weight.
            return loss
        return super().post_process_loss(loss, args, timesteps, noise_scheduler)

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        super().on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=is_train)
        if os.environ.get(edm_playground.ASSERT_ENV) != "1":
            return
        if not getattr(args, "playground_v25", False):
            return
        raw = batch.get("latents")
        captions = batch.get("captions")
        if raw is None or captions is None:
            raise RuntimeError("PGV25_ASSERT_LATENT_ID: batch is missing latents or captions")
        for i, caption in enumerate(captions):
            match = re.match(r"id(\d+)", caption.strip())
            if match is None:
                raise RuntimeError(
                    f"PGV25 pairing: caption {caption!r} does not start with the image id. "
                    "keep_tokens/shuffle_caption dropped the id, or the caption was paired with the wrong image."
                )
            expected = float(int(match.group(1)))
            got = float(raw[i, 0, 0, 0].item())
            if abs(got - expected) > 1e-3:
                raise RuntimeError(
                    f"PGV25 pairing: caption {caption!r} expects latent stamp {expected}, found {got}. "
                    "The cached latent does not belong to this caption."
                )

    def update_metadata(self, metadata, args):
        metadata["ss_playground_v25"] = bool(getattr(args, "playground_v25", False))
        if getattr(args, "playground_v25", False):
            metadata["ss_pgv25_sigma_sampling"] = getattr(
                args, "pgv25_sigma_sampling", edm_playground.SIGMA_SAMPLING_KARRAS
            )
            metadata["ss_pgv25_loss_weighting"] = getattr(
                args, "pgv25_loss_weighting", edm_playground.LOSS_WEIGHTING_NONE
            )
            metadata["ss_pgv25_sigma_mean"] = getattr(args, "pgv25_sigma_mean", edm_playground.DEFAULT_P_MEAN)
            metadata["ss_pgv25_sigma_std"] = getattr(args, "pgv25_sigma_std", edm_playground.DEFAULT_P_STD)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        if getattr(args, "playground_v25", False):
            # The training loop calls this every step. Warn only when sampling was requested.
            if getattr(args, "sample_prompts", None) and not getattr(self, "_playground_sample_warned", False):
                logger.warning(
                    "playground_v25: skipping sample image generation. "
                    "The SDXL sampler is DDPM/epsilon and does not implement EDM (c_noise, c_in, sigma)."
                )
                self._playground_sample_warned = True
            return
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    trainer.train(args)

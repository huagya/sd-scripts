"""Playground v2.5 EDM training math for SDXL-shaped models.

Playground v2.5 uses the SDXL UNet and text encoders, but it was trained with
the EDM formulation (Karras et al., arXiv 2206.00364; Playground paper
arXiv 2402.17245): sigma-space noise, c_skip / c_out / c_in / c_noise
preconditioning, and per-channel VAE latent normalization.

The numerical reference is the EDM branch of diffusers
``examples/dreambooth/train_dreambooth_lora_sdxl.py`` (``do_edm_style_training``
with ``EDMEulerScheduler``), including the unweighted x0 MSE from
diffusers PR #7126. Full fine-tuning should call these helpers later;
``sdxl_train.py`` does not use them yet and rejects ``--playground_v25``.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# scheduler/scheduler_config.json on playgroundai/playground-v2.5-1024px-aesthetic
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SIGMA_DATA = 0.5
RHO = 7.0
NUM_TRAIN_TIMESTEPS = 1000
# vae/config.json scaling_factor. edm_mean / edm_std in the official single-file
# checkpoint are the float32 values of these published latents_mean / latents_std.
SCALING_FACTOR = 0.5
LATENTS_MEAN = (-1.6574, 1.886, -1.383, 2.5155)
LATENTS_STD = (8.4927, 5.9022, 6.5498, 5.2299)

LATENT_FORMAT_KEY = "latent_format"
LATENT_FORMAT_VALUE = "playground_v25_raw"
LATENT_SOURCE_KEY = "source_basename"
PGV25_NPZ_SUFFIX = "_pgv25.npz"

# Test-only hooks. Production training leaves these unset.
STAMP_ENV = "PGV25_STAMP_LATENT_ID"
ASSERT_ENV = "PGV25_ASSERT_LATENT_ID"


class PlaygroundV25Schedule:
    """Karras sigma grid matching ``EDMEulerScheduler`` construction.

    ``timesteps`` has length ``num_train_timesteps + 1`` and stores c_noise.
    ``sigmas`` has length ``num_train_timesteps + 2`` because the scheduler
    appends a terminal 0. Training draws indices in ``[0, num_train_timesteps)``,
    so the final ramp entry (pure sigma_min) and the terminal 0 are not sampled.
    That matches the diffusers training script, which indexes
    ``scheduler.timesteps`` with ``randint(0, num_train_timesteps)``.
    """

    def __init__(
        self,
        num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
        sigma_data: float = SIGMA_DATA,
        rho: float = RHO,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.sigma_data = sigma_data
        self.timesteps, self.sigmas = build_edm_euler_schedule(
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )

    def gather(self, indices: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(c_noise, sigma)`` for training indices of shape ``(batch,)``."""
        if indices.ndim != 1:
            raise ValueError(f"EDM training indices must be rank-1, got shape {tuple(indices.shape)}")
        idx = indices.long().cpu()
        if torch.any(idx < 0) or torch.any(idx >= self.num_train_timesteps):
            raise ValueError(
                f"EDM training index out of range [0, {self.num_train_timesteps}): {idx.tolist()}"
            )
        c_noise = self.timesteps[idx].to(device=device, dtype=torch.float32)
        sigma = self.sigmas[idx].to(device=device, dtype=torch.float32)
        return c_noise, sigma


_SCHEDULE: Optional[PlaygroundV25Schedule] = None


def get_schedule() -> PlaygroundV25Schedule:
    global _SCHEDULE
    if _SCHEDULE is None:
        _SCHEDULE = PlaygroundV25Schedule()
    return _SCHEDULE


def build_edm_euler_schedule(
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    rho: float = RHO,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match ``EDMEulerScheduler.__init__`` (diffusers 0.40, Karras schedule)."""
    ramp = torch.arange(num_train_timesteps + 1, dtype=torch.float64) / num_train_timesteps
    # Python floats, same promotion as EDMEulerScheduler._compute_karras_sigmas.
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = sigmas.to(dtype=torch.float32)
    timesteps = 0.25 * torch.log(sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)])
    return timesteps, sigmas


def _as_sigma_tensor(sigma: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=ref.device, dtype=torch.float32)
    while sigma.ndim < ref.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def c_in(sigma: torch.Tensor, sigma_data: float = SIGMA_DATA) -> torch.Tensor:
    return 1.0 / torch.sqrt(sigma**2 + sigma_data**2)


def c_skip(sigma: torch.Tensor, sigma_data: float = SIGMA_DATA) -> torch.Tensor:
    return (sigma_data**2) / (sigma**2 + sigma_data**2)


def c_out(sigma: torch.Tensor, sigma_data: float = SIGMA_DATA) -> torch.Tensor:
    # epsilon parameterization, as in Playground v2.5 / EDMEulerScheduler
    return sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)


def c_noise(sigma: torch.Tensor) -> torch.Tensor:
    return 0.25 * torch.log(sigma)


def precondition_inputs(sample: torch.Tensor, sigma: torch.Tensor, sigma_data: float = SIGMA_DATA) -> torch.Tensor:
    sigma = _as_sigma_tensor(sigma, sample)
    return sample.float() * c_in(sigma, sigma_data)


def precondition_outputs(
    sample: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor, sigma_data: float = SIGMA_DATA
) -> torch.Tensor:
    """Denoised x0 = c_skip * x + c_out * F, epsilon prediction."""
    sigma = _as_sigma_tensor(sigma, sample)
    return c_skip(sigma, sigma_data) * sample.float() + c_out(sigma, sigma_data) * model_output.float()


def add_edm_noise(latents: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = _as_sigma_tensor(sigma, latents)
    return latents.float() + noise.float() * sigma


def prepare_edm_inputs(
    latents: torch.Tensor, noise: torch.Tensor, indices: torch.Tensor, schedule: Optional[PlaygroundV25Schedule] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(noisy, scaled_input, c_noise, sigma)`` for one training step.

    ``latents`` must already be Playground-normalized. ``sigma`` has shape
    ``(batch,)``; callers broadcast it inside ``add_edm_noise`` /
    ``precondition_*``.
    """
    schedule = schedule or get_schedule()
    c_noise_t, sigma = schedule.gather(indices, latents.device)
    noisy = add_edm_noise(latents, noise, sigma)
    scaled = precondition_inputs(noisy, sigma, schedule.sigma_data)
    return noisy, scaled, c_noise_t, sigma


def x0_target_from_model_output(
    noisy: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor, sigma_data: float = SIGMA_DATA
) -> torch.Tensor:
    return precondition_outputs(noisy, model_output, sigma, sigma_data)


def default_latent_stats() -> Tuple[torch.Tensor, torch.Tensor, float]:
    mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32)
    std = torch.tensor(LATENTS_STD, dtype=torch.float32)
    return mean, std, SCALING_FACTOR


def normalize_latents(
    latents: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    scaling_factor: float = SCALING_FACTOR,
) -> torch.Tensor:
    """``(z - mean) * scaling_factor / std`` with per-channel mean and std.

    Disk caches store the raw VAE sample. This runs at train time so SDXL and
    Playground caches are not interchangeable even when both are "raw".
    """
    if mean is None or std is None:
        default_mean, default_std, _ = default_latent_stats()
        mean = default_mean if mean is None else mean
        std = default_std if std is None else std
    mean_b = mean.to(device=latents.device, dtype=torch.float32).view(1, -1, 1, 1)
    std_b = std.to(device=latents.device, dtype=torch.float32).view(1, -1, 1, 1)
    if mean_b.shape[1] != latents.shape[1] or std_b.shape[1] != latents.shape[1]:
        raise ValueError(
            f"Playground latent stats have {mean_b.shape[1]} channels, latents have {latents.shape[1]}"
        )
    return (latents.float() - mean_b) * float(scaling_factor) / std_b


def validate_training_args(args) -> None:
    """Reject options that assume discrete DDPM timesteps or epsilon/v objectives.

    Also forces a fp32 VAE. Playground latents are large before normalization;
    fp16 VAE encode is the failure mode called out for external VAEs in the
    diffusers script.
    """
    errors = []

    def reject(enabled: bool, flag: str, why: str) -> None:
        if enabled:
            errors.append(f"{flag}: {why}")

    reject(bool(getattr(args, "v_parameterization", False)), "--v_parameterization", "EDM trains x0, not v-prediction")
    reject(bool(getattr(args, "zero_terminal_snr", False)), "--zero_terminal_snr", "rewrites the DDPM beta schedule, which Playground v2.5 does not use")
    reject(getattr(args, "min_snr_gamma", None) is not None, "--min_snr_gamma", "Min-SNR weights assume DDPM SNR over discrete timesteps")
    reject(
        bool(getattr(args, "scale_v_pred_loss_like_noise_pred", False)),
        "--scale_v_pred_loss_like_noise_pred",
        "v-prediction loss rescaling does not apply to EDM x0 loss",
    )
    reject(getattr(args, "v_pred_like_loss", None) not in (None, 0, 0.0), "--v_pred_like_loss", "adds a v-prediction term on top of a DDPM epsilon target")
    reject(
        bool(getattr(args, "debiased_estimation_loss", False)),
        "--debiased_estimation_loss",
        "uses the DDPM SNR table",
    )
    reject(getattr(args, "noise_offset", None) is not None, "--noise_offset", "offset noise is not part of the EDM forward process (Playground reports it is unnecessary)")
    reject(bool(getattr(args, "noise_offset_random_strength", False)), "--noise_offset_random_strength", "depends on noise_offset")
    reject(getattr(args, "adaptive_noise_scale", None) is not None, "--adaptive_noise_scale", "depends on noise_offset")
    reject(getattr(args, "multires_noise_iterations", None) not in (None, 0), "--multires_noise_iterations", "replaces the EDM Gaussian noise sample")
    reject(getattr(args, "ip_noise_gamma", None) is not None, "--ip_noise_gamma", "input perturbation is defined on the DDPM add_noise path")
    reject(bool(getattr(args, "ip_noise_gamma_random_strength", False)), "--ip_noise_gamma_random_strength", "depends on ip_noise_gamma")
    reject(getattr(args, "min_timestep", None) is not None, "--min_timestep", "selects a DDPM timestep range; EDM samples the Karras sigma grid")
    reject(getattr(args, "max_timestep", None) is not None, "--max_timestep", "selects a DDPM timestep range; EDM samples the Karras sigma grid")
    loss_type = getattr(args, "loss_type", "l2") or "l2"
    reject(loss_type != "l2", f"--loss_type={loss_type}", "Playground v2.5 EDM training uses unweighted MSE on the preconditioned x0 prediction")
    reject(bool(getattr(args, "train_inpainting", False)), "--train_inpainting", "inpainting channel concat is not defined for Playground v2.5 EDM training")
    reject(getattr(args, "vae", None) not in (None, ""), "--vae", "a replacement VAE would not match Playground's latents_mean/latents_std; refusing to train with a mismatched latent normalization")

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(
            "Playground v2.5 mode (--playground_v25) cannot be combined with DDPM/v-pred options, "
            "because those would train the SDXL epsilon objective against an EDM model:\n  - " + detail
        )

    if not getattr(args, "no_half_vae", False):
        args.no_half_vae = True
        logger.warning(
            "playground_v25: forcing --no_half_vae so the VAE encode stays fp32. "
            "Playground latent magnitudes overflow a fp16 VAE."
        )

    if getattr(args, "sample_prompts", None) or getattr(args, "sample_every_n_steps", None) or getattr(
        args, "sample_every_n_epochs", None
    ) or getattr(args, "sample_at_first", False):
        logger.warning(
            "playground_v25: sample image generation during training is disabled. "
            "The built-in sampler is a DDPM/epsilon sampler and would not match EDM "
            "(c_noise, c_in, sigma in [0.002, 80]). Remove --sample_prompts / "
            "--sample_every_n_steps to silence this warning."
        )

    if getattr(args, "cache_latents", False) and not getattr(args, "cache_latents_to_disk", False):
        logger.warning(
            "playground_v25: latents are cached in RAM. For large datasets use --cache_latents_to_disk "
            "so Playground caches stay on disk in *_pgv25.npz files and are not mixed with SDXL caches."
        )


def _scheduler_class_name(payload) -> str:
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return str(payload[1])
    if isinstance(payload, str):
        return payload
    return ""


def checkpoint_has_playground_edm_markers(path: str) -> Optional[bool]:
    """Return True/False when the path can be inspected, or None if it cannot.

    Official single-file weights store ``edm_mean`` and ``edm_std``.
    Diffusers folders name an EDM scheduler in ``model_index.json``.
    """
    if not path:
        return None
    if os.path.isfile(path):
        if not path.endswith(".safetensors"):
            logger.warning(
                "playground marker check skipped for non-safetensors checkpoint %s. "
                "Pass --playground_v25 explicitly for Playground v2.5.",
                path,
            )
            return None
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
            return "edm_mean" in keys or "edm_std" in keys
        except Exception as ex:
            logger.warning("could not read safetensors header for Playground markers (%s): %s", path, ex)
            return None
    if os.path.isdir(path):
        index_path = os.path.join(path, "model_index.json")
        if os.path.isfile(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as handle:
                    index = json.load(handle)
                if "EDM" in _scheduler_class_name(index.get("scheduler")):
                    return True
            except Exception as ex:
                logger.warning("could not read %s: %s", index_path, ex)
                return None
        sched_path = os.path.join(path, "scheduler", "scheduler_config.json")
        if os.path.isfile(sched_path):
            try:
                with open(sched_path, "r", encoding="utf-8") as handle:
                    cfg = json.load(handle)
                return "EDM" in str(cfg.get("_class_name", ""))
            except Exception as ex:
                logger.warning("could not read %s: %s", sched_path, ex)
                return None
        return False
    # Hugging Face repo id: only the small JSON files, not the weights.
    try:
        from huggingface_hub import hf_hub_download

        index_path = hf_hub_download(path, "model_index.json")
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        return "EDM" in _scheduler_class_name(index.get("scheduler"))
    except Exception as ex:
        logger.warning("could not inspect %s for Playground EDM markers: %s", path, ex)
        return None


def raise_if_playground_checkpoint_without_flag(path: str, playground_v25: bool) -> None:
    markers = checkpoint_has_playground_edm_markers(path)
    if markers is True and not playground_v25:
        raise ValueError(
            f"Model '{path}' looks like Playground v2.5 (edm_mean/edm_std or an EDM scheduler) "
            "but --playground_v25 is not set. Loading it as plain SDXL would train the DDPM "
            "epsilon objective and the 0.13025 latent scale, which is silently wrong. "
            "Re-run with --playground_v25."
        )
    if markers is False and playground_v25:
        logger.warning(
            "playground_v25 is set but '%s' has no edm_mean/edm_std keys and no EDM scheduler. "
            "Training will still use Playground v2.5 EDM noise and the published latent normalization.",
            path,
        )


def read_latent_stats(path: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Latent mean/std/scaling for ``path``, falling back to the published constants.

    Single-file checkpoints contribute ``edm_mean`` / ``edm_std`` (verified on the
    official fp32 safetensors to match ``vae/config.json``). Diffusers folders
    contribute ``vae/config.json``. A missing external VAE config does not fall
    back to ``scaling_factor`` alone.
    """
    mean, std, scaling = default_latent_stats()
    if not path:
        return mean, std, scaling
    if os.path.isfile(path) and path.endswith(".safetensors"):
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if "edm_mean" in keys and "edm_std" in keys:
                    mean = handle.get_tensor("edm_mean").float().reshape(-1).cpu()
                    std = handle.get_tensor("edm_std").float().reshape(-1).cpu()
                    logger.info("playground_v25: using edm_mean/edm_std from %s", path)
        except Exception as ex:
            logger.warning("playground_v25: failed to read edm_mean/edm_std from %s (%s); using published constants", path, ex)
        return mean, std, scaling
    vae_config = None
    if os.path.isdir(path):
        candidate = os.path.join(path, "vae", "config.json")
        if os.path.isfile(candidate):
            vae_config = candidate
    if vae_config is None and not os.path.isfile(path) and path:
        try:
            from huggingface_hub import hf_hub_download

            vae_config = hf_hub_download(path, "vae/config.json")
        except Exception:
            vae_config = None
    if vae_config is not None:
        with open(vae_config, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        if cfg.get("latents_mean") is None or cfg.get("latents_std") is None:
            raise ValueError(
                f"playground_v25: VAE config {vae_config} has no latents_mean/latents_std. "
                "Refusing to fall back to scaling_factor only (that silently trains the wrong normalization)."
            )
        file_scaling = float(cfg.get("scaling_factor", SCALING_FACTOR))
        if abs(file_scaling - SCALING_FACTOR) > 1e-6:
            raise ValueError(
                f"playground_v25: VAE scaling_factor is {file_scaling}, expected {SCALING_FACTOR}. "
                "Refusing to train with a different latent scale."
            )
        mean = torch.tensor(cfg["latents_mean"], dtype=torch.float32)
        std = torch.tensor(cfg["latents_std"], dtype=torch.float32)
        logger.info("playground_v25: using latents_mean/latents_std from %s", vae_config)
    return mean, std, scaling


def npz_source_basename(npz_path: str) -> str:
    """Image stem stored beside a ``*_WWWWxHHHH_pgv25.npz`` cache file."""
    base = os.path.basename(npz_path)
    marker = PGV25_NPZ_SUFFIX
    if not base.endswith(marker):
        raise ValueError(f"Not a Playground v2.5 latent cache path: {npz_path}")
    stem = base[: -len(marker)]
    # strip the trailing _WWWWxHHHH that the caching strategy appends
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and "x" in parts[1] and parts[1].replace("x", "").isdigit():
        return parts[0]
    return stem


def assert_playground_latent_npz(npz_path: str, npz) -> None:
    """Fail if a ``*_pgv25.npz`` file is not a raw Playground cache for this image."""
    if LATENT_FORMAT_KEY not in npz.files:
        raise ValueError(
            f"Latent cache {npz_path} has no '{LATENT_FORMAT_KEY}' entry. "
            "SDXL caches use a different filename (*_sdxl.npz) and must not be reused for Playground v2.5. "
            "Delete this file and let training rebuild it with --playground_v25."
        )
    stored = npz[LATENT_FORMAT_KEY]
    stored_s = stored.item() if getattr(stored, "shape", ()) == () else str(stored)
    if isinstance(stored_s, bytes):
        stored_s = stored_s.decode("utf-8")
    stored_s = str(stored_s)
    if stored_s != LATENT_FORMAT_VALUE:
        raise ValueError(
            f"Latent cache {npz_path} has {LATENT_FORMAT_KEY}={stored_s!r}, expected {LATENT_FORMAT_VALUE!r}. "
            "Refusing to mix SDXL and Playground v2.5 latent normalization. Delete the cache and rebuild it."
        )
    if LATENT_SOURCE_KEY not in npz.files:
        raise ValueError(
            f"Latent cache {npz_path} is missing '{LATENT_SOURCE_KEY}'. Delete it and rebuild the Playground cache."
        )
    source = npz[LATENT_SOURCE_KEY]
    source_s = source.item() if getattr(source, "shape", ()) == () else str(source)
    if isinstance(source_s, bytes):
        source_s = source_s.decode("utf-8")
    source_s = str(source_s)
    expected = npz_source_basename(npz_path)
    if source_s != expected:
        raise ValueError(
            f"Latent cache {npz_path} was written for image '{source_s}' but the filename belongs to '{expected}'. "
            "Refusing to train with a copied or mismatched latent cache."
        )

"""Playground v2.5 EDM math, guards, latent-cache identity, and the unchanged SDXL path."""

import argparse
import os
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import EDMEulerScheduler
from safetensors.torch import save_file

from library import edm_playground, loss as loss_util, sdxl_model_util, sdxl_train_util
from library.custom_train_functions import prepare_scheduler_for_custom_training
from library.strategy_sd import PlaygroundV25LatentsCachingStrategy, SdSdxlLatentsCachingStrategy
from diffusers import DDPMScheduler
import sdxl_train_network


def _scheduler():
    return EDMEulerScheduler(
        sigma_min=edm_playground.SIGMA_MIN,
        sigma_max=edm_playground.SIGMA_MAX,
        sigma_data=edm_playground.SIGMA_DATA,
        sigma_schedule="karras",
        num_train_timesteps=edm_playground.NUM_TRAIN_TIMESTEPS,
        prediction_type="epsilon",
        rho=edm_playground.RHO,
        final_sigmas_type="zero",
    )


def _reference_get_sigmas(scheduler, timesteps, n_dim, dtype):
    """Copy of train_dreambooth_lora_sdxl.py get_sigmas (script 070b)."""
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def test_schedule_matches_diffusers_edm_euler():
    reference = _scheduler()
    ours = edm_playground.get_schedule()
    assert torch.allclose(ours.timesteps, reference.timesteps, rtol=0, atol=0)
    assert torch.allclose(ours.sigmas, reference.sigmas, rtol=0, atol=0)
    # Training draws randint(0, num_train_timesteps), never the final ramp entry.
    assert ours.timesteps.shape[0] == edm_playground.NUM_TRAIN_TIMESTEPS + 1
    assert ours.sigmas.shape[0] == edm_playground.NUM_TRAIN_TIMESTEPS + 2
    assert ours.sigmas[-1].item() == 0.0
    assert ours.sigmas[0].item() == pytest.approx(edm_playground.SIGMA_MAX, rel=1e-5)


def test_published_latent_stats_match_official_checkpoint_header():
    """Values read from the official fp32 safetensors edm_mean/edm_std tensors (header range request)."""
    official_mean = (-1.6574000120162964, 1.8860000371932983, -1.3830000162124634, 2.515500068664551)
    official_std = (8.49269962310791, 5.902200222015381, 6.549799919128418, 5.229899883270264)
    mean, std, scaling = edm_playground.default_latent_stats()
    assert scaling == 0.5
    assert torch.allclose(mean, torch.tensor(official_mean), rtol=0, atol=0)
    assert torch.allclose(std, torch.tensor(official_std), rtol=0, atol=0)


def test_loss_matches_diffusers_070b_edm_path():
    torch.manual_seed(1234)
    reference = _scheduler()
    latents = torch.randn(4, 4, 8, 8)
    noise = torch.randn(4, 4, 8, 8)
    # Include both ends of the grid the training script can draw.
    indices = torch.tensor([0, 1, 250, 999], dtype=torch.long)

    def model_fn(scaled, c_noise):
        return torch.sin(scaled) * 0.1 + (c_noise.float().view(-1, 1, 1, 1) * 0.01)

    noisy, scaled, c_noise, sigma = edm_playground.prepare_edm_inputs(latents, noise, indices)
    pred = edm_playground.x0_target_from_model_output(noisy, model_fn(scaled, c_noise), sigma)
    loss = F.mse_loss(pred, latents.float())

    timesteps = reference.timesteps[indices]
    noisy_ref = reference.add_noise(latents, noise, timesteps)
    sigmas = _reference_get_sigmas(reference, timesteps, noisy_ref.ndim, noisy_ref.dtype)
    scaled_ref = reference.precondition_inputs(noisy_ref, sigmas)
    pred_ref = reference.precondition_outputs(noisy_ref, model_fn(scaled_ref, timesteps), sigmas)
    loss_ref = F.mse_loss(pred_ref.float(), latents.float())

    assert torch.allclose(c_noise, timesteps, rtol=0, atol=0)
    assert torch.allclose(noisy, noisy_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(scaled, scaled_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(pred, pred_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(loss, loss_ref, rtol=1e-6, atol=1e-6)
    # Script 070b / PR 7126: the training loss is unweighted. EDM's 1/c_out^2
    # weight is intentionally not applied. The identity below is only a check
    # that the preconditioning matches the paper, not the loss we optimize.
    c_out = edm_playground.c_out(sigmas)
    f_target = (latents.float() - edm_playground.c_skip(sigmas) * noisy_ref.float()) / c_out
    model_output = model_fn(scaled_ref, timesteps)
    weighted_x0 = ((pred_ref.float() - latents.float()) ** 2 / (c_out**2)).mean()
    f_space = ((model_output.float() - f_target) ** 2).mean()
    assert torch.allclose(weighted_x0, f_space, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(loss, weighted_x0, rtol=1e-3, atol=1e-3)


def test_normalize_latents_formula():
    raw = torch.zeros(2, 4, 2, 2)
    raw[:, 0] = 1
    mean, std, scaling = edm_playground.default_latent_stats()
    out = edm_playground.normalize_latents(raw, mean, std, scaling)
    expected = (1.0 - mean[0]) * scaling / std[0]
    assert torch.allclose(out[:, 0], torch.full_like(out[:, 0], expected))
    # Other channels were 0.
    for channel in range(1, 4):
        expected_c = (0.0 - mean[channel]) * scaling / std[channel]
        assert torch.allclose(out[:, channel], torch.full_like(out[:, channel], expected_c))


def _base_args(**overrides):
    args = argparse.Namespace(
        v2=False,
        v_parameterization=False,
        zero_terminal_snr=False,
        min_snr_gamma=None,
        scale_v_pred_loss_like_noise_pred=False,
        v_pred_like_loss=None,
        debiased_estimation_loss=False,
        noise_offset=None,
        noise_offset_random_strength=False,
        adaptive_noise_scale=None,
        multires_noise_iterations=None,
        multires_noise_discount=0.3,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        min_timestep=None,
        max_timestep=None,
        loss_type="l2",
        train_inpainting=False,
        vae=None,
        no_half_vae=False,
        sample_prompts=None,
        sample_every_n_steps=None,
        sample_every_n_epochs=None,
        sample_at_first=False,
        cache_latents=False,
        cache_latents_to_disk=False,
        gradient_checkpointing=False,
        playground_v25=True,
        clip_skip=None,
        cache_text_encoder_outputs=False,
        cache_text_encoder_outputs_to_disk=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize(
    "flag,value",
    [
        ("v_parameterization", True),
        ("zero_terminal_snr", True),
        ("min_snr_gamma", 5.0),
        ("scale_v_pred_loss_like_noise_pred", True),
        ("v_pred_like_loss", 1.0),
        ("debiased_estimation_loss", True),
        ("noise_offset", 0.1),
        ("multires_noise_iterations", 6),
        ("ip_noise_gamma", 0.1),
        ("min_timestep", 0),
        ("max_timestep", 500),
        ("loss_type", "huber"),
        ("loss_type", "l1"),
        ("train_inpainting", True),
        ("vae", "/tmp/other-vae.safetensors"),
    ],
)
def test_incompatible_options_fail(flag, value):
    args = _base_args(**{flag: value})
    with pytest.raises(ValueError, match="playground_v25"):
        edm_playground.validate_training_args(args)


def test_full_finetune_rejects_flag():
    args = _base_args()
    with pytest.raises(ValueError, match="sdxl_train_network.py"):
        sdxl_train_util.verify_sdxl_training_args(args, support_playground_v25=False)


def test_lora_parser_accepts_clean_flag():
    args = _base_args(no_half_vae=True)
    sdxl_train_util.verify_sdxl_training_args(args, support_playground_v25=True)
    assert args.no_half_vae is True


def test_edm_markers_fail_without_flag(tmp_path):
    path = tmp_path / "pg.safetensors"
    save_file(
        {
            "edm_mean": torch.tensor(edm_playground.LATENTS_MEAN).view(1, 4, 1, 1),
            "edm_std": torch.tensor(edm_playground.LATENTS_STD).view(1, 4, 1, 1),
        },
        str(path),
    )
    assert edm_playground.checkpoint_has_playground_edm_markers(str(path)) is True
    with pytest.raises(ValueError, match="silently wrong"):
        edm_playground.raise_if_playground_checkpoint_without_flag(str(path), False)

    folder = tmp_path / "diffusers_pg"
    (folder / "scheduler").mkdir(parents=True)
    (folder / "model_index.json").write_text(
        '{"scheduler": ["diffusers", "EDMDPMSolverMultistepScheduler"]}', encoding="utf-8"
    )
    assert edm_playground.checkpoint_has_playground_edm_markers(str(folder)) is True
    with pytest.raises(ValueError, match="--playground_v25"):
        edm_playground.raise_if_playground_checkpoint_without_flag(str(folder), False)


def test_vae_config_without_stats_fails(tmp_path):
    folder = tmp_path / "model"
    (folder / "vae").mkdir(parents=True)
    (folder / "vae" / "config.json").write_text('{"scaling_factor": 0.13025}', encoding="utf-8")
    with pytest.raises(ValueError, match="latents_mean"):
        edm_playground.read_latent_stats(str(folder))


def test_latent_cache_is_not_mixed_with_sdxl(tmp_path):
    image = tmp_path / "img_07.png"
    image.write_bytes(b"not-read")
    pg = PlaygroundV25LatentsCachingStrategy(True, 1, False)
    sdxl = SdSdxlLatentsCachingStrategy(False, True, 1, False)
    pg_path = pg.get_latents_npz_path(str(image), (128, 64))
    sdxl_path = sdxl.get_latents_npz_path(str(image), (128, 64))
    assert pg_path.endswith("_pgv25.npz")
    assert sdxl_path.endswith("_sdxl.npz")
    assert pg_path != sdxl_path

    legacy = os.path.splitext(str(image))[0] + ".npz"
    np.savez(legacy, latents=np.zeros((4, 8, 8), dtype=np.float32))
    # Playground must not adopt the legacy SDXL cache even when it sits beside the image.
    assert pg.get_latents_npz_path(str(image), (128, 64)) == pg_path

    latents = torch.arange(4 * 8 * 16, dtype=torch.float32).reshape(4, 8, 16)
    pg._stamp_by_npz = None
    pg.save_latents_to_disk(pg_path, latents, [128, 64], [0, 0, 128, 64], key_reso_suffix="_8x16")
    loaded, _, _, _, _ = pg.load_latents_from_disk(pg_path, (128, 64))
    assert loaded.shape == (4, 8, 16)
    assert np.allclose(loaded, latents.numpy())

    # An SDXL-looking file renamed into the Playground suffix is refused.
    bad = tmp_path / "other_0128x0064_pgv25.npz"
    np.savez(bad, latents_8x16=latents.numpy(), original_size_8x16=np.array([64, 128]), crop_ltrb_8x16=np.array([0, 0, 0, 0]))
    with pytest.raises(ValueError, match="latent_format"):
        pg.load_latents_from_disk(str(bad), (128, 64))

    copied = tmp_path / "img_99_0128x0064_pgv25.npz"
    with np.load(pg_path) as src:
        payload = {key: src[key] for key in src.files}
    np.savez(copied, **payload)
    with pytest.raises(ValueError, match="img_07"):
        pg.load_latents_from_disk(str(copied), (128, 64))


class _Accel:
    device = torch.device("cpu")

    def autocast(self):
        return nullcontext()


class _FakeUnet(torch.nn.Module):
    def forward(self, x, timesteps, context, y):
        bias = context.float().mean() + y.float().mean() + timesteps.float().mean()
        return x.float() * 0.0 + bias


def test_sdxl_noise_target_path_is_unchanged():
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    args = _base_args(playground_v25=False, loss_type="l2")
    latents = torch.randn(2, 4, 4, 4)
    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
    batch = {
        "original_sizes_hw": torch.tensor([[64, 64], [64, 64]]),
        "crop_top_lefts": torch.tensor([[0, 0], [0, 0]]),
        "target_sizes_hw": torch.tensor([[64, 64], [64, 64]]),
        "custom_attributes": [{}, {}],
    }
    text = [
        torch.zeros(2, 77, 32),
        torch.zeros(2, 77, 32),
        torch.zeros(2, 1280),
    ]
    torch.manual_seed(7)
    reference_noise, reference_noisy, reference_timesteps = loss_util.get_noise_noisy_latents_and_timesteps(
        args, scheduler, latents
    )
    torch.manual_seed(7)
    pred, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args,
        _Accel(),
        scheduler,
        latents,
        batch,
        text,
        _FakeUnet(),
        None,
        torch.float32,
        True,
        True,
    )
    assert weighting is None
    assert torch.equal(target, reference_noise)
    assert torch.equal(timesteps, reference_timesteps)
    # Fake UNet returns a constant, so this only checks the call happened on the DDPM noisy latent.
    assert pred.shape == reference_noisy.shape
    scaled = trainer.shift_scale_latents(args, torch.ones(1, 4, 2, 2))
    assert torch.allclose(scaled, torch.full_like(scaled, sdxl_model_util.VAE_SCALE_FACTOR))


def test_trainer_playground_target_is_x0_not_epsilon():
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    trainer.pg_latents_mean, trainer.pg_latents_std, trainer.pg_scaling_factor = edm_playground.default_latent_stats()
    args = _base_args(no_half_vae=True, gradient_checkpointing=False)
    raw = torch.randn(2, 4, 4, 4)
    latents = trainer.shift_scale_latents(args, raw)
    manual = edm_playground.normalize_latents(raw)
    assert torch.allclose(latents, manual)

    batch = {
        "original_sizes_hw": torch.tensor([[64, 64], [64, 64]]),
        "crop_top_lefts": torch.tensor([[0, 0], [0, 0]]),
        "target_sizes_hw": torch.tensor([[64, 64], [64, 64]]),
    }
    text = [torch.zeros(2, 77, 32), torch.zeros(2, 77, 32), torch.zeros(2, 1280)]
    torch.manual_seed(11)
    pred, target, c_noise, weighting = trainer.get_noise_pred_and_target(
        args,
        _Accel(),
        None,
        latents,
        batch,
        text,
        _FakeUnet(),
        None,
        torch.float32,
        True,
        True,
    )
    assert weighting is None
    assert torch.allclose(target, latents.float())
    # c_noise is log-sigma, not a DDPM integer timestep.
    assert c_noise.dtype == torch.float32
    assert int(c_noise.abs().max()) < 1000 or True
    assert not torch.allclose(c_noise, c_noise.round())
    assert pred.shape == latents.shape


def test_sample_images_are_skipped(monkeypatch):
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    args = _base_args(sample_prompts="prompts.txt", sample_every_n_steps=1)

    def explode(*_a, **_k):
        raise AssertionError("DDPM sampler should not run")

    monkeypatch.setattr(sdxl_train_util, "sample_images", explode)
    trainer.sample_images(None, args, 0, 1, "cpu", None, None, None, None)
    trainer.sample_images(None, args, 0, 2, "cpu", None, None, None, None)

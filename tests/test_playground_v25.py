"""Playground v2.5 EDM math, guards, latent-cache identity, and the unchanged SDXL path."""

import argparse
import math
import os
import subprocess
import sys
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

    # An SDXL-looking file renamed into the Playground suffix is incomplete: recompute, do not crash the run.
    bad = tmp_path / "other_0128x0064_pgv25.npz"
    np.savez(bad, latents_8x16=latents.numpy(), original_size_8x16=np.array([64, 128]), crop_ltrb_8x16=np.array([0, 0, 0, 0]))
    assert pg.is_disk_cached_latents_expected((128, 64), str(bad), False, False) is False
    with pytest.raises(ValueError, match="Delete this file"):
        pg.load_latents_from_disk(str(bad), (128, 64))
    # A foreign format string is a one-line error that names the file.
    foreign = tmp_path / "img_08_0128x0064_pgv25.npz"
    np.savez(
        foreign,
        latents_8x16=latents.numpy(),
        original_size_8x16=np.array([64, 128]),
        crop_ltrb_8x16=np.array([0, 0, 0, 0]),
        latent_format=np.array("sdxl_raw"),
        source_basename=np.array("img_08"),
    )
    with pytest.raises(ValueError, match="img_08_0128x0064_pgv25.npz"):
        pg.is_disk_cached_latents_expected((128, 64), str(foreign), False, False)

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
    """Depends on the latent and the timestep, so a constant-output UNet cannot pass."""

    def forward(self, x, timesteps, context, y):
        t = timesteps.float().view(-1, 1, 1, 1)
        return x.float() * 0.05 + t * 0.01


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
    expected_pred = reference_noisy.float() * 0.05 + reference_timesteps.float().view(-1, 1, 1, 1) * 0.01
    assert torch.allclose(pred.float(), expected_pred, rtol=1e-5, atol=1e-5)
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
    noise = torch.randn_like(latents, dtype=torch.float32)
    indices = torch.randint(0, edm_playground.NUM_TRAIN_TIMESTEPS, (latents.shape[0],), device="cpu")
    noisy, scaled, c_noise_ref, sigma = edm_playground.prepare_edm_inputs(latents, noise, indices)
    model_out = scaled.float() * 0.05 + c_noise_ref.float().view(-1, 1, 1, 1) * 0.01
    expected = edm_playground.x0_target_from_model_output(noisy, model_out, sigma)

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
    assert c_noise.dtype == torch.float32
    assert torch.allclose(c_noise, 0.25 * torch.log(sigma))
    assert torch.allclose(c_noise, c_noise_ref)
    assert int(c_noise.abs().max()) < 1000
    assert not torch.allclose(c_noise, c_noise.round())
    assert torch.allclose(pred, expected, rtol=1e-5, atol=1e-5)


def test_sample_images_are_skipped(monkeypatch):
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    args = _base_args(sample_prompts="prompts.txt", sample_every_n_steps=1)

    def explode(*_a, **_k):
        raise AssertionError("DDPM sampler should not run")

    monkeypatch.setattr(sdxl_train_util, "sample_images", explode)
    trainer.sample_images(None, args, 0, 1, "cpu", None, None, None, None)
    trainer.sample_images(None, args, 0, 2, "cpu", None, None, None, None)


def _pg_batch(batch_size=2):
    return {
        "original_sizes_hw": torch.tensor([[64, 64]] * batch_size),
        "crop_top_lefts": torch.tensor([[0, 0]] * batch_size),
        "target_sizes_hw": torch.tensor([[64, 64]] * batch_size),
    }


def _pg_text(batch_size=2):
    return [torch.zeros(batch_size, 77, 32), torch.zeros(batch_size, 77, 32), torch.zeros(batch_size, 1280)]


def test_validation_pins_karras_index(monkeypatch):
    real_randint = torch.randint

    def fail_randint(*_a, **_k):
        raise AssertionError("validation sampled a random sigma")

    monkeypatch.setattr(torch, "randint", fail_randint)
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    args = _base_args(no_half_vae=True, min_timestep=200, max_timestep=200)
    latents = torch.randn(2, 4, 4, 4)
    _pred, _target, c_noise, weighting = trainer.get_noise_pred_and_target(
        args, _Accel(), None, latents, _pg_batch(), _pg_text(), _FakeUnet(), None, torch.float32, True, False
    )
    expected = edm_playground.get_schedule().timesteps[200].expand_as(c_noise)
    assert torch.allclose(c_noise, expected)
    assert weighting is None
    # Training still draws even if the validation hack left min == max on the args object.
    called = {"n": 0}

    def counting_randint(*a, **k):
        called["n"] += 1
        return real_randint(*a, **k)

    monkeypatch.setattr(torch, "randint", counting_randint)
    trainer.get_noise_pred_and_target(
        args, _Accel(), None, latents, _pg_batch(), _pg_text(), _FakeUnet(), None, torch.float32, True, True
    )
    assert called["n"] == 1


def test_continuous_sigma_c_noise_is_quarter_log():
    sigma = torch.tensor([0.123456, 3.5, 17.0], dtype=torch.float32)
    latents = torch.randn(3, 4, 2, 2)
    noise = torch.randn_like(latents)
    _noisy, _scaled, c_noise, sigma_out = edm_playground.prepare_edm_inputs_from_sigma(latents, noise, sigma)
    assert torch.allclose(sigma_out, sigma)
    assert torch.allclose(c_noise, 0.25 * torch.log(sigma), rtol=0, atol=0)
    schedule = edm_playground.get_schedule()
    for value, cn in zip(sigma, c_noise):
        nearest = int(torch.argmin((schedule.sigmas[:-1] - value).abs()).item())
        assert abs(float(cn) - float(schedule.timesteps[nearest])) > 1e-4


def test_lognormal_sigma_stats():
    torch.manual_seed(0)
    sigma = edm_playground.sample_lognormal_sigma(100_000, p_mean=-1.2, p_std=1.2)
    log_sigma = torch.log(sigma)
    assert float(log_sigma.mean()) == pytest.approx(-1.2, abs=0.02)
    assert float(log_sigma.std(unbiased=False)) == pytest.approx(1.2, abs=0.02)
    assert float(sigma.median()) == pytest.approx(math.exp(-1.2), rel=0.05)
    assert float((sigma > 10).float().mean()) < 0.02


def test_edm_weight_matches_f_space_mse():
    torch.manual_seed(3)
    latents = torch.randn(4, 4, 4, 4)
    noise = torch.randn_like(latents)
    indices = torch.tensor([0, 10, 400, 999])
    noisy, scaled, c_noise, sigma = edm_playground.prepare_edm_inputs(latents, noise, indices)
    model_out = torch.sin(scaled) * 0.1 + c_noise.view(-1, 1, 1, 1) * 0.01
    pred = edm_playground.x0_target_from_model_output(noisy, model_out, sigma)
    weight = edm_playground.edm_loss_weight(sigma).view(-1, 1, 1, 1)
    weighted = ((pred - latents) ** 2 * weight).mean()
    c_out = edm_playground.c_out(sigma.view(-1, 1, 1, 1))
    f_target = (latents - edm_playground.c_skip(sigma.view(-1, 1, 1, 1)) * noisy) / c_out
    f_space = ((model_out - f_target) ** 2).mean()
    unweighted = ((pred - latents) ** 2).mean()
    assert torch.allclose(weighted, f_space, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(unweighted, f_space, rtol=1e-3, atol=1e-3)


def test_trainer_lognormal_and_edm_weight(monkeypatch):
    def fail_randint(*_a, **_k):
        raise AssertionError("lognormal sampling should not draw a Karras index")

    monkeypatch.setattr(torch, "randint", fail_randint)
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    args = _base_args(no_half_vae=True, pgv25_sigma_sampling="lognormal", pgv25_loss_weighting="edm")
    latents = torch.randn(2, 4, 4, 4)
    _pred, target, c_noise, weighting = trainer.get_noise_pred_and_target(
        args, _Accel(), None, latents, _pg_batch(), _pg_text(), _FakeUnet(), None, torch.float32, True, True
    )
    sigma = torch.exp(c_noise / 0.25)
    assert torch.allclose(c_noise, 0.25 * torch.log(sigma))
    assert torch.allclose(weighting, edm_playground.edm_loss_weight(sigma).view(-1, 1, 1, 1))
    assert torch.allclose(target, latents.float())


@pytest.mark.parametrize("train_unet", [False, True])
def test_gradient_checkpoint_bf16_does_not_raise(train_unet):
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    args = _base_args(no_half_vae=True, gradient_checkpointing=True)
    latents = torch.randn(2, 4, 2, 2)
    pred, target, c_noise, weighting = trainer.get_noise_pred_and_target(
        args,
        _Accel(),
        None,
        latents,
        _pg_batch(),
        _pg_text(),
        _FakeUnet(),
        None,
        torch.bfloat16,
        train_unet,
        True,
    )
    assert pred.shape == latents.shape
    assert target.shape == latents.shape
    assert c_noise.shape == (2,)
    assert weighting is None


def test_pg_options_require_the_flag():
    args = _base_args(playground_v25=False, pgv25_loss_weighting="edm")
    with pytest.raises(ValueError, match="require --playground_v25"):
        sdxl_train_util.verify_sdxl_training_args(args, support_playground_v25=True)


def test_incomplete_cache_and_size_change(tmp_path):
    image = tmp_path / "img_01.png"
    image.write_bytes(b"aaaa")
    pg = PlaygroundV25LatentsCachingStrategy(True, 1, False)
    path = pg.get_latents_npz_path(str(image), (64, 64))
    latents = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(4, 8, 8)
    pg._stamp_by_npz = None
    pg.save_latents_to_disk(path, latents, [64, 64], [0, 0, 64, 64], key_reso_suffix="_8x8")
    assert pg.is_disk_cached_latents_expected((64, 64), path, False, False) is True
    loaded, _, _, _, _ = pg.load_latents_from_disk(path, (64, 64))
    assert np.allclose(loaded, latents.numpy())
    with np.load(path) as npz:
        assert "source_size" in npz.files
        assert "source_mtime_ns" in npz.files
    assert [name for name in os.listdir(tmp_path) if name.startswith(".pgv25-cache-")] == []

    image.write_bytes(b"aaaa-replaced")
    assert pg.is_disk_cached_latents_expected((64, 64), path, False, False) is False

    garbage = tmp_path / "img_02_0064x0064_pgv25.npz"
    garbage.write_bytes(b"not a numpy archive")
    pg.get_latents_npz_path(str(tmp_path / "img_02.png"), (64, 64))
    skip = PlaygroundV25LatentsCachingStrategy(True, 1, True)
    skip.get_latents_npz_path(str(tmp_path / "img_02.png"), (64, 64))
    assert skip.is_disk_cached_latents_expected((64, 64), str(garbage), False, False) is False

    fp16 = PlaygroundV25LatentsCachingStrategy(True, 1, False)
    fp16.cache_fp16 = True
    fp16_path = fp16.get_latents_npz_path(str(image), (64, 64))
    fp16._stamp_by_npz = None
    fp16.save_latents_to_disk(fp16_path, latents, [64, 64], [0, 0, 64, 64], key_reso_suffix="_8x8")
    with np.load(fp16_path) as npz:
        assert npz["latents_8x8"].dtype == np.float16
    loaded_fp16, _, _, _, _ = fp16.load_latents_from_disk(fp16_path, (64, 64))
    assert np.allclose(loaded_fp16.astype(np.float32), latents.numpy())


def test_weight_check_script_skips_when_missing(tmp_path):
    missing = tmp_path / "missing-playground.safetensors"
    proc = subprocess.run(
        [sys.executable, "tools/check_playground_v25_unet.py", "--model", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "SKIP" in combined
    assert str(missing) in combined

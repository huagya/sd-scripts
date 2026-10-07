#!/usr/bin/env python3
"""Compare the kohya SDXL UNet load with the diffusers UNet load on a local Playground v2.5 checkpoint.

The script is meant for a Windows GPU machine with the weights already downloaded.
It does not download the weights. If ``--model`` is missing, it prints SKIP and
exits 0 so a CPU test run can call it without the file.

Low memory:
  * only UNet tensors are read (not the text encoders or the VAE)
  * weights are cast to fp16 or bf16
  * the latent is small (default 1x4x8x8)
  * the diffusers UNet is freed before the kohya UNet is built

``edm_mean`` / ``edm_std`` are reported and are not passed into the UNet.
A missing model path is a skip. A present path that fails to load is exit 1.

Example (Command Prompt or PowerShell)::

    python tools\\check_playground_v25_unet.py --model C:\\models\\playground-v2.5-1024px-aesthetic.safetensors --dtype bf16
"""

from __future__ import annotations

import argparse
import os
import sys


def _dtype_from_name(name: str):
    import torch

    table = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if name not in table:
        raise SystemExit(f"--dtype must be fp16 or bf16, got {name}")
    return table[name]


def _report_edm_keys(path: str) -> None:
    """Print whether edm_mean / edm_std are in a safetensors file. Directories skip this."""
    if not os.path.isfile(path):
        print("edm_mean: not a single file (directory load)")
        print("edm_std: not a single file (directory load)")
        return
    if not path.lower().endswith(".safetensors"):
        print("edm_mean: skipped (not safetensors)")
        print("edm_std: skipped (not safetensors)")
        return
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name in ("edm_mean", "edm_std"):
            if name not in keys:
                print(f"{name}: absent")
                continue
            tensor = handle.get_tensor(name).float().reshape(-1)
            values = ", ".join(f"{v:.6g}" for v in tensor.tolist())
            print(f"{name}: present shape={tuple(handle.get_tensor(name).shape)} values=[{values}]")
    print("extra keys edm_mean/edm_std are not UNet weights; the kohya loader ignores them.")


def _load_kohya_unet_from_safetensors(path: str, dtype, device):
    """Same UNet class and state-dict load as ``load_models_from_sdxl_checkpoint``, UNet keys only."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from library.sdxl_model_util import _load_state_dict_on_device
    from library.sdxl_original_unet import SdxlUNet2DConditionModel

    unet_sd = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model.diffusion_model."):
                unet_sd[key[len("model.diffusion_model.") :]] = handle.get_tensor(key)
    if "input_blocks.0.0.weight" not in unet_sd:
        raise RuntimeError(
            f"{path} has no model.diffusion_model.input_blocks.0.0.weight. "
            "Expected an SDXL-layout Playground safetensors file."
        )
    in_channels = int(unet_sd["input_blocks.0.0.weight"].shape[1])
    with init_empty_weights():
        unet = SdxlUNet2DConditionModel(in_channels=in_channels)
    info = _load_state_dict_on_device(unet, unet_sd, device="cpu", dtype=dtype)
    print(f"kohya UNet load: {info} in_channels={in_channels} dtype={dtype}")
    unet.to(device)
    unet.eval()
    return unet


def _load_kohya_unet_from_diffusers_dir(path: str, dtype, device):
    """Load ``unet/`` with diffusers, then the same conversion the kohya SDXL loader uses."""
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel

    from library.sdxl_model_util import _load_state_dict_on_device, convert_diffusers_unet_state_dict_to_sdxl
    from library.sdxl_original_unet import SdxlUNet2DConditionModel

    diffusers_unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=True)
    state_dict = convert_diffusers_unet_state_dict_to_sdxl(diffusers_unet.state_dict())
    in_channels = int(state_dict["input_blocks.0.0.weight"].shape[1])
    del diffusers_unet
    with init_empty_weights():
        unet = SdxlUNet2DConditionModel(in_channels=in_channels)
    info = _load_state_dict_on_device(unet, state_dict, device="cpu", dtype=dtype)
    print(f"kohya UNet load (converted from diffusers folder): {info} dtype={dtype}")
    unet.to(device)
    unet.eval()
    return unet


def _load_diffusers_unet(path: str, dtype, device):
    from diffusers import UNet2DConditionModel

    if os.path.isdir(path):
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=True)
    else:
        # May download the small SDXL UNet config JSON, not the weights.
        unet = UNet2DConditionModel.from_single_file(path, torch_dtype=dtype)
    unet.to(device)
    unet.eval()
    print(f"diffusers UNet load: {type(unet).__name__} dtype={dtype}")
    return unet


def _matching_inputs(diffusers_unet, dtype, device, latent_h: int, latent_w: int, seed: int):
    """One latent, one c_noise, and the same 2816-d vector both UNets consume."""
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    latents = torch.randn(1, 4, latent_h, latent_w, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    context = torch.randn(1, 77, 2048, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    text_embeds = torch.randn(1, 1280, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    # original h/w, crop top/left, target h/w
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=dtype)
    # c_noise = 0.25 * ln(sigma) for sigma=1
    timestep = torch.zeros(1, device=device, dtype=dtype)
    time_embeds = diffusers_unet.add_time_proj(time_ids.flatten())
    time_embeds = time_embeds.reshape(text_embeds.shape[0], -1).to(dtype=dtype)
    y = torch.cat([text_embeds, time_embeds], dim=1)
    return latents, timestep, context, text_embeds, time_ids, y


def compare(path: str, dtype_name: str, latent_h: int, latent_w: int, seed: int, device_name: str) -> int:
    import torch

    if not os.path.exists(path):
        print(f"SKIP: model not found: {path}")
        print("Place the Playground v2.5 weights at that path and run this script again.")
        return 0

    dtype = _dtype_from_name(dtype_name)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"model: {path}")
    print(f"device: {device} dtype: {dtype} latent: 1x4x{latent_h}x{latent_w}")
    _report_edm_keys(path)

    diffusers_unet = _load_diffusers_unet(path, dtype, device)
    latents, timestep, context, text_embeds, time_ids, y = _matching_inputs(
        diffusers_unet, dtype, device, latent_h, latent_w, seed
    )
    with torch.no_grad():
        diffusers_out = diffusers_unet(
            latents,
            timestep,
            encoder_hidden_states=context,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample.float().cpu()
    del diffusers_unet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if os.path.isdir(path):
        kohya_unet = _load_kohya_unet_from_diffusers_dir(path, dtype, device)
    else:
        kohya_unet = _load_kohya_unet_from_safetensors(path, dtype, device)
    with torch.no_grad():
        kohya_out = kohya_unet(latents, timestep, context, y).float().cpu()
    del kohya_unet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    diff = (kohya_out - diffusers_out).abs()
    print(f"max abs diff: {float(diff.max()):.8g}")
    print(f"mean abs diff: {float(diff.mean()):.8g}")
    print(f"kohya shape: {tuple(kohya_out.shape)} diffusers shape: {tuple(diffusers_out.shape)}")
    if kohya_out.shape != diffusers_out.shape:
        print("ERROR: output shapes differ")
        return 1
    if not torch.isfinite(diff).all():
        print("ERROR: non-finite diff")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare kohya vs diffusers Playground v2.5 UNet on one forward.")
    parser.add_argument("--model", required=True, help="Local safetensors file or diffusers folder. Missing path exits 0 (skip).")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"], help="UNet dtype. Default fp16.")
    parser.add_argument("--height", type=int, default=8, help="Latent height. Default 8.")
    parser.add_argument("--width", type=int, default=8, help="Latent width. Default 8.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda. Default auto.")
    args = parser.parse_args(argv)
    # Accept Windows paths as the user typed them.
    model = os.path.expanduser(args.model)
    return compare(model, args.dtype, args.height, args.width, args.seed, args.device)


if __name__ == "__main__":
    sys.exit(main())

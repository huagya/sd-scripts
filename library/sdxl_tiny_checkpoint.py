"""Randomly initialized SDXL-shaped checkpoint for CPU tests.

Production SDXL is about 3.5B parameters and does not fit this environment.
``ss_sdxl_arch=tiny-test-v1`` in the safetensors metadata selects this loader.
Real training must not set that metadata; the default loader is unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load_file, save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from library.clip_text_model import wrap_clip_text_model
from library.sdxl_original_unet import SdxlUNet2DConditionModel

TINY_ARCH_KEY = "ss_sdxl_arch"
TINY_ARCH_VALUE = "tiny-test-v1"
TINY_ARCH_JSON_KEY = "ss_sdxl_arch_json"

# TE1 hidden + TE2 hidden must equal UNet context_dim.
# TE2 projection_dim + 256*6 size embeddings must equal adm_in_channels (1280+1536).
DEFAULT_TINY_ARCH: Dict[str, Any] = {
    "unet": {
        "in_channels": 4,
        "model_channels": 32,
        "context_dim": 64,
        "adm_in_channels": 2816,
        "attention_head_dim": 64,
        "transformer_depth_level1": 1,
        "transformer_depth_level2": 1,
    },
    "te1": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "projection_dim": 32,
    },
    "te2": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "projection_dim": 1280,
    },
    "vae": {
        "block_out_channels": [32, 32, 32, 32],
        "layers_per_block": 1,
    },
}


def is_tiny_sdxl_checkpoint(path: str) -> bool:
    if not path or not path.endswith(".safetensors"):
        return False
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return metadata.get(TINY_ARCH_KEY) == TINY_ARCH_VALUE
    except Exception:
        return False


def _clip_config(spec: Dict[str, Any], hidden_act: str) -> CLIPTextConfig:
    return CLIPTextConfig(
        vocab_size=49408,
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        max_position_embeddings=77,
        hidden_act=hidden_act,
        layer_norm_eps=1e-5,
        projection_dim=spec["projection_dim"],
        bos_token_id=0,
        eos_token_id=49407,
        pad_token_id=0,
    )


def build_tiny_models(arch: Dict[str, Any] | None = None, seed: int = 0):
    arch = arch or DEFAULT_TINY_ARCH
    generator = torch.Generator().manual_seed(seed)
    # torch.manual_seed covers nn.init inside from_config
    torch.manual_seed(seed)

    te1 = wrap_clip_text_model(CLIPTextModel(_clip_config(arch["te1"], "quick_gelu")))
    te2 = CLIPTextModelWithProjection(_clip_config(arch["te2"], "gelu"))
    unet = SdxlUNet2DConditionModel(**arch["unet"])

    from diffusers import AutoencoderKL

    vae_channels = tuple(arch["vae"]["block_out_channels"])
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * len(vae_channels),
        up_block_types=("UpDecoderBlock2D",) * len(vae_channels),
        block_out_channels=vae_channels,
        layers_per_block=arch["vae"]["layers_per_block"],
        latent_channels=4,
        norm_num_groups=32,
        sample_size=64,
    )
    # Re-seed parameters deterministically in case from_config draws extra RNG.
    _reinit(te1, generator)
    _reinit(te2, generator)
    _reinit(unet, generator)
    _reinit(vae, generator)
    te1.eval()
    te2.eval()
    unet.eval()
    vae.eval()
    return te1, te2, vae, unet


def _reinit(module: torch.nn.Module, generator: torch.Generator) -> None:
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02, generator=generator)
        parameter.data.mul_(0.1)


def save_tiny_sdxl_checkpoint(path: str, seed: int = 0, arch: Dict[str, Any] | None = None) -> Dict[str, Any]:
    arch = json.loads(json.dumps(arch or DEFAULT_TINY_ARCH))
    te1, te2, vae, unet = build_tiny_models(arch, seed=seed)
    state = {}
    for key, value in te1.state_dict().items():
        state["te1." + key] = value.contiguous()
    for key, value in te2.state_dict().items():
        state["te2." + key] = value.contiguous()
    for key, value in unet.state_dict().items():
        state["unet." + key] = value.contiguous()
    for key, value in vae.state_dict().items():
        state["vae." + key] = value.contiguous()
    save_file(
        state,
        path,
        metadata={
            TINY_ARCH_KEY: TINY_ARCH_VALUE,
            TINY_ARCH_JSON_KEY: json.dumps(arch),
        },
    )
    return arch


def load_tiny_sdxl_checkpoint(path: str, device="cpu", dtype=None) -> Tuple[Any, Any, Any, Any, None, None]:
    """Return the same tuple as ``load_models_from_sdxl_checkpoint`` (no logit scale)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    arch = json.loads(metadata[TINY_ARCH_JSON_KEY])
    te1, te2, vae, unet = build_tiny_models(arch, seed=0)
    state = load_file(path, device="cpu")

    def take(prefix: str, module: torch.nn.Module) -> None:
        subset = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        missing, unexpected = module.load_state_dict(subset, strict=False)
        # position_ids is a buffer on some transformers builds and is not trained.
        missing = [key for key in missing if not key.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"tiny checkpoint {prefix} missing={missing} unexpected={unexpected}")

    take("te1.", te1)
    take("te2.", te2)
    take("unet.", unet)
    take("vae.", vae)
    if dtype is not None:
        unet.to(dtype=dtype)
        vae.to(dtype=dtype)
    if device is not None:
        te1.to(device)
        te2.to(device)
        unet.to(device)
        vae.to(device)
    return te1, te2, vae, unet, None, None

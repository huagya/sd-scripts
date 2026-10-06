"""End-to-end sdxl_train_network.py smoke test on a tiny random SDXL-shaped model.

No Playground / SDXL weight download. Set PGV25_STAMP_LATENT_ID so the disk cache
records the image's red-channel mean (the caption id) and PGV25_ASSERT_LATENT_ID
so every training batch checks that the caption and the latent still match.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open

from library.sdxl_tiny_checkpoint import is_tiny_sdxl_checkpoint, load_tiny_sdxl_checkpoint, save_tiny_sdxl_checkpoint


pytestmark = pytest.mark.smoke


def _write_dataset(root: str, count: int, plant_legacy_npz: bool = False) -> None:
    subset = os.path.join(root, "1_ids")
    os.makedirs(subset, exist_ok=True)
    sizes = [
        (64, 64),
        (64, 96),
        (96, 64),
        (96, 96),
        (128, 64),
        (64, 128),
        (128, 96),
        (96, 128),
        (128, 128),
        (160, 96),
    ]
    words = " ".join(["watercolor"] * 30)
    for i in range(1, count + 1):
        width, height = sizes[(i - 1) % len(sizes)]
        image = Image.new("RGB", (width, height), (i, 40, 90))
        path = os.path.join(subset, f"img_{i:02d}.png")
        image.save(path)
        tags = [f"id{i:03d}"] + [f"tag{n:03d}" for n in range(60)]
        caption = ", ".join(tags) + ", " + words
        with open(os.path.join(subset, f"img_{i:02d}.txt"), "w", encoding="utf-8") as handle:
            handle.write(caption)
        if plant_legacy_npz:
            # A legacy SDXL cache sits beside the image. Playground training must ignore it.
            legacy = os.path.join(subset, f"img_{i:02d}.npz")
            np.savez(legacy, latents=np.ones((4, 4, 4), dtype=np.float32))


def _run(cmd, env, log_path):
    print("RUN", " ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        with open(log_path, "r", encoding="utf-8", errors="replace") as log:
            text = log.read()
        tail = text[-8000:]
        raise AssertionError(f"command failed ({proc.returncode}). Log tail:\n{tail}")
    return log_path


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny") / "tiny_sdxl.safetensors"
    save_tiny_sdxl_checkpoint(str(path), seed=0)
    assert is_tiny_sdxl_checkpoint(str(path))
    return str(path)


def _train_args(checkpoint, data_dir, out_dir, name, steps, playground: bool):
    cmd = [
        sys.executable,
        "sdxl_train_network.py",
        f"--pretrained_model_name_or_path={checkpoint}",
        f"--train_data_dir={data_dir}",
        f"--output_dir={out_dir}",
        f"--output_name={name}",
        "--save_model_as=safetensors",
        "--network_module=networks.lora",
        "--network_dim=2",
        "--network_alpha=1",
        "--network_train_unet_only",
        "--resolution=128,128",
        "--enable_bucket",
        "--min_bucket_reso=64",
        "--max_bucket_reso=160",
        "--bucket_reso_steps=32",
        "--train_batch_size=1",
        f"--max_train_steps={steps}",
        "--learning_rate=1e-4",
        "--optimizer_type=AdamW",
        "--mixed_precision=no",
        "--cache_latents_to_disk",
        "--caption_extension=.txt",
        "--shuffle_caption",
        "--keep_tokens=1",
        "--max_token_length=225",
        "--gradient_checkpointing",
        "--sdpa",
        "--seed=42",
        "--max_data_loader_n_workers=0",
        "--no_half_vae",
        "--logging_dir=" + os.path.join(out_dir, "logs"),
    ]
    if playground:
        cmd.append("--playground_v25")
    return cmd


def test_playground_v25_lora_smoke(tiny_checkpoint, tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    _write_dataset(str(data), 20, plant_legacy_npz=True)
    env = os.environ.copy()
    env["PGV25_STAMP_LATENT_ID"] = "1"
    env["PGV25_ASSERT_LATENT_ID"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log_path = str(tmp_path / "playground_train.log")
    _run(_train_args(tiny_checkpoint, str(data), str(out), "pg_lora", 20, True), env, log_path)

    lora_path = out / "pg_lora.safetensors"
    assert lora_path.is_file(), f"LoRA was not saved. See {log_path}"
    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    assert metadata.get("ss_playground_v25") == "True"
    assert metadata.get("ss_network_module") == "networks.lora"
    assert metadata.get("ss_max_token_length") == "225"
    assert any(key.startswith("lora_unet_") for key in keys)
    assert not any(key in ("edm_mean", "edm_std") for key in keys)

    pg_caches = []
    for dirpath, _, files in os.walk(data):
        for name in files:
            if name.endswith("_pgv25.npz"):
                pg_caches.append(os.path.join(dirpath, name))
            if name.endswith("_sdxl.npz"):
                raise AssertionError(f"SDXL latent cache was written during Playground training: {name}")
    assert len(pg_caches) == 20

    from library.edm_playground import LATENT_FORMAT_VALUE, assert_playground_latent_npz

    for i in range(1, 21):
        image_path = data / "1_ids" / f"img_{i:02d}.png"
        with Image.open(image_path) as image:
            red = float(np.array(image.convert("RGB"))[:, :, 0].mean())
        assert red == pytest.approx(float(i))
        caption = (data / "1_ids" / f"img_{i:02d}.txt").read_text(encoding="utf-8")
        assert caption.startswith(f"id{i:03d},")
        matches = [path for path in pg_caches if os.path.basename(path).startswith(f"img_{i:02d}_")]
        assert len(matches) == 1, matches
        with np.load(matches[0]) as npz:
            assert_playground_latent_npz(matches[0], npz)
            assert str(npz["latent_format"].item()) == LATENT_FORMAT_VALUE
            # The training process already asserted caption/latent stamps. Check the file too.
            latent_key = [key for key in npz.files if key.startswith("latents_")][0]
            assert float(npz[latent_key][0, 0, 0]) == pytest.approx(red)
        legacy = data / "1_ids" / f"img_{i:02d}.npz"
        with np.load(legacy) as old:
            assert old["latents"].shape == (4, 4, 4)
            assert list(old.files) == ["latents"]

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        log_text = handle.read()
    assert "PGV25 pairing" not in log_text
    assert "steps:" in log_text or "num epochs" in log_text

    te1, te2, vae, unet, _, _ = load_tiny_sdxl_checkpoint(tiny_checkpoint)
    from networks.lora import create_network_from_weights

    network, weights = create_network_from_weights(1.0, str(lora_path), vae, [te1, te2], unet, for_inference=True)
    assert network is not None
    assert weights is not None
    assert len(weights) == len(keys)
    # Applying the LoRA must run without a shape error. Weights are random and tiny.
    network.merge_to([te1, te2], unet, weights, torch.float32, torch.device("cpu"))


def test_plain_sdxl_lora_smoke_unchanged(tiny_checkpoint, tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    _write_dataset(str(data), 2)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path = str(tmp_path / "sdxl_train.log")
    _run(_train_args(tiny_checkpoint, str(data), str(out), "sdxl_lora", 2, False), env, log_path)

    lora_path = out / "sdxl_lora.safetensors"
    assert lora_path.is_file()
    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    assert metadata.get("ss_playground_v25") == "False"
    assert any(key.startswith("lora_unet_") for key in keys)
    pg = []
    sdxl = []
    for dirpath, _, files in os.walk(data):
        for name in files:
            if name.endswith("_pgv25.npz"):
                pg.append(name)
            if name.endswith("_sdxl.npz"):
                sdxl.append(name)
    assert pg == []
    assert len(sdxl) == 2

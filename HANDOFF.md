# Playground v2.5 LoRA handoff

SDXL LoRA training (`sdxl_train_network.py`) can now train against Playground v2.5's EDM objective instead of SDXL's DDPM epsilon objective. Full fine-tuning and a GUI are not in this change. The math lives in `library/edm_playground.py` so `sdxl_train.py` can call the same helpers later. `sdxl_train.py` rejects `--playground_v25` rather than ignoring it.

## Flag

`--playground_v25`

What it switches:

* Noise is `x = x0 + σ n`. The default sampler (`--pgv25_sigma_sampling=karras_uniform`) draws an index on the Karras grid used by diffusers `EDMEulerScheduler` (σ from 80 down toward 0.002, ρ = 7, 1000 training bins). The UNet timestep is `c_noise = 0.25 ln σ`, including when σ is continuous. It is not a DDPM integer.
* `--pgv25_sigma_sampling=lognormal` draws `ln(σ) ~ Normal(P_mean, P_std)` with `--pgv25_sigma_mean` / `--pgv25_sigma_std` (defaults -1.2 / 1.2, the EDM paper). `c_noise` is still `0.25 ln σ` of that continuous σ, not a grid lookup.
* The UNet input is `c_in * x`. The training target is `x0`. The prediction is `c_skip * x + c_out * F` with `σ_data = 0.5` and epsilon preconditioning (Playground's `prediction_type`).
* Loss default (`--pgv25_loss_weighting=none`) is unweighted MSE on that `x0` prediction (diffusers PR #7126 / `train_dreambooth_lora_sdxl.py` with `--do_edm_style_training`). `--pgv25_loss_weighting=edm` multiplies by `λ(σ) = (σ² + σ_data²) / (σ · σ_data)²`, which is `1/c_out²` and equals an F-space MSE. The default stays the diffusers reference until a real-GPU A/B picks one.
* Latents are normalized as `(z - mean) * 0.5 / std` at train time. Disk caches store the raw VAE sample. `--pgv25_cache_fp16` stores that raw sample as fp16 (about half the disk). Normalization at train time is still fp32.

Published constants (also the official single-file `edm_mean` / `edm_std` tensors):

* mean `[-1.6574, 1.886, -1.383, 2.5155]`
* std `[8.4927, 5.9022, 6.5498, 5.2299]`
* scaling `0.5`

If the checkpoint or Diffusers `vae/config.json` contains these stats, those values are used. A VAE config that has only `scaling_factor` is an error. `--vae` is an error, so an sdxl-vae-fp16-fix file cannot silently replace Playground's normalization.

EDM coefficients are computed in fp32 even when the UNet is bf16. The reference script multiplies σ in the latent dtype; this code does not.

## How to run

Both of these load through the existing SDXL paths:

* **Single-file safetensors** in the SDXL checkpoint layout (`model.diffusion_model.*`, `conditioner.embedders.*`, `first_stage_model.*`). The official `playground-v2.5-1024px-aesthetic.safetensors` is this layout. Its header contains `edm_mean` and `edm_std` with the values above (checked by reading the safetensors header and those 32 bytes, not by loading the weights). Extra keys are ignored by the SDXL loader.
* **Diffusers folder** (or a Hugging Face repo id) via `StableDiffusionXLPipeline`. `model_index.json` names `EDMDPMSolverMultistepScheduler`. `vae/config.json` supplies `latents_mean` / `latents_std`.

A file or folder that looks like Playground and is missing `--playground_v25` raises before the weights are used as an SDXL DDPM model.

Latent caches are `*_WWWWxHHHH_pgv25.npz`. Each write is a temp file in the same directory plus `os.replace`, so a crash does not leave a half-written npz in the final name. The file stores `latent_format=playground_v25_raw`, the source image stem, and the source file size and mtime in nanoseconds. `*_sdxl.npz` and legacy `.npz` are not read.

On the next launch:

* A missing marker, an unreadable file, or a size/mtime that does not match the image is logged in one line that names the file, and that file is recomputed.
* A `latent_format` that is present but not `playground_v25_raw`, or a source stem that does not match the filename, raises one line that names the file and says to delete it.
* These checks run even with `--skip_cache_check`.

`source_basename` only repeats the stem already encoded in the filename, so it cannot see a replaced image. Size and mtime can. A replacement that keeps both the same byte size and the same mtime (a copy that restores the old timestamp) is not detected. The cache does not store a pixel hash.

`--cache_latents` without `--cache_latents_to_disk` still normalizes correctly but keeps latents in RAM. For ~400k images use the disk cache. At 1024px the raw fp32 latent is 4 × 128 × 128 × 4 bytes ≈ 256 KiB, so 400k images are about 105 GB, and `--flip_aug` stores a second latent and roughly doubles that. `--pgv25_cache_fp16` cuts the latent arrays in half. Values are the same magnitude as SDXL latents (std about 6–8), so fp16 is quantization, not an overflow fix. Leave it off until you need the disk.

LoRA files are ordinary kohya LoRA (`lora_unet_*` keys). Metadata includes `ss_playground_v25=True`. `edm_mean` / `edm_std` are not written into the LoRA. ComfyUI's automatic Playground detection looks for those keys on a full checkpoint, which matters for a later full fine-tune, not for this LoRA.

Sample images during training are skipped. The built-in sampler is DDPM/epsilon and would denoise with the wrong timestep and the wrong latent scale. If `--sample_prompts` is set, training logs a warning and continues.

Validation is the SDXL loop: it pins `min_timestep = max_timestep` to 200, 400, 600, and 800 (four interior points of `linspace(0, 1000, 6)`). In Playground mode those integers are indices on the Karras training grid, so each validation pass uses that fixed σ for the whole batch. The Gaussian noise is still drawn, but the validation RNG is switched to `--validation_seed` (or `--seed`) by the existing loop, so the pair (σ, noise) repeats. Lognormal sampling is not used during validation. `--min_timestep` / `--max_timestep` on the command line are still rejected; only the validation loop sets them, and only while `is_train` is false.

### Starting point for 16GB VRAM (not tuned)

These are starting points, not measured settings. Nothing here was run on a 16GB GPU.

```bash
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
  --pretrained_model_name_or_path="playground-v2.5-1024px-aesthetic.safetensors" \
  --train_data_dir="data" \
  --output_dir="output" \
  --output_name="pgv25_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --network_dim=16 \
  --network_alpha=16 \
  --network_train_unet_only \
  --resolution=1024,1024 \
  --enable_bucket \
  --min_bucket_reso=640 \
  --max_bucket_reso=1536 \
  --bucket_reso_steps=64 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --optimizer_type=AdamW8bit \
  --learning_rate=1e-5 \
  --lr_scheduler=constant \
  --cache_latents_to_disk \
  --caption_extension=.txt \
  --shuffle_caption \
  --keep_tokens=1 \
  --max_token_length=225 \
  --sdpa \
  --max_data_loader_n_workers=2 \
  --seed=42 \
  --playground_v25
```

| Setting | Starting point | Why |
|---|---|---|
| `network_dim` / `network_alpha` | 16 / 16 | SDXL LoRA rank that usually fits in 16GB with batch 1. Try 8 if VRAM is tight, 32 if it fits. |
| optimizer | AdamW8bit, lr `1e-5` | 8-bit AdamW is the usual 16GB LoRA choice. `1e-5` is a starting UNet LoRA rate, not a tuned one. Prodigy (`--optimizer_type=Prodigy --learning_rate=1.0`) is another starting point people use for LoRA; it is not the default here. |
| resolution | 1024, buckets 640–1536, step 64 | Playground v2.5 is a 1024px model. The paper's non-square buckets (1152×896, 1216×832, 1344×768, and the 1254×836 eval crop) fit under a 1536 long side. A short side below 640 is dropped. |
| sigma / loss | `karras_uniform` + `none` | Diffusers reference. The Playground report (pgv2.5 paper PDF, p.6) says high-resolution training skewed noise toward noisier levels. `P_mean` / `P_std` and λ were not published. Switch to `lognormal` and/or `edm` only for an A/B. |
| caching | `--cache_latents_to_disk` | ~105 GB raw fp32 at 400k × 1024, about twice that with `flip_aug`. `--pgv25_cache_fp16` halves the latent arrays. Do not keep 400k latents in RAM. |
| VAE | fp32 (`--no_half_vae` is forced) | The Playground VAE config sets `force_upcast: true`, and diffusers PR #7126 keeps the encode in fp32. Raw latents are a similar magnitude to SDXL (std about 6–8). fp32 is that upcast, not because these latents uniquely overflow fp16. |
| workers | 2 or 0 on Windows | Windows spawn duplicates the dataset. |
| captions | `shuffle_caption` + `keep_tokens` | Works, because text-encoder outputs are not cached. Do not combine `--cache_text_encoder_outputs` with shuffle (existing SDXL assert). |

Do not pass `--v_parameterization`, `--min_snr_gamma`, `--scale_v_pred_loss_like_noise_pred`, `--v_pred_like_loss`, `--debiased_estimation_loss`, `--noise_offset`, `--multires_noise_iterations`, `--ip_noise_gamma`, `--min_timestep`, `--max_timestep`, `--zero_terminal_snr`, `--loss_type` other than `l2`, `--train_inpainting`, or `--vae`. Those raise.

## Design

* `library/edm_playground.py` — schedule, preconditioning, latent norm, argument checks, checkpoint marker checks. No training-loop policy.
* `SdxlNetworkTrainer` overrides scale, noise/target, loss post-processing, the cache strategy, metadata, and sampling only when the flag is set. The flag-off path calls the previous SDXL methods.
* `PlaygroundV25LatentsCachingStrategy` uses a different filename and refuses the wrong file. Normalization stays out of the cache so a later full fine-tune can share the files.
* `sdxl_train.py` and the other SDXL scripts share the argument, and `verify_sdxl_training_args` rejects it unless the caller passes `support_playground_v25=True` (only the LoRA trainer does).

Rejected or deferred:

* Making `lognormal` + `edm` the default. Both are flags now. The default stays script 070b (`karras_uniform`, unweighted x0 MSE) until a real-GPU A/B. Playground's report (https://marketing-cdn.playground.com/research/pgv2.5_compressed.pdf p.6) says they skewed noise toward noisier levels at high resolution (Hoogeboom-style). They did not publish `P_mean`, `P_std`, or λ. Uniform Karras puts more mass on high σ than `Normal(-1.2, 1.2)` does (median σ around 2.5 vs about 0.3).
* The advanced Diffusers script (cache/caption index bug, clip_skip argument bug). Not used.
* In-training previews. Implementing an EDM sampler (and the Playground VAE decode, including the mean/std denormalization) is a separate change.
* Full fine-tune, GUI, `edm_mean`/`edm_std` on a saved checkpoint, DoRA, CAME.
* Text-encoder-only training is not rejected. Gradient checkpointing casts the UNet input to bf16 before `requires_grad_`, including when the UNet is frozen, so that combination does not hit the non-leaf `requires_grad_` error. It has not been run as a real training job.

## Tests

Linux CPU, no GPU. Setup: `tools/setup_cpu_test_env.sh` (CPU torch, the packages in `requirements-test-cpu.txt`, CLIP tokenizers under `tokenizer_cache/`). The smoke test uses a randomly initialized SDXL-shaped UNet / text encoders / VAE (`library/sdxl_tiny_checkpoint.py`, metadata `ss_sdxl_arch=tiny-test-v1`). That loader is only for this metadata. A normal checkpoint never takes it.

```text
python3 -m pytest tests/test_playground_v25.py tests/test_playground_v25_smoke.py -v --tb=line
python3 -m pytest tests -q --tb=line --ignore=tests/local --ignore=tests/test_optimizer.py
```

Latest CPU run: Playground file + smoke are included in `322 passed, 21 skipped` (42.10s). `tests/test_optimizer.py` does not import without `bitsandbytes` (not installed here; that file was not changed). Covered:

* σ grid, `add_noise`, preconditioning, and unweighted MSE match diffusers `EDMEulerScheduler` and the 070b `get_sigmas` path (exact σ/c_noise, loss within 1e-6).
* `λ(σ) * ||x0 error||²` matches F-space MSE, and the unweighted loss does not. Lognormal draws match `Normal(-1.2, 1.2)` within sampling noise (median near `exp(-1.2)`, few σ > 10). Continuous σ uses `c_noise = 0.25 ln σ`, not the nearest grid bin.
* Published mean/std match the official fp32 safetensors `edm_mean`/`edm_std` values.
* Incompatible flags, a VAE config without stats, pgv25 options without `--playground_v25`, and a full-fine-tune parse all raise.
* `edm_mean` in a safetensors file, and an EDM scheduler in `model_index.json`, raise without the flag.
* `*_pgv25.npz` is distinct from `*_sdxl.npz` and legacy `.npz`. Writes are atomic. A marker-less or unreadable file is recomputed. A wrong `latent_format` or a copied stem raises one line. A size change recomputes. fp16 cache round-trips values that are exact in fp16.
* Validation with `min_timestep == max_timestep == 200` uses Karras index 200 and does not call `randint`. A training step still does.
* Flag off: the UNet output is `0.05 * noisy + 0.01 * timestep` (the fake UNet depends on its input) and the target is still ε. The latent scale is still 0.13025, same noise as `loss_util.get_noise_noisy_latents_and_timesteps`.
* Text-encoder-only and UNet training with gradient checkpointing and bf16 both return a prediction on CPU without the non-leaf `requires_grad_` error.
* CLI smoke, 20 solid-color images, disk cache, buckets, `shuffle_caption`, `keep_tokens=1`, `max_token_length=225`, gradient checkpointing, SDPA: saves `pg_lora.safetensors` with `ss_pgv25_sigma_sampling=karras_uniform` and `ss_pgv25_loss_weighting=none`, reloads it onto the tiny UNet, and checks each caption id against the latent stamp. Separate 2-step runs pass `--pgv25_sigma_sampling=lognormal` and `--pgv25_loss_weighting=edm`. A 2-step flag-off run writes `*_sdxl.npz` only.
* `tools/check_playground_v25_unet.py` with a missing path prints SKIP and exits 0.

## Real-weights check

`tools/check_playground_v25_unet.py` loads the UNet only (not the text encoders or VAE), in fp16 or bf16, on a small latent. It loads through the kohya SDXL UNet class (`model.diffusion_model.*` only, so `edm_mean` / `edm_std` are ignored) and through diffusers (`from_single_file` or `unet/` in a folder), runs the same input, and prints the max absolute difference. A missing path prints `SKIP` and exits 0. The first diffusers load may download the small SDXL UNet config, not the weights.

```text
python tools\check_playground_v25_unet.py --model C:\models\playground-v2.5-1024px-aesthetic.safetensors --dtype bf16
```

## 未確認 / unverified

* Loading the real ~3.5B weights into the trainer, and the max abs diff from the check script above. Only the safetensors header and the 32-byte `edm_mean`/`edm_std` payload were read. The fp16 single-file was not opened.
* Diffusers-folder load of the real Playground repo (`StableDiffusionXLPipeline.from_pretrained`). The marker check for a local `model_index.json` was tested; the weight conversion was not.
* Any GPU, 16GB VRAM, a real bf16 training step, 8-bit AdamW, Prodigy, xformers, or images-per-second. The bf16 + gradient-checkpointing test is a tiny fake UNet on CPU.
* Which of `karras_uniform`/`none` vs `lognormal`/`edm` matches Playground. The paper's high-resolution noise skew is not in the default.
* Image quality, overfit tests, or color cast after a real VAE decode.
* ComfyUI loading the kohya LoRA on top of Playground v2.5. ComfyUI detects the base by `edm_mean`/`edm_std` on the checkpoint, not on the LoRA.
* Whether every non-official single-file (Civitai repacks, `.ckpt`) uses the SDXL key prefixes. A `.ckpt` is not scanned for `edm_mean`.
* Windows DataLoader spawn, 400k-image cache time, disk, and RAM high-water mark. Atomic replace and the size/mtime check are unit-tested, not at 400k.
* That fp32 EDM coefficients match a bf16 run of script 070b. The numeric test is fp32.
* Full fine-tune, GUI, and writing `edm_mean`/`edm_std` into a saved checkpoint.
* A same-size, same-mtime image replacement is not detected (no pixel hash).

## Next steps

1. On the 16GB Windows machine: run `tools/check_playground_v25_unet.py` on the official safetensors and record the max abs diff. Then load it with `--playground_v25`, run ~10 steps, record VRAM and seconds/step. Then a small memorize test and a ComfyUI load of the kohya LoRA (base model detected via `edm_mean`/`edm_std`, LoRA via `LoraLoaderModelOnly`).
2. A/B `--pgv25_sigma_sampling=lognormal` and `--pgv25_loss_weighting=edm` against the defaults on a real GPU before changing the default.
3. Confirm the fp16 single-file opens with the same key prefixes.
4. GUI: a preset that adds `--playground_v25` and the 16GB flags above. kohya_ss "Additional parameters" can pass the flag without a new GUI.
5. Full fine-tune: call `prepare_edm_inputs` / `x0_target_from_model_output` / `normalize_latents` from the `sdxl_train.py` loop, allow the flag in `verify_sdxl_training_args`, and write `edm_mean` / `edm_std` into the saved checkpoint so ComfyUI detects it. Reuse `PlaygroundV25LatentsCachingStrategy`.

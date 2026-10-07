import glob
import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from library.clip_tokenizer import CLIPTokenizer  # transformers.CLIPTokenizer with the legacy (ftfy) text normalization
from library import accelerator_setup
import library.device_utils as device_utils
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncodingStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TOKENIZER_ID = "openai/clip-vit-large-patch14"

# The SD2.x tokenizer (previously loaded from stabilityai/stable-diffusion-2, which is no longer accessible on the Hub)
# is the v1 (OpenAI CLIP) tokenizer with "!" (id 0) as the pad token instead of <|endoftext|>: same vocabulary,
# same merges, same token ids. So it is built from the v1 tokenizer. v2 and v2.1 use the same tokenizer.
V2_TOKENIZER_PAD_TOKEN = "!"
# directory name under --tokenizer_cache_dir, kept so that existing caches of the v2 tokenizer are still used
V2_TOKENIZER_CACHE_NAME = "stabilityai_stable-diffusion-2"


class SdTokenizeStrategy(TokenizeStrategy):
    def __init__(self, v2: bool, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None) -> None:
        """
        max_length does not include <BOS> and <EOS> (None, 75, 150, 225)
        """
        logger.info(f"Using {'v2' if v2 else 'v1'} tokenizer")
        if v2:
            self.tokenizer = self._load_tokenizer(
                CLIPTokenizer,
                TOKENIZER_ID,
                tokenizer_cache_dir=tokenizer_cache_dir,
                cache_name=V2_TOKENIZER_CACHE_NAME,
                pad_token=V2_TOKENIZER_PAD_TOKEN,
            )
        else:
            self.tokenizer = self._load_tokenizer(CLIPTokenizer, TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length + 2

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        return [torch.stack([self._get_input_ids(self.tokenizer, t, self.max_length) for t in text], dim=0)]

    def tokenize_with_weights(self, text: str | List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        text = [text] if isinstance(text, str) else text
        tokens_list = []
        weights_list = []
        for t in text:
            tokens, weights = self._get_input_ids(self.tokenizer, t, self.max_length, weighted=True)
            tokens_list.append(tokens)
            weights_list.append(weights)
        return [torch.stack(tokens_list, dim=0)], [torch.stack(weights_list, dim=0)]


class SdTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, clip_skip: Optional[int] = None) -> None:
        self.clip_skip = clip_skip

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        text_encoder = models[0]
        tokens = tokens[0]
        sd_tokenize_strategy = tokenize_strategy  # type: SdTokenizeStrategy

        # tokens: b,n,77
        b_size = tokens.size()[0]
        max_token_length = tokens.size()[1] * tokens.size()[2]
        model_max_length = sd_tokenize_strategy.tokenizer.model_max_length
        tokens = tokens.reshape((-1, model_max_length))  # batch_size*3, 77

        tokens = tokens.to(text_encoder.device)

        if self.clip_skip is None:
            encoder_hidden_states = text_encoder(tokens)[0]
        else:
            enc_out = text_encoder(tokens, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out["hidden_states"][-self.clip_skip]
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

        # bs*3, 77, 768 or 1024
        encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

        if max_token_length != model_max_length:
            v1 = sd_tokenize_strategy.tokenizer.pad_token_id == sd_tokenize_strategy.tokenizer.eos_token_id
            if not v1:
                # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    chunk = encoder_hidden_states[:, i : i + model_max_length - 2]  # <BOS> の後から 最後の前まで
                    if i > 0:
                        for j in range(len(chunk)):
                            if tokens[j, 1] == sd_tokenize_strategy.tokenizer.eos_token:
                                # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                                chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                    states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
                encoder_hidden_states = torch.cat(states_list, dim=1)
            else:
                # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
                states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
                for i in range(1, max_token_length, model_max_length):
                    states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])  # <BOS> の後から <EOS> の前まで
                states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
                encoder_hidden_states = torch.cat(states_list, dim=1)

        return [encoder_hidden_states]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        encoder_hidden_states = self.encode_tokens(tokenize_strategy, models, tokens_list)[0]

        weights = weights_list[0].to(encoder_hidden_states.device)

        # apply weights
        if weights.shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            encoder_hidden_states = encoder_hidden_states * weights.squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for i in range(weights.shape[1]):
                encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] = encoder_hidden_states[:, i * 75 + 1 : i * 75 + 76] * weights[
                    :, i, 1:-1
                ].unsqueeze(-1)

        return [encoder_hidden_states]


class SdSdxlLatentsCachingStrategy(LatentsCachingStrategy):
    # sd and sdxl share the same strategy. we can make them separate, but the difference is only the suffix.
    # and we keep the old npz for the backward compatibility.

    SD_OLD_LATENTS_NPZ_SUFFIX = ".npz"
    SD_LATENTS_NPZ_SUFFIX = "_sd.npz"
    SDXL_LATENTS_NPZ_SUFFIX = "_sdxl.npz"

    def __init__(self, sd: bool, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.sd = sd
        self.suffix = (
            SdSdxlLatentsCachingStrategy.SD_LATENTS_NPZ_SUFFIX if sd else SdSdxlLatentsCachingStrategy.SDXL_LATENTS_NPZ_SUFFIX
        )

    @property
    def cache_suffix(self) -> str:
        return self.suffix

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        # support old .npz
        old_npz_file = os.path.splitext(absolute_path)[0] + SdSdxlLatentsCachingStrategy.SD_OLD_LATENTS_NPZ_SUFFIX
        if os.path.exists(old_npz_file):
            return old_npz_file
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.suffix

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).latent_dist.sample()
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not accelerator_setup.HIGH_VRAM:
            device_utils.clean_memory_on_device(vae.device)


class PlaygroundV25LatentsCachingStrategy(SdSdxlLatentsCachingStrategy):
    """Raw VAE latents for Playground v2.5, in ``*_pgv25.npz`` files.

    Normalization ``(z - mean) * 0.5 / std`` is applied at train time, not in
    the cache. The file records ``latent_format=playground_v25_raw``, the
    source image stem, and the source file size and mtime. SDXL ``*_sdxl.npz``
    / legacy ``.npz`` files are never selected. An incomplete cache is
    recomputed. A file whose format or source stem does not match raises one
    line that names the file. Writes are a temp file plus ``os.replace``.
    """

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(False, cache_to_disk, batch_size, skip_disk_cache_validity_check)
        from library.edm_playground import PGV25_NPZ_SUFFIX

        self.suffix = PGV25_NPZ_SUFFIX
        self._stamp_by_npz: Optional[dict] = None
        self._image_path_by_npz: dict = {}
        self._image_stat_by_npz: dict = {}
        # Off by default. fp16 only stores the raw VAE sample; train time still normalizes in fp32.
        self.cache_fp16 = False

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        # Do not fall back to legacy .npz or *_sdxl.npz. Those are a different normalization.
        path = os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.suffix
        self._image_path_by_npz[path] = absolute_path
        return path

    def _image_path_for(self, npz_path: str) -> Optional[str]:
        return self._image_path_by_npz.get(npz_path)

    def _classify(self, npz_path: str) -> str:
        """``missing``, ``ok``, or ``recompute``. May raise one line for a bad identity."""
        from library.edm_playground import classify_playground_cache

        if not os.path.exists(npz_path):
            return "missing"
        image_path = self._image_path_for(npz_path)
        try:
            loaded = np.load(npz_path)
        except Exception as ex:
            # Includes a truncated zip and numpy's "pickled data" ValueError.
            # A format/source ValueError is raised later, after the file opens.
            logger.warning(
                "playground_v25: cannot read %s (%s). It will be recomputed. Delete this file if you meant to keep it.",
                npz_path,
                ex,
            )
            return "recompute"
        with loaded as npz:
            return classify_playground_cache(npz_path, npz, image_path)

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        # Format, source stem, and size/mtime are never skipped, including with --skip_cache_check.
        status = self._classify(npz_path)
        if status != "ok":
            return False
        if self.skip_disk_cache_validity_check:
            return True
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        status = self._classify(npz_path)
        if status != "ok":
            raise ValueError(
                f"Latent cache {npz_path} is incomplete or does not match its image. "
                "Delete this file and rerun so training rebuilds it."
            )
        # Read and close the archive. Leaving np.load open locks the file on Windows.
        expected_latents_size = (bucket_reso[1] // 8, bucket_reso[0] // 8)  # bucket_reso is (W, H)
        key_reso_suffix = f"_{expected_latents_size[0]}x{expected_latents_size[1]}"
        with np.load(npz_path) as npz:
            if "latents" + key_reso_suffix not in npz:
                if "latents" not in npz:
                    raise ValueError(
                        f"latents not found in {npz_path} (with or without resolution suffix {key_reso_suffix}). "
                        "Delete this file and rerun so training rebuilds it."
                    )
                key_reso_suffix = ""
            latents = npz["latents" + key_reso_suffix]
            original_size = npz["original_size" + key_reso_suffix].tolist()
            crop_ltrb = npz["crop_ltrb" + key_reso_suffix].tolist()
            flipped_key = "latents_flipped" + key_reso_suffix
            alpha_key = "alpha_mask" + key_reso_suffix
            flipped_latents = npz[flipped_key] if flipped_key in npz else None
            alpha_mask = npz[alpha_key] if alpha_key in npz else None
            # Copy out of the zip before it closes.
            latents = np.array(latents)
            if flipped_latents is not None:
                flipped_latents = np.array(flipped_latents)
            if alpha_mask is not None:
                alpha_mask = np.array(alpha_mask)
        return latents, original_size, crop_ltrb, flipped_latents, alpha_mask

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        from library.edm_playground import STAMP_ENV, image_file_stat

        self._stamp_by_npz = {}
        self._image_stat_by_npz = {}
        for info in image_infos:
            if getattr(info, "latents_npz", None) and getattr(info, "absolute_path", None):
                self._image_path_by_npz[info.latents_npz] = info.absolute_path
                try:
                    self._image_stat_by_npz[info.latents_npz] = image_file_stat(info.absolute_path)
                except OSError as ex:
                    logger.warning("playground_v25: could not stat %s (%s)", info.absolute_path, ex)
        if os.environ.get(STAMP_ENV) == "1":
            from PIL import Image

            for info in image_infos:
                with Image.open(info.absolute_path) as image:
                    rgb = np.array(image.convert("RGB"), dtype=np.float32)
                # Solid-color test images keep this mean through bucket crops.
                self._stamp_by_npz[info.latents_npz] = float(rgb[:, :, 0].mean())
        try:
            super().cache_batch_latents(vae, image_infos, flip_aug, alpha_mask, random_crop)
        finally:
            self._stamp_by_npz = None

    def _store_latent_array(self, latents_tensor):
        array = latents_tensor.float().cpu().numpy()
        if self.cache_fp16:
            array = array.astype(np.float16)
        return array

    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        alpha_mask=None,
        key_reso_suffix="",
    ):
        from library.edm_playground import (
            LATENT_FORMAT_KEY,
            LATENT_FORMAT_VALUE,
            LATENT_SOURCE_KEY,
            LATENT_SOURCE_MTIME_NS_KEY,
            LATENT_SOURCE_SIZE_KEY,
            atomic_savez,
            npz_source_basename,
        )

        if self._stamp_by_npz and npz_path in self._stamp_by_npz:
            stamp = self._stamp_by_npz[npz_path]
            latents_tensor = latents_tensor.clone()
            latents_tensor[0, 0, 0] = stamp
            if flipped_latents_tensor is not None:
                flipped_latents_tensor = flipped_latents_tensor.clone()
                flipped_latents_tensor[0, 0, 0] = stamp

        kwargs = {}
        if os.path.exists(npz_path):
            # Keep other resolutions already stored in a valid cache. An incomplete file is replaced.
            try:
                if self._classify(npz_path) == "ok":
                    with np.load(npz_path) as npz:
                        kwargs = {key: npz[key] for key in npz.files}
            except ValueError:
                kwargs = {}
            except Exception:
                kwargs = {}

        kwargs["latents" + key_reso_suffix] = self._store_latent_array(latents_tensor)
        kwargs["original_size" + key_reso_suffix] = np.array(original_size)
        kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = self._store_latent_array(flipped_latents_tensor)
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = alpha_mask.float().cpu().numpy()
        kwargs[LATENT_FORMAT_KEY] = np.array(LATENT_FORMAT_VALUE)
        kwargs[LATENT_SOURCE_KEY] = np.array(npz_source_basename(npz_path))
        stat = self._image_stat_by_npz.get(npz_path)
        if stat is None:
            image_path = self._image_path_for(npz_path)
            if image_path and os.path.isfile(image_path):
                from library.edm_playground import image_file_stat

                try:
                    stat = image_file_stat(image_path)
                except OSError:
                    stat = None
        if stat is not None:
            kwargs[LATENT_SOURCE_SIZE_KEY] = np.int64(stat[0])
            kwargs[LATENT_SOURCE_MTIME_NS_KEY] = np.int64(stat[1])
        atomic_savez(npz_path, **kwargs)

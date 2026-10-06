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
    the cache. The file records ``latent_format=playground_v25_raw`` and the
    source image stem. SDXL ``*_sdxl.npz`` / legacy ``.npz`` files are never
    selected, and a mismatched Playground file raises instead of being reused.
    """

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(False, cache_to_disk, batch_size, skip_disk_cache_validity_check)
        from library.edm_playground import PGV25_NPZ_SUFFIX

        self.suffix = PGV25_NPZ_SUFFIX
        self._stamp_by_npz: Optional[dict] = None

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        # Do not fall back to legacy .npz or *_sdxl.npz. Those are a different normalization.
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.suffix

    def _assert_cache_file(self, npz_path: str) -> None:
        from library.edm_playground import assert_playground_latent_npz

        if not os.path.exists(npz_path):
            return
        with np.load(npz_path) as npz:
            assert_playground_latent_npz(npz_path, npz)

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        # Format/source checks are never skipped, including with --skip_cache_check.
        self._assert_cache_file(npz_path)
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        self._assert_cache_file(npz_path)
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        from library.edm_playground import STAMP_ENV

        self._stamp_by_npz = {}
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
        from library.edm_playground import LATENT_FORMAT_KEY, LATENT_FORMAT_VALUE, LATENT_SOURCE_KEY, npz_source_basename

        if self._stamp_by_npz and npz_path in self._stamp_by_npz:
            stamp = self._stamp_by_npz[npz_path]
            latents_tensor = latents_tensor.clone()
            latents_tensor[0, 0, 0] = stamp
            if flipped_latents_tensor is not None:
                flipped_latents_tensor = flipped_latents_tensor.clone()
                flipped_latents_tensor[0, 0, 0] = stamp
        super().save_latents_to_disk(
            npz_path,
            latents_tensor,
            original_size,
            crop_ltrb,
            flipped_latents_tensor,
            alpha_mask,
            key_reso_suffix,
        )
        with np.load(npz_path) as npz:
            kwargs = {key: npz[key] for key in npz.files}
        kwargs[LATENT_FORMAT_KEY] = np.array(LATENT_FORMAT_VALUE)
        kwargs[LATENT_SOURCE_KEY] = np.array(npz_source_basename(npz_path))
        np.savez(npz_path, **kwargs)

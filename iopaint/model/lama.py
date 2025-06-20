
import os
import numpy as np
import torch

from iopaint.helper import (
    norm_img,
    get_cache_path_by_url,
    load_jit_model,
    download_model,
)
from iopaint.schema import InpaintRequest
from .base import InpaintModel

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

ANIME_LAMA_MODEL_URL = os.environ.get(
    "ANIME_LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
)
ANIME_LAMA_MODEL_MD5 = os.environ.get(
    "ANIME_LAMA_MODEL_MD5", "29f284f36a0a510bcacf39ecf4c4d54f"
)


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8
    is_erase_model = True

    @staticmethod
    def download():
        download_model(LAMA_MODEL_URL, LAMA_MODEL_MD5)

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: RGB IMAGE
        """
        image = norm_img(image)
        mask = norm_img(mask)
        mask = ((mask > 0) * 1)

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        return cur_res


class AnimeLaMa(LaMa):
    name = "anime-lama"

    @staticmethod
    def download():
        download_model(ANIME_LAMA_MODEL_URL, ANIME_LAMA_MODEL_MD5)

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(
            ANIME_LAMA_MODEL_URL, device, ANIME_LAMA_MODEL_MD5
        ).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(ANIME_LAMA_MODEL_URL))

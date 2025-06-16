import abc
from typing import Optional

import cv2
import torch
import numpy as np
import logging

from iopaint.helper import (
    boxes_from_mask,
    resize_max_size,
    pad_img_to_modulo,
    switch_mps_device,
)
from iopaint.schema import InpaintRequest, HDStrategy
from .helper.g_diffuser_bot import expand_image
from iopaint.plugins import RealESRGANUpscaler
import iopaint.predict

class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    is_erase_model = False

    def __init__(self, device, **kwargs):
        """
        Args:
            device:
        """
        device = switch_mps_device(self.name, device)
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        return False

    @abc.abstractmethod
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W, 1] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    @staticmethod
    def download(): ...

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]

        # パディング
        pad_image = pad_img_to_modulo(image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size)
        pad_mask = pad_img_to_modulo(mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size)

        # モデルへ
        result = self.forward(pad_image, pad_mask, config)

        # パディング外し
        result = result[0:origin_height, 0:origin_width]

        # マスク領域だけ反映
        if config.sd_keep_unmasked_area:
            mask = mask[:, :, np.newaxis].astype(np.float32) / 255
            result = result * mask + image * (1 - mask)
        return result

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB
        masks: [H, W]
        return: RGB IMAGE
        """
        inpaint_result = None
        
        # logging.info(f"hd_strategy: {config.hd_strategy}")
        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logging.info("Run crop strategy")
                boxes = boxes_from_mask(mask)
                crop_result = []
                for box in boxes:
                    crop_image, crop_box = self._run_box(image, mask, box, config)
                    crop_result.append((crop_image, crop_box))

                #inpaint_result = image[:, :, ::-1]
                inpaint_result = image
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        elif config.hd_strategy == HDStrategy.RESIZE:
            if max(image.shape) > config.hd_strategy_resize_limit:
                origin_size = image.shape[:2]

                # 実行メモリの都合上縮小画像で実行する
                downsize_image = resize_max_size(image, size_limit=config.hd_strategy_resize_limit)
                downsize_mask = resize_max_size(mask, size_limit=config.hd_strategy_resize_limit)
                logging.info(f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}")

                # 実行
                inpaint_result = self._pad_forward(downsize_image, downsize_mask, config)

                # AIによる拡大
                if config.hd_strategy_resize_use_realesrgan == True:
                    inpaint_result = iopaint.predict.predict_plugin(inpaint_result, RealESRGANUpscaler.name)
                    
                # 元のサイズに戻す
                inpaint_result = cv2.resize(inpaint_result, (origin_size[1], origin_size[0]), interpolation=cv2.INTER_CUBIC)

                #mask2 = mask.astype(np.float32) / 255
                #inpaint_result = mask2[:, :, np.newaxis] * inpaint_result + (1.0 - mask2[:, :, np.newaxis]) * image
                #original_pixel_indices = mask <127
                #inpaint_result[original_pixel_indices] = image[original_pixel_indices]

        if inpaint_result is None:
            inpaint_result = self._pad_forward(image, mask, config)

        return inpaint_result

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        img_h, img_w = image.shape[:2]

        w = box_w + config.hd_strategy_crop_margin * 2
        h = box_h + config.hd_strategy_crop_margin * 2

        _l = cx - w // 2
        _r = cx + w // 2
        _t = cy - h // 2
        _b = cy + h // 2

        l = max(_l, 0)
        r = min(_r, img_w)
        t = max(_t, 0)
        b = min(_b, img_h)

        # try to get more context when crop around image edge
        if _l < 0:
            r += abs(_l)
        if _r > img_w:
            l -= _r - img_w
        if _t < 0:
            b += abs(_t)
        if _b > img_h:
            t -= _b - img_h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]

        # logging.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

        return crop_img, crop_mask, [l, t, r, b]

    def _calculate_cdf(self, histogram):
        cdf = histogram.cumsum()
        normalized_cdf = cdf / float(cdf.max())
        return normalized_cdf

    def _calculate_lookup(self, source_cdf, reference_cdf):
        lookup_table = np.zeros(256)
        lookup_val = 0
        for source_index, source_val in enumerate(source_cdf):
            for reference_index, reference_val in enumerate(reference_cdf):
                if reference_val >= source_val:
                    lookup_val = reference_index
                    break
            lookup_table[source_index] = lookup_val
        return lookup_table

    def _match_histograms(self, source, reference, mask):
        transformed_channels = []
        if len(mask.shape) == 3:
            mask = mask[:, :, -1]

        for channel in range(source.shape[-1]):
            source_channel = source[:, :, channel]
            reference_channel = reference[:, :, channel]

            # only calculate histograms for non-masked parts
            source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
            reference_histogram, _ = np.histogram(
                reference_channel[mask == 0], 256, [0, 256]
            )

            source_cdf = self._calculate_cdf(source_histogram)
            reference_cdf = self._calculate_cdf(reference_histogram)

            lookup = self._calculate_lookup(source_cdf, reference_cdf)

            transformed_channels.append(cv2.LUT(source_channel, lookup))

        result = cv2.merge(transformed_channels)
        result = cv2.convertScaleAbs(result)

        return result

    def _apply_cropper(self, image, mask, config: InpaintRequest):
        img_h, img_w = image.shape[:2]
        l, t, w, h = (
            config.croper_x,
            config.croper_y,
            config.croper_width,
            config.croper_height,
        )
        r = l + w
        b = t + h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        return crop_img, crop_mask, (l, t, r, b)

    def _run_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE
        """
        crop_img, crop_mask, [l, t, r, b] = self._crop_box(image, mask, box, config)

        return self._pad_forward(crop_img, crop_mask, config), [l, t, r, b]


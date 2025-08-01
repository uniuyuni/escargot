from typing import Dict

import logging

from .gfpgan_plugin import GFPGANPlugin
from .interactive_seg import InteractiveSeg
from .realesrgan import RealESRGANUpscaler
from .remove_bg import RemoveBG
from .restoreformer import RestoreFormerPlugin
from ..schema import InteractiveSegModel, Device, RealESRGANModel


def build_plugins(
    enable_interactive_seg: bool,
    interactive_seg_model: InteractiveSegModel,
    interactive_seg_device: Device,
    enable_remove_bg: bool,
    remove_bg_model: str,
    enable_realesrgan: bool,
    realesrgan_device: Device,
    realesrgan_model: RealESRGANModel,
    enable_gfpgan: bool,
    gfpgan_device: Device,
    enable_restoreformer: bool,
    restoreformer_device: Device,
    no_half: bool,
) -> Dict:
    plugins = {}
    if enable_interactive_seg:
        logging.info(f"Initialize {InteractiveSeg.name} plugin")
        plugins[InteractiveSeg.name] = InteractiveSeg(
            interactive_seg_model, interactive_seg_device
        )

    if enable_remove_bg:
        logging.info(f"Initialize {RemoveBG.name} plugin")
        plugins[RemoveBG.name] = RemoveBG(remove_bg_model)

    if enable_realesrgan:
        logging.info(
            f"Initialize {RealESRGANUpscaler.name} plugin: {realesrgan_model}, {realesrgan_device}"
        )
        plugins[RealESRGANUpscaler.name] = RealESRGANUpscaler(
            realesrgan_model,
            realesrgan_device,
            no_half=no_half,
        )

    if enable_gfpgan:
        logging.info(f"Initialize {GFPGANPlugin.name} plugin")
        if enable_realesrgan:
            logging.info("Use realesrgan as GFPGAN background upscaler")
        else:
            logging.info(
                f"GFPGAN no background upscaler, use --enable-realesrgan to enable it"
            )
        plugins[GFPGANPlugin.name] = GFPGANPlugin(
            gfpgan_device,
            upscaler=plugins.get(RealESRGANUpscaler.name, None),
        )

    if enable_restoreformer:
        logging.info(f"Initialize {RestoreFormerPlugin.name} plugin")
        plugins[RestoreFormerPlugin.name] = RestoreFormerPlugin(
            restoreformer_device,
            upscaler=plugins.get(RealESRGANUpscaler.name, None),
        )
    return plugins

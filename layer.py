
import cv2
import skimage
import numpy as np
import colorsys

import noise2void
import DRBNet
import colorcorrect.algorithm as cca

import core

#補正既定クラス
class AdjustmentLayer():

    def __init__(self, **kwargs):
        self.diff = None

    def set_param(self, param, widget):
        return param

    # 差分の作成
    def make_diff(self, img, param):
        self.diff = img

    # 差分の適用
    def apply_diff(self, img):
        img += self.diff

class NoiseLayer(AdjustmentLayer):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if NoiseLayer.__net is None:
            NoiseLayer.__net = noise2void.setup_predict()

    def set_param(self, param, widget):
        param['noise'] = 0 if widget.ids["toggle_noise"].state == "normal" else 1

    def make_diff(self, img, param):
        if param.get('noise', 0) > 0:
            self.diff = noise2void.predict(img, NoiseLayer.__net, 'mps')
        else:
            self.diff = None

class DefocusLayer(AdjustmentLayer):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if DefocusLayer.__net is None:
            DefocusLayer.__net = DRBNet.setup_predict()

    def set_param(self, param, widget):
        param['defocus'] = 0 if widget.ids["toggle_defocus"].state == "normal" else 1

    def make_diff(self, img, param):
        if param.get('defocus', 0) > 0:
            self.diff = DRBNet.predict(img, DefocusLayer.__net, 'mps')
        else:
            self.diff = None

class ColorCorrectLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['color_correct'] = 0 if widget.ids["toggle_color_correct"].state == "normal" else 1

    def make_diff(self, img, param):
        if param.get('color_correct', 0) > 0:
            self.diff = img-cca.automatic_color_equalization(img)
        else:
            self.diff = None

class BilateralFilterLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['bilateral_filter'] = widget.ids["slider_bilateral_filter"].value
        param['bilateral_filter_color'] = widget.ids["slider_bilateral_filter_color"].value
        param['bilateral_filter_space'] = widget.ids["slider_bilateral_filter_space"].value
        param['highpass_filter'] = widget.ids["slider_highpass_filter"].value

    def make_diff(self, img, param):
        if param.get('bilateral_filter', 0) > 0:
            bilateral = cv2.bilateralFilter(img, int(param.get('bilateral_filter', 1)), int(param.get('bilateral_filter_color', 75)), int(param.get('bilateral_filter_space', 75)))
            highpass = core.highpass_filter(img, param.get('highpass_filter', 0))
            self.diff = core.blend_overlay(bilateral, highpass)
            #self.diff = highpass
        else:
            self.diff = None

class ExposureLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['exposure'] = widget.ids["slider_exposure"].value

    def make_diff(self, img, param):
        rgb = core.adjust_exposure(img, param.get('exposure', 0))
        self.diff = rgb-img    # RGB保存

class ContrastLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['contrast'] = widget.ids["slider_contrast"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.adjust_contrast(hls[:,:,1], param.get('contrast', 0))
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class ToneLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['black'] = widget.ids["slider_black"].value*100.0
        param['white'] = widget.ids["slider_white"].value*100.0
        param['shadow'] = widget.ids["slider_shadow"].value
        param['hilight'] = widget.ids["slider_hilight"].value

    def make_diff(self, img, param):
        # 制御点とエクスポージャー補正値を設定
        points = np.array([0.0, 13107.0, 52429.0, 65535.0])
        values = np.array([0.0, 13107.0, 52429.0, 65535.0])
        values[1] += param.get('black', 0)
        values[2] += param.get('white', 0)
        points /= 65535.0
        values /= 65535.0

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_curve(hls[:,:,1], points, values, False)
        hls_l = core.adjust_shadow(hls_l, param.get('shadow', 0))
        hls_l = core.adjust_hilight(hls_l, param.get('hilight', 0))
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class SaturationLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['saturation'] = widget.ids["slider_saturation"].value
        param['vibrance'] = widget.ids["slider_vibrance"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.adjust_saturation(hls[:,:,2], param.get('saturation', 0), param.get('vibrance', 0))
        self.diff = hls_s/hls[:,:,2]    # Sのみ保存

class DehazeLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['dehaze'] = widget.ids["slider_dehaze"].value

    def make_diff(self, img, param):
        img2 = core.apply_dehaze(img, param.get('dehaze', 0))
        self.diff = img2-img     # RGBのみ保存

class LevelLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['black_level'] = widget.ids["slider_black_level"].value
        param['white_level'] = widget.ids["slider_white_level"].value
        param['mid_level'] = widget.ids["slider_mid_level"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_level_adjustment(hls[:,:,1], param.get('black_level', 0), param.get('white_level', 255), param.get('mid_level', 127))
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class TonecurveLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_spline()

    def make_diff(self, img, param):
        img2 = core.apply_spline(img, param.get('tonecurve', None))
        if img2 is not None:
            self.diff = img2-img            # RGB保存
        else:
            self.diff = None

class TonecurveRedLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_spline()

    def make_diff(self, img, param):
        img2 = core.apply_spline(img[:,:,0], param.get('tonecurve_red', None))
        if img2 is not None:
            self.diff = img2-img[:,:,0]            # R保存
        else:
            self.diff = None

class TonecurveGreenLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_spline()

    def make_diff(self, img, param):
        img2 = core.apply_spline(img[:,:,1], param.get('tonecurve_green', None))
        if img2 is not None:
            self.diff = img2-img[:,:,1]            # G保存
        else:
            self.diff = None

class TonecurveBlueLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_spline()

    def make_diff(self, img, param):
        img2 = core.apply_spline(img[:,:,2], param.get('tonecurve_blue', None))
        if img2 is not None:
            self.diff = img2-img[:,:,2]            # B保存
        else:
            self.diff = None

class LumvsLumLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['LumvsLum'] = widget.ids["LumvsLum"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_spline(hls[:,:,1], param.get("LumvsLum", None))
        self.diff = 2.0**((hls_l-0.5)*2.0)   # Lのみ保存

class HuevsHueLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['HuevsHue'] = widget.ids["HuevsHue"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_h = core.apply_spline(hls[:,:,0]/360.0, param.get("HuevsHue", None))
        self.diff = (hls_h-0.5)*360.0    # Hのみ保存

class HuevsSatLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.apply_spline(hls[:,:,0], param.get("HuevsSat", None))
        self.diff = 2.0**((hls_s-0.5)*2.0)     # Sのみ保存

class HuevsLumLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_spline(hls[:,:,0], param.get("HuevsLum", None))
        self.diff = 2.0**((hls_l-0.5)*2.0)     # Lのみ保存

class LumvsSatLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['LumvsSat'] = widget.ids["LumvsSat"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.apply_spline(hls[:,:,1], param.get("LumvsSat", None))
        self.diff = 2.0**((hls_s-0.5)*2.0)     # Sのみ保存

class SatvsSatLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['SatvsSat'] = widget.ids["SatvsSat"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.apply_spline(hls[:,:,2], param.get("SatvsSat", None))
        self.diff = 2.0**((hls_s-0.5)*2.0)     # Sのみ保存

class SatvsLumLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['SatvsLum'] = widget.ids["SatvsLum"].get_spline()

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_spline(hls[:,:,2], param.get("SatvsLum", None))
        self.diff = 2.0**((hls_l-0.5)*2.0)     # Lのみ保存

class GradingLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['grading'] = widget.ids["grading"].get_spline()
        param['grading_hue'] = widget.ids["slider_grading_hue"].value
        param['grading_lum'] = widget.ids["slider_grading_lum"].value
        param['grading_sat'] = widget.ids["slider_grading_sat"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        blend = core.apply_spline(hls[:,:,1], param.get("grading", None))
        rgb = np.array(colorsys.hls_to_rgb(param.get("grading_hue", 0)/360.0, param.get("grading_lum", 50)/100.0, param.get("grading_sat", 50)/100.0), dtype=np.float32)
        blend_inv = 1-blend
        self.diff = (img*blend_inv[:, :, np.newaxis] + rgb*blend[:, :, np.newaxis]) - img

class HLSRedLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['red_hue'] = widget.ids["slider_red_hue"].value
        param['red_sat'] = widget.ids["slider_red_sat"].value
        param['red_val'] = widget.ids["slider_red_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_red(hls, (param.get("red_hue", 0), param.get("red_sat", 0), param.get("red_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSOrangeLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['orange_hue'] = widget.ids["slider_orange_hue"].value
        param['orange_sat'] = widget.ids["slider_orange_sat"].value
        param['orange_val'] = widget.ids["slider_orange_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2= core.adjust_hls_orange(hls, (param.get("orange_hue", 0), param.get("orange_sat", 0), param.get("orange_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSYellowLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['yellow_hue'] = widget.ids["slider_yellow_hue"].value
        param['yellow_sat'] = widget.ids["slider_yellow_sat"].value
        param['yellow_val'] = widget.ids["slider_yellow_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_yellow(hls, (param.get("yellow_hue", 0), param.get("yellow_sat", 0), param.get("yellow_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSGreenLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['green_hue'] = widget.ids["slider_green_hue"].value
        param['green_sat'] = widget.ids["slider_green_sat"].value
        param['green_val'] = widget.ids["slider_green_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_green(hls, (param.get("green_hue", 0), param.get("green_sat", 0), param.get("green_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSCyanLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['cyan_hue'] = widget.ids["slider_cyan_hue"].value
        param['cyan_sat'] = widget.ids["slider_cyan_sat"].value
        param['cyan_val'] = widget.ids["slider_cyan_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_cyan(hls, (param.get("cyan_hue", 0), param.get("cyan_sat", 0), param.get("cyan_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSBlueLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['blue_hue'] = widget.ids["slider_blue_hue"].value
        param['blue_sat'] = widget.ids["slider_blue_sat"].value
        param['blue_val'] = widget.ids["slider_blue_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_blue(hls, (param.get("blue_hue", 0), param.get("blue_sat", 0), param.get("blue_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSPurpleLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['purple_hue'] = widget.ids["slider_purple_hue"].value
        param['purple_sat'] = widget.ids["slider_purple_sat"].value
        param['purple_val'] = widget.ids["slider_purple_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_purple(hls, (param.get("purple_hue", 0), param.get("purple_sat", 0), param.get("purple_val", 0)))
        self.diff = hls2-hls            # HLS保存

class HLSMagentaLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['magenta_hue'] = widget.ids["slider_magenta_hue"].value
        param['magenta_sat'] = widget.ids["slider_magenta_sat"].value
        param['magenta_val'] = widget.ids["slider_magenta_val"].value

    def make_diff(self, img, param):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_magenta(hls, (param.get("magenta_hue", 0), param.get("magenta_sat", 0), param.get("magenta_val", 0)))
        self.diff = hls2-hls            # HLS保存

def create_layer(layer):
    layer['noise'] = NoiseLayer()
    layer['defocus'] = DefocusLayer()
    layer['color_correct'] = ColorCorrectLayer()
    layer['bilateral_filter'] = BilateralFilterLayer()
    layer['exposure'] = ExposureLayer()
    layer['contrast'] = ContrastLayer()
    layer['tone'] = ToneLayer()
    layer['saturation'] = SaturationLayer()
    layer['dehaze'] = DehazeLayer()
    layer['level'] = LevelLayer()
    layer['tonecurve'] = TonecurveLayer()
    layer['tonecurve_red'] = TonecurveRedLayer()
    layer['tonecurve_green'] = TonecurveGreenLayer()
    layer['tonecurve_blue'] = TonecurveBlueLayer()
    layer['LumvsLum'] = LumvsLumLayer()
    layer['HuevsHue'] = HuevsHueLayer()
    layer['HuevsSat'] = HuevsSatLayer()
    layer['HuevsLum'] = HuevsLumLayer()
    layer['LumvsSat'] = LumvsSatLayer()
    layer['SatvsSat'] = SatvsSatLayer()
    layer['SatvsLum'] = SatvsLumLayer()
    layer['grading'] = GradingLayer()
    layer['hls_red'] = HLSRedLayer()
    layer['hls_orange'] = HLSOrangeLayer()
    layer['hls_yellow'] = HLSYellowLayer()
    layer['hls_green'] = HLSGreenLayer()
    layer['hls_cyan'] = HLSCyanLayer()
    layer['hls_blue'] = HLSBlueLayer()
    layer['hls_purple'] = HLSPurpleLayer()
    layer['hls_magenta'] = HLSMagentaLayer()


def pipeline(img, layer):

    diff = layer['noise'].diff
    if diff is not None:
        rgb = diff
    else:
        rgb = img.copy()
    diff = layer['defocus'].diff
    if diff is not None:
        rgb = diff
    diff = layer['bilateral_filter'].diff
    if diff is not None:
        rgb = diff

    # RGB
    diff = layer['color_correct'].diff
    if diff is not None: rgb += diff
    diff = layer['dehaze'].diff
    if diff is not None: rgb += diff
    diff = layer['tonecurve'].diff
    if diff is not None: rgb += diff
    diff = layer['tonecurve_red'].diff
    if diff is not None: rgb[:,:,0] += diff
    diff = layer['tonecurve_green'].diff
    if diff is not None: rgb[:,:,1] += diff
    diff = layer['tonecurve_blue'].diff
    if diff is not None: rgb[:,:,2] += diff
    diff = layer['grading'].diff
    if diff is not None: rgb += diff
    diff = layer['exposure'].diff
    if diff is not None: rgb += diff

    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)

    # Hのみ
    diff = layer['HuevsHue'].diff
    if diff is not None: hls[:,:,0] += diff

    # Lのみ
    diff = layer['contrast'].diff
    if diff is not None: hls[:,:,1] += diff
    diff = layer['tone'].diff
    if diff is not None: hls[:,:,1] += diff
    diff = layer['level'].diff
    if diff is not None: hls[:,:,1] += diff
    diff = layer['LumvsLum'].diff
    if diff is not None: hls[:,:,1] *= diff
    diff = layer['HuevsLum'].diff
    if diff is not None: hls[:,:,1] *= diff
    diff = layer['SatvsLum'].diff
    if diff is not None: hls[:,:,1] *= diff

    # Sのみ
    diff = layer['HuevsSat'].diff
    if diff is not None: hls[:,:,2] *= diff
    diff = layer['LumvsSat'].diff
    if diff is not None: hls[:,:,2] *= diff
    diff = layer['SatvsSat'].diff
    if diff is not None: hls[:,:,2] *= diff


    # HLS
    diff = layer['hls_red'].diff
    if diff is not None: hls += diff
    diff = layer['hls_orange'].diff
    if diff is not None: hls += diff
    diff = layer['hls_yellow'].diff
    if diff is not None: hls += diff
    diff = layer['hls_green'].diff
    if diff is not None: hls += diff
    diff = layer['hls_cyan'].diff
    if diff is not None: hls += diff
    diff = layer['hls_blue'].diff
    if diff is not None: hls += diff
    diff = layer['hls_purple'].diff
    if diff is not None: hls += diff
    diff = layer['hls_magenta'].diff
    if diff is not None: hls += diff

    # Sのみ
    diff = layer['saturation'].diff
    if diff is not None: hls[:,:,2] *= diff

    # 合成
    hls[:,:,1] = np.clip(hls[:,:,1], 0, 1.0)
    hls[:,:,2] = np.clip(hls[:,:,2], 0, 1.0)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)

    return rgb

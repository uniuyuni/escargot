
import cv2
import numpy as np
import colorsys

import noise2void
import DRBNet
import colorcorrect.algorithm as cca
import perlin
import lama

import core
import cubelut
import mask_editor

#補正既定クラス
class AdjustmentLayer():

    def __init__(self, **kwargs):
        self.diff = None
        self.hash = None

    def reset(self):
        self.diff = None
        self.hash = None

    def set_param(self, param, widget):
        return param

    # 差分の作成
    def make_diff(self, img, param):
        self.diff = img


class NoiseLayer(AdjustmentLayer):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if NoiseLayer.__net is None:
            NoiseLayer.__net = noise2void.setup_predict()

    def set_param(self, param, widget):
        param['noise_reduction'] = 0 if widget.ids["toggle_noise_reduction"].state == "normal" else 1

    def make_diff(self, img, param):
        nr = param.get('noise_reduction', 0)
        if nr <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((nr))
            if self.hash != param_hash:
                self.diff = noise2void.predict(img, NoiseLayer.__net, 'mps')
                self.hash = param_hash
        
        return self.diff

class InpaintLayer(AdjustmentLayer):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if InpaintLayer.__net is None:
            InpaintLayer.__net = lama.setup_predict()
        
        self.mask_editor = None

    def set_param(self, param, widget):
        param['inpaint'] = 0 if widget.ids["toggle_inpaint"].state == "normal" else 1
        param['inpaint_predict'] = 0 if widget.ids["button_inpaint"].state == "normal" else 1

        if param['inpaint'] > 0:
            if self.mask_editor is None:
                self.mask_editor = mask_editor.MaskEditor(param['src_size'][1], param['src_size'][0])
                self.mask_editor.zoom = widget.scale
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)
            
        if param['inpaint'] <= 0:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None

    def make_diff(self, img, param):
        ip = param.get('inpaint', 0)
        ipp = param.get('inpaint_predict', 0)
        if (ip > 0 and ipp > 0) is True:
            self.diff = lama.predict(img, self.mask_editor.get_mask(), InpaintLayer.__net)
            self.mask_editor.clear_mask()
            self.mask_editor.update_canvas()
        
        return self.diff

class DefocusLayer(AdjustmentLayer):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if DefocusLayer.__net is None:
            DefocusLayer.__net = DRBNet.setup_predict()

    def set_param(self, param, widget):
        param['defocus'] = 0 if widget.ids["toggle_defocus"].state == "normal" else 1

    def make_diff(self, img, param):
        df = param.get('defocus', 0)
        if df <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((df))
            if self.hash != param_hash:
                self.diff = DRBNet.predict(img, DefocusLayer.__net, 'mps')
                self.hash = param_hash

        return self.diff

class ColorCorrectLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['color_correct'] = 0 if widget.ids["toggle_color_correct"].state == "normal" else 1

    def make_diff(self, rgb, param):
        cc = param.get('color_correct', 0)
        if cc <= 0:
            self.diff = None
            self.hash = None
        
        else:
            param_hash = hash((cc))
            if self.hash != param_hash:
                self.diff = cca.automatic_color_equalization(rgb)-rgb
                self.hash = param_hash

        return self.diff

class LUTLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['lut'] = 0 if widget.ids["toggle_lut"].state == "normal" else 1
        if param['lut'] > 0:
            self.lut = cubelut.read_lut("Retro 2023.CUBE")

    def make_diff(self, rgb, param):
        lt = param.get('lut', 0)
        if lt <= 0:
            self.diff = None
            self.hash = None
        
        else:
            param_hash = hash((lt))
            if self.hash != param_hash:
                self.diff = cubelut.process_image(rgb, self.lut)
                self.hash = param_hash

        return self.diff

class LowpassFilterLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['lowpass_filter'] = widget.ids["slider_lowpass_filter"].value
        param['highpass_filter'] = widget.ids["slider_highpass_filter"].value

    def make_diff(self, img, param):
        lpfr = int(param.get('lowpass_filter', 0))
        hpfr = int(param.get('highpass_filter', 0))
        if lpfr == 0 and hpfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((lpfr, hpfr))
            if self.hash != param_hash:
                if lpfr > 0 and hpfr == 0:
                    self.diff = core.lowpass_filter(img, lpfr-1)
                elif lpfr == 0 and hpfr > 0:
                    self.diff = core.highpass_filter(img, hpfr-1)
                    #self.diff = highpass
                else:
                    lowpass = core.lowpass_filter(img, lpfr-1)
                    highpass = core.highpass_filter(img, hpfr-1)
                    self.diff = core.blend_overlay(lowpass, highpass)
                self.hash = param_hash

        return self.diff

class PerlinNoiseLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['perlin_noise'] = widget.ids["slider_perlin_noise"].value
        param['perlin_noise_opacity'] = widget.ids["slider_perlin_noise_opacity"].value

    def make_diff(self, img, param):
        pn = int(param.get('perlin_noise', 0))
        pno = param.get('perlin_noise_opacity', 100)
        if pn == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((pn, pno))
            if self.hash != param_hash:
                img2 = perlin.make_perlin_noise(img.shape[1], img.shape[0], pn)
                img2 = ((img2*(pno/100.0))+1.0)/2.0
                img2 = np.stack( [img2, img2, img2], axis=2)
                img2 = core.blend_overlay(img, img2)
                self.diff = img2
                self.hash = param_hash

        return self.diff

class ColorTemperatureLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['color_temperature'] = widget.ids["slider_color_temperature"].value
        param['color_temperature_strength'] = widget.ids["slider_color_temperature_strength"].value

    def make_diff(self, rgb, param):
        ct = param.get('color_temperature', 0)
        cts = param.get('color_temperature_strength', 0)/100.0
        if ct == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((ct, cts))
            if self.hash != param_hash:
                r, g, b = core.convert_Kelvin2RGB(ct)
                self.diff = np.array([r, g, b])    # RGB保存
                self.hash = param_hash

        return self.diff

class ExposureLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['exposure'] = widget.ids["slider_exposure"].value

    def make_diff(self, rgb, param):
        ev = param.get('exposure', 0)
        param_hash = hash((ev))
        if ev == 0:
            self.diff = None
            self.hash = None
        
        elif self.hash != param_hash:
            rgb2 = core.adjust_exposure(rgb, ev)
            self.diff = rgb2-rgb    # RGB保存
            self.hash = param_hash

        return self.diff

class ContrastLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['contrast'] = widget.ids["slider_contrast"].value

    def make_diff(self, rgb, param):
        con = param.get('contrast', 0)
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb2 = core.adjust_contrast(rgb, con)
            self.diff = rgb2-rgb
            self.hash = param_hash

        return self.diff

class ToneLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['black'] = widget.ids["slider_black"].value*100.0
        param['white'] = widget.ids["slider_white"].value*100.0
        param['shadow'] = widget.ids["slider_shadow"].value
        param['hilight'] = widget.ids["slider_hilight"].value

    def make_diff(self, hls_l, param):
        black = param.get('black', 0)
        white = param.get('white', 0)
        shadow = param.get('shadow', 0)
        hilight =  param.get('hilight', 0)
        param_hash = hash((black, white, shadow, hilight))
        if black == 0 and white == 0 and shadow == 0 and hilight == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            # 制御点とエクスポージャー補正値を設定
            points = np.array([0.0, 13107.0, 52429.0, 65535.0])
            values = np.array([0.0, 13107.0, 52429.0, 65535.0])
            values[1] += black
            values[2] += white
            points /= 65535.0
            values /= 65535.0

            hls2_l = core.apply_curve(hls_l, points, values, False)
            hls2_l = core.adjust_shadow(hls2_l, shadow)
            hls2_l = core.adjust_hilight(hls2_l, hilight)
            self.diff = hls2_l-hls_l    # Lのみ保存
            self.hash = param_hash

        return self.diff

class SaturationLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['saturation'] = widget.ids["slider_saturation"].value
        param['vibrance'] = widget.ids["slider_vibrance"].value

    def make_diff(self, hls, param):
        sat = param.get('saturation', 0)
        vib = param.get('vibrance', 0)
        param_hash = hash((sat, vib))
        if sat == 0 and vib == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls_s = hls[:,:,2]
            hls2_s = core.adjust_saturation(hls_s, sat, vib)
            self.diff = hls2_s/hls_s    # Sのみ保存
            self.hash = param_hash
        
        return self.diff

class DehazeLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['dehaze'] = widget.ids["slider_dehaze"].value

    def make_diff(self, rgb, param):
        de = param.get('dehaze', 0)
        if de == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((de))
            if self.hash != param_hash:
                rgb2 = core.apply_dehaze(rgb, de)
                self.diff = rgb2-rgb     # RGBのみ保存
                self.hash = param_hash

        return self.diff

class LevelLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['black_level'] = widget.ids["slider_black_level"].value
        param['white_level'] = widget.ids["slider_white_level"].value
        param['mid_level'] = widget.ids["slider_mid_level"].value

    def make_diff(self, hls_l, param):
        bl = param.get('black_level', 0)
        wl = param.get('white_level', 255)
        ml = param.get('mid_level', 127)
        param_hash = hash((bl, wl, ml))
        if bl == 0 and wl == 255 and ml == 127:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2_l = core.apply_level_adjustment(hls_l, bl, wl, ml)
            self.diff = hls2_l-hls_l    # Lのみ保存
            self.hash = param_hash

        return self.diff

class TonecurveLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_spline()

    def make_diff(self, img, param):
        tc = param.get('tonecurve', None)
        if tc is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(tc))
            if self.hash != param_hash:
                img2 = core.apply_spline(img, tc)
                self.diff = img2-img        # RGB保存
                self.hash = param_hash

        return self.diff

class TonecurveRedLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_spline()

    def make_diff(self, img, param):
        tred = param.get('tonecurve_red', None)
        if tred is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(tred))
            if self.hash != param_hash:
                img2 = core.apply_spline(img[:,:,0], tred)
                self.diff = img2-img[:,:,0]            # R保存
                self.hash = param_hash

        return self.diff

class TonecurveGreenLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_spline()

    def make_diff(self, img, param):   
        tgreen = param.get('tonecurve_green', None)
        if tgreen is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(tgreen))
            if self.hash != param_hash:
                img2 = core.apply_spline(img[:,:,1], tgreen)
                self.diff = img2-img[:,:,1]            # G保存
                self.hash = param_hash

        return self.diff

class TonecurveBlueLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_spline()

    def make_diff(self, img, param):
        tblue = param.get('tonecurve_blue', None)
        if tblue is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(tblue))
            if self.hash != param_hash:
                img2 = core.apply_spline(img[:,:,2], tblue)
                self.diff = img2-img[:,:,2]        # B保存
                self.hash = param_hash

        return self.diff

class LumvsLumLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['LumvsLum'] = widget.ids["LumvsLum"].get_spline()

    def make_diff(self, hls_l, param):
        ll = param.get("LumvsLum", None)
        if ll is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(ll))
            if self.hash != param_hash:
                hls2_l = core.apply_spline(hls_l, ll)
                self.diff = 2.0**((hls2_l-0.5)*2.0)   # Lのみ保存
                self.hash = param_hash

        return self.diff

class HuevsHueLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['HuevsHue'] = widget.ids["HuevsHue"].get_spline()

    def make_diff(self, hls_h, param):
        hh = param.get("HuevsHue", None)
        if hh is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hh))
            if self.hash != param_hash:
                hls2_h = core.apply_spline(hls_h/360.0, hh)
                self.diff = (hls2_h-0.5)*360.0    # Hのみ保存
                self.hash = param_hash

        return self.diff

class HuevsSatLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_spline()

    def make_diff(self, hls_h, param):
        hs = param.get("HuevsSat", None)
        if hs is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hs))
            if self.hash != param_hash:
                hls2_s = core.apply_spline(hls_h, hs)
                self.diff = 2.0**((hls2_s-0.5)*2.0)     # Sのみ保存
                self.hash = param_hash

        return self.diff

class HuevsLumLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_spline()

    def make_diff(self, hls_h, param):
        hl = param.get("HuevsLum", None)
        if hl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hl))
            if self.hash != param_hash:
                hls2_l = core.apply_spline(hls_h, hl)
                self.diff = 2.0**((hls2_l-0.5)*2.0)     # Lのみ保存
                self.hash = param_hash

        return self.diff

class LumvsSatLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['LumvsSat'] = widget.ids["LumvsSat"].get_spline()

    def make_diff(self, hls_l, param):
        ls = param.get("LumvsSat", None)
        if ls is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(ls))
            if self.hash != param_hash:
                hls2_s = core.apply_spline(hls_l, ls)
                self.diff = 2.0**((hls2_s-0.5)*2.0)     # Sのみ保存
                self.hash = param_hash

        return self.diff

class SatvsSatLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['SatvsSat'] = widget.ids["SatvsSat"].get_spline()

    def make_diff(self, hls_s, param):
        ss = param.get("SatvsSat", None)
        if ss is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(ss))
            if self.hash != param_hash:
                hls2_s = core.apply_spline(hls_s, ss)
                self.diff = 2.0**((hls2_s-0.5)*2.0)     # Sのみ保存
                self.hash = param_hash

        return self.diff

class SatvsLumLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['SatvsLum'] = widget.ids["SatvsLum"].get_spline()

    def make_diff(self, hls_s, param):
        sl = param.get("SatvsLum", None)
        if sl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(sl))
            if self.hash != param_hash:
                hls2_l = core.apply_spline(hls_s, sl)
                self.diff = 2.0**((hls2_l-0.5)*2.0)     # Lのみ保存
                self.hash = param_hash

        return self.diff

class GradingLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['grading'] = widget.ids["grading"].get_spline()
        param['grading_hue'] = widget.ids["slider_grading_hue"].value
        param['grading_lum'] = widget.ids["slider_grading_lum"].value
        param['grading_sat'] = widget.ids["slider_grading_sat"].value

    def make_diff(self, hls, param):
        gr = param.get("grading", None)
        gh = param.get("grading_hue", 0)
        gl = param.get("grading_lum", 0)
        gs = param.get("grading_sat", 0)
        if gr is None and gh == 0 and gl == 0 and gs == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((np.sum(gr), gh, gl, gs))
            if self.hash != param_hash:
                blend = core.apply_spline(hls[:,:,1], gr)
                rgb = np.array(colorsys.hls_to_rgb(gh/360.0, gl/100.0, gs/100.0), dtype=np.float32)
                blend_inv = 1-blend
                self.diff = (hls*blend_inv[:, :, np.newaxis] + rgb*blend[:, :, np.newaxis]) - hls
                self.hash = param_hash

        return self.diff

class HLSRedLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_red_hue'] = widget.ids["slider_hls_red_hue"].value
        param['hls_red_lum'] = widget.ids["slider_hls_red_lum"].value
        param['hls_red_sat'] = widget.ids["slider_hls_red_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_red_hue", 0)
        lum = param.get("hls_red_lum", 0)
        sat = param.get("hls_red_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_red(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

class HLSOrangeLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_orange_hue'] = widget.ids["slider_hls_orange_hue"].value
        param['hls_orange_lum'] = widget.ids["slider_hls_orange_lum"].value
        param['hls_orange_sat'] = widget.ids["slider_hls_orange_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_orange_hue", 0)
        lum = param.get("hls_orange_lum", 0)
        sat = param.get("hls_orange_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_orange(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

class HLSYellowLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_yellow_hue'] = widget.ids["slider_hls_yellow_hue"].value
        param['hls_yellow_lum'] = widget.ids["slider_hls_yellow_lum"].value
        param['hls_yellow_sat'] = widget.ids["slider_hls_yellow_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_yellow_hue", 0)
        lum = param.get("hls_yellow_lum", 0)
        sat = param.get("hls_yellow_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_yellow(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = hlparam_hashs

        return self.diff

class HLSGreenLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_green_hue'] = widget.ids["slider_hls_green_hue"].value
        param['hls_green_lum'] = widget.ids["slider_hls_green_lum"].value
        param['hls_green_sat'] = widget.ids["slider_hls_green_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_green_hue", 0)
        lum = param.get("hls_green_lum", 0)
        sat = param.get("hls_green_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_green(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

class HLSCyanLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_cyan_hue'] = widget.ids["slider_hls_cyan_hue"].value
        param['hls_cyan_lum'] = widget.ids["slider_hls_cyan_lum"].value
        param['hls_cyan_sat'] = widget.ids["slider_hls_cyan_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_cyan_hue", 0)
        lum = param.get("hls_cyan_lum", 0)
        sat = param.get("hls_cyan_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_cyan(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

class HLSBlueLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_blue_hue'] = widget.ids["slider_hls_blue_hue"].value
        param['hls_blue_lum'] = widget.ids["slider_hls_blue_lum"].value
        param['hls_blue_sat'] = widget.ids["slider_hls_blue_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_blue_hue", 0)
        lum = param.get("hls_blue_lum", 0)
        sat = param.get("hls_blue_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_blue(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

class HLSPurpleLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_purple_hue'] = widget.ids["slider_hls_purple_hue"].value
        param['hls_purple_lum'] = widget.ids["slider_hls_purple_lum"].value
        param['hls_purple_sat'] = widget.ids["slider_hls_purple_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_purple_hue", 0)
        lum = param.get("hls_purple_lum", 0)
        sat = param.get("hls_purple_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_purple(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

class HLSMagentaLayer(AdjustmentLayer):

    def set_param(self, param, widget):
        param['hls_magenta_hue'] = widget.ids["slider_hls_magenta_hue"].value
        param['hls_magenta_lum'] = widget.ids["slider_hls_magenta_lum"].value
        param['hls_magenta_sat'] = widget.ids["slider_hls_magenta_sat"].value

    def make_diff(self, hls, param):
        hue = param.get("hls_magenta_hue", 0)
        lum = param.get("hls_magenta_lum", 0)
        sat = param.get("hls_magenta_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_hls_magenta(hls, (hue, lum, sat))
            self.diff = hls2-hls    # HLS差分
            self.hash = param_hash

        return self.diff

def create_layer():
    layer = [{}, {}, {}]

    lv0 = layer[0]
    lv0['inpaint'] = InpaintLayer()    

    lv1 = layer[1]
    lv1['noise_reduction'] = NoiseLayer()
    lv1['defocus'] = DefocusLayer()
    lv1['lowpass_filter'] = LowpassFilterLayer()
    lv1['perlin_noise'] = PerlinNoiseLayer()
    lv1['lut'] = LUTLayer()
    
    lv2 = layer[2]
    lv2['color_temperature'] = ColorTemperatureLayer()
    lv2['color_correct'] = ColorCorrectLayer()
    lv2['exposure'] = ExposureLayer()
    lv2['contrast'] = ContrastLayer()
    lv2['tone'] = ToneLayer()
    lv2['saturation'] = SaturationLayer()
    lv2['dehaze'] = DehazeLayer()
    lv2['level'] = LevelLayer()
    lv2['tonecurve'] = TonecurveLayer()
    lv2['tonecurve_red'] = TonecurveRedLayer()
    lv2['tonecurve_green'] = TonecurveGreenLayer()
    lv2['tonecurve_blue'] = TonecurveBlueLayer()
    lv2['LumvsLum'] = LumvsLumLayer()
    lv2['HuevsHue'] = HuevsHueLayer()
    lv2['HuevsSat'] = HuevsSatLayer()
    lv2['HuevsLum'] = HuevsLumLayer()
    lv2['LumvsSat'] = LumvsSatLayer()
    lv2['SatvsSat'] = SatvsSatLayer()
    lv2['SatvsLum'] = SatvsLumLayer()
    lv2['grading'] = GradingLayer()
    lv2['hls_red'] = HLSRedLayer()
    lv2['hls_orange'] = HLSOrangeLayer()
    lv2['hls_yellow'] = HLSYellowLayer()
    lv2['hls_green'] = HLSGreenLayer()
    lv2['hls_cyan'] = HLSCyanLayer()
    lv2['hls_blue'] = HLSBlueLayer()
    lv2['hls_purple'] = HLSPurpleLayer()
    lv2['hls_magenta'] = HLSMagentaLayer()

    return layer

def pipeline_lv0(img, layer, param):
    lv0 = layer[0]
    lv1reset = False

    l = lv0['inpaint']
    pre_diff = l.diff
    diff = l.make_diff(img, param)
    if diff is not None:
        rgb = diff
    else:
        rgb = img.copy()
    if pre_diff is not diff:
        lv1reset = True

    if lv1reset == True:
        for v in layer[1].values():
            v.reset()
        for v in layer[2].values():
            v.reset()

    return rgb, lv1reset

def pipeline_lv1(img, layer, param):
    lv1 = layer[1]
    lv2reset = False

    l = lv1['noise_reduction']
    pre_diff = l.diff
    diff = l.make_diff(img, param)
    if diff is not None:
        rgb = diff
    else:
        rgb = img.copy()
    if pre_diff is not diff:
        lv1['lut'].reset()
        lv1['defocus'].reset()
        lv1['lowpass_filter'].reset()
        lv1['perlin_noise'].reset()
        lv1reset = True

    l = lv1['lut']
    pre_diff = l.diff
    diff = l.make_diff(rgb, param)
    if diff is not None:
        rgb = diff
    if pre_diff is not diff:
        lv1['defocus'].reset()
        lv1['lowpass_filter'].reset()
        lv1['perlin_noise'].reset()
        lv1reset = True

    l = lv1['defocus']
    pre_diff = l.diff
    diff = l.make_diff(rgb, param)
    if diff is not None:
        rgb = diff
    if pre_diff is not diff:
        lv1['lowpass_filter'].reset()
        lv1['perlin_noise'].reset()
        lv1reset = True

    l = lv1['lowpass_filter']
    pre_diff = l.diff
    diff = l.make_diff(rgb, param)
    if diff is not None:
        rgb = diff
    if pre_diff is not diff:
        lv1['perlin_noise'].reset()
        lv1reset = True

    l = lv1['perlin_noise']
    pre_diff = l.diff
    diff = l.make_diff(rgb, param)
    if diff is not None:
        rgb = diff
    if pre_diff is not diff:
        lv1reset = True

    if lv2reset == True:
        for v in layer[1].values():
            v.reset()

    return rgb


def pipeline_lv2(img, layer, param):
    lv2 = layer[2]

    rgb = img.copy()

    # RGB
    diff = lv2['color_correct'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['dehaze'].make_diff(rgb, param)
    if diff is not None: rgb += diff

    rgb2 = rgb.copy()
    diff = lv2['tonecurve'].make_diff(rgb, param)
    if diff is not None: rgb2 += diff
    diff = lv2['tonecurve_red'].make_diff(rgb, param)
    if diff is not None: rgb2[:,:,0] += diff
    diff = lv2['tonecurve_green'].make_diff(rgb, param)
    if diff is not None: rgb2[:,:,1] += diff
    diff = lv2['tonecurve_blue'].make_diff(rgb, param)
    if diff is not None: rgb2[:,:,2] += diff
    diff = lv2['grading'].make_diff(rgb, param)
    if diff is not None: rgb2 += diff
    rgb = rgb2

    diff = lv2['contrast'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['exposure'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['color_temperature'].make_diff(rgb, param)
    if diff is not None: rgb *= diff

    # 以降HLS
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)

    # Lのみ
    hls_l = hls[:, :, 1]
    hls2_l = hls_l.copy()
    diff = lv2['tone'].make_diff(hls_l, param)
    if diff is not None: hls2_l += diff
    diff = lv2['level'].make_diff(hls_l, param)
    if diff is not None: hls2_l += diff
    hls[:, :, 1] = hls2_l

    # HLS
    hls2 = hls.copy()
    diff = lv2['hls_red'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_orange'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_yellow'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_green'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_cyan'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_blue'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_purple'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    diff = lv2['hls_magenta'].make_diff(hls, param)
    if diff is not None: hls2 += diff
    hls = hls2

    # Hのみ
    hls_h = hls[:, :, 0]
    hls2_h = hls_h.copy()
    diff = lv2['HuevsHue'].make_diff(hls_h, param)
    if diff is not None: hls2_h += diff
    hls[:, :, 0] = hls2_h

    #　Lのみ
    hls_l = hls[:, :, 1]
    hls2_l = hls_l.copy()
    diff = lv2['HuevsLum'].make_diff(hls_l, param)
    if diff is not None: hls2_l *= diff
    diff = lv2['LumvsLum'].make_diff(hls_l, param)
    if diff is not None: hls2_l *= diff
    diff = lv2['SatvsLum'].make_diff(hls_l, param)
    if diff is not None: hls2_l *= diff
    hls[:, :, 1] = hls2_l

    # Sのみ
    hls_s = hls[:, :, 2]
    hls2_s = hls_s.copy()
    diff = lv2['HuevsSat'].make_diff(hls_s, param)
    if diff is not None: hls2_s *= diff
    diff = lv2['LumvsSat'].make_diff(hls_s, param)
    if diff is not None: hls2_s *= diff
    diff = lv2['SatvsSat'].make_diff(hls_s, param)
    if diff is not None: hls2_s *= diff
    hls[:, :, 2] = hls2_s

    diff = lv2['saturation'].make_diff(hls, param)
    if diff is not None: hls2_s *= diff
    hls[:, :, 2] = hls2_s

    # 合成
    hls[:,:,1] = np.clip(hls[:,:,1], 0, 1.0)
    hls[:,:,2] = np.clip(hls[:,:,2], 0, 1.0)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)

    return rgb

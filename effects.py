
import cv2
import numpy as np
import colorsys
import skimage

import noise2void
import DRBNet
import colorcorrect.algorithm as cca
import perlin
import lama
import deepfillv2

import core
import cubelut
import mask_editor
import crop_editor

#補正既定クラス
class Effect():

    def __init__(self, **kwargs):
        self.reeffect()

    def reeffect(self):
        self.diff = None
        self.hash = None

    def set2widget(self, widget, param):
        pass

    def set2param(self, param, widget):
        pass

    # 差分の作成
    def make_diff(self, img, param):
        self.diff = img

    def apply_diff(self, img):
        pass

class RotationEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_rotation"].set_slider_value(param.get('rotation', 0))

    def set2param(self, param, widget):
        param['rotation'] = widget.ids["slider_rotation"].value

    def make_diff(self, img, param):
        ang = param.get('rotation', 0)
        if ang == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ang))
            if self.hash != param_hash:
                self.diff = core.rotation(img, ang)
                self.hash = param_hash
        
        return self.diff
    
    def apply_diff(self, img):
        return self.diff

class AINoiseReductonEffect(Effect):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if AINoiseReductonEffect.__net is None:
            AINoiseReductonEffect.__net = noise2void.setup_predict()

    def set2widget(self, widget, param):
        widget.ids["switch_ai_noise_reduction"].active = False if param.get('ai_noise_reduction', 0) == 0 else True

    def set2param(self, param, widget):
        param['ai_noise_reduction'] = 0 if widget.ids["switch_ai_noise_reduction"].active == False else 1

    def make_diff(self, img, param):
        nr = param.get('ai_noise_reduction', 0)
        if nr <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((nr))
            if self.hash != param_hash:
                self.diff = noise2void.predict(img, AINoiseReductonEffect.__net, 'mps')
                self.hash = param_hash
        
        return self.diff

class NLMNoiseReductionEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_nlm_noise_reduction"].set_slider_value(param.get('nlm_noise_reduction', 0))

    def set2param(self, param, widget):
        param['nlm_noise_reduction'] = widget.ids["slider_nlm_noise_reduction"].value

    def make_diff(self, img, param):
        nlm = int(param.get('nlm_noise_reduction', 0))
        if nlm == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((nlm))
            if self.hash != param_hash:
                sigma_est = np.mean(skimage.restoration.estimate_sigma(img, channel_axis=2))
                self.diff = skimage.restoration.denoise_nl_means(img, h=nlm/100.0*sigma_est, sigma=sigma_est, fast_mode=True, channel_axis=2)
                self.hash = param_hash

        return self.diff
    
class DeblurFilterEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_deblur_filter"].set_slider_value(param.get('deblur_filter', 0))

    def set2param(self, param, widget):
        param['deblur_filter'] = widget.ids["slider_deblur_filter"].value

    def make_diff(self, img, param):
        dbfr = int(param.get('deblur_filter', 0))
        if dbfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((dbfr))
            if self.hash != param_hash:
                self.diff = core.lucy_richardson_gauss(img, dbfr)
                self.hash = param_hash

        return self.diff

"""
class InpaintEffect(Effect):
    __generator= None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if InpaintEffect.__generator is None:
            InpaintEffect.__generator = deepfillv2.setup_predict('cpu')
        
        self.mask_editor = None
        self.crop_editor = None

    def set2widget(self, widget, param):
        widget.ids["switch_inpaint"].active = False if param.get('inpaint', 0) == 0 else True
        widget.ids["button_inpaint_predict"].state = "normal" if param.get('inpaint_predict', 0) == 0 else "down"

    def set2param(self, param, widget):
        param['inpaint'] = 0 if widget.ids["switch_inpaint"].active == False else 1
        param['inpaint_predict'] = 0 if widget.ids["button_inpaint_predict"].state == "normal" else 1

        if param['inpaint'] > 0:
            if self.mask_editor is None:
                self.mask_editor = mask_editor.MaskEditor(param['img_size'][1], param['img_size'][0])
                self.mask_editor.zoom = widget.get_scale()
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)

            if self.crop_editor is None:
                self.crop_editor = crop_editor.CropEditor(input_width=param['img_size'][1], input_height=param['img_size'][0], scale=widget.get_scale())
                widget.ids["preview_widget"].add_widget(self.crop_editor)
            
        if param['inpaint'] <= 0:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None

            if self.crop_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.crop_editor)
                self.crop_editor = None

    def make_diff(self, img, param):
        ip = param.get('inpaint', 0)
        ipp = param.get('inpaint_predict', 0)
        if (ip > 0 and ipp > 0) is True:
            cx, cy, cw, ch, sc = self.crop_editor.get_crop_info()
            img2 = deepfillv2.predict(img[cy:cy+ch, cx:cx+cw], self.mask_editor.get_mask()[cy:cy+ch, cx:cx+cw], InpaintEffect.__generator, 'cpu')
            img3 = img.copy()
            img3[cy:cy+ch, cx:cx+cw] = img2
            self.diff = img3
            self.mask_editor.clear_mask()
            self.mask_editor.update_canvas()
        
        return self.diff
"""

class InpaintDiff:
    def __init__(self, **kwargs):
        self.crop_info = kwargs.get('crop_info', None)
        self.image = kwargs.get('image', None)

class InpaintEffect(Effect):

    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if InpaintEffect.__net is None:
            InpaintEffect.__net = lama.setup_predict()
        
        self.crop_info_list = []
        self.mask_editor = None
        self.crop_editor = None

    def set2widget(self, widget, param):
        widget.ids["switch_inpaint"].active = False if param.get('inpaint', 0) == 0 else True
        widget.ids["button_inpaint_predict"].state = "normal" if param.get('inpaint_predict', 0) == 0 else "down"

    def set2param(self, param, widget):
        param['inpaint'] = 0 if widget.ids["switch_inpaint"].active == False else 1
        param['inpaint_predict'] = 0 if widget.ids["button_inpaint_predict"].state == "normal" else 1

        if param['inpaint'] > 0:
            if self.mask_editor is None:
                self.mask_editor = mask_editor.MaskEditor(param['img_size'][1], param['img_size'][0])
                self.mask_editor.zoom = widget.get_scale()
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)

            if self.crop_editor is None:
                self.crop_editor = crop_editor.CropEditor(input_width=param['img_size'][1], input_height=param['img_size'][0], scale=widget.get_scale())
                widget.ids["preview_widget"].add_widget(self.crop_editor)
            
        if param['inpaint'] <= 0:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None

            if self.crop_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.crop_editor)
                self.crop_editor = None

    def make_diff(self, img, param):
        ip = param.get('inpaint', 0)
        ipp = param.get('inpaint_predict', 0)
        if (ip > 0 and ipp > 0) is True:
            cx, cy, cw, ch, sc = self.crop_editor.get_crop_info()
            img2 = lama.predict(img[cy:cy+ch, cx:cx+cw], self.mask_editor.get_mask()[cy:cy+ch, cx:cx+cw], InpaintEffect.__net)
            self.crop_info_list.append(InpaintDiff(crop_info=(cx, cy, cw, ch), image=img2))
            self.diff = self.crop_info_list
            self.mask_editor.clear_mask()
            self.mask_editor.update_canvas()
        
        return self.diff
    
    def apply_diff(self, img):
        for i in range(len(self.crop.info_list)):
            cx, cy, cw, ch = self.crop_info_list[i].crop_info
            img[cy:cy+ch, cx:cx+cw] = self.crop_info_list[i].image
        
        return img


class DefocusEffect(Effect):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if DefocusEffect.__net is None:
            DefocusEffect.__net = DRBNet.setup_predict()

    def set2widget(self, widget, param):
        widget.ids["switch_defocus"].active = False if param.get('defocus', 0) == 0 else True

    def set2param(self, param, widget):
        param['defocus'] = 0 if widget.ids["switch_defocus"].active == False else 1

    def make_diff(self, img, param):
        df = param.get('defocus', 0)
        if df <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((df))
            if self.hash != param_hash:
                self.diff = DRBNet.predict(img, DefocusEffect.__net, 'mps')
                self.hash = param_hash

        return self.diff

class ColorCorrectEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_color_correct"].active = False if param.get('defcolor_correctocus', 0) == 0 else True

    def set2param(self, param, widget):
        param['color_correct'] = 0 if widget.ids["switch_color_correct"].active == False else 1

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

class DensityEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_density"].set_slider_value(param.get('density', 0))

    def set2param(self, param, widget):
        param['density'] = widget.ids["slider_density"].value

    def make_diff(self, hls, param):
        den = param.get('density', 0)
        param_hash = hash((den))
        if den == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_density(hls, den)
            self.diff = np.zeros_like(hls)
            self.diff[:,:,1] = hls2[:,:,1]-hls[:,:,1]
            self.diff[:,:,2] = np.divide(hls2[:,:,2], hls[:,:,2], out=self.diff[:,:,2], where=hls[:,:,2]!=0.0)
            self.hash = param_hash

        return self.diff

class ClearColorEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_clear_color"].set_slider_value(param.get('clear_color', 0))

    def set2param(self, param, widget):
        param['clear_color'] = widget.ids["slider_clear_color"].value

    def make_diff(self, hls, param):
        cc = param.get('clear_color', 0)
        param_hash = hash((cc))
        if cc == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2 = core.adjust_clear_color(hls, cc)
            self.diff = hls2-hls
            self.hash = param_hash

        return self.diff

class LUTEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["toggle_lut"].state = "normal" if param.get('lut', 0) == 0 else "down"

    def set2param(self, param, widget):
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

class LowpassFilterEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_lowpass_filter"].set_slider_value(param.get('lowpass_filter', 0))
        widget.ids["slider_highpass_filter"].set_slider_value(param.get('highpass_filter', 0))

    def set2param(self, param, widget):
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

class PerlinNoiseEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_perlin_noise"].set_slider_value(param.get('perlin_noise', 0))
        widget.ids["slider_perlin_noise_opacity"].set_slider_value(param.get('perlin_noise_opacity', 100))

    def set2param(self, param, widget):
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

class ColorTemperatureEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_color_temperature"].set_slider_value(param.get('color_temperature', 6400))
        widget.ids["slider_color_tint"].set_slider_value(param.get('color_tint', 0))
        widget.ids["slider_color_temperature"].set_slider_reset(param.get('color_temperature_reset', 6400))
        widget.ids["slider_color_tint"].set_slider_reset(param.get('color_tint_reset', 0))
 
    def set2param(self, param, widget):
        param['color_temperature'] = widget.ids["slider_color_temperature"].value
        param['color_tint'] = widget.ids["slider_color_tint"].value

    def make_diff(self, rgb, param):
        temp = param.get('color_temperature', 6400)
        tint = param.get('color_tint', 0)
        Y = param.get('color_Y', 1.0)
        param_hash = hash((temp, tint))
        if self.hash != param_hash:
            self.diff = core.convert_TempTint2RGB(temp, -tint, Y)
            self.hash = param_hash

        return self.diff

class ExposureEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_exposure"].set_slider_value(param.get('exposure', 0))

    def set2param(self, param, widget):
        param['exposure'] = widget.ids["slider_exposure"].value

    def make_diff(self, rgb, param):
        ev = param.get('exposure', 0)
        param_hash = hash((ev))
        if ev == 0:
            self.diff = None
            self.hash = None
        
        elif self.hash != param_hash:
            self.diff = core.adjust_exposure(rgb, ev)
            self.hash = param_hash

        return self.diff

class ContrastEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_contrast"].set_slider_value(param.get('contrast', 0))

    def set2param(self, param, widget):
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

class MidtoneEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_midtone"].set_slider_value(param.get('midtone', 0))

    def set2param(self, param, widget):
        param['midtone'] = widget.ids["slider_midtone"].value

    def make_diff(self, hls_l, param):
        mt = param.get('midtone', 0)
        param_hash = hash((mt))
        if mt == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2_l = core.adjust_shadow(hls_l, mt)
            hls2_l = core.adjust_hilight(hls2_l, mt)
            self.diff = hls2_l-hls_l    # Lのみ保存
            self.hash = param_hash

        return self.diff

class ToneEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_black"].set_slider_value(param.get('black', 0))
        widget.ids["slider_white"].set_slider_value(param.get('white', 0))
        widget.ids["slider_shadow"].set_slider_value(param.get('shadow', 0))
        widget.ids["slider_hilight"].set_slider_value(param.get('hilight', 0))

    def set2param(self, param, widget):
        param['black'] = widget.ids["slider_black"].value
        param['white'] = widget.ids["slider_white"].value
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
            values[1] += black * 100.0
            values[2] += white * 100.0
            points /= 65535.0
            values /= 65535.0

            hls2_l = core.apply_curve(hls_l, points, values, False)
            hls2_l = core.adjust_shadow(hls2_l, shadow)
            hls2_l = core.adjust_hilight(hls2_l, hilight)
            self.diff = hls2_l-hls_l    # Lのみ保存
            self.hash = param_hash

        return self.diff

class SaturationEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_saturation"].set_slider_value(param.get('saturation', 0))
        widget.ids["slider_vibrance"].set_slider_value(param.get('vibrance', 0))

    def set2param(self, param, widget):
        param['saturation'] = widget.ids["slider_saturation"].value
        param['vibrance'] = widget.ids["slider_vibrance"].value

    def make_diff(self, hls_s, param):
        sat = param.get('saturation', 0)
        vib = param.get('vibrance', 0)
        param_hash = hash((sat, vib))
        if sat == 0 and vib == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2_s = core.adjust_saturation(hls_s, sat, vib)
            self.diff = np.divide(hls2_s, hls_s, out=np.zeros_like(hls_s), where=hls_s!=0.0)    # Sのみ保存
            self.hash = param_hash
        
        return self.diff

class DehazeEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_dehaze"].set_slider_value(param.get('dehaze', 0))

    def set2param(self, param, widget):
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

class LevelEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_black_level"].set_slider_value(param.get('black_level', 0))
        widget.ids["slider_white_level"].set_slider_value(param.get('white_level', 255))
        widget.ids["slider_mid_level"].set_slider_value(param.get('mid_level',127))

    def set2param(self, param, widget):
        bl = widget.ids["slider_black_level"].value
        wl = widget.ids["slider_white_level"].value
        ml = widget.ids["slider_mid_level"].value
        param['black_level'] = bl
        param['white_level'] = wl
        param['mid_level'] = ml

    def make_diff(self, hls_l, param):
        bl = param.get('black_level', 0)
        wl = param.get('white_level', 255)
        ml = param.get('mid_level', 127)
        param_hash = hash((bl, wl, ml))
        if bl == 0 and wl == 255 and ml == 127:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2_l = core.apply_level_adjustment(hls_l, bl, ml, wl)
            self.diff = hls2_l-hls_l    # Lのみ保存
            self.hash = param_hash

        return self.diff

class TonecurveEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve"].set_point_list(param.get('tonecurve', None))

    def set2param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_point_list()

    def make_diff(self, img, param):
        pl = param.get('tonecurve', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                img2 = core.apply_point_list(img, pl)
                self.diff = img2-img        # RGB保存
                self.hash = param_hash

        return self.diff

class TonecurveRedEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_red"].set_point_list(param.get('tonecurve_red', None))

    def set2param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_point_list()

    def make_diff(self, img, param):
        pl = param.get('tonecurve_red', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                img2 = core.apply_point_list(img[:,:,0], pl)
                self.diff = img2-img[:,:,0]            # R保存
                self.hash = param_hash

        return self.diff

class TonecurveGreenEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_green"].set_point_list(param.get('tonecurve_green', None))

    def set2param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_point_list()

    def make_diff(self, img, param):   
        pl = param.get('tonecurve_green', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                img2 = core.apply_point_list(img[:,:,1], pl)
                self.diff = img2-img[:,:,1]            # G保存
                self.hash = param_hash

        return self.diff

class TonecurveBlueEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_blue"].set_point_list(param.get('tonecurve_blue', None))

    def set2param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_point_list()

    def make_diff(self, img, param):
        pl = param.get('tonecurve_blue', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                img2 = core.apply_point_list(img[:,:,2], pl)
                self.diff = img2-img[:,:,2]        # B保存
                self.hash = param_hash

        return self.diff

class LumvsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["LumvsLum"].set_point_list(param.get('LumvsLum', None))

    def set2param(self, param, widget):
        param['LumvsLum'] = widget.ids["LumvsLum"].get_point_list()

    def make_diff(self, hls_l, param):
        ll = param.get("LumvsLum", None)
        if ll is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(ll))
            if self.hash != param_hash:
                hls2_l = core.apply_point_list(hls_l, ll)
                self.diff = 2.0**((hls2_l-0.5)*2.0)   # Lのみ保存
                self.hash = param_hash

        return self.diff

class HuevsHueEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsHue"].set_point_list(param.get('HuevsHue', None))

    def set2param(self, param, widget):
        param['HuevsHue'] = widget.ids["HuevsHue"].get_point_list()

    def make_diff(self, hls_h, param):
        hh = param.get("HuevsHue", None)
        if hh is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hh))
            if self.hash != param_hash:
                hls2_h = core.apply_point_list(hls_h/360.0, hh)
                self.diff = (hls2_h-0.5)*360.0    # Hのみ保存
                self.hash = param_hash

        return self.diff

class HuevsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsSat"].set_point_list(param.get('HuevsSat', None))

    def set2param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_point_list()

    def make_diff(self, hls_h, param):
        hs = param.get("HuevsSat", None)
        if hs is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hs))
            if self.hash != param_hash:
                hls2_s = core.apply_point_list(hls_h, hs)
                self.diff = 2.0**((hls2_s-0.5)*2.0)     # Sのみ保存
                self.hash = param_hash

        return self.diff

class HuevsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsLum"].set_point_list(param.get('HuevsLum', None))

    def set2param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_point_list()

    def make_diff(self, hls_h, param):
        hl = param.get("HuevsLum", None)
        if hl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hl))
            if self.hash != param_hash:
                hls2_l = core.apply_point_list(hls_h, hl)
                self.diff = 2.0**((hls2_l-0.5)*2.0)     # Lのみ保存
                self.hash = param_hash

        return self.diff

class LumvsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["LumvsSat"].set_point_list(param.get('LumvsSat', None))

    def set2param(self, param, widget):
        param['LumvsSat'] = widget.ids["LumvsSat"].get_point_list()

    def make_diff(self, hls_l, param):
        ls = param.get("LumvsSat", None)
        if ls is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(ls))
            if self.hash != param_hash:
                hls2_s = core.apply_point_list(hls_l, ls)
                self.diff = 2.0**((hls2_s-0.5)*2.0)     # Sのみ保存
                self.hash = param_hash

        return self.diff

class SatvsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["SatvsSat"].set_point_list(param.get('SatvsSat', None))

    def set2param(self, param, widget):
        param['SatvsSat'] = widget.ids["SatvsSat"].get_point_list()

    def make_diff(self, hls_s, param):
        ss = param.get("SatvsSat", None)
        if ss is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(ss))
            if self.hash != param_hash:
                hls2_s = core.apply_point_list(hls_s, ss)
                self.diff = 2.0**((hls2_s-0.5)*2.0)     # Sのみ保存
                self.hash = param_hash

        return self.diff

class SatvsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["SatvsLum"].set_point_list(param.get('SatvsLum', None))

    def set2param(self, param, widget):
        param['SatvsLum'] = widget.ids["SatvsLum"].get_point_list()

    def make_diff(self, hls_s, param):
        sl = param.get("SatvsLum", None)
        if sl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(sl))
            if self.hash != param_hash:
                hls2_l = core.apply_point_list(hls_s, sl)
                self.diff = 2.0**((hls2_l-0.5)*2.0)     # Lのみ保存
                self.hash = param_hash

        return self.diff

class GradingEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["grading"].set_point_list(param.get('grading', None))
        widget.ids["slider_grading_hue"].set_slider_value(param.get('grading_hue', 0))
        widget.ids["slider_grading_lum"].set_slider_value(param.get('grading_lum', 0))
        widget.ids["slider_grading_sat"].set_slider_value(param.get('grading_sat', 0))

    def set2param(self, param, widget):
        param['grading'] = widget.ids["grading"].get_point_list()
        param['grading_hue'] = widget.ids["slider_grading_hue"].value
        param['grading_lum'] = widget.ids["slider_grading_lum"].value
        param['grading_sat'] = widget.ids["slider_grading_sat"].value

    def make_diff(self, hls, param):
        pl = param.get("grading", None)
        gh = param.get("grading_hue", 0)
        gl = param.get("grading_lum", 0)
        gs = param.get("grading_sat", 0)
        if pl is None and gh == 0 and gl == 0 and gs == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((np.sum(pl), gh, gl, gs))
            if self.hash != param_hash:
                blend = core.apply_point_list(hls[:,:,1], pl)
                rgb = np.array(colorsys.hls_to_rgb(gh/360.0, gl/100.0, gs/100.0), dtype=np.float32)
                blend_inv = 1-blend
                self.diff = (hls*blend_inv[:, :, np.newaxis] + rgb*blend[:, :, np.newaxis]) - hls
                self.hash = param_hash

        return self.diff

class HLSRedEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_red_hue"].set_slider_value(param.get('hls_red_hue', 0))
        widget.ids["slider_hls_red_lum"].set_slider_value(param.get('hls_red_lum', 0))
        widget.ids["slider_hls_red_sat"].set_slider_value(param.get('hls_red_sat', 0))

    def set2param(self, param, widget):
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

class HLSOrangeEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_orange_hue"].set_slider_value(param.get('hls_orange_hue', 0))
        widget.ids["slider_hls_orange_lum"].set_slider_value(param.get('hls_orange_lum', 0))
        widget.ids["slider_hls_orange_sat"].set_slider_value(param.get('hls_orange_sat', 0))

    def set2param(self, param, widget):
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

class HLSYellowEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_yellow_hue"].set_slider_value(param.get('hls_yellow_hue', 0))
        widget.ids["slider_hls_yellow_lum"].set_slider_value(param.get('hls_yellow_lum', 0))
        widget.ids["slider_hls_yellow_sat"].set_slider_value(param.get('hls_yellow_sat', 0))

    def set2param(self, param, widget):
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
            self.hash = param_hash

        return self.diff

class HLSGreenEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_green_hue"].set_slider_value(param.get('hls_green_hue', 0))
        widget.ids["slider_hls_green_lum"].set_slider_value(param.get('hls_green_lum', 0))
        widget.ids["slider_hls_green_sat"].set_slider_value(param.get('hls_green_sat', 0))

    def set2param(self, param, widget):
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

class HLSCyanEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_cyan_hue"].set_slider_value(param.get('hls_cyan_hue', 0))
        widget.ids["slider_hls_cyan_lum"].set_slider_value(param.get('hls_cyan_lum', 0))
        widget.ids["slider_hls_cyan_sat"].set_slider_value(param.get('hls_cyan_sat', 0))

    def set2param(self, param, widget):
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

class HLSBlueEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_blue_hue"].set_slider_value(param.get('hls_blue_hue', 0))
        widget.ids["slider_hls_blue_lum"].set_slider_value(param.get('hls_blue_lum', 0))
        widget.ids["slider_hls_blue_sat"].set_slider_value(param.get('hls_blue_sat', 0))

    def set2param(self, param, widget):
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

class HLSPurpleEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_purple_hue"].set_slider_value(param.get('hls_purple_hue', 0))
        widget.ids["slider_hls_purple_lum"].set_slider_value(param.get('hls_purple_lum', 0))
        widget.ids["slider_hls_purple_sat"].set_slider_value(param.get('hls_purple_sat', 0))

    def set2param(self, param, widget):
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

class HLSMagentaEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_hls_magenta_hue"].set_slider_value(param.get('hls_magenta_hue', 0))
        widget.ids["slider_hls_magenta_lum"].set_slider_value(param.get('hls_magenta_lum', 0))
        widget.ids["slider_hls_magenta_sat"].set_slider_value(param.get('hls_magenta_sat', 0))

    def set2param(self, param, widget):
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

class GlowEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_glow_black"].set_slider_value(param.get('glow_black', 0))
        widget.ids["slider_glow_gauss"].set_slider_value(param.get('glow_gauss', 0))
        widget.ids["slider_glow_opacity"].set_slider_value(param.get('glow_opacity',0))

    def set2param(self, param, widget):
        param['glow_black'] = widget.ids["slider_glow_black"].value
        param['glow_gauss'] = widget.ids["slider_glow_gauss"].value
        param['glow_opacity'] = widget.ids["slider_glow_opacity"].value

    def make_diff(self, rgb, param):
        gb = param.get('glow_black', 0)
        gg = int(param.get('glow_gauss', 0))
        go = param.get('glow_opacity', 0)
        if gb == 0 and gg == 0 and go == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((gb, gg, go))
            if self.hash != param_hash:
                hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
                hls[:,:,1] = core.apply_level_adjustment(hls[:,:,1], gb, 127+gg/2, 255)
                rgb2 = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
                if gg > 0:
                    rgb2 = core.lowpass_filter(rgb2, gg*2-1)
                go = go/100.0
                self.diff = cv2.addWeighted(rgb, 1.0-go, core.blend_screen(rgb, rgb2), go, 0)
                self.hash = param_hash

        return self.diff

def create_effects():
    effects = [{}, {}, {}]

    lv0 = effects[0]
    lv0['inpaint'] = InpaintEffect()
    lv0['rotation'] = RotationEffect()  

    lv1 = effects[1]
    lv1['ai_noise_reduction'] = AINoiseReductonEffect()
    lv1['nlm_noise_reduction'] = NLMNoiseReductionEffect()
    lv1['deblur_filter'] = DeblurFilterEffect()
    lv1['lut'] = LUTEffect()
    lv1['defocus'] = DefocusEffect()
    lv1['lowpass_filter'] = LowpassFilterEffect()
    lv1['perlin_noise'] = PerlinNoiseEffect()
    lv1['glow'] = GlowEffect()
    
    lv2 = effects[2]
    lv2['color_temperature'] = ColorTemperatureEffect()
    lv2['color_correct'] = ColorCorrectEffect()
    lv2['density'] = DensityEffect()
    lv2['clear_color'] = ClearColorEffect()
    lv2['exposure'] = ExposureEffect()
    lv2['contrast'] = ContrastEffect()
    lv2['midtone'] = MidtoneEffect()
    lv2['tone'] = ToneEffect()
    lv2['saturation'] = SaturationEffect()
    lv2['dehaze'] = DehazeEffect()
    lv2['level'] = LevelEffect()
    lv2['tonecurve'] = TonecurveEffect()
    lv2['tonecurve_red'] = TonecurveRedEffect()
    lv2['tonecurve_green'] = TonecurveGreenEffect()
    lv2['tonecurve_blue'] = TonecurveBlueEffect()
    lv2['LumvsLum'] = LumvsLumEffect()
    lv2['HuevsHue'] = HuevsHueEffect()
    lv2['HuevsSat'] = HuevsSatEffect()
    lv2['HuevsLum'] = HuevsLumEffect()
    lv2['LumvsSat'] = LumvsSatEffect()
    lv2['SatvsSat'] = SatvsSatEffect()
    lv2['SatvsLum'] = SatvsLumEffect()
    lv2['grading'] = GradingEffect()
    lv2['hls_red'] = HLSRedEffect()
    lv2['hls_orange'] = HLSOrangeEffect()
    lv2['hls_yellow'] = HLSYellowEffect()
    lv2['hls_green'] = HLSGreenEffect()
    lv2['hls_cyan'] = HLSCyanEffect()
    lv2['hls_blue'] = HLSBlueEffect()
    lv2['hls_purple'] = HLSPurpleEffect()
    lv2['hls_magenta'] = HLSMagentaEffect()

    return effects

def reeffect_all(effect):
    for dict in effect:
        for l in dict.values():
            l.reeffect()

def pipeline_lv0(img, effects, param):
    lv0 = effects[0]
    lv1reset = False

    rgb = img.copy()
    for i, n in enumerate(lv0):
        if lv1reset == True:
            lv0[n].reeffect()
            
        pre_diff = lv0[n].diff
        diff = lv0[n].make_diff(rgb, param)
        if diff is not None:
            rgb = lv0[n].apply_diff(rgb)

        if pre_diff is not diff:
            lv1reset = True

    if lv1reset == True:
        for v in effects[1].values():
            v.reeffect()
        for v in effects[2].values():
            v.reeffect()

    return rgb, lv1reset

def pipeline_lv1(img, effects, param):
    lv1 = effects[1]
    lv2reset = False

    rgb = img.copy()
    for i, n in enumerate(lv1):
        if lv2reset == True:
            lv1[n].reeffect()
            
        pre_diff = lv1[n].diff
        diff = lv1[n].make_diff(rgb, param)
        if diff is not None:
            rgb = diff

        if pre_diff is not diff:
            lv2reset = True
            
    if lv2reset == True:
        for l in effects[2].values():
            l.reeffect()

    return rgb


def pipeline_lv2(img, effects, param):
    lv2 = effects[2]

    rgb = img.copy()


    # 以降HLS
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)

    # Lのみ
    hls_l = hls[:, :, 1]
    hls2_l = hls_l.copy()
    diff = lv2['midtone'].make_diff(hls_l, param)
    if diff is not None: hls2_l += diff
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

    diff = lv2['density'].make_diff(hls, param)
    if diff is not None:
        hls[:,:,1] += diff[:,:,1]
        hls[:,:,2] *= diff[:,:,2]

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

    diff = lv2['saturation'].make_diff(hls2_s, param)
    if diff is not None: hls2_s *= diff
    hls[:, :, 2] = hls2_s

    # 合成
    hls[:,:,1] = np.clip(hls[:,:,1], 0, 1.0)
    hls[:,:,2] = np.clip(hls[:,:,2], 0, 1.0)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)

    # RGB
    diff = lv2['color_correct'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['dehaze'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['clear_color'].make_diff(rgb, param)
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
    if diff is not None: rgb *= diff
    diff = lv2['color_temperature'].make_diff(rgb, param)
    if diff is not None: rgb *= diff

    rgb = np.clip(rgb, 0, 1.0)
    return rgb
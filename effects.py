
import cv2
import numpy as np
import importlib
import bz2

#import colorsys
#import skimage

#import noise2void
#import DRBNet
#import iopaint.predict
#import dehazing.dehaze

import core
import cubelut
import mask_editor
import crop_editor
import microcontrast
import subpixel_shift
import film_simulation
import lens_simulator
import config
import pipeline

# 補正基底クラス
class Effect():

    def __init__(self, **kwargs):
        self.diff = None
        self.hash = None

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
        if self.diff is not None:
            return self.diff
        return img

    def finalize(self, param, widget):
        pass


# レンズモディファイア
class LensModifierEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_color_modification"].active = False if param.get('color_modification', 0) == 0 else True
        widget.ids["switch_subpixel_distortion"].active = False if param.get('subpixel_distortion', 0) == 0 else True
        widget.ids["switch_geometry_distortion"].active = False if param.get('geometry_distortion', 0) == 0 else True

    def set2param(self, param, widget):
        param['color_modification'] = 0 if widget.ids["switch_color_modification"].active == False else 1
        param['subpixel_distortion'] = 0 if widget.ids["switch_subpixel_distortion"].active == False else 1
        param['geometry_distortion'] = 0 if widget.ids["switch_geometry_distortion"].active == False else 1

    def make_diff(self, img, param):
        cd = param.get('color_modification', 0)
        sd = param.get('subpixel_distortion', 0)
        gd = param.get('geometry_distortion', 0)
        if cd <= 0 and sd <= 0 and gd <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((cd, sd, gd))
            if self.hash != param_hash:
                self.diff = core.modify_lensimage(img, None, cd > 0, sd > 0, gd > 0)
                self.hash = param_hash
        
        return self.diff
    

# サブピクセルシフト合成
class SubpixelShiftEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_subpixel_shift"].active = False if param.get('subpixel_shift', 0) == 0 else True

    def set2param(self, param, widget):
        param['subpixel_shift'] = 0 if widget.ids["switch_subpixel_shift"].active == False else 1

    def make_diff(self, img, param):
        ss = param.get('subpixel_shift', 0)
        if ss <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ss))
            if self.hash != param_hash:
                self.diff = subpixel_shift.create_enhanced_image(img)
                self.hash = param_hash
        
        return self.diff
    
class InpaintDiff:
    def __init__(self, **kwargs):
        self.crop_info = kwargs.get('crop_info', None)
        self.image = kwargs.get('image', None)

    def image2list(self):
        if type(self.image) is np.ndarray:
            self.image = (self.image.shape, list(bz2.compress(self.image.tobytes(), 1)))
            #self.image = self.image.tolist()

    def list2image(self):
        if type(self.image) is list:
            self.image = np.reshape(np.frombuffer(bz2.decompress(bytearray(self.image[1])), dtype=np.float32), self.image[0])
            #self.image = np.array(self.image)

class InpaintEffect(Effect):
    __iopaint = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.inpaint_diff_list = []
        self.mask_editor = None

    @staticmethod
    def dump(param):
        inpaint_diff_list = param.get('inpaint_diff_list', None)
        if inpaint_diff_list is not None:
            inpaint_diff_list_dumps = []
            for inpaint_diff in inpaint_diff_list:
                inpaint_diff.image2list()
                inpaint_diff_list_dumps.append((inpaint_diff.crop_info, inpaint_diff.image))
            param['inpaint_diff_list'] = inpaint_diff_list_dumps

    @staticmethod
    def load(param):
        inpaint_diff_list_dumps = param.get('inpaint_diff_list', None)
        if inpaint_diff_list_dumps is not None:
            inpaint_diff_list = []
            for inpaint_diff_dump in inpaint_diff_list_dumps:
                inpaint_diff = InpaintDiff(crop_info=inpaint_diff_dump[0], image=inpaint_diff_dump[1])
                inpaint_diff.list2image()
                inpaint_diff_list.append(inpaint_diff)
            param['inpaint_diff_list'] = inpaint_diff_list

    def set2widget(self, widget, param):
        widget.ids["switch_inpaint"].active = False if param.get('inpaint', 0) == 0 else True
        widget.ids["button_inpaint_predict"].state = "normal" if param.get('inpaint_predict', 0) == 0 else "down"

    def set2param(self, param, widget):
        param['inpaint'] = 0 if widget.ids["switch_inpaint"].active == False else 1
        param['inpaint_predict'] = 0 if widget.ids["button_inpaint_predict"].state == "normal" else 1

        if param['inpaint'] > 0:
            if self.mask_editor is None:
                self.mask_editor = mask_editor.MaskEditor(param['img_size'][0], param['img_size'][1])
                self.mask_editor.zoom = param['crop_info'][4]
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)
            
        if param['inpaint'] <= 0:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None

    def make_diff(self, img, param):
        self.inpaint_diff_list = param.get('inpaint_diff_list', [])
        ip = param.get('inpaint', 0)
        ipp = param.get('inpaint_predict', 0)
        if (ip > 0 and ipp > 0) is True:
            if InpaintEffect.__iopaint is None:
                InpaintEffect.__iopaint = importlib.import_module('iopaint.predict')

            mask = cv2.GaussianBlur(self.mask_editor.get_mask(), (63, 63), 0)
            w, h = param['original_img_size']
            eh, ew = img.shape[:2]
            x, y = (ew-w)//2, (eh-h)//2
            img2 = InpaintEffect.__iopaint.predict(img[y:y+h, x:x+w], mask, model=config.get_config('iopaint_model'), resize_limit=config.get_config('iopaint_resize_limit'), use_realesrgan=config.get_config('iopaint_use_realesrgan'))
            img2 = img2 #/ param.get('white_balance', [1, 1, 1])
            bboxes = core.get_multiple_mask_bbox(self.mask_editor.get_mask())
            for bbox in bboxes:
                self.inpaint_diff_list.append(InpaintDiff(crop_info=(bbox[0] + x, bbox[1] + y, bbox[2], bbox[3]), image=img2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]))
            param['inpaint_diff_list'] = self.inpaint_diff_list
            self.mask_editor.clear_mask()
            self.mask_editor.update_canvas()
        
        if len(self.inpaint_diff_list) > 0:
            img2 = img.copy()
            for inpaint_diff in self.inpaint_diff_list:
                cx, cy, cw, ch = inpaint_diff.crop_info
                img2[cy:cy+ch, cx:cx+cw] = inpaint_diff.image
            self.diff = img2
        else:
            self.diff = None

        return self.diff
    

# 画像回転、反転
class RotationEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_rotation"].set_slider_value(param.get('rotation', 0))

    def set2param(self, param, widget):
        param['rotation'] = widget.ids["slider_rotation"].value
        
    def set2param2(self, param, arg):
        if arg == 'hflip':
            param['flip_mode'] = param.get('flip_mode', 0) ^ 1

        elif arg == 'vflip':
            param['flip_mode'] = param.get('flip_mode', 0) ^ 2

        elif arg == 90:
            rot = param.get('rotation2', 0) + 90.0
            if rot >= 90*4:
                rot = 0
            param['rotation2'] = rot

        elif arg == -90:
            rot = param.get('rotation2', 0) - 90.0
            if rot < 0:
                rot = 90*3
            param['rotation2'] = rot

        else:
            pass

    def make_diff(self, img, param):
        ang = param.get('rotation', 0)
        ang2 = param.get('rotation2', 0)
        flp = param.get('flip_mode', 0)
        if ang == 0 and ang2 == 0 and flp == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ang, ang2, flp))
            if self.hash != param_hash:
                self.diff = core.rotation(img, ang + ang2, flp)
                self.hash = param_hash
        
        return self.diff


# クロップ
class CropEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.crop_editor = None

    def set2widget(self, widget, param):
        widget.ids["spinner_acpect_ratio"].text = param.get('aspect_ratio', "None")

    def set2param(self, param, widget):
        param['crop_enable'] = False if widget.ids["effects"].current_tab.text != "Geometry" else True
        param['aspect_ratio'] = widget.ids["spinner_acpect_ratio"].text

        # crop_info がないのはマスク
        if param.get('crop_rect', None) is not None:

            # クロップエディタを開く
            if param['crop_enable'] == True:
                self._open_crop_editor(param, widget)

            # クロップエディタを閉じる
            elif param['crop_enable'] == False:
                self._close_crop_editor(param, widget)

            # クロップ範囲をリセット
            if self.crop_editor is not None:
                if widget.ids["button_crop_reset"].state == "down":
                    self.crop_editor.set_to_local_crop_rect([0, 0, 0, 0])
                    self.crop_editor.update_crop_size()

                self.crop_editor.input_angle = param.get('rotation', 0) + param.get('rotation2', 0)


    def make_diff(self, img, param):
        ce = param.get('crop_enable', False)
        crop_info = param.get('crop_info', (0, 0, 0, 0, 1))
        if ce == True or crop_info == (0, 0, 0, 0, 1):
            self.diff = None
            self.hash = None
            param['img_size'] = (param['original_img_size'][0], param['original_img_size'][1])
        else:
            param_hash = hash((ce))
            if self.hash != param_hash:
                self.diff = crop_info
                self.hash = param_hash
                param['img_size'] = (crop_info[2], crop_info[3])
        return self.diff
    
    def apply_diff(self, img):
        return img

    def _open_crop_editor(self, param, widget):
        if self.crop_editor is None:
            input_width, input_height = param['original_img_size']
            x1, y1, x2, y2 = param['crop_rect']
            scale = config.get_config('preview_size')/max(input_width, input_height)
            self.crop_editor = crop_editor.CropEditor(input_width=input_width, input_height=input_height, scale=scale, crop_rect=(x1, y1, x2, y2))
            widget.ids["preview_widget"].add_widget(self.crop_editor)

            # 編集中は一時的に変更
            param['crop_info'] = crop_editor.CropEditor.get_initial_crop_info(input_width, input_height, scale)

    def _close_crop_editor(self, param, widget):
        if self.crop_editor is not None:
            param['crop_rect'] = self.crop_editor.get_crop_rect()
            param['crop_info'] = self.crop_editor.get_crop_info()

            widget.ids["preview_widget"].remove_widget(self.crop_editor)
            self.crop_editor = None

    def finalize(self, param, widget):
        self._close_crop_editor(param, widget)


# AI ノイズ除去
class AINoiseReductonEffect(Effect):
    __net = None
    __noise2void = None

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
                if AINoiseReductonEffect.__noise2void is None:
                    AINoiseReductonEffect.__noise2void = importlib.import_module('noise2void')
                if AINoiseReductonEffect.__net is None:
                    AINoiseReductonEffect.__net = AINoiseReductonEffect.__noise2void.setup_predict()

                self.diff = AINoiseReductonEffect.__noise2void.predict(img, AINoiseReductonEffect.__net, config.get_config('gpu_type'))
                self.hash = param_hash
        
        return self.diff


# NLMノイズ除去
class NLMNoiseReductionEffect(Effect):
    __skimage = None

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
                if NLMNoiseReductionEffect.__skimage is None:
                    NLMNoiseReductionEffect.__skimage = importlib.import_module('skimage')

                sigma_est = np.mean(NLMNoiseReductionEffect.__skimage.restoration.estimate_sigma(img, channel_axis=2))
                self.diff = NLMNoiseReductionEffect.__skimage.restoration.denoise_nl_means(img, h=nlm/100.0*sigma_est, sigma=sigma_est, fast_mode=True, channel_axis=2)
                self.hash = param_hash

        return self.diff


# デブラーフィルタ
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


class DefocusEffect(Effect):
    __net = None
    __DRBNet = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                if DefocusEffect.__DRBNet is None:
                    DefocusEffect.__DRBNet = importlib.import_module('DRBNet')

                if DefocusEffect.__net is None:
                    DefocusEffect.__net = DefocusEffect.__DRBNet.setup_predict()

                self.diff = DefocusEffect.__DRBNet.predict(img, DefocusEffect.__net, 'mps')
                self.hash = param_hash

        return self.diff


class LensblurFilterEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_lensblur_filter"].set_slider_value(param.get('lensblur_filter', 0))

    def set2param(self, param, widget):
        param['lensblur_filter'] = widget.ids["slider_lensblur_filter"].value

    def make_diff(self, img, param):
        lpfr = int(param.get('lensblur_filter', 0))
        if lpfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((lpfr))
            if self.hash != param_hash:
                self.diff = core.lensblur_filter(img, lpfr-1)
                self.hash = param_hash

        return self.diff


class LUTEffect(Effect):
    file_pathes = { '---': None }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lut = None

    def set2widget(self, widget, param):
        widget.ids["lut_spinner"].text = param.get('lut_name', 'None')

    def set2param(self, param, widget):
        if param.get('lut_name', "") != widget.ids["lut_spinner"].text:
            self.lut = None
        param['lut_name'] = widget.ids["lut_spinner"].text
        param['lut_path'] = LUTEffect.file_pathes.get(param['lut_name'], None)

    def make_diff(self, rgb, param):
        lut_path = param.get('lut_path', None)
        if lut_path is None:
            self.diff = None
            self.hash = None
        
        else:
            param_hash = hash((lut_path))
            if self.hash != param_hash:
                if self.lut is None:
                    self.lut = cubelut.read_lut(lut_path)
                self.diff = self.lut
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return cubelut.process_image(rgb, self.diff)

class FilmSimulationEffect(Effect):
 
    def set2widget(self, widget, param):
        widget.ids["spinner_film_preset"].text = param.get('film_preset', 'None')
        widget.ids["slider_film_intensity"].set_slider_value(param.get('film_intensity', 100))

    def set2param(self, param, widget):
        param['film_preset'] = widget.ids["spinner_film_preset"].text
        param['film_intensity'] = widget.ids["slider_film_intensity"].value

    def make_diff(self, rgb, param):
        preset = param.get('film_preset', 'None')
        intensity = param.get('film_intensity', 100)
        if preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity))
            if self.hash != param_hash:
                self.diff = (preset, intensity)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        film = film_simulation.simulator.apply_simulation(rgb, self.diff[0])
        per = self.diff[1] / 100.0
        return film * per + rgb * (1-per)

class LensSimulatorEffect(Effect):
 
    def set2widget(self, widget, param):
        widget.ids["spinner_lens_preset"].text = param.get('lens_preset', 'None')
        widget.ids["slider_lens_intensity"].set_slider_value(param.get('lens_intensity', 100))

    def set2param(self, param, widget):
        param['lens_preset'] = widget.ids["spinner_lens_preset"].text
        param['lens_intensity'] = widget.ids["slider_lens_intensity"].value

    def make_diff(self, rgb, param):
        preset = param.get('lens_preset', 'None')
        intensity = param.get('lens_intensity', 100)
        if preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity))
            if self.hash != param_hash:
                self.diff = (preset, intensity)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        lens = lens_simulator.process_image(rgb, self.diff[0])
        #lens = lens_simulator.apply_old_lens_effect(rgb, self.diff[0])
        per = self.diff[1] / 100.0
        return lens * per + rgb * (1-per)


class ColorTemperatureEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_color_temperature"].set_slider_value(param.get('color_temperature', 5000))
        widget.ids["slider_color_tint"].set_slider_value(param.get('color_tint', 0))
        widget.ids["slider_color_temperature"].set_slider_reset(param.get('color_temperature_reset', 5000))
        widget.ids["slider_color_tint"].set_slider_reset(param.get('color_tint_reset', 0))
 
    def set2param(self, param, widget):
        param['color_temperature'] = widget.ids["slider_color_temperature"].value
        param['color_tint'] = widget.ids["slider_color_tint"].value

    @staticmethod
    def apply_color_temperature(rgb, param):
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        return rgb * core.invert_TempTint2RGB(temp, tint, Y, 5000)

    def make_diff(self, rgb, param):
        sw = param.get('color_temperature_switch', True)
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        if sw == False:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((temp, tint))
            if self.hash != param_hash:
                self.diff = core.invert_TempTint2RGB(temp, tint, Y, 5000)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, img):
        return img * self.diff

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
            self.diff = core.calc_exposure(rgb, ev)
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return rgb * self.diff

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
            self.diff = con
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return core.adjust_contrast(rgb, self.diff, 0.5)

class MicroContrastEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_microcontrast"].set_slider_value(param.get('microcontrast', 0))

    def set2param(self, param, widget):
        param['microcontrast'] = widget.ids["slider_microcontrast"].value

    def make_diff(self, rgb, param):
        con = param.get('microcontrast', 0)
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = con
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        rgb2, _ = microcontrast.calculate_microcontrast(rgb, 7, self.diff)
        return rgb2

class ToneEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_black"].set_slider_value(param.get('black', 0))
        widget.ids["slider_white"].set_slider_value(param.get('white', 0))
        widget.ids["slider_shadow"].set_slider_value(param.get('shadow', 0))
        widget.ids["slider_highlight"].set_slider_value(param.get('highlight', 0))
        widget.ids["slider_midtone"].set_slider_value(param.get('midtone', 0))

    def set2param(self, param, widget):
        param['black'] = widget.ids["slider_black"].value
        param['white'] = widget.ids["slider_white"].value
        param['shadow'] = widget.ids["slider_shadow"].value
        param['highlight'] = widget.ids["slider_highlight"].value
        param['midtone'] = widget.ids["slider_midtone"].value

    def make_diff(self, rgb, param):
        black = param.get('black', 0)
        white = param.get('white', 0)
        shadow = param.get('shadow', 0)
        highlight =  param.get('highlight', 0)
        mt = param.get('midtone', 0)
        param_hash = hash((black, white, shadow, highlight, mt))
        if black == 0 and white == 0 and shadow == 0 and highlight == 0 and mt == 0:
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
            #self.diff = (highlight, shadow, points, values)
            self.diff = (highlight, shadow, mt, white, black)
            self.hash = param_hash
        return self.diff
    
    def apply_diff(self, rgb):
        #rgb2 = core.adjust_shadow_highlight(rgb, self.diff[0], self.diff[1])
        #rgb2 = core.apply_curve(rgb2, self.diff[2], self.diff[3])
        rgb2 = core.adjust_tone(rgb, self.diff[0], self.diff[1], self.diff[2], self.diff[3], self.diff[4])
        return rgb2

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
            hls2_s = core.calc_saturation(hls_s, sat, vib)
            self.diff = np.divide(hls2_s, hls_s, where=hls_s!=0.0)    # Sのみ保存
            self.hash = param_hash
        
        return self.diff
    
    def apply_diff(self, hls_s):
        return hls_s * self.diff

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
                self.diff = core.dehaze_image(rgb, de/100)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
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

    def make_diff(self, rgb, param):
        bl = param.get('black_level', 0)
        wl = param.get('white_level', 255)
        ml = param.get('mid_level', 127)
        param_hash = hash((bl, wl, ml))
        if bl == 0 and wl == 255 and ml == 127:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = (bl, ml, wl)
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return core.apply_level_adjustment(rgb, self.diff[0], self.diff[1], self.diff[2])

class TonecurveEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve"].set_point_list(param.get('tonecurve', None))

    def set2param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_point_list()

    def make_diff(self, rgb, param):
        pl = param.get('tonecurve', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return core.apply_lut(rgb, self.diff)

class TonecurveRedEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_red"].set_point_list(param.get('tonecurve_red', None))

    def set2param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_point_list()

    def make_diff(self, rgb_r, param):
        pl = param.get('tonecurve_red', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_r):
        return core.apply_lut(rgb_r, self.diff)

class TonecurveGreenEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_green"].set_point_list(param.get('tonecurve_green', None))

    def set2param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_point_list()

    def make_diff(self, rgb_g, param):   
        pl = param.get('tonecurve_green', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_g):
        return core.apply_lut(rgb_g, self.diff)

class TonecurveBlueEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_blue"].set_point_list(param.get('tonecurve_blue', None))

    def set2param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_point_list()

    def make_diff(self, rgb_b, param):
        pl = param.get('tonecurve_blue', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_g):
        return core.apply_lut(rgb_g, self.diff)

class GradingEffect(Effect):
    __colorsys = None

    def __init__(self, numstr, **kwargs):
        super().__init__(**kwargs)

        self.numstr = numstr

    def set2widget(self, widget, param):
        widget.ids["grading" + self.numstr].set_point_list(param.get('grading' + self.numstr, None))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].set_slider_value(param.get('grading' + self.numstr + '_hue', 0))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].set_slider_value(param.get('grading' + self.numstr + '_lum', 0))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].set_slider_value(param.get('grading' + self.numstr + '_sat', 0))

    def set2param(self, param, widget):
        param["grading" + self.numstr] = widget.ids["grading" + self.numstr].get_point_list()
        param["grading" + self.numstr + "_hue"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].value
        param["grading" + self.numstr + "_lum"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].value
        param["grading" + self.numstr + "_sat"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].value

    def make_diff(self, rgb, param):
        pl = param.get("grading" + self.numstr, None)
        gh = param.get("grading" + self.numstr + "_hue", 0)
        gl = param.get("grading" + self.numstr + "_lum", 0)
        gs = param.get("grading" + self.numstr + "_sat", 0)
        if pl is None and gh == 0 and gl == 0 and gs == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((np.sum(pl), gh, gl, gs))
            if self.hash != param_hash:
                if GradingEffect.__colorsys is None:
                    GradingEffect.__colorsys = importlib.import_module('colorsys')

                lut = core.calc_point_list_to_lut(pl)
                rgbs = np.array(GradingEffect.__colorsys.hls_to_rgb(gh/360.0, gl/100.0, gs/100.0), dtype=np.float32)
                self.diff = (lut, rgbs)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        lut, rgbs = self.diff
        blend = core.apply_lut(rgb, lut)
        blend_inv = 1-blend
        return (rgb*blend_inv + rgb*rgbs*blend)

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
                lut = core.calc_point_list_to_lut(hh)
                self.diff = ((lut-0.5)*2.0)*360
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_h):
        return core.apply_lut_add(hls_h, self.diff, 359)

class HuevsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsLum"].set_point_list(param.get('HuevsLum', None))

    def set2param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_point_list()

    def make_diff(self, hls_l, param):
        hl = param.get("HuevsLum", None)
        if hl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hl))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hl)
                self.diff = 2.0**((lut-0.5)*2.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        return core.apply_lut_mul(hls_l, self.diff, 1.0)

class HuevsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsSat"].set_point_list(param.get('HuevsSat', None))

    def set2param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_point_list()

    def make_diff(self, hls_s, param):
        hs = param.get("HuevsSat", None)
        if hs is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(hs))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hs)
                self.diff = (lut-0.5)*2.0+1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        return core.apply_lut_mul(hls_s, self.diff, 1.0)

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
                lut = core.calc_point_list_to_lut(ll)
                self.diff = 2.0**((lut-0.5)*2.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        return core.apply_lut_mul(hls_l, self.diff, 1.0)

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
                lut = core.calc_point_list_to_lut(ls)
                self.diff = (lut-0.5)*2.0+1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        return core.apply_lut_mul(hls_s, self.diff, 1.0)

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
                lut = core.calc_point_list_to_lut(sl)
                self.diff = 2.0**((lut-0.5)*2.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        return core.apply_lut_mul(hls_l, self.diff, 1.0)

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
                lut = core.calc_point_list_to_lut(ss)
                self.diff = (lut-0.5)*2.0+1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        return core.apply_lut_mul(hls_s, self.diff, 1.0)

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
            self.diff = core.adjust_hls_red_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_orange_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_yellow_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff

    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_green_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff

    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_cyan_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff

    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_blue_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff

    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_purple_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff

    def apply_diff(self, hls):
        return hls + self.diff

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
            self.diff = core.adjust_hls_magenta_smooth(hls, (hue, lum, sat)) - hls
            self.hash = param_hash

        return self.diff

    def apply_diff(self, hls):
        return hls + self.diff

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
                    rgb2 = core.lensblur_filter(rgb2, gg*2-1)
                go = go/100.0
                self.diff = cv2.addWeighted(rgb, 1.0-go, core.blend_screen(rgb, rgb2), go, 0)
                self.hash = param_hash

        return self.diff

class Mask2Effect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_mask2_hue_min"].set_slider_value(param.get('mask2_hue_min', 0))
        widget.ids["slider_mask2_hue_max"].set_slider_value(param.get('mask2_hue_max', 359))
        widget.ids["slider_mask2_lum_min"].set_slider_value(param.get('mask2_lum_min', 0))
        widget.ids["slider_mask2_lum_max"].set_slider_value(param.get('mask2_lum_max', 255))
        widget.ids["slider_mask2_sat_min"].set_slider_value(param.get('mask2_sat_min', 0))
        widget.ids["slider_mask2_sat_max"].set_slider_value(param.get('mask2_sat_max', 255))
        widget.ids["slider_mask2_blur"].set_slider_value(param.get('mask2_blur', 0))

    def set2param(self, param, widget):
        param['mask2_hue_min'] = widget.ids["slider_mask2_hue_min"].value
        param['mask2_hue_max'] = widget.ids["slider_mask2_hue_max"].value
        param['mask2_lum_min'] = widget.ids["slider_mask2_lum_min"].value
        param['mask2_lum_max'] = widget.ids["slider_mask2_lum_max"].value
        param['mask2_sat_min'] = widget.ids["slider_mask2_sat_min"].value
        param['mask2_sat_max'] = widget.ids["slider_mask2_sat_max"].value
        param['mask2_blur'] = widget.ids["slider_mask2_blur"].value

    def make_diff(self, rgb, param):
        hmin = param.get('mask2_hue_min', 0)
        hmax = param.get('mask2_hue_max', 359)
        lmin = param.get('mask2_lum_min', 0)
        lmax = param.get('mask2_lum_max', 255)
        smin = param.get('mask2_sat_min', 0)
        smax = param.get('mask2_sat_max', 255)
        blur = param.get('mask2_blur', 0)
        if hmin == 0 and hmax == 359 and lmin == 0 and lmax == 255 and smin == 0 and smax == 255 and blur == 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((hmin, hmax, lmin, lmax, smin, smax, blur))
            if self.hash != param_hash:
                self.diff = None
                self.hash = param_hash

        return self.diff

class HighlightCompressEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_highlight_compress"].active = False if param.get('highlight_compress', 0) == 0 else True

    def set2param(self, param, widget):
        param['highlight_compress'] = 0 if widget.ids["switch_highlight_compress"].active == False else 1

    def make_diff(self, img, param):
        hc = param.get('highlight_compress', 0)
        if hc <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((hc))
            if self.hash != param_hash:
                self.diff = 1.0
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return core.highlight_compress(rgb)

class CorrectOverexposedAreas(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_correct_overexposed_areas"].active = False if param.get('correct_overexposed_areas', 0) == 0 else True
        widget.ids["slider_correct_overexposed_areas_blur"].set_slider_value(param.get('correct_overexposed_areas_blur', 0))
        widget.ids["cp_correct_overexposed_areas"].ids['slider_red'].set_slider_value(param.get('correct_overexposed_areas_red', 0))
        widget.ids["cp_correct_overexposed_areas"].ids['slider_green'].set_slider_value(param.get('correct_overexposed_areas_green', 0))
        widget.ids["cp_correct_overexposed_areas"].ids['slider_blue'].set_slider_value(param.get('correct_overexposed_areas_blue', 0))

    def set2param(self, param, widget):
        param['correct_overexposed_areas'] = 0 if widget.ids["switch_correct_overexposed_areas"].active == False else 1
        param['correct_overexposed_areas_blur'] = widget.ids["slider_correct_overexposed_areas_blur"].value
        param["correct_overexposed_areas_red"] = widget.ids["cp_correct_overexposed_areas"].ids['slider_red'].value
        param["correct_overexposed_areas_green"] = widget.ids["cp_correct_overexposed_areas"].ids['slider_green'].value
        param["correct_overexposed_areas_blue"] = widget.ids["cp_correct_overexposed_areas"].ids['slider_blue'].value

    def make_diff(self, img, param):
        coa = param.get('correct_overexposed_areas', 0)
        coabl = param.get('correct_overexposed_areas_blur', 0)
        coar = param.get("correct_overexposed_areas_red", 0)
        coag = param.get("correct_overexposed_areas_green", 0)
        coab = param.get("correct_overexposed_areas_blue", 0)
        if coa <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((coa, coabl, coar, coag, coab))
            if self.hash != param_hash:
                self.diff = (coabl, (coar/255, coag/255, coab/255))
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return core.correct_overexposed_areas(rgb, blur_sigma=self.diff[0], correction_color=self.diff[1])

class VignetteEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_vignette_intensity"].set_slider_value(param.get('vignette_intensity', 0))
        widget.ids["slider_vignette_radius_percent"].set_slider_value(param.get('vignette_radius_percent', 0))

    def set2param(self, param, widget):
        param['vignette_intensity'] = widget.ids["slider_vignette_intensity"].value
        param['vignette_radius_percent'] = widget.ids["slider_vignette_radius_percent"].value

    def make_diff(self, rgb, crop_info, param):
        vi = param.get('vignette_intensity', 0)
        vr = param.get('vignette_radius_percent', 0)
        param_hash = hash((vi, vr))
        if vi == 0 and vr == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = (vi, vr, param['img_size'])
            self.hash = param_hash
        
        return self.diff
    
    def apply_diff(self, rgb, crop_info):
        imax = max(self.diff[2])
        return core.apply_vignette(rgb, self.diff[0], self.diff[1], crop_info, (self.diff[2][0], self.diff[2][1]))

class CLAHEEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_clahe_intensity"].set_slider_value(param.get('clahe_intensity', 0))

    def set2param(self, param, widget):
        param['clahe_intensity'] = widget.ids["slider_clahe_intensity"].value

    def make_diff(self, img, param):
        ci = param.get('clahe_intensity', 0)
        if ci <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((ci))
            if self.hash != param_hash:
                clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
                target = np.zeros_like(img, dtype=np.uint16)
                img2 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
                for i in range(3):
                    target[..., i] = clahe.apply(img2[..., i])
                target = target.astype(np.float32) / 65535
                ci = ci / 100
                self.diff = target * ci + img * (1-ci)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return self.diff

class RGB2HLSEffect(Effect):

    def make_diff(self, rgb, param):
        if self.diff is None:
            self.diff = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
        return self.diff
    
    def apply_diff(self, rgb):
        return self.diff

class HLS2RGBEffect(Effect):

    def make_diff(self, hls, param):
        if self.diff is None:
            self.diff = cv2.cvtColor(np.array(hls), cv2.COLOR_HLS2RGB_FULL)
        return self.diff
    
    def apply_diff(self, hls):
        return self.diff
    

class HLSEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['hls_red'] = HLSRedEffect()
        effecs['hls_orange'] = HLSOrangeEffect()
        effecs['hls_yellow'] = HLSYellowEffect()
        effecs['hls_green'] = HLSGreenEffect()
        effecs['hls_cyan'] = HLSCyanEffect()
        effecs['hls_blue'] = HLSBlueEffect()
        effecs['hls_purple'] = HLSPurpleEffect()
        effecs['hls_magenta'] = HLSMagentaEffect()
        self.hls_effects = effecs

    def reeffect(self):
        for n in self.hls_effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.hls_effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.hls_effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param):
        self.diff = pipeline.pipeline_hls(hls, self.hls_effects, param)

        return self.diff
    
    def apply_diff(self, hls):
        return self.diff

class CurveEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['tonecurve'] = TonecurveEffect()
        effecs['tonecurve_red'] = TonecurveRedEffect()
        effecs['tonecurve_green'] = TonecurveGreenEffect()
        effecs['tonecurve_blue'] = TonecurveBlueEffect()
        effecs['grading1'] = GradingEffect("1")
        effecs['grading2'] = GradingEffect("2")
        self.effects = effecs

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, rgb, param):
        self.diff = pipeline.pipeline_curve(rgb, self.effects, param)

        return self.diff
    
    def apply_diff(self, rgb):
        return self.diff

class VSandSaturationEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['HuevsHue'] = HuevsHueEffect()
        effecs['HuevsLum'] = HuevsLumEffect()
        effecs['LumvsLum'] = LumvsLumEffect()
        effecs['SatvsLum'] = SatvsLumEffect()
        effecs['HuevsSat'] = HuevsSatEffect()
        effecs['LumvsSat'] = LumvsSatEffect()
        effecs['SatvsSat'] = SatvsSatEffect()
        effecs['saturation'] = SaturationEffect()
        self.effects = effecs

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param):
        self.diff = pipeline.pipeline_vs_and_saturation(hls, self.effects, param)

        return self.diff
    
    def apply_diff(self, rgb):
        return self.diff

def create_effects():
    effects = [{}, {}, {}, {}, {}]

    lv0 = effects[0]
    #lv0['lens_modifier'] = LensModifierEffect()
    lv0['subpixel_shift'] = SubpixelShiftEffect()
    lv0['inpaint'] = InpaintEffect()
    lv0['rotation'] = RotationEffect()
    lv0['crop'] = CropEffect()

    lv1 = effects[1]
    lv1['ai_noise_reduction'] = AINoiseReductonEffect()
    lv1['nlm_noise_reduction'] = NLMNoiseReductionEffect()
    lv1['deblur_filter'] = DeblurFilterEffect()
    lv1['defocus'] = DefocusEffect()
    lv1['lensblur_filter'] = LensblurFilterEffect()
    lv1['glow'] = GlowEffect()
    
    lv2 = effects[2]
    lv2['color_temperature'] = ColorTemperatureEffect()
    lv2['dehaze'] = DehazeEffect()

    lv2['rgb2hls1'] = RGB2HLSEffect()
    lv2['hls'] = HLSEffect()
    lv2['hls2rgb1'] = HLS2RGBEffect()
 
    lv2['exposure'] = ExposureEffect()
    lv2['contrast'] = ContrastEffect()
    lv2['microcontrast'] = MicroContrastEffect()
    lv2['tone'] = ToneEffect()

    lv2['highlight_compress'] = HighlightCompressEffect()
    lv2['level'] = LevelEffect()
    lv2['clahe'] = CLAHEEffect()

    lv2['curve'] = CurveEffect()

    lv2['rgb2hls2'] = RGB2HLSEffect()
    lv2['vs_and_saturation'] = VSandSaturationEffect()
    lv2['hls2rgb2'] = HLS2RGBEffect()

    lv2['lut'] = LUTEffect()
    lv2['lens_simulator'] = LensSimulatorEffect()
    lv2['film_simulation'] = FilmSimulationEffect()

    lv3 = effects[3]
    lv3['mask2'] = Mask2Effect()
    lv3['correct_overexposed_areas'] = CorrectOverexposedAreas()

    lv4 = effects[4]
    lv4['vignette'] = VignetteEffect()

    return effects

def set2widget_all(widget, effects, param):
    for dict in effects:
        for l in dict.values():
            l.set2widget(widget, param)
            #l.set2param(param, self)
            l.reeffect()

def reeffect_all(effects):
    for dict in effects:
        for l in dict.values():
            l.reeffect()

def finalize_all(effects, param, widget):
    for dict in effects:
        for l in dict.values():
            l.finalize(param, widget)


import cv2
from cv2.gapi import crop
import numpy as np
import jax.numpy as jnp
import importlib
from enum import Enum

#import colorsys
#import skimage
#import DRBNet
#import iopaint.predict
#import dehazing.dehaze

import core
import cubelut
import mask_editor
import crop_editor
import subpixel_shift
import film_emulator
import lens_simulator
import config
import pipeline
import filter
import local_contrast
import params
import utils
import mediapipe_util
import distortion_painter

class EffectMode(Enum):
    PREVIEW = 0
    LOUPE = 1
    EXPORT = 2

class EffectConfig():

    def __init__(self, **kwargs):
        self.disp_info = None
        self.is_zoom = False
        self.mode = EffectMode.PREVIEW
        self.resolution_scale = 1.0

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
    def make_diff(self, img, param, efconfig):
        self.diff = img

    def apply_diff(self, img):
        if self.diff is not None:
            return self.diff
        return img

    def finalize(self, param, widget):
        pass

    def get_param_dict(self, param):
        return {}

    def get_param(self, param, key):
        return param.get(key, self.get_param_dict(param)[key])

    def delete_default_param(self, param):
        for p in self.get_param_dict(param).items():
            try:
                if param[p[0]] == p[1]:
                    del param[p[0]]
            except:
                pass

# レンズモディファイア
class LensModifierEffect(Effect):

    def get_param_dict(self, param):
        return {
            'lens_modifier': True,
            'color_modification': True,
            'subpixel_distortion': True,
            'geometry_distortion': True,
        }

    def set2widget(self, widget, param):
        widget.ids["checkbox_color_modification"].active = self.get_param(param, 'color_modification')
        widget.ids["checkbox_subpixel_distortion"].active = self.get_param(param, 'subpixel_distortion')
        widget.ids["checkbox_geometry_distortion"].active = self.get_param(param, 'geometry_distortion')

    def set2param(self, param, widget):
        param['color_modification'] = widget.ids["checkbox_color_modification"].active
        param['subpixel_distortion'] = widget.ids["checkbox_subpixel_distortion"].active
        param['geometry_distortion'] = widget.ids["checkbox_geometry_distortion"].active

    def make_diff(self, img, param, efconfig):
        lm = self.get_param(param, 'lens_modifier')
        cd = self.get_param(param, 'color_modification')
        sd = self.get_param(param, 'subpixel_distortion')
        gd = self.get_param(param, 'geometry_distortion')
        if lm == False or (cd == False and sd == False and gd == False):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((cd, sd, gd))
            if self.hash != param_hash:
                self.diff = core.modify_lensfun(img, cd, sd, gd)
                self.hash = param_hash
        
        return self.diff
    

# サブピクセルシフト合成
class SubpixelShiftEffect(Effect):

    def get_param_dict(self, param):
        return {
            'subpixel_shift': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_subpixel_shift"].active = False if self.get_param(param, 'subpixel_shift') == 0 else True

    def set2param(self, param, widget):
        param['subpixel_shift'] = 0 if widget.ids["switch_subpixel_shift"].active == False else 1

    def make_diff(self, img, param, efconfig):
        ss = self.get_param(param, 'subpixel_shift')
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
        self.disp_info = kwargs.get('disp_info', None)
        self.image = kwargs.get('image', None)

    def image2list(self):
        if type(self.image) is np.ndarray:
            self.image = utils.convert_image_to_list(self.image)
            #self.image = (self.image.shape, list(bz2.compress(self.image.tobytes(), 1)))

    def list2image(self):
        if type(self.image) is list or type(self.image) is tuple:
            self.image = utils.convert_image_from_list(self.image)
            #self.image = np.reshape(np.frombuffer(bz2.decompress(bytearray(self.image[1])), dtype=np.float32), self.image[0])

class InpaintEffect(Effect):
    __iopaint = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.inpaint_diff_list = []
        self.mask_editor = None

    def get_param_dict(self, param):
        return {
            'inpaint': 0,
            'inpaint_predict': 0,
            'inpaint_diff_list': [],
        }

    def set2widget(self, widget, param):
        widget.ids["switch_inpaint"].active = False if self.get_param(param, 'inpaint') == 0 else True
        widget.ids["button_inpaint_predict"].state = "normal" if self.get_param(param, 'inpaint_predict') == 0 else "down"

    def set2param(self, param, widget):
        param['inpaint'] = 0 if widget.ids["switch_inpaint"].active == False else 1
        param['inpaint_predict'] = 0 if widget.ids["button_inpaint_predict"].state == "normal" else 1

        if param['inpaint'] > 0:
            if self.mask_editor is None:
                self.mask_editor = mask_editor.MaskEditor(param['img_size'][0], param['img_size'][1])
                self.mask_editor.zoom = params.get_disp_info(param)[4]
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)
            
        if param['inpaint'] <= 0:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None

    def make_diff(self, img, param, efconfig):
        self.inpaint_diff_list = self.get_param(param, 'inpaint_diff_list')
        ip = self.get_param(param, 'inpaint')
        ipp = self.get_param(param, 'inpaint_predict')
        if (ip > 0 and ipp > 0) is True:
            param['inpaint_predict'] = 0 # なぜか二重起動するときがあるので予防

            if InpaintEffect.__iopaint is None:
                InpaintEffect.__iopaint = importlib.import_module('iopaint.predict')

            mask = cv2.GaussianBlur(self.mask_editor.get_mask(), (63, 63), 0)
            w, h = param['original_img_size']
            eh, ew = img.shape[:2]
            x, y = (ew-w)//2, (eh-h)//2
            img2 = InpaintEffect.__iopaint.predict(img[y:y+h, x:x+w], mask,
                            model=config.get_config('iopaint_model'),
                            resize_limit=config.get_config('iopaint_resize_limit'),
                            use_realesrgan=config.get_config('iopaint_use_realesrgan'))
            img2 = img2 #/ param.get('white_balance', [1, 1, 1])
            bboxes = core.get_multiple_mask_bbox(self.mask_editor.get_mask())
            for bbox in bboxes:
                self.inpaint_diff_list.append(InpaintDiff(disp_info=(bbox[0] + x, bbox[1] + y, bbox[2], bbox[3]), image=img2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]))
            param['inpaint_diff_list'] = self.inpaint_diff_list
            self.mask_editor.clear_mask()
            self.mask_editor.update_canvas()
        
        param_hash = hash((len(self.inpaint_diff_list)))
        if self.hash != param_hash:
            if len(self.inpaint_diff_list) > 0:
                img2 = img.copy()
                for inpaint_diff in self.inpaint_diff_list:
                    inpaint_diff.list2image()   # データを変換する必要があるときがある
                    cx, cy, cw, ch = inpaint_diff.disp_info
                    img2[cy:cy+ch, cx:cx+cw] = inpaint_diff.image
                self.diff = img2
            else:
                self.diff = None
            self.hash = param_hash

        return self.diff
    
# 画像回転、反転
class DistortionEffect(Effect):

    def __init__(self, distortion_callback=None, **kwargs):
        super().__init__(**kwargs)
        
        self.distortion_painter = None
        self.is_initial_open = 0
        self.is_initial_close = 0
        self.effect_type = 'forward_warp'
        self.set_distortion_callback(distortion_callback)

    def set_distortion_callback(self, callback):
        self.distortion_callback = callback

    def get_param_dict(self, param):
        return {
            'distortion_recorded': [],
        }    

    def set2widget(self, widget, param):
        pass

    def set2param(self, param, widget):
        distortion_enable = False if widget.ids["effects"].current_tab.text != "Liquify" else True

        # クロップエディタを開く
        if distortion_enable == True:
            self._open_distortion_painter(param, widget)

        # クロップエディタを閉じる
        elif distortion_enable == False:
            self._close_distortion_painter(param, widget)

        if self.distortion_painter is not None:
            self.distortion_painter.set_brush_size(widget.ids["slider_distortion_brush_size"].value)
            self.distortion_painter.set_strength(widget.ids["slider_distortion_strength"].value)

            # クロップ範囲をリセット
            if widget.ids["button_distortion_reset"].state == "down":
                widget.ids["button_distortion_reset"].state = "normal" # 無限ルーぷ防止
                self.distortion_painter.reset_image()


    def set2param2(self, param, arg):
        if self.distortion_painter is not None:
            self.distortion_painter.set_effect(arg)
            self.effect_type = arg

    def make_diff(self, img, param, efconfig):
        if self.is_initial_open > 0:
            if self.is_initial_open > 1:
                self.distortion_painter.set_effect(self.effect_type)
                self.distortion_painter.set_ref_image(img, True)
                self.distortion_painter.set_primary_param(param)
                self.distortion_painter.remap_recorded()
            else:
                self.distortion_painter.set_primary_param(param)
            self.is_initial_open -= 1

        if self.is_initial_close > 0:
            self.diff = None
            self.hash = None
            self.is_initial_close -= 1

        if self.distortion_painter is not None:
            if self.hash is not img:
                self.distortion_painter.set_ref_image(img, False)
                self.distortion_painter.remap_image()
                self.hash = img
            self.diff = 0 if self.diff is None else self.diff + 1
        
        else:
            dr = self.get_param(param, 'distortion_recorded')
            if len(dr) > 0:
                param_hash = hash((len(dr)))
                if self.hash != param_hash:
                    tcg_info = core.param_to_tcg_info(param)
                    self.diff = distortion_painter.DistortionCanvas.replay_recorded(img, param['distortion_recorded'], tcg_info)
                    self.hash = param_hash

        return self.diff

    def apply_diff(self, img):
        if self.diff is not None:
            if self.distortion_painter is not None:
                return self.distortion_painter.get_current_image()
            else:
                return self.diff
        return img

    def finalize(self, param, widget):
        self._close_distortion_painter(param, widget)

    def _open_distortion_painter(self, param, widget):
        if self.distortion_painter is None:
            self.distortion_painter = distortion_painter.DistortionCanvas(image_widget=widget.ids["preview"],
                    recorded=self.get_param(param, 'distortion_recorded'),
                    callback=self._painter_callback,
                    effect_type=self.effect_type,
                    brush_size=widget.ids["slider_distortion_brush_size"].value,
                    strength=widget.ids["slider_distortion_strength"].value)
            widget.ids["preview_widget"].add_widget(self.distortion_painter)
            self.is_initial_open = 2
            self.is_initial_close = 0

    def _close_distortion_painter(self, param, widget):
        if self.distortion_painter is not None:
            widget.ids["preview_widget"].remove_widget(self.distortion_painter)
            param['distortion_recorded'] = self.distortion_painter.get_recorded()
            self.distortion_painter = None
            self.is_initial_open = 0
            self.is_initial_close = 1

    def _painter_callback(self):
        if self.distortion_callback is not None:
            self.distortion_callback()

# 画像回転、反転
class RotationEffect(Effect):

    def get_param_dict(self, param):
        return {
            'rotation': 0,
            'rotation2': 0,
            'flip_mode': 0,
            'crop_enable': False,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_rotation"].set_slider_value(self.get_param(param, 'rotation'))

    def set2param(self, param, widget):
        param['rotation'] = widget.ids["slider_rotation"].value
        
    def set2param2(self, param, arg):
        if arg == 'hflip':
            param['flip_mode'] = self.get_param(param, 'flip_mode') ^ 1

        elif arg == 'vflip':
            param['flip_mode'] = self.get_param(param, 'flip_mode') ^ 2

        elif arg == 90:
            rot = self.get_param(param, 'rotation2') + 90.0
            if rot >= 90*4:
                rot = 0
            param['rotation2'] = rot

        elif arg == -90:
            rot = self.get_param(param, 'rotation2') - 90.0
            if rot < 0:
                rot = 90*3
            param['rotation2'] = rot

        else:
            pass

    def make_diff(self, img, param, efconfig):
        ang = self.get_param(param, 'rotation')
        ang2 = self.get_param(param, 'rotation2')
        flp = self.get_param(param, 'flip_mode')
        crop_enable = self.get_param(param, 'crop_enable')

        param_hash = hash((ang, ang2, flp, crop_enable))
        if self.hash != param_hash:
            self.diff = core.rotation(img, ang + ang2, flp,
                                        inter_mode=0 if efconfig.mode == EffectMode.EXPORT else 1,
                                        border_mode="refrect" if crop_enable == False else "constant")
            self.hash = param_hash
        
        return self.diff

# クロップ
class CropEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.crop_editor = None
        self.crop_editor_callback = None

    def set_editing_callback(self, callback):
        self.crop_editor_callback = callback

    def _param_to_aspect_ratio(self, param):
        ar = self.get_param(param, 'aspect_ratio')
        return eval(ar if ar != "None" else "0")

    def get_param_dict(self, param):
        param2 = param.copy()
        params.set_crop_rect(param2, core.get_initial_crop_rect(*param['original_img_size']))
        #params.set_disp_info(param2, core.get_initial_disp_info(*param['original_img_size'], config.get_config('preview_size')/max(param['original_img_size'])))
        return {
            'rotation': 0,
            'rotation2': 0,
            'crop_enable': False,
            'crop_rect': param2['crop_rect'],
            #'disp_info': param2['disp_info'],
            'aspect_ratio': "None",
        }

    def set2widget(self, widget, param):
        widget.ids["spinner_acpect_ratio"].text = param.get('aspect_ratio', "None")

    def set2param(self, param, widget):
        param['crop_enable'] = False if widget.ids["effects"].current_tab.text != "Geometry" else True
        param['aspect_ratio'] = widget.ids["spinner_acpect_ratio"].text

        # crop_rect がないのはマスク
        if params.get_crop_rect(param) is not None:

            # クロップエディタを開く
            if param['crop_enable'] == True:
                self._open_crop_editor(param, widget)

            # クロップエディタを閉じる
            elif param['crop_enable'] == False:
                self._close_crop_editor(param, widget)

            # クロップ範囲をリセット
            if self.crop_editor is not None:
                if widget.ids["button_crop_reset"].state == "down":
                    self.reset_crop_editor()

                self.reset2_crop_editor(param)


    def make_diff(self, img, param, efconfig):
        ce = self.get_param(param, 'crop_enable')
        disp_info = params.get_disp_info(param)

        if ce == True or disp_info is None:
            self.diff = None
            self.hash = None
            param['img_size'] = (param['original_img_size'][0], param['original_img_size'][1])
            msize = max(param['original_img_size'][0], param['original_img_size'][1])
            params.set_disp_info(param, (0, 0, msize, msize, disp_info[4]))
        else:
            param_hash = hash((ce))
            if self.hash != param_hash:
                self.diff = disp_info
                self.hash = param_hash
                param['img_size'] = (disp_info[2], disp_info[3])
        return self.diff
    
    def apply_diff(self, img):
        return img

    def _open_crop_editor(self, param, widget):
        if self.crop_editor is None:
            input_width, input_height = param['original_img_size']
            x1, y1, x2, y2 = params.get_crop_rect(param)
            scale = config.get_config('preview_size')/max(input_width, input_height)
            self.crop_editor = crop_editor.CropEditor(input_width=input_width, input_height=input_height, scale=scale, crop_rect=[x1, y1, x2, y2], aspect_ratio=self._param_to_aspect_ratio(param))
            self.crop_editor.set_editing_callback(self._crop_editing)
            widget.ids["preview_widget"].add_widget(self.crop_editor)

            # 編集中は一時的に変更
            params.set_disp_info(param, core.get_initial_disp_info(input_width, input_height, scale))

            # 保存しておく
            self.param = param

    def _close_crop_editor(self, param, widget):
        if self.crop_editor is not None:
            params.set_crop_rect(param, self.crop_editor.get_crop_rect())
            params.set_disp_info(param, self.crop_editor.get_disp_info())

            widget.ids["preview_widget"].remove_widget(self.crop_editor)
            self.crop_editor = None

    def _crop_editing(self):
        params.set_crop_rect(self.param, self.crop_editor.get_crop_rect())
        if self.crop_editor_callback is not None:
            self.crop_editor_callback()

    def reset_crop_editor(self):
        if self.crop_editor is not None:
            self.crop_editor._set_to_local_crop_rect((0, 0, 0, 0))
            self.crop_editor.update_crop_size()

    def reset2_crop_editor(self, param):
        if self.crop_editor is not None:
            self.crop_editor.input_angle = self.get_param(param, 'rotation') + self.get_param(param, 'rotation2')
            self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))

    def finalize(self, param, widget):
        self._close_crop_editor(param, widget)


# AI ノイズ除去
class AINoiseReductonEffect(Effect):
    __net = None
    __module = None

    def get_param_dict(self, param):
        return {
            'ai_noise_reduction': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_ai_noise_reduction"].active = False if self.get_param(param, 'ai_noise_reduction') == 0 else True

    def set2param(self, param, widget):
        param['ai_noise_reduction'] = 0 if widget.ids["switch_ai_noise_reduction"].active == False else 1

    def make_diff(self, img, param, efconfig):
        nr = self.get_param(param, 'ai_noise_reduction')
        if nr <= 0 or efconfig.mode == EffectMode.PREVIEW:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((nr))
            if self.hash != param_hash:
                if AINoiseReductonEffect.__module is None:
                    AINoiseReductonEffect.__module = importlib.import_module('SCUNet')
                if AINoiseReductonEffect.__net is None:
                    AINoiseReductonEffect.__net = AINoiseReductonEffect.__module.setup_model(device=config.get_config('gpu_device'))

                #img = np.clip(img, 0, 1)
                self.diff = AINoiseReductonEffect.__module.denoise_image_helper(AINoiseReductonEffect.__net, img, config.get_config('gpu_device'))
                self.hash = param_hash
        
        return self.diff


# BM3Dノイズ除去
class BM3DNoiseReductionEffect(Effect):
    __bm3d = None

    def get_param_dict(self, param):
        return {
            'bm3d_noise_reduction': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_bm3d_noise_reduction"].set_slider_value(self.get_param(param, 'bm3d_noise_reduction'))

    def set2param(self, param, widget):
        param['bm3d_noise_reduction'] = widget.ids["slider_bm3d_noise_reduction"].value

    def make_diff(self, img, param, efconfig):
        bm3d = int(self.get_param(param, 'bm3d_noise_reduction'))
        if bm3d == 0 or efconfig.disp_info[4] < config.get_config('scale_threshold'):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((bm3d))
            if self.hash != param_hash:
                if BM3DNoiseReductionEffect.__bm3d is None:
                    BM3DNoiseReductionEffect.__bm3d = importlib.import_module('bm3dcl')
                
                self.diff = BM3DNoiseReductionEffect.__bm3d.bm3d_denoise(img, bm3d/100.0 * efconfig.disp_info[4])
                self.hash = param_hash

        return self.diff

class LightNoiseReductionEffect(Effect):

    def get_param_dict(self, param):
        return {
            'light_noise_reduction': 0,
            'light_color_noise_reduction': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_light_noise_reduction"].set_slider_value(self.get_param(param, 'light_noise_reduction'))
        widget.ids["slider_light_color_noise_reduction"].set_slider_value(self.get_param(param, 'light_color_noise_reduction'))

    def set2param(self, param, widget):
        param['light_noise_reduction'] = widget.ids["slider_light_noise_reduction"].value
        param['light_color_noise_reduction'] = widget.ids["slider_light_color_noise_reduction"].value

    def make_diff(self, img, param, efconfig):
        its = int(self.get_param(param, 'light_noise_reduction'))
        col = int(self.get_param(param, 'light_color_noise_reduction'))
        if its == 0 and col == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((its, col))
            if self.hash != param_hash:  
                its = its * efconfig.disp_info[4]
                col = col * efconfig.disp_info[4]
                self.diff = core.light_denoise(img, its, col)
                #color_denise = core.denoise_like_lightroom(img, its, col)
                #denoise_img = core.lab_multiscale_denoise(color_denise, "bilateral")
                #self.diff = img * (1.0 - its) + denoise_img * its
                self.hash = param_hash

        return self.diff

# デブラーフィルタ
class DeblurFilterEffect(Effect):

    def get_param_dict(self, param):
        return {
            'deblur_filter': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_deblur_filter"].set_slider_value(self.get_param(param, 'deblur_filter'))

    def set2param(self, param, widget):
        param['deblur_filter'] = widget.ids["slider_deblur_filter"].value

    def make_diff(self, img, param, efconfig):
        dbfr = int(self.get_param(param, 'deblur_filter'))
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

    def get_param_dict(self, param):
        return {
            'defocus': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_defocus"].active = False if self.get_param(param, 'defocus') == 0 else True

    def set2param(self, param, widget):
        param['defocus'] = 0 if widget.ids["switch_defocus"].active == False else 1

    def make_diff(self, img, param, efconfig):
        df = self.get_param(param, 'defocus')
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

                self.diff = DefocusEffect.__DRBNet.predict(img, DefocusEffect.__net, config.get_config('gpu_device'))
                self.hash = param_hash

        return self.diff


class LensblurFilterEffect(Effect):

    def get_param_dict(self, param):
        return {
            'lensblur_filter': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_lensblur_filter"].set_slider_value(self.get_param(param, 'lensblur_filter'))

    def set2param(self, param, widget):
        param['lensblur_filter'] = widget.ids["slider_lensblur_filter"].value

    def make_diff(self, img, param, efconfig):
        lpfr = int(self.get_param(param, 'lensblur_filter'))
        if lpfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((lpfr))
            if self.hash != param_hash:
                self.diff = filter.lensblur_filter(img, int(round(lpfr-1) * 4 * efconfig.resolution_scale))
                self.hash = param_hash

        return self.diff

class ScratchEffect(Effect):

    def get_param_dict(self, param):
        return {
            'scratch': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_scratch"].set_slider_value(self.get_param(param, 'scratch'))

    def set2param(self, param, widget):
        param['scratch'] = widget.ids["slider_scratch"].value

    def make_diff(self, img, param, efconfig):
        fr = int(self.get_param(param, 'scratch'))
        if fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.diff = filter.scratch_effect(img, 1.0, fr / 100 * efconfig.resolution_scale)
                self.hash = param_hash

        return self.diff

class FrostedGlassEffect(Effect):

    def get_param_dict(self, param):
        return {
            'frosted_glass': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_frosted_glass"].set_slider_value(self.get_param(param, 'frosted_glass'))

    def set2param(self, param, widget):
        param['frosted_glass'] = widget.ids["slider_frosted_glass"].value

    def make_diff(self, img, param, efconfig):
        fr = int(self.get_param(param, 'frosted_glass'))
        if fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.diff = filter.frosted_glass_effect(img, fr / 100 * efconfig.resolution_scale, fr / 1000 * efconfig.resolution_scale)
                self.hash = param_hash

        return self.diff

class MosaicEffect(Effect):

    def get_param_dict(self, param):
        return {
            'mosaic': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_mosaic"].set_slider_value(self.get_param(param, 'mosaic'))

    def set2param(self, param, widget):
        param['mosaic'] = widget.ids["slider_mosaic"].value

    def make_diff(self, img, param, efconfig):
        fr = int(self.get_param(param, 'mosaic'))
        if fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.diff = filter.mosaic_effect(img, int(fr * efconfig.resolution_scale))
                self.hash = param_hash

        return self.diff

class GlowEffect(Effect):

    def get_param_dict(self, param):
        return {
            'glow_black': 0,
            'glow_gauss': 0,
            'glow_opacity': 0,
        }    

    def set2widget(self, widget, param):
        widget.ids["slider_glow_black"].set_slider_value(self.get_param(param, 'glow_black'))
        widget.ids["slider_glow_gauss"].set_slider_value(self.get_param(param, 'glow_gauss'))
        widget.ids["slider_glow_opacity"].set_slider_value(self.get_param(param, 'glow_opacity'))

    def set2param(self, param, widget):
        param['glow_black'] = widget.ids["slider_glow_black"].value
        param['glow_gauss'] = widget.ids["slider_glow_gauss"].value
        param['glow_opacity'] = widget.ids["slider_glow_opacity"].value

    def make_diff(self, rgb, param, efconfig):
        gb = self.get_param(param, 'glow_black')
        gg = int(self.get_param(param, 'glow_gauss'))
        go = self.get_param(param, 'glow_opacity')
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
                    rgb2 = filter.lensblur_filter(rgb2, (gg * 2 * efconfig.resolution_scale) | 1) 
                go = go/100.0
                self.diff = cv2.addWeighted(rgb, 1.0-go, core.blend_screen(rgb, rgb2), go, 0)
                self.hash = param_hash

        return self.diff

class FaceEffect(Effect):

    def get_param_dict(self, param):
        return {
            'jawline_scale': 0,
            'jaw_scale': 0,
            'left_eye_scale': 0,
            'right_eye_scale': 0,
            'lips_scale': 0,
        }    

    def set2widget(self, widget, param):
        widget.ids["slider_jawline_scale"].set_slider_value(self.get_param(param, 'jawline_scale'))
        widget.ids["slider_jaw_scale"].set_slider_value(self.get_param(param, 'jaw_scale'))
        widget.ids["slider_left_eye_scale"].set_slider_value(self.get_param(param, 'left_eye_scale'))
        widget.ids["slider_right_eye_scale"].set_slider_value(self.get_param(param, 'right_eye_scale'))
        widget.ids["slider_lips_scale"].set_slider_value(self.get_param(param, 'lips_scale'))

    def set2param(self, param, widget):
        param['jawline_scale'] = widget.ids["slider_jawline_scale"].value
        param['jaw_scale'] = widget.ids["slider_jaw_scale"].value
        param['left_eye_scale'] = widget.ids["slider_left_eye_scale"].value
        param['right_eye_scale'] = widget.ids["slider_right_eye_scale"].value
        param['lips_scale'] = widget.ids["slider_lips_scale"].value

    def make_diff(self, rgb, param, efconfig):
        jls = self.get_param(param, 'jawline_scale')
        js = self.get_param(param, 'jaw_scale')
        ls = self.get_param(param, 'left_eye_scale')
        rs = self.get_param(param, 'right_eye_scale')
        lipss = self.get_param(param, 'lips_scale')
        if ls == 0 and rs == 0 and jls == 0 and js == 0 and lipss == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((jls, js, ls, rs, lipss))
            if self.hash != param_hash:
                fms = mediapipe_util.setup_face_mesh(rgb)
                rgb = mediapipe_util.adjust_face_jawline(fms, rgb, jls/100, False) #efconfig.mode == EffectMode.PREVIEW)
                rgb = mediapipe_util.adjust_face_jaw(fms, rgb, js/100, False)
                rgb = mediapipe_util.adjust_left_eye(fms, rgb, ls/100, False)
                rgb = mediapipe_util.adjust_right_eye(fms, rgb, rs/100, False)
                rgb = mediapipe_util.adjust_lips(fms, rgb, lipss/100, False)
                mediapipe_util.clear_face_mesh(fms)
                self.diff = rgb
                self.hash = param_hash

        return self.diff

class ColorTemperatureEffect(Effect):

    def get_param_dict(self, param):
        return {
            'color_temperature_reset': 5000,
            'color_temperature': param.get('color_temperature_reset', 5000),
            'color_tint_reset': 0,
            'color_tint': param.get('color_tint_reset', 0),
            'color_Y': 1.0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_color_temperature"].set_slider_value(self.get_param(param, 'color_temperature'))
        widget.ids["slider_color_tint"].set_slider_value(self.get_param(param, 'color_tint'))
        widget.ids["slider_color_temperature"].set_slider_reset(self.get_param(param, 'color_temperature_reset'))
        widget.ids["slider_color_tint"].set_slider_reset(self.get_param(param, 'color_tint_reset'))
 
    def set2param(self, param, widget):
        param['color_temperature'] = widget.ids["slider_color_temperature"].value
        param['color_tint'] = widget.ids["slider_color_tint"].value

    @staticmethod
    def apply_color_temperature(rgb, param):
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        return rgb * core.invert_TempTint2RGB(temp, tint, Y, 5000)

    def make_diff(self, rgb, param, efconfig):
        temp = self.get_param(param, 'color_temperature')
        tint = self.get_param(param, 'color_tint')
        Y = self.get_param(param, 'color_Y')
        if False:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((temp, tint))
            if self.hash != param_hash:
                trgb = core.convert_TempTint2RGB(param['color_temperature_reset'], param['color_tint_reset'], self.get_param(param, 'color_Y'))
                self.diff = rgb * (trgb / core.convert_TempTint2RGB(temp, tint, Y))
                #self.diff = rgb * np.array(core.invert_TempTint2RGB(temp, tint, Y, 5000), dtype=np.float32)
                self.hash = param_hash

        return self.diff

class DehazeEffect(Effect):

    def get_param_dict(self, param):
        return {
            'dehaze': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_dehaze"].set_slider_value(self.get_param(param, 'dehaze'))

    def set2param(self, param, widget):
        param['dehaze'] = widget.ids["slider_dehaze"].value

    def make_diff(self, rgb, param, efconfig):
        de = self.get_param(param, 'dehaze')
        if de == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((de))
            if self.hash != param_hash:
                self.diff = core.dehaze_image(rgb, de/100)
                self.hash = param_hash

        return self.diff

class RGB2HLSEffect(Effect):

    def make_diff(self, rgb, param, efconfig):
        if self.diff is None:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
            #self.diff = hlsrgb.rgb_to_hls(np.array(rgb))
        return self.diff

class HLS2RGBEffect(Effect):

    def make_diff(self, hls, param, efconfig):
        if self.diff is None:
            hls = core.type_convert(hls, np.ndarray)
            self.diff = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
            #self.diff = hlsrgb.hls_to_rgb(np.array(hls))
        return self.diff

class HLSEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['hls_red'] = HLSColorEffect('red')
        effecs['hls_orange'] = HLSColorEffect('orange')
        effecs['hls_yellow'] = HLSColorEffect('yellow')
        effecs['hls_green'] = HLSColorEffect('green')
        effecs['hls_cyan'] = HLSColorEffect('cyan')
        effecs['hls_blue'] = HLSColorEffect('blue')
        effecs['hls_purple'] = HLSColorEffect('purple')
        effecs['hls_magenta'] = HLSColorEffect('magenta')
        self.hls_effects = effecs

    def delete_default_param(self, param):
        for n in self.hls_effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        for n in self.hls_effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.hls_effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.hls_effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param, efconfig):
        self.diff = pipeline.pipeline_hls(hls, self.hls_effects, param, efconfig)

        return self.diff
    
class HLSColorEffect(Effect):

    def __init__(self, color_name, **kwargs):
        super().__init__(**kwargs)

        self.color_name = color_name
    
    def get_param_dict(self, param):
        return {
            "hls_" + self.color_name + "_hue": 0,
            "hls_" + self.color_name + "_lum": 0,
            "hls_" + self.color_name + "_sat": 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_hls_" + self.color_name + "_hue"].set_slider_value(self.get_param(param, "hls_" + self.color_name + "_hue"))
        widget.ids["slider_hls_" + self.color_name + "_lum"].set_slider_value(self.get_param(param, "hls_" + self.color_name + "_lum"))
        widget.ids["slider_hls_" + self.color_name + "_sat"].set_slider_value(self.get_param(param, "hls_" + self.color_name + "_sat"))

    def set2param(self, param, widget):
        param["hls_" + self.color_name + "_hue"] = widget.ids["slider_hls_" + self.color_name + "_hue"].value
        param["hls_" + self.color_name + "_lum"] = widget.ids["slider_hls_" + self.color_name + "_lum"].value
        param["hls_" + self.color_name + "_sat"] = widget.ids["slider_hls_" + self.color_name + "_sat"].value

    def make_diff(self, hls, param, efconfig):
        hue = self.get_param(param, "hls_" + self.color_name + "_hue")
        lum = self.get_param(param, "hls_" + self.color_name + "_lum")
        sat = self.get_param(param, "hls_" + self.color_name + "_sat")
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = core.adjust_hls_color_one(hls, self.color_name, hue, lum/100, sat/100, efconfig.resolution_scale) - hls
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, hls):
        return hls + self.diff

class ExposureEffect(Effect):

    def get_param_dict(self, param):
        return {
            'exposure': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_exposure"].set_slider_value(self.get_param(param, 'exposure'))

    def set2param(self, param, widget):
        param['exposure'] = widget.ids["slider_exposure"].value

    def make_diff(self, rgb, param, efconfig):
        ev = self.get_param(param, 'exposure')
        param_hash = hash((ev))
        if ev == 0:
            self.diff = None
            self.hash = None
        
        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = core.adjust_exposure(rgb, ev)
            self.hash = param_hash

        return self.diff
    
class ContrastEffect(Effect):

    def get_param_dict(self, param):
        return {
            'contrast': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_contrast"].set_slider_value(self.get_param(param, 'contrast'))

    def set2param(self, param, widget):
        param['contrast'] = widget.ids["slider_contrast"].value

    def make_diff(self, rgb, param, efconfig):
        con = self.get_param(param, 'contrast')
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, jnp.ndarray)
            self.diff, _ = core.adjust_tone(rgb, con, -con)
            self.hash = param_hash

        return self.diff

class ClarityEffect(Effect):

    def get_param_dict(self, param):
        return {
            'clarity': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_clarity"].set_slider_value(self.get_param(param, 'clarity'))

    def set2param(self, param, widget):
        param['clarity'] = widget.ids["slider_clarity"].value

    def make_diff(self, rgb, param, efconfig):
        con = self.get_param(param, 'clarity')
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = local_contrast.apply_clarity_luminance(rgb, con * 2 * efconfig.resolution_scale)
            self.hash = param_hash

        return self.diff

class TextureEffect(Effect):

    def get_param_dict(self, param):
        return {
            'texture': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_texture"].set_slider_value(self.get_param(param, 'texture'))

    def set2param(self, param, widget):
        param['texture'] = widget.ids["slider_texture"].value

    def make_diff(self, rgb, param, efconfig):
        con = self.get_param(param, 'texture')
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = local_contrast.apply_texture_advanced(rgb, con * 0.5 * efconfig.resolution_scale)
            self.hash = param_hash

        return self.diff
    
class MicroContrastEffect(Effect):

    def get_param_dict(self, param):
        return {
            'microcontrast': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_microcontrast"].set_slider_value(self.get_param(param, 'microcontrast'))

    def set2param(self, param, widget):
        param['microcontrast'] = widget.ids["slider_microcontrast"].value

    def make_diff(self, rgb, param, efconfig):
        con = self.get_param(param, 'microcontrast')
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = local_contrast.apply_microcontrast(rgb, con * 0.5 * efconfig.resolution_scale)
            self.hash = param_hash

        return self.diff
    
class ToneEffect(Effect):

    def get_param_dict(self, param):
        return {
            'shadow': 0,
            'highlight': 0,
            'midtone': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_shadow"].set_slider_value(self.get_param(param, 'shadow'))
        widget.ids["slider_highlight"].set_slider_value(self.get_param(param, 'highlight'))
        widget.ids["slider_midtone"].set_slider_value(self.get_param(param, 'midtone'))

    def set2param(self, param, widget):
        param['shadow'] = widget.ids["slider_shadow"].value
        param['highlight'] = widget.ids["slider_highlight"].value
        param['midtone'] = widget.ids["slider_midtone"].value

    def make_diff(self, rgb, param, efconfig):
        shadow = self.get_param(param, 'shadow')
        highlight = self.get_param(param, 'highlight')
        mt = self.get_param(param, 'midtone')
        param_hash = hash((shadow, highlight, mt))
        if shadow == 0 and highlight == 0 and mt == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, jnp.ndarray)
            self.diff, masks = core.adjust_tone(rgb, highlight, shadow, mt, 0, 0)
            """
            if masks[0] is not None:
                mask = np.array(masks[0])
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = 1 - mask
                threshold = np.max(mask)
                mask[mask > threshold / 2] = 0.0
                source = np.array(self.diff)
                #cv2.imwrite("mask.jpg", cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                #cv2.imwrite("mask.jpg", (mask * 255).astype(np.uint8))
                target = local_contrast.apply_microcontrast(source, 150)
                mask = mask[..., np.newaxis]
                self.diff = source * (1-mask) + target * mask
            """
            if masks[1] is not None:
                mask = np.array(masks[1])
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                threshold = np.max(mask)
                mask[mask < threshold * 3 / 4] = 0.0
                source = np.array(self.diff)
                #cv2.imwrite("mask.jpg", cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                #cv2.imwrite("mask.jpg", (mask * 255).astype(np.uint8))
                target = local_contrast.apply_microcontrast(source, 200 * efconfig.resolution_scale)
                mask = mask[..., np.newaxis]
                self.diff = source * (1-mask) + target * mask

            self.hash = param_hash
        return self.diff
    
class HighlightCompressEffect(Effect):

    def get_param_dict(self, param):
        return {
            'highlight_compress': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_highlight_compress"].active = True if self.get_param(param, 'highlight_compress') == 1 else False

    def set2param(self, param, widget):
        param['highlight_compress'] = 1 if widget.ids["switch_highlight_compress"].active == True else 0

    def make_diff(self, rgb, param, efconfig):
        hc = self.get_param(param, 'highlight_compress')
        if hc <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((hc))
            if self.hash != param_hash:
                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.highlight_compress(rgb)
                self.hash = param_hash

        return self.diff

class LevelEffect(Effect):

    def get_param_dict(self, param):
        return {
            'black_level': 0,
            'white_level': 255,
            'mid_level': 127,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_black_level"].set_slider_value(self.get_param(param, 'black_level'))
        widget.ids["slider_white_level"].set_slider_value(self.get_param(param, 'white_level'))
        widget.ids["slider_mid_level"].set_slider_value(self.get_param(param, 'mid_level'))

    def set2param(self, param, widget):
        bl = widget.ids["slider_black_level"].value
        wl = widget.ids["slider_white_level"].value
        ml = widget.ids["slider_mid_level"].value
        param['black_level'] = bl
        param['white_level'] = wl
        param['mid_level'] = ml

    def make_diff(self, rgb, param, efconfig):
        bl = self.get_param(param, 'black_level')
        wl = self.get_param(param, 'white_level')
        ml = self.get_param(param, 'mid_level')
        param_hash = hash((bl, wl, ml))
        if bl == 0 and wl == 255 and ml == 127:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            rgb = core.type_convert(rgb, jnp.ndarray)
            self.diff = core.apply_level_adjustment(rgb, bl, ml, wl)
            self.hash = param_hash

        return self.diff
    
class CLAHEEffect(Effect):

    def get_param_dict(self, param):
        return {
            'clahe_intensity': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_clahe_intensity"].set_slider_value(self.get_param(param, 'clahe_intensity'))

    def set2param(self, param, widget):
        param['clahe_intensity'] = widget.ids["slider_clahe_intensity"].value

    def make_diff(self, img, param, efconfig):
        ci = self.get_param(param, 'clahe_intensity')
        if ci <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((ci))
            if self.hash != param_hash:
                img = core.type_convert(img, np.ndarray)
                clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
                target = np.empty_like(img, dtype=np.uint16)
                img2 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
                for i in range(3):
                    target[..., i] = clahe.apply(img2[..., i])
                target = target.astype(np.float32) / 65535
                ci = ci / 100
                self.diff = target * ci + img * (1-ci)
                self.hash = param_hash

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

    def delete_default_param(self, param):
        for n in self.effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, rgb, param, efconfig):
        self.diff = pipeline.pipeline_curve(rgb, self.effects, param, efconfig)

        return self.diff
    
class TonecurveEffect(Effect):

    def get_param_dict(self, param):
        return {
            'tonecurve': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve"].set_point_list(self.get_param(param, 'tonecurve'))

    def set2param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_point_list()

    def make_diff(self, rgb, param, efconfig):
        pl = self.get_param(param, 'tonecurve')
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
        rgb =  core.type_convert(rgb, jnp.ndarray)
        return core.apply_lut(rgb, self.diff)

class TonecurveRedEffect(Effect):

    def get_param_dict(self, param):
        return {
            'tonecurve_red': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve_red"].set_point_list(self.get_param(param, 'tonecurve_red'))

    def set2param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_point_list()

    def make_diff(self, rgb_r, param, efconfig):
        pl = self.get_param(param, 'tonecurve_red')
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
        rgb_r =  core.type_convert(rgb_r, jnp.ndarray)
        return core.apply_lut(rgb_r, self.diff)

class TonecurveGreenEffect(Effect):

    def get_param_dict(self, param):
        return {
            'tonecurve_green': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve_green"].set_point_list(self.get_param(param, 'tonecurve_green'))

    def set2param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_point_list()

    def make_diff(self, rgb_g, param, efconfig):   
        pl = self.get_param(param, 'tonecurve_green')
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
        rgb_g =  core.type_convert(rgb_g, jnp.ndarray)
        return core.apply_lut(rgb_g, self.diff)

class TonecurveBlueEffect(Effect):

    def get_param_dict(self, param):
        return {
            'tonecurve_blue': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve_blue"].set_point_list(self.get_param(param, 'tonecurve_blue'))

    def set2param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_point_list()

    def make_diff(self, rgb_b, param, efconfig):
        pl = self.get_param(param, 'tonecurve_blue')
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_b):
        rgb_b =  core.type_convert(rgb_b, jnp.ndarray)
        return core.apply_lut(rgb_b, self.diff)

class GradingEffect(Effect):
    __colorsys = None

    def __init__(self, numstr, **kwargs):
        super().__init__(**kwargs)

        self.numstr = numstr

    def get_param_dict(self, param):
        return {
            'grading' + self.numstr: None,
            'grading' + self.numstr + '_hue': 0,
            'grading' + self.numstr + '_lum': 50,
            'grading' + self.numstr + '_sat': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["grading" + self.numstr].set_point_list(self.get_param(param, 'grading' + self.numstr))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].set_slider_value(self.get_param(param, 'grading' + self.numstr + '_hue'))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].set_slider_value(self.get_param(param, 'grading' + self.numstr + '_lum'))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].set_slider_value(self.get_param(param, 'grading' + self.numstr + '_sat'))

    def set2param(self, param, widget):
        param["grading" + self.numstr] = widget.ids["grading" + self.numstr].get_point_list(True)
        param["grading" + self.numstr + "_hue"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].value
        param["grading" + self.numstr + "_lum"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].value
        param["grading" + self.numstr + "_sat"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].value

    def make_diff(self, rgb, param, efconfig):
        pl = self.get_param(param, "grading" + self.numstr)
        gh = self.get_param(param, "grading" + self.numstr + "_hue")
        gl = self.get_param(param, "grading" + self.numstr + "_lum")
        gs = self.get_param(param, "grading" + self.numstr + "_sat")
        if gh == 0 and gl == 50 and gs == 0:
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
        rgb = core.type_convert(rgb, jnp.ndarray)
        gray = core.cvtColorRGB2Gray(rgb)
        blend = core.apply_lut(gray, lut)
        blend = jnp.array(blend)
        return core.apply_mask(rgb, blend, rgb * rgbs)

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

    def delete_default_param(self, param):
        for n in self.effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param, efconfig):
        self.diff = pipeline.pipeline_vs_and_saturation(hls, self.effects, param, efconfig)

        return self.diff
    
class HuevsHueEffect(Effect):

    def get_param_dict(self, param):
        return {
            'HuevsHue': None,
        }

    def set2widget(self, widget, param):
        widget.ids["HuevsHue"].set_point_list(self.get_param(param, 'HuevsHue'))

    def set2param(self, param, widget):
        param['HuevsHue'] = widget.ids["HuevsHue"].get_point_list()

    def make_diff(self, hls_h, param, efconfig):
        hh = self.get_param(param, "HuevsHue")
        if hh is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hh))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hh)
                self.diff = ((lut - 0.5) * 2.0) * 360
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_h):
        hls_h = core.type_convert(hls_h, jnp.ndarray)
        return core.apply_lut(hls_h / 360, self.diff, 1.0) + hls_h

class HuevsLumEffect(Effect):

    def get_param_dict(self, param):
        return {
            'HuevsLum': None,
        }

    def set2widget(self, widget, param):
        widget.ids["HuevsLum"].set_point_list(self.get_param(param, 'HuevsLum'))

    def set2param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_point_list()

    def make_diff(self, hls_l, param, efconfig):
        hl = self.get_param(param, "HuevsLum")
        if hl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hl))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hl)
                self.diff = 2.0 ** ((lut - 0.5) * 4.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_hl):
        hls_hl = core.type_convert(hls_hl, jnp.ndarray)
        return core.apply_lut(hls_hl[0] / 360, self.diff, 1.0) * hls_hl[1]

class HuevsSatEffect(Effect):

    def get_param_dict(self, param):
        return {
            'HuevsSat': None,
        }

    def set2widget(self, widget, param):
        widget.ids["HuevsSat"].set_point_list(self.get_param(param, 'HuevsSat'))

    def set2param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_point_list()

    def make_diff(self, hls_s, param, efconfig):
        hs = self.get_param(param, "HuevsSat")
        if hs is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hs))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hs)
                self.diff = (lut - 0.5) * 2.0 + 1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_hs):
        hls_hs = core.type_convert(hls_hs, jnp.ndarray)
        return core.apply_lut(hls_hs[0] / 360.0, self.diff, 1.0) * hls_hs[1]

class LumvsLumEffect(Effect):

    def get_param_dict(self, param):
        return {
            'LumvsLum': None,
        }

    def set2widget(self, widget, param):
        widget.ids["LumvsLum"].set_point_list(self.get_param(param, 'LumvsLum'))

    def set2param(self, param, widget):
        param['LumvsLum'] = widget.ids["LumvsLum"].get_point_list()

    def make_diff(self, hls_l, param, efconfig):
        ll = self.get_param(param, "LumvsLum")
        if ll is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ll))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(ll)
                self.diff = 2.0 ** ((lut - 0.5) * 4.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        hls_l = core.type_convert(hls_l, jnp.ndarray)
        return core.apply_lut(hls_l, self.diff, 1.0) * hls_l

class LumvsSatEffect(Effect):

    def get_param_dict(self, param):
        return {
            'LumvsSat': None,
        }

    def set2widget(self, widget, param):
        widget.ids["LumvsSat"].set_point_list(self.get_param(param, 'LumvsSat'))

    def set2param(self, param, widget):
        param['LumvsSat'] = widget.ids["LumvsSat"].get_point_list()

    def make_diff(self, hls_l, param, efconfig):
        ls = self.get_param(param, "LumvsSat")
        if ls is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ls))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(ls)
                self.diff = (lut - 0.5) * 2.0 + 1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_ls):
        hls_ls = core.type_convert(hls_ls, jnp.ndarray)
        return core.apply_lut(hls_ls[0], self.diff, 1.0) * hls_ls[1]

class SatvsLumEffect(Effect):

    def get_param_dict(self, param):
        return {
            'SatvsLum': None,
        }

    def set2widget(self, widget, param):
        widget.ids["SatvsLum"].set_point_list(self.get_param(param, 'SatvsLum'))

    def set2param(self, param, widget):
        param['SatvsLum'] = widget.ids["SatvsLum"].get_point_list()

    def make_diff(self, hls_s, param, efconfig):
        sl = self.get_param(param, "SatvsLum")
        if sl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(sl))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(sl)
                self.diff = 2.0 ** ((lut - 0.5) * 4.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_sl):
        hls_sl = core.type_convert(hls_sl, jnp.ndarray)
        return core.apply_lut(hls_sl[0], self.diff, 1.0) * hls_sl[1]

class SatvsSatEffect(Effect):

    def get_param_dict(self, param):
        return {
            'SatvsSat': None,
        }

    def set2widget(self, widget, param):
        widget.ids["SatvsSat"].set_point_list(self.get_param(param, 'SatvsSat'))

    def set2param(self, param, widget):
        param['SatvsSat'] = widget.ids["SatvsSat"].get_point_list()

    def make_diff(self, hls_s, param, efconfig):
        ss = self.get_param(param, "SatvsSat")
        if ss is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ss))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(ss)
                self.diff = (lut - 0.5) * 2.0 + 1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        hls_s = core.type_convert(hls_s, jnp.ndarray)
        return core.apply_lut(hls_s, self.diff, 1.0) * hls_s

class SaturationEffect(Effect):

    def get_param_dict(self, param):
        return {
            'saturation': 0,
            'vibrance': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_saturation"].set_slider_value(self.get_param(param, 'saturation'))
        widget.ids["slider_vibrance"].set_slider_value(self.get_param(param, 'vibrance'))

    def set2param(self, param, widget):
        param['saturation'] = widget.ids["slider_saturation"].value
        param['vibrance'] = widget.ids["slider_vibrance"].value

    def make_diff(self, hls_s, param, efconfig):
        sat = self.get_param(param, 'saturation')
        vib = self.get_param(param, 'vibrance')
        param_hash = hash((sat, vib))
        if sat == 0 and vib == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls_s = core.type_convert(hls_s, np.ndarray)
            hls2_s = core.calc_saturation(hls_s, sat, vib)
            self.diff = np.divide(hls2_s, hls_s, where=hls_s!=0.0)    # Sのみ保存
            self.hash = param_hash
        
        return self.diff
    
    def apply_diff(self, hls_s):
        return hls_s * self.diff

class LUTEffect(Effect):
    file_pathes = { '---': None }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lut = None

    def get_param_dict(self, param):
        return {
            'lut_name': 'None',
            'lut_path': None,
        }

    def set2widget(self, widget, param):
        widget.ids["lut_spinner"].text = self.get_param(param, 'lut_name')

    def set2param(self, param, widget):
        spinner = widget.ids["lut_spinner"]
        name = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        if self.get_param(param, 'lut_name') != name:
            self.lut = None
        param['lut_name'] = name
        param['lut_path'] = LUTEffect.file_pathes.get(param['lut_name'], None)

    def make_diff(self, rgb, param, efconfig):
        lut_path = self.get_param(param, 'lut_path')
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
        rgb = core.type_convert(rgb, np.ndarray)
        return cubelut.process_image(rgb, self.diff)

class LensSimulatorEffect(Effect):

    def get_param_dict(self, param):
        return {
            'lens_preset': 'None',
            'lens_intensity': 100,
        }
 
    def set2widget(self, widget, param):
        widget.ids["spinner_lens_preset"].text = self.get_param(param, 'lens_preset')
        widget.ids["slider_lens_intensity"].set_slider_value(self.get_param(param, 'lens_intensity'))

    def set2param(self, param, widget):
        spinner = widget.ids["spinner_lens_preset"]
        param['lens_preset'] = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        param['lens_intensity'] = widget.ids["slider_lens_intensity"].value

    def make_diff(self, rgb, param, efconfig):
        preset = self.get_param(param, 'lens_preset')
        intensity = self.get_param(param, 'lens_intensity')
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
        rgb = core.type_convert(rgb, np.ndarray)
        lens = lens_simulator.process_image(rgb, self.diff[0])
        #lens = lens_simulator.apply_old_lens_effect(rgb, self.diff[0])
        per = self.diff[1] / 100.0
        return lens * per + rgb * (1-per)

class FilmSimulationEffect(Effect):

    def get_param_dict(self, param):
        return {
            'film_preset': 'None',
            'film_intensity': 100,
            'film_expired': 0,
        }
 
    def set2widget(self, widget, param):
        widget.ids["spinner_film_preset"].text = self.get_param(param, 'film_preset')
        widget.ids["slider_film_intensity"].set_slider_value(self.get_param(param, 'film_intensity'))
        widget.ids["slider_film_expired"].set_slider_value(self.get_param(param, 'film_expired'))

    def set2param(self, param, widget):
        spinner = widget.ids["spinner_film_preset"]
        param['film_preset'] = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        param['film_intensity'] = widget.ids["slider_film_intensity"].value
        param['film_expired'] = widget.ids["slider_film_expired"].value

    def make_diff(self, rgb, param, efconfig):
        preset = self.get_param(param, 'film_preset')
        intensity = self.get_param(param, 'film_intensity')
        expired = self.get_param(param, 'film_expired')
        if preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity, expired))
            if self.hash != param_hash:
                self.diff = (preset, intensity, expired)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        rgb = core.type_convert(rgb, np.ndarray)
        film = film_emulator.emulator.apply_film_effect(rgb, self.diff[0], self.diff[2])
        per = self.diff[1] / 100.0
        return film * per + rgb * (1-per)


class SolidColorEffect(Effect):

    def get_param_dict(self, param):
        return {
            'solid_color': 0,
            'solid_color_red': 0,
            'solid_color_green': 0,
            'solid_color_blue': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_solid_color"].active = False if self.get_param(param, 'solid_color') == 0 else True
        widget.ids["cp_solid_color"].ids['slider_red'].set_slider_value(self.get_param(param, 'solid_color_red'))
        widget.ids["cp_solid_color"].ids['slider_green'].set_slider_value(self.get_param(param, 'solid_color_green'))
        widget.ids["cp_solid_color"].ids['slider_blue'].set_slider_value(self.get_param(param, 'solid_color_blue'))

    def set2param(self, param, widget):
        param['solid_color'] = 0 if widget.ids["switch_solid_color"].active == False else 1
        param["solid_color_red"] = widget.ids["cp_solid_color"].ids['slider_red'].value
        param["solid_color_green"] = widget.ids["cp_solid_color"].ids['slider_green'].value
        param["solid_color_blue"] = widget.ids["cp_solid_color"].ids['slider_blue'].value

    def make_diff(self, rgb, param, efconfig):
        coa = self.get_param(param, 'solid_color')
        coar = self.get_param(param, "solid_color_red")
        coag = self.get_param(param, "solid_color_green")
        coab = self.get_param(param, "solid_color_blue")
        if coa <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((coa, coar, coag, coab))
            if self.hash != param_hash:
                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.apply_solid_color(rgb, solid_color=(coar/255, coag/255, coab/255))
                self.hash = param_hash

        return self.diff

class UnsharpMaskEffect(Effect):

    def get_param_dict(self, param):
        return {
            'unsharp_mask_amount': 0,
            'unsharp_mask_sigma': 50,
        }
 
    def set2widget(self, widget, param):
        widget.ids["slider_unsharp_mask_amount"].set_slider_value(self.get_param(param, 'unsharp_mask_amount'))
        widget.ids["slider_unsharp_mask_sigma"].set_slider_value(self.get_param(param, 'unsharp_mask_sigma'))

    def set2param(self, param, widget):
        param['unsharp_mask_amount'] = widget.ids["slider_unsharp_mask_amount"].value
        param['unsharp_mask_sigma'] = widget.ids["slider_unsharp_mask_sigma"].value

    def make_diff(self, rgb, param, efconfig):
        amount = self.get_param(param, 'unsharp_mask_amount')
        sigma = self.get_param(param, 'unsharp_mask_sigma')
        if amount <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((amount, sigma))
            if self.hash != param_hash:
                rgb = core.type_convert(rgb, np.ndarray)
                amount = amount / 100.0 * 1.5
                sigma = sigma / 100.0 * 3.0
                self.diff = core.unsharp_mask(rgb, amount, sigma)
                self.hash = param_hash

        return self.diff


class Mask2Effect(Effect):

    def get_param_dict(self, param):
        return {
            'mask2_depth_min': 0,
            'mask2_depth_max': 255,
            'mask2_hue_distance': 179,
            'mask2_hue_min': 0,
            'mask2_hue_max': 359,
            'mask2_lum_distance': 127,
            'mask2_lum_min': 0,
            'mask2_lum_max': 255,
            'mask2_sat_distance': 127,
            'mask2_sat_min': 0,
            'mask2_sat_max': 255,
            'mask2_blur': 0,
            'mask2_face_face': True,
            'mask2_face_brows': True,
            'mask2_face_eyes': True,
            'mask2_face_nose': True,
            'mask2_face_mouth': True,
            'mask2_face_lips': True,
            'mask2_open_space': 0,
            'mask2_close_space': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_mask2_depth_min"].set_slider_value(self.get_param(param, 'mask2_depth_min'))
        widget.ids["slider_mask2_depth_max"].set_slider_value(self.get_param(param, 'mask2_depth_max'))
        widget.ids["slider_mask2_hue_distance"].set_slider_value(self.get_param(param, 'mask2_hue_distance'))
        widget.ids["slider_mask2_hue_min"].set_slider_value(self.get_param(param, 'mask2_hue_min'))
        widget.ids["slider_mask2_hue_max"].set_slider_value(self.get_param(param, 'mask2_hue_max'))
        widget.ids["slider_mask2_lum_distance"].set_slider_value(self.get_param(param, 'mask2_lum_distance'))
        widget.ids["slider_mask2_lum_min"].set_slider_value(self.get_param(param, 'mask2_lum_min'))
        widget.ids["slider_mask2_lum_max"].set_slider_value(self.get_param(param, 'mask2_lum_max'))
        widget.ids["slider_mask2_sat_distance"].set_slider_value(self.get_param(param, 'mask2_sat_distance'))
        widget.ids["slider_mask2_sat_min"].set_slider_value(self.get_param(param, 'mask2_sat_min'))
        widget.ids["slider_mask2_sat_max"].set_slider_value(self.get_param(param, 'mask2_sat_max'))
        widget.ids["slider_mask2_blur"].set_slider_value(self.get_param(param, 'mask2_blur'))
        widget.ids["checkbox_mask2_face_face"].active = self.get_param(param, 'mask2_face_face')
        widget.ids["checkbox_mask2_face_brows"].active = self.get_param(param, 'mask2_face_brows')
        widget.ids["checkbox_mask2_face_eyes"].active = self.get_param(param, 'mask2_face_eyes')
        widget.ids["checkbox_mask2_face_nose"].active = self.get_param(param, 'mask2_face_nose')
        widget.ids["checkbox_mask2_face_mouth"].active = self.get_param(param, 'mask2_face_mouth')
        widget.ids["checkbox_mask2_face_lips"].active = self.get_param(param, 'mask2_face_lips')
        widget.ids["slider_mask2_open_space"].set_slider_value(self.get_param(param, 'mask2_open_space'))
        widget.ids["slider_mask2_close_space"].set_slider_value(self.get_param(param, 'mask2_close_space'))

    def set2param(self, param, widget):
        param['mask2_depth_min'] = widget.ids["slider_mask2_depth_min"].value
        param['mask2_depth_max'] = widget.ids["slider_mask2_depth_max"].value
        param['mask2_hue_distance'] = widget.ids["slider_mask2_hue_distance"].value
        param['mask2_hue_min'] = widget.ids["slider_mask2_hue_min"].value
        param['mask2_hue_max'] = widget.ids["slider_mask2_hue_max"].value
        param['mask2_lum_distance'] = widget.ids["slider_mask2_lum_distance"].value
        param['mask2_lum_min'] = widget.ids["slider_mask2_lum_min"].value
        param['mask2_lum_max'] = widget.ids["slider_mask2_lum_max"].value
        param['mask2_sat_distance'] = widget.ids["slider_mask2_sat_distance"].value
        param['mask2_sat_min'] = widget.ids["slider_mask2_sat_min"].value
        param['mask2_sat_max'] = widget.ids["slider_mask2_sat_max"].value
        param['mask2_blur'] = widget.ids["slider_mask2_blur"].value
        param['mask2_face_face'] = widget.ids["checkbox_mask2_face_face"].active
        param['mask2_face_brows'] = widget.ids["checkbox_mask2_face_brows"].active
        param['mask2_face_eyes'] = widget.ids["checkbox_mask2_face_eyes"].active
        param['mask2_face_nose'] = widget.ids["checkbox_mask2_face_nose"].active
        param['mask2_face_mouth'] = widget.ids["checkbox_mask2_face_mouth"].active
        param['mask2_face_lips'] = widget.ids["checkbox_mask2_face_lips"].active
        param['mask2_open_space'] = widget.ids["slider_mask2_open_space"].value
        param['mask2_close_space'] = widget.ids["slider_mask2_close_space"].value

    def make_diff(self, rgb, param, efconfig):
        dmin = self.get_param(param, 'mask2_depth_min')
        dmax = self.get_param(param, 'mask2_depth_max')
        hdis = self.get_param(param, 'mask2_hue_distance')
        hmin = self.get_param(param, 'mask2_hue_min')
        hmax = self.get_param(param, 'mask2_hue_max')
        ldis = self.get_param(param, 'mask2_lum_distance')
        lmin = self.get_param(param, 'mask2_lum_min')
        lmax = self.get_param(param, 'mask2_lum_max')
        sdis = self.get_param(param, 'mask2_sat_distance')
        smin = self.get_param(param, 'mask2_sat_min')
        smax = self.get_param(param, 'mask2_sat_max')
        blur = self.get_param(param, 'mask2_blur')
        face_face = self.get_param(param, 'mask2_face_face')
        face_brows = self.get_param(param, 'mask2_face_brows')
        face_eyes = self.get_param(param, 'mask2_face_eyes')
        face_nose = self.get_param(param, 'mask2_face_nose')
        face_mouth = self.get_param(param, 'mask2_face_mouth')
        face_lips = self.get_param(param, 'mask2_face_lips')
        open_space = self.get_param(param, 'mask2_open_space')
        close_space = self.get_param(param, 'mask2_close_space')
        if  (dmin == 0 and dmax == 255 and
             hdis == 179 and hmin == 0 and hmax == 359 and
             ldis == 127 and lmin == 0 and lmax == 255 and
             sdis == 127 and smin == 0 and smax == 255 and
             blur == 0):
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((dmin, dmax, hdis, hmin, hmax, ldis, lmin, lmax, sdis, smin, smax, blur))
            if self.hash != param_hash:
                self.diff = None
                self.hash = param_hash

        return self.diff

class GrainEffect(Effect):

    def get_param_dict(self, param):
        return {
            'grain_intensity': 0,
            'grain_size': 0,
            'grain_blue_bias': 0,
            'grain_shadow_boost': 0,
            'grain_color_noise_ratio': 0
        }

    def set2widget(self, widget, param):
        widget.ids["slider_grain_intensity"].set_slider_value(self.get_param(param, 'grain_intensity'))
        widget.ids["slider_grain_size"].set_slider_value(self.get_param(param, 'grain_size'))
        widget.ids["slider_grain_blue_bias"].set_slider_value(self.get_param(param, 'grain_blue_bias'))
        widget.ids["slider_grain_shadow_boost"].set_slider_value(self.get_param(param, 'grain_shadow_boost'))
        widget.ids["slider_grain_color_noise_ratio"].set_slider_value(self.get_param(param, 'grain_color_noise_ratio'))

    def set2param(self, param, widget):
        param['grain_intensity'] = widget.ids["slider_grain_intensity"].value
        param['grain_size'] = widget.ids["slider_grain_size"].value
        param['grain_blue_bias'] = widget.ids["slider_grain_blue_bias"].value
        param['grain_shadow_boost'] = widget.ids["slider_grain_shadow_boost"].value        
        param['grain_color_noise_ratio'] = widget.ids["slider_grain_color_noise_ratio"].value

    def make_diff(self, rgb, param, efconfig):
        gi = self.get_param(param, 'grain_intensity')
        gs = self.get_param(param, 'grain_size')
        gbb = self.get_param(param, 'grain_blue_bias')
        gsb = self.get_param(param, 'grain_shadow_boost')
        gcnr = self.get_param(param, 'grain_color_noise_ratio')
        if gi == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((gi, gs, gbb, gsb, gcnr))
            if self.hash != param_hash:
                rgb = core.type_convert(rgb, np.ndarray)
                gi = gi / 100.0                 # 0.0-1.0
                gs = gs / 100.0 * 4.0 + 1.0     # 1.0-5.0
                gbb = gbb / 100.0 + 1.0         # 1.0-2.0
                gsb = gsb / 100.0 * 1.5 + 0.5   # 0.5-2.0          
                gcnr = gcnr / 100.0             # 0.0-1.0
                self.diff = core.apply_film_grain(rgb, gi * efconfig.disp_info[4], gs * efconfig.resolution_scale , gbb, gsb, gcnr)
                self.hash = param_hash
        
        return self.diff
    
class VignetteEffect(Effect):

    def get_param_dict(self, param):
        return {
            'vignette_intensity': 0,
            'vignette_radius_percent': 0,
            'vignette_softness': 100,
            'crop_enable': False,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_vignette_intensity"].set_slider_value(self.get_param(param, 'vignette_intensity'))
        widget.ids["slider_vignette_radius_percent"].set_slider_value(self.get_param(param, 'vignette_radius_percent'))
        widget.ids["slider_vignette_softness"].set_slider_value(self.get_param(param, 'vignette_softness'))

    def set2param(self, param, widget):
        param['vignette_intensity'] = widget.ids["slider_vignette_intensity"].value
        param['vignette_radius_percent'] = widget.ids["slider_vignette_radius_percent"].value
        param['vignette_softness'] = widget.ids["slider_vignette_softness"].value

    def make_diff(self, rgb, param, efconfig):
        vi = self.get_param(param, 'vignette_intensity')
        vr = self.get_param(param, 'vignette_radius_percent')
        vs = self.get_param(param, 'vignette_softness')
        pce = self.get_param(param, 'crop_enable')
        if (vi == 0 and vr == 0) or pce == True:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((vi, vr, vs))
            if self.hash != param_hash:
                _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(config.get_config('preview_width'), config.get_config('preview_height'), efconfig.disp_info)
                rgb = core.type_convert(rgb, jnp.ndarray)
                vs = (100 - vs) / 100.0 * 3.0 + 1.0  # 1.0-4.0
                self.diff = core.apply_vignette(rgb, vi, vr, efconfig.disp_info, params.get_crop_rect(param), (offset_x, offset_y), vs)
                self.hash = param_hash
        
        return self.diff
    

def create_effects(distortion_callback=None):
    effects = [{}, {}, {}, {}, {}]

    lv0 = effects[0]
    lv0['lens_modifier'] = LensModifierEffect()
    lv0['subpixel_shift'] = SubpixelShiftEffect()
    lv0['inpaint'] = InpaintEffect()
    lv0['distortion'] = DistortionEffect(distortion_callback=distortion_callback)
    lv0['rotation'] = RotationEffect()
    lv0['crop'] = CropEffect()

    lv1 = effects[1]
    lv1['ai_noise_reduction'] = AINoiseReductonEffect()
    lv1['bm3d_noise_reduction'] = BM3DNoiseReductionEffect()
    lv1['light_noise_reduction'] = LightNoiseReductionEffect()
    lv1['deblur_filter'] = DeblurFilterEffect()
    lv1['defocus'] = DefocusEffect()
    lv1['lensblur_filter'] = LensblurFilterEffect()
    lv1['scratch'] = ScratchEffect()
    lv1['frosted_glass'] = FrostedGlassEffect()
    lv1['mosaic'] = MosaicEffect()
    lv1['glow'] = GlowEffect()
    lv1['face'] = FaceEffect()
    
    lv2 = effects[2]
    lv2['color_temperature'] = ColorTemperatureEffect()
    lv2['dehaze'] = DehazeEffect()
 
    lv2['exposure'] = ExposureEffect()
    lv2['contrast'] = ContrastEffect()
    lv2['clarity'] = ClarityEffect()
    lv2['texture'] = TextureEffect()
    lv2['microcontrast'] = MicroContrastEffect()
    lv2['tone'] = ToneEffect()

    lv2['highlight_compress'] = HighlightCompressEffect()

    # ここでクリッピング

    #lv2['rgb2hls1'] = RGB2HLSEffect()
    #lv2['hls2rgb1'] = HLS2RGBEffect()

    lv2['clahe'] = CLAHEEffect()

    lv2['rgb2hls2'] = RGB2HLSEffect()
    lv2['hls'] = HLSEffect()
    lv2['vs_and_saturation'] = VSandSaturationEffect()
    lv2['hls2rgb2'] = HLS2RGBEffect()

    lv2['curve'] = CurveEffect()

    lv2['lut'] = LUTEffect()
    lv2['lens_simulator'] = LensSimulatorEffect()
    lv2['film_emulation'] = FilmSimulationEffect()
    lv2['level'] = LevelEffect()
    lv2['solid_color'] = SolidColorEffect()
    lv2['unsharp_mask'] = UnsharpMaskEffect()

    lv3 = effects[3]
    lv3['mask2'] = Mask2Effect()

    lv4 = effects[4]
    lv4['grain'] = GrainEffect()
    lv4['vignette'] = VignetteEffect()

    return effects

def set2widget_all(widget, effects, param):
    for dict in effects:
        for l in dict.values():
            l.set2widget(widget, param)
            #l.set2param(param, self)
            l.reeffect()

def reeffect_all(effects, lv=0):
    for i, dict in enumerate(effects):
        if i >= lv:
            for l in dict.values():
               l.reeffect()

def finalize_all(effects, param, widget):
    for dict in effects:
        for l in dict.values():
            l.finalize(param, widget)

def delete_default_param_all(effects, param):
    param2 = param.copy()
    for dict in effects:
        for l in dict.values():
            l.delete_default_param(param2)
    return param2


if __name__ == '__main__':
    param = {'test1': 0, 'test2': 0}

    for p in {'test1': 0, 'test2': 1}.items():
        try:
            if param[p[0]] == p[1]:
                del param[p[0]]
        except:
            pass

    print(param)



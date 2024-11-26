import os
import numpy as np
import pyautogui as pag
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.uix.popup import Popup as KVPopup
from kivy.core.window import Window as KVWindow
from kivy.graphics.texture import Texture as KVTexture
from kivy.clock import Clock, mainthread
from kivy.metrics import Metrics, dp
from functools import partial
import math
from datetime import datetime as dt
import json
import joblib
import threading

import core
import imageset
import curve
import effects
import pipeline
import param_slider
import viewer_widget
import histogram_widget
import metainfo
import util
import mask_editor2
import color_picker
import macos

class MainWidget(MDWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.texture_height = 1024
        self.texture_width = 1024
        self.texture = None
        self.imgset = None
        self.click_x = 0
        self.click_y = 0        
        self.crop_image = None
        self.crop_info = None
        self.is_zoomed = False
        self.drag_start_point = None
        self.primary_param = {}
        self.primary_effects = effects.create_effects()
        self.apply_thread = None
        self.is_draw_image = False

    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.mask_editor2 = self.ids['mask_editor2']
        self.ids['preview_widget'].remove_widget(self.mask_editor2)
        self.mask_editor2.disabled = True

    def serialize(self):
        tdatetime = dt.now()
        tstr = tdatetime.strftime('%Y/%m/%d')
        mask_dict = self.ids['mask_editor2'].serialize()

        dict = {
            'make': "escargo",
            'date': tstr,
            'version': "0.4.1",
            'primary_param': self.primary_param,
        }
        if mask_dict is not None:
            dict.update(mask_dict)

        return dict

    def deserialize(self, dict):
        self.primary_param = dict['primary_param']
        mask_dict = dict.get('mask2', None)
        if mask_dict is not None:
            self.ids['mask_editor2'].deserialize(dict)

    def save_json(self):
        if self.imgset is not None:
            file_path = self.imgset.file_path + '.json'
            with open(file_path, 'w') as f:
                dict = self.serialize()
                json.dump(dict, f)

    def load_json(self):
        if self.imgset is not None:
            file_path = self.imgset.file_path + '.json'
            try:
                with open(file_path, 'r') as f:
                    dict = json.load(f)
                    self.deserialize(dict)
            except FileNotFoundError as e:
               pass
 
    def load_image(self, file_path, exif_data):
        self.imgset = imageset.ImageSet()
        self.imgset.load(file_path, exif_data, self.primary_param, self.start_draw_image_and_crop)

        self.texture = KVTexture.create(size=(self.texture_width, self.texture_height), colorfmt='rgb', bufferfmt='float')
        self.texture.flip_vertical()

        self.click_x = 0
        self.click_y = 0
        self.is_zoomed = False
        self.crop_image = None
        _, self.crop_info = core.crop_image(self.imgset.img, self.texture_width, self.texture_height, self.click_x, self.click_y, (0, 0), self.is_zoomed)
        self.start_draw_image()
    
    def start_draw_image_and_crop(self, imgset, offset=(0, 0)):
        if self.is_draw_image == False:
            if self.imgset == imgset:
                self.crop_image = None
                self.start_draw_image(offset)

    def empty_image(self):
        self.imgset = None
        self.texture = KVTexture.create(size=(self.texture_width, self.texture_height), colorfmt='rgb', bufferfmt='float')
        self.texture.flip_vertical()
        self.ids["preview"].texture = None

    # @mainthread
    def blit_image(self, img):
        self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.ids["preview"].texture = None # 更新のために必要
        self.ids["preview"].texture = self.texture

    def draw_histogram(self, img):
        self.ids["histogram"].draw_histogram(img)

    def draw_image(self, offset, dt):
        if self.imgset is not None:
            img, self.crop_image, self.crop_info = pipeline.process_pipeline(self.imgset.img, self.crop_info, offset, self.crop_image, self.is_zoomed, self.texture_width, self.texture_height, self.click_x, self.click_y, self.primary_effects, self.primary_param, self.ids['mask_editor2'])
            img = core.apply_gamma(img, 1.0/2.222)
            self.draw_histogram(img)
            img = np.clip(img, 0, 1)
            self.blit_image(img)
        self.is_draw_image = False

    def start_draw_image(self, offset=(0, 0)):
        if self.is_draw_image == False: #２重コール防止
            self.is_draw_image = True
            Clock.schedule_once(partial(self.draw_image, offset), -1)
        #self.apply_thread = threading.Thread(target=self.draw_image, daemon=True)
        #self.apply_thread.start()
    
    def apply_effects_lv(self, lv, effect):
        mask = self.ids['mask_editor2'].get_active_mask()
        if mask is None:
            self.primary_effects[lv][effect].set2param(self.primary_param, self)
        else:
            mask.effects[lv][effect].set2param(mask.effects_param, self)

        self.ids['mask_editor2'].set_draw_mask(False)
        self.start_draw_image()
    
    def reset_param(self, param):
        param.clear()

    def set2widget_all(self, effects, param):
        if effects is None:
            effects = self.primary_effects
            param = self.primary_param
            
        for dict in effects:
            for l in dict.values():
                l.set2widget(self, param)
                l.reeffect()
    
    @mainthread
    def on_select(self, card):
        self.reset_param(self.primary_param)
        self.ids["mask_editor2"].clear_mask()
        if card is not None:
            self.load_image(card.file_path, card.exif_data)
            self.load_json()
            self.set2widget_all(self.primary_effects, self.primary_param)
            self.set_exif_data(card.exif_data)
        else:
            self.empty_image()

    def on_image_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # ズーム操作
            if touch.is_double_tap == True:
                self.is_zoomed = not self.is_zoomed
                if self.is_zoomed == False:
                    self.click_x, self.click_y = 0, 0
                else:
                    # ウィンドウ座標からローカルイメージ座標に変換
                    self.click_x, self.click_y = util.to_texture(touch.pos, self.ids['preview'])

                _, self.crop_info = core.crop_image(self.imgset.img, self.texture_width, self.texture_height, self.click_x, self.click_y, (0, 0), self.is_zoomed)
                effects.reeffect_all(self.primary_effects)
                self.start_draw_image_and_crop()

            # ドラッグ操作
            elif self.is_zoomed == True:
                self.drag_start_point = touch.pos

    def on_image_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            if self.is_zoomed == True:
                if self.drag_start_point != None:
                    offset_x = touch.pos[0] - self.drag_start_point[0]
                    offset_y = touch.pos[1] - self.drag_start_point[1]
                    offset_x = -offset_x

                    #_, self.crop_info = core.crop_image_info(self.imgset.img, self.crop_info, (offset_x, offset_y))
                    effects.reeffect_all(self.primary_effects)
                    self.start_draw_image_and_crop((offset_x, offset_y))

                    self.drag_start_point = touch.pos
                
    def on_image_touch_up(self, touch):
        if self.is_zoomed == True:
            if self.drag_start_point != None:
                self.drag_start_point = None

    def on_select_press(self):
        macos.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self.handle_for_dir_selection).run()

    def delay_set_image(self, dt):
        self.mask_editor2.set_image(self.imgset.img, self.texture_width, self.texture_height, self.crop_info, math.radians(self.primary_param.get('rotation', 0)), -1)
        
    def on_mask2_press(self, value):
        if value == "down":
            self.mask_editor2.disabled = False
            self.ids['preview_widget'].add_widget(self.mask_editor2)
            Clock.schedule_once(self.delay_set_image, -1)   # editor2のサイズが未決定なので遅らせる
        else:
            self.mask_editor2.disabled = True
            self.ids['preview_widget'].remove_widget(self.mask_editor2)
            self.ids['mask_editor2'].set_active_mask(None)

    def handle_for_dir_selection(self, selection):
        if selection is not None:
            self.ids['viewer'].set_path(selection[0].decode())

    def on_lut_select_folder(self):
        macos.FileChooser(title="Select LUT Folder", mode="dir", filters=[("CUBE Files", "*.cube")], on_selection=self.handle_for_lut).run()

    def handle_for_lut(self, selection):
        if selection is not None:
            self.set_lut_path(selection[0].decode())

    def set_lut_path(self, path):
        lut_values = ['None']
        effects.LUTEffect.file_pathes = { 'None': None, }

        file_list = os.listdir(path)
        file_list.sort()
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if file_name.lower().endswith(('.cube')):
                lut_values.append(file_name)
                effects.LUTEffect.file_pathes[file_name] = file_path
        self.ids['lut_spinner'].values = lut_values

    def set_exif_data(self, exif_data):
        self.ids['exif_file_name'].value = exif_data.get("FileName", "-")
        self.ids['exif_create_date'].value = exif_data.get("CreateDate", "-")
        self.ids['exif_image_size'].value = exif_data.get("ImageSize", "-")
        self.ids['exif_iso_speed'].value = str(exif_data.get("ISO", "-"))
        self.ids['exif_aperture'].value = str(exif_data.get("ApertureValue", "-"))
        self.ids['exif_shutter_speed'].value = exif_data.get("ShutterSpeedValue", "-")
        self.ids['exif_brightness'].value = str(exif_data.get("BrightnessValue", "-"))
        self.ids['exif_flash'].value = exif_data.get("Flash", "-")
        self.ids['exif_white_balance'].value = exif_data.get("WhiteBalance", "-")
        self.ids['exif_focal_length'].value = exif_data.get("FocalLength", "-")
        self.ids['exif_exposure_program'].value = exif_data.get("PictureMode", "-")
        self.ids['exif_make'].value = exif_data.get("Make", "-")
        self.ids['exif_model'].value = exif_data.get("Model", "-")
        self.ids['exif_lens_model'].value = exif_data.get("LensModel", "-")
        self.ids['exif_software'].value = exif_data.get("Software", "-")
        #self.ids['exif_'].value = exif_data.get("", "-")

    def get_scale(self):
        return self.crop_info[4]
    
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = "escargot"
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Blue'

    def build(self): 
        widget = MainWidget()

        KVWindow.size = (dp(600), dp(400))
        scr_w,scr_h = pag.size()
        KVWindow.left = (scr_w - dp(600)) // 2
        KVWindow.top = (scr_h - dp(400)) // 2

        # testcode
        #widget.load_image("DSCF0090.raf")
        #widget.load_image(os.getcwd() + "/picture/DSCF0002-small.tif")
        widget.ids['viewer'].set_path(os.getcwd() + "/picture")

        return widget

if __name__ == '__main__':
    """
    import imageio
    import imageio.v3 as iio

    imageio.plugins.freeimage.download()

    print(imageio.plugins.formats)

    metadata = iio.immeta(os.getcwd() + "/picture/DSCF0002.jpg")
    metadata = iio.immeta(os.getcwd() + "/picture/DSCF0002.raf", plugin='rawpy', exclude_applied=False)

    reader = imageio.get_reader(os.getcwd() + "/picture/DSCF0002.raf")
    metadata = reader.get_meta_data(0)
    """
    MainApp().run()



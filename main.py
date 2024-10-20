import os
import numpy as np
import pyautogui as pag
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.core.window import Window as KVWindow
from kivy.graphics.texture import Texture as KVTexture
from kivy.clock import Clock
from functools import partial

import core
import imageset
import curve
import effects
import param_slider
import viewer_widget
import histogram_widget
import metainfo
import util

class MainWidget(MDWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.texture_height = 1024
        self.texture_width = 1024
        self.texture = None
        self.imgset = None
        self.tex_x = 0
        self.tex_y = 0        
        self.is_zoomed = False
        self.crop_info = None
        self.drag_start_point = None
        self.current_param = {}
        self.effects = effects.create_effects()
        self.apply_thread = None
 
    def load_image(self, file_path, exif_data):
        self.imgset = imageset.ImageSet()
        self.imgset.load(file_path, exif_data, self.current_param)

        self.texture = KVTexture.create(size=(self.texture_width, self.texture_height), bufferfmt='ushort')
        self.texture.flip_vertical()

        self.tex_x = 0
        self.tex_y = 0
        self.is_zoomed = False
        _, self.crop_info = core.crop_image(self.imgset.img, self.texture_width, self.texture_height, self.tex_x, self.tex_y, (0, 0), self.is_zoomed)
        self.start_draw_image()

    # @mainthread
    def blit_image(self, img):
        self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.ids["preview"].texture = None # 更新のために必要
        self.ids["preview"].texture = self.texture

    def draw_histogram(self, img):
        self.ids["histogram"].draw_histogram(img)

    def draw_image(self, offset, dt):
        if self.imgset is not None:
            img, reset = effects.pipeline_lv0(self.imgset.img, self.effects, self.current_param)
            if self.is_zoomed:
                img, self.crop_info = core.crop_image_info(img, self.crop_info, offset)
            else:
                img, self.crop_info = core.crop_image(img, self.texture_width, self.texture_height, self.tex_x, self.tex_y, offset, self.is_zoomed)

            img = effects.pipeline_lv1(img, self.effects, self.current_param)
            img = effects.pipeline_lv2(img, self.effects, self.current_param)
            self.draw_histogram(img)
            img = core.apply_gamma(img, 1.0/2.222)
            self.blit_image(img)

    def start_draw_image(self, offset=(0, 0)):
        Clock.schedule_once(partial(self.draw_image, offset), -1)
        #self.apply_thread = threading.Thread(target=self.draw_image, daemon=True)
        #self.apply_thread.start()
    
    def apply_effects_lv(self, lv, effect):
        #try:
            self.effects[lv][effect].set2param(self.current_param, self)
            self.start_draw_image()

        #except AttributeError:
        #    print('AttributeError: ' + effect)
        #return True
    
    def reset_param(self, param):
        param.clear()

    def set2widget_all(self, effect, param):
        self.ids['effects'].disabled = True
        for dict in effect:
            for l in dict.values():
                l.set2widget(self, param)
                l.reeffect()
        self.ids['effects'].disabled = False
    
    def on_select(self, file_path, exif_data):
        self.reset_param(self.current_param)
        self.load_image(file_path, exif_data)
        #self.imgset.img = core.modify_lens(self.imgset.img, exif_data)
        self.set2widget_all(self.effects, self.current_param)
        self.set_exif_data(exif_data)

    def on_image_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # ズーム操作
            if touch.is_double_tap:
                self.is_zoomed = not self.is_zoomed
                if self.is_zoomed == False:
                    self.tex_x, self.tex_y = 0, 0
                else:
                    # ウィンドウ座標からローカルイメージ座標に変換
                    self.tex_x, self.tex_y = util.to_texture(touch.pos, self.ids['preview'])

                _, self.crop_info = core.crop_image(self.imgset.img, self.texture_width, self.texture_height, self.tex_x, self.tex_y, (0, 0), self.is_zoomed)
                effects.reeffect_all(self.effects)
                self.start_draw_image()

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
                    effects.reeffect_all(self.effects)
                    self.start_draw_image((offset_x, offset_y))

                    self.drag_start_point = touch.pos
                

    def on_image_touch_up(self, touch):
        if self.is_zoomed == True:
            if self.drag_start_point != None:
                self.drag_start_point = None

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

        KVWindow.size = (1200, 800)
        scr_w,scr_h= pag.size()
        KVWindow.left = (scr_w - 1200) / 2
        KVWindow.top = (scr_h - 800) / 2

        # testcode
        #widget.load_image("DSCF0090.raf")
        #widget.load_image(os.getcwd() + "/picture/DSCF0002-small.tif")
        widget.ids['viewer'].set_path(os.getcwd() + "/picture")

        return widget

if __name__ == '__main__':
    MainApp().run()



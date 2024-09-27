import os
import threading
import numpy as np
import pyautogui as pag
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.core.window import Window as KVWindow
from kivy.graphics.texture import Texture as KVTexture
#from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.clock import mainthread

import core
import imageset
import curve
import effects
import param_slider
import viewer_widget
import histogram_widget

class MainWidget(MDWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tex = None
        self.imgset = None
        self.is_zoomed = False
        self.drag_start_point = None
        self.current_param = {}
        self.effects = effects.create_effects()
        self.apply_thread = None
 
    def load_image(self, file_path):
        self.imgset = imageset.ImageSet()
        self.imgset.load(file_path, self.current_param)

        self.tex = KVTexture.create(size=(1024, 1024), bufferfmt='ushort')
        self.tex.flip_vertical()

        self.imgset.crop_image(0, 0, 1024, 1024, False)
        self.is_zoomed = False
        self.apply_effects()

    # @mainthread
    def blit_image(self, img):
        self.tex.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.ids["preview"].texture = self.tex

    def draw_histogram(self, img):
        self.ids["histogram"].draw_histogram(img)

    def draw_image(self, dt):
        if self.imgset is not None:
            self.imgset.img, reset = effects.pipeline_lv0(self.imgset.img, self.effects, self.current_param)
            if reset == True:
                pass
            img = effects.pipeline_lv1(self.imgset.prv, self.effects, self.current_param)
            img = effects.pipeline_lv2(img, self.effects, self.current_param)
            #self.draw_histogram(img)
            img = core.apply_gamma(img, 1.0/2.222)
            self.blit_image(img)

    def apply_effects(self):
        Clock.schedule_once(self.draw_image, -1)
        #self.apply_thread = threading.Thread(target=self.draw_image, daemon=True)
        #self.apply_thread.start()
    
    def apply_effects_lv(self, lv, effect):
        try:
            self.effects[lv][effect].set2param(self.current_param, self)
            self.apply_effects()
        except AttributeError:
            print('AttributeError: ' + effect)
        return True
    
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
        self.load_image(file_path)
        self.imgset.img = core.modify_lens(self.imgset.img, exif_data)
        self.set2widget_all(self.effects, self.current_param)

    def on_image_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # ズーム操作
            if touch.is_double_tap:
                self.is_zoomed = not self.is_zoomed
                if self.is_zoomed == False:
                    tex_x, tex_y = 0, 0
                else:
                    # ウィンドウ座標からローカルイメージ座標に変換
                    tex_x, tex_y = core.to_texture(touch.pos, self.ids['preview'])

                self.imgset.crop_image(tex_x, tex_y, 1024, 1024, self.is_zoomed)
                effects.reeffect_all(self.effects)
                self.apply_effects()

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

                    self.imgset.crop_image2((offset_x, offset_y))
                    effects.reeffect_all(self.effects)
                    self.apply_effects()

                    self.drag_start_point = touch.pos

    def on_image_touch_up(self, touch):
        if self.is_zoomed == True:
            if self.drag_start_point != None:
                self.drag_start_point = None


    
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



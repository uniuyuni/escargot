import os
import numpy as np
import matplotlib.pyplot as plt
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock

import core
import imageset
import curve
import layer
import param_slider
import viewer_widget

class MainWidget(MDWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tex = None
        self.imgset = None
        self.scale = 1.0
        self.prv_x = 0
        self.prv_y = 0
        self.ax = None
        self.tcax = None
        self.param = {}
        self.imglayer = layer.create_layer()
        self.tmblayer = layer.create_layer()

    def load_image(self, filename):
        self.imgset = imageset.ImageSet()
        self.imgset.load(filename, filename + '.mask')
        self.param['src_size'] = self.imgset.img.shape

        # self.texture = Texture.create(size=(img.shape[1], img.shape[0]))
        shape = self.imgset.img.shape
        if shape[1] >= shape[0]:
            w = 1024
            s = 1024.0/shape[1]
            h = int(1024*shape[0]/shape[1])
        else:
            w = int(1024*shape[1]/shape[0])
            s = 1024.0/shape[0]
            h = 1024
        self.scale = s
        
        self.tex = Texture.create(size=(w, h), bufferfmt='ushort')
        self.tex.flip_vertical()

        self.imgset.make_clip(self.scale, self.prv_x, self.prv_y, self.tex.width, self.tex.height)
        self.adjust_all()

    def draw_histogram(self, img):
        # ヒストグラムの取得
        img = core.apply_gamma(img, 1.0/2.222)
        hist, bins = np.histogram(img.ravel(), 256, [0, 1.0])

        # ヒストグラムの表示
        if self.ax is None:
            # 描画する領域を用意する
            self.fig, self.ax = plt.subplots()
            self.ids["info"].add_widget(FigureCanvasKivyAgg(self.fig))

        self.ax.clear()
        self.ax.plot(hist)
        self.fig.gca().axis('off')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def blit_image(self, img):
        img = np.clip(img, 0.0, 1.0)
        img = core.apply_gamma(img, 1.0/2.222)
        self.tex.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.ids["preview"].texture = self.tex
    
    def draw_histogram(self):
        tmb = layer.pipeline_lv1(self.imgset.tmb, self.tmblayer, self.param)
        self.draw_histogram(tmb)

    def draw_image(self, dt):
        if self.imgset is not None:
            self.imgset.img, reset = layer.pipeline_lv0(self.imgset.img, self.imglayer, self.param)
            if reset == True:
                self.imgset.make_clip(self.scale, self.prv_x, self.prv_y, self.tex.width, self.tex.height)
            img = layer.pipeline_lv1(self.imgset.prv, self.imglayer, self.param)
            img = layer.pipeline_lv2(img, self.imglayer, self.param)
            self.blit_image(img)

    def adjust_all(self):
        Clock.schedule_once(self.draw_image, -1)

    def adjust_lv0(self, layer):
        self.imglayer[0][layer].set2param(self.param, self)
        self.adjust_all()
        return True

    def adjust_lv1(self, layer):
        self.imglayer[1][layer].set2param(self.param, self)
        self.adjust_all()
        return True
    
    def adjust_lv2(self, layer):
        try:
            self.imglayer[2][layer].set2param(self.param, self)
            self.adjust_all()
        except AttributeError:
            print('AttributeError: ' + layer)
        return True
    
    def reset_param(self, param):
        param = {}

    def set2widget_all(self, layer, param):
        for dict in layer:
            for l in dict.values():
                l.set2widget(self, param)
    
    def on_select(self, file_path, exif_data):
        self.reset_param(self.param)
        self.load_image(file_path)
        self.set2widget_all(self.imglayer, self.param)
    
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = 'escargot'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = MainWidget()

        Window.size = (1024, 800)

        # testcode
        #widget.load_image("DSCF0090.raf")
        widget.load_image(os.getcwd() + "/picture/DSCF0002-small.tif")
        widget.ids['viewer'].set_path("/Users/uniuyuni/PythonProjects/escargot/picture")

        return widget


if __name__ == '__main__':
    MainApp().run()



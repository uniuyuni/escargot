import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import kivy
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.graphics.texture import Texture
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

import core
import imageset
import curve
import layer

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
        self.imglayer = {}
        layer.create_layer(self.imglayer)
        self.tmblayer = {}
        layer.create_layer(self.tmblayer)


    def load_image(self, filename):
        self.imgset = imageset.ImageSet()
        self.imgset.load(filename, filename + '.mask')

        # self.texture = Texture.create(size=(img.shape[1], img.shape[0]))
        self.tex = Texture.create(size=(1024, 1024), bufferfmt='ushort')
        self.tex.flip_vertical()

        self.scale = 1024.0/max(self.imgset.src.shape)

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

    def draw_tonecurve0(self, cs):
        if self.tcax is None:
            self.tcfig, self.tcax = plt.subplots()
            self.ids["info"].add_widget(FigureCanvasKivyAgg(self.tcfig))

        self.tcax.clear()
        
        # プロット用のX値を生成
        x_new = np.linspace(0, 65535, 100)
        y_new = cs(x_new)

        # スプライン曲線のプロット
        self.tcax.plot(x_new, y_new)

        self.tcfig.canvas.draw()
        self.tcfig.canvas.flush_events()

    def blit_image(self, img):
        img = np.clip(img, 0.0, 1.0)
        img = core.apply_gamma(img, 1.0/2.222)
        self.tex.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.ids["preview"].texture = self.tex
    
    async def async_draw_histogram(self):
        tmb = layer.pipeline(self.imgset.tmb, self.tmblayer)
        self.draw_histogram(tmb)

    async def async_blt_image(self):
        img = layer.pipeline(self.imgset.prv, self.imglayer)
        self.blit_image(img)

        asyncio.create_task(self.async_draw_histogram())

    def adjust_all(self):
        imgtask = asyncio.run(self.async_blt_image())                 

    def adjust_key(self, layer):
        self.imglayer[layer].set_param(self.param, self)
        self.imglayer[layer].make_diff(self.imgset.prv, self.param)
        self.tmblayer[layer].make_diff(self.imgset.tmb, self.param)
        self.adjust_all()
        return True
    
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = 'escargot'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = MainWidget()

        # testcode
        #widget.load_image("DSCF0090.raf")
        widget.load_image(os.getcwd() + "/DSCF0002.tif")
        
        return widget

if __name__ == '__main__':
    MainApp().run()



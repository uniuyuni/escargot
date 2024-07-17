import os
import cv2
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

import noise2void
import DRBNet

import core
import imageset
import curve


#補正既定クラス
class AdjustmentLayer():

    def __init__(self, **kwargs):
        self.diff = None

    # 差分の作成
    def make_diff(self, img, widget):
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

    def make_diff(self, img, widget):
        state = widget.ids["toggle_noise"].state
        if state == "down":
            self.diff = noise2void.predict(img, NoiseLayer.__net, 'mps')
        if state == "normal":
            self.diff = img

class DefocusLayer(AdjustmentLayer):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if DefocusLayer.__net is None:
            DefocusLayer.__net = DRBNet.setup_predict()

    def make_diff(self, img, widget):
        state = widget.ids["toggle_defocus"].state
        if state == "down":
            self.diff = DRBNet.predict(img, DefocusLayer.__net, 'mps')
        if state == "normal":
            self.diff = img

class ExposureLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.adjust_exposure(hls[:,:,1], widget.ids["slider_exposure"].value)
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class ContrastLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.adjust_contrast(hls[:,:,1], widget.ids["slider_contrast"].value)
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class ToneLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        # 制御点とエクスポージャー補正値を設定
        points = np.array([0.0, 13107.0, 52429.0, 65535.0])
        values = np.array([0.0, 13107.0, 52429.0, 65535.0])
        values[1] += widget.ids["slider_black"].value*100.0
        values[2] += widget.ids["slider_white"].value*100.0
        points /= 65535.0
        values /= 65535.0

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_curve(hls[:,:,1], points, values, False)
        hls_l = core.adjust_shadow(hls_l, widget.ids["slider_shadow"].value)
        hls_l = core.adjust_hilight(hls_l, widget.ids["slider_hilight"].value)
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class SaturationLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.adjust_saturation(hls[:,:,2], widget.ids["slider_saturation"].value, widget.ids["slider_vibrance"].value)
        self.diff = hls_s/hls[:,:,2]    # Sのみ保存

class DehazeLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        img2 = core.apply_dehaze(img, widget.ids["slider_dehaze"].value)
        self.diff = img2-img     # RGBのみ保存

class LevelLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_level_adjustment(hls[:,:,1], widget.ids["slider_black_level"].value, widget.ids["slider_white_level"].value, widget.ids["slider_mid"].value)
        self.diff = hls_l-hls[:,:,1]    # Lのみ保存

class TonecurveLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        img2 = core.apply_spline(img, widget.ids["tonecurve"].get_spline())
        self.diff = img2-img            # RGB保存

class TonecurveRedLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        img2 = core.apply_spline(img[:,:,0], widget.ids["tonecurve_red"].get_spline())
        self.diff = img2-img[:,:,0]            # R保存

class TonecurveGreenLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        img2 = core.apply_spline(img[:,:,1], widget.ids["tonecurve_green"].get_spline())
        self.diff = img2-img[:,:,1]            # G保存

class TonecurveBlueLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        img2 = core.apply_spline(img[:,:,2], widget.ids["tonecurve_blue"].get_spline())
        self.diff = img2-img[:,:,2]            # B保存

class LumvsLumLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_spline(hls[:,:,1], widget.ids["LumvsLum"].get_spline())
        self.diff = 2.0**((hls_l-0.5)*2.0)   # Lのみ保存

class HuevsHueLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_h = core.apply_spline(hls[:,:,0]/360.0, widget.ids["HuevsHue"].get_spline())
        self.diff = (hls_h-0.5)*360.0    # Hのみ保存

class HuevsSatLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.apply_spline(hls[:,:,0], widget.ids["HuevsSat"].get_spline())
        self.diff = 2.0**((hls_s-0.5)*2.0)     # Sのみ保存

class HuevsLumLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_spline(hls[:,:,0], widget.ids["HuevsLum"].get_spline())
        self.diff = 2.0**((hls_l-0.5)*2.0)     # Lのみ保存

class LumvsSatLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.apply_spline(hls[:,:,1], widget.ids["LumvsSat"].get_spline())
        self.diff = 2.0**((hls_s-0.5)*2.0)     # Sのみ保存

class SatvsSatLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_s = core.apply_spline(hls[:,:,2], widget.ids["SatvsSat"].get_spline())
        self.diff = 2.0**((hls_s-0.5)*2.0)     # Sのみ保存

class SatvsLumLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls_l = core.apply_spline(hls[:,:,2], widget.ids["SatvsLum"].get_spline())
        self.diff = 2.0**((hls_l-0.5)*2.0)     # Lのみ保存

class HLSRedLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_red(hls, (widget.ids["slider_red_hue"].value, widget.ids["slider_red_sat"].value, widget.ids["slider_red_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSOrangeLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2= core.adjust_hls_orange(hls, (widget.ids["slider_orange_hue"].value, widget.ids["slider_orange_sat"].value, widget.ids["slider_orange_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSYellowLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_yellow(hls, (widget.ids["slider_yellow_hue"].value, widget.ids["slider_yellow_sat"].value, widget.ids["slider_yellow_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSGreenLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_green(hls, (widget.ids["slider_green_hue"].value, widget.ids["slider_green_sat"].value, widget.ids["slider_green_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSCyanLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_cyan(hls, (widget.ids["slider_cyan_hue"].value, widget.ids["slider_cyan_sat"].value, widget.ids["slider_cyan_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSBlueLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_blue(hls, (widget.ids["slider_blue_hue"].value, widget.ids["slider_blue_sat"].value, widget.ids["slider_blue_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSPurpleLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_purple(hls, (widget.ids["slider_purple_hue"].value, widget.ids["slider_purple_sat"].value, widget.ids["slider_purple_val"].value))
        self.diff = hls2-hls            # HLS保存

class HLSMagentaLayer(AdjustmentLayer):

    def make_diff(self, img, widget):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS_FULL)
        hls2 = core.adjust_hls_magenta(hls, (widget.ids["slider_magenta_hue"].value, widget.ids["slider_magenta_sat"].value, widget.ids["slider_magenta_val"].value))
        self.diff = hls2-hls            # HLS保存


class MainWidget(Widget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tex = None
        self.imgset = None
        self.scale = 1.0
        self.prv_x = 0
        self.prv_y = 0
        self.ax = None
        self.tcax = None
        self.imglyr = {}
        MainWidget.__create_layer(self.imglyr)
        self.tmblyr = {}
        MainWidget.__create_layer(self.tmblyr)

    def __create_layer(layer):
        layer['noise'] = NoiseLayer()
        layer['defocus'] = DefocusLayer()
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
        layer['hls_red'] = HLSRedLayer()
        layer['hls_orange'] = HLSOrangeLayer()
        layer['hls_yellow'] = HLSYellowLayer()
        layer['hls_green'] = HLSGreenLayer()
        layer['hls_cyan'] = HLSCyanLayer()
        layer['hls_blue'] = HLSBlueLayer()
        layer['hls_purple'] = HLSPurpleLayer()
        layer['hls_magenta'] = HLSMagentaLayer()


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
    
    def adjust_dict(self, img, dic):

        diff = dic['noise'].diff
        if diff is not None:
            rgb = diff
        else:
            rgb = img.copy()
        diff = dic['defocus'].diff
        if diff is not None:
            rgb = diff
    
        # RGB
        diff = dic['dehaze'].diff
        if diff is not None: rgb += diff
        diff = dic['tonecurve'].diff
        if diff is not None: rgb += diff
        diff = dic['tonecurve_red'].diff
        if diff is not None: rgb[:,:,0] += diff
        diff = dic['tonecurve_green'].diff
        if diff is not None: rgb[:,:,1] += diff
        diff = dic['tonecurve_blue'].diff
        if diff is not None: rgb[:,:,2] += diff

        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)

        # Hのみ
        diff = dic['HuevsHue'].diff
        if diff is not None: hls[:,:,0] += diff

        # Lのみ
        diff = dic['exposure'].diff
        if diff is not None: hls[:,:,1] += diff
        diff = dic['contrast'].diff
        if diff is not None: hls[:,:,1] += diff
        diff = dic['tone'].diff
        if diff is not None: hls[:,:,1] += diff
        diff = dic['level'].diff
        if diff is not None: hls[:,:,1] += diff
        diff = dic['LumvsLum'].diff
        if diff is not None: hls[:,:,1] *= diff
        diff = dic['HuevsLum'].diff
        if diff is not None: hls[:,:,1] *= diff
        diff = dic['SatvsLum'].diff
        if diff is not None: hls[:,:,1] *= diff

        # Sのみ
        diff = dic['HuevsSat'].diff
        if diff is not None: hls[:,:,2] *= diff
        diff = dic['LumvsSat'].diff
        if diff is not None: hls[:,:,2] *= diff
        diff = dic['SatvsSat'].diff
        if diff is not None: hls[:,:,2] *= diff


        # HLS
        diff = dic['hls_red'].diff
        if diff is not None: hls += diff
        diff = dic['hls_orange'].diff
        if diff is not None: hls += diff
        diff = dic['hls_yellow'].diff
        if diff is not None: hls += diff
        diff = dic['hls_green'].diff
        if diff is not None: hls += diff
        diff = dic['hls_cyan'].diff
        if diff is not None: hls += diff
        diff = dic['hls_blue'].diff
        if diff is not None: hls += diff
        diff = dic['hls_purple'].diff
        if diff is not None: hls += diff
        diff = dic['hls_magenta'].diff
        if diff is not None: hls += diff

        # Sのみ
        diff = dic['saturation'].diff
        if diff is not None: hls[:,:,2] *= diff

        # 合成
        hls[:,:,1] = np.clip(hls[:,:,1], 0, 1.0)
        hls[:,:,2] = np.clip(hls[:,:,2], 0, 1.0)
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)

        return rgb

    async def async_draw_histogram(self):
        tmb = self.adjust_dict(self.imgset.tmb, self.tmblyr)
        self.draw_histogram(tmb)

    async def async_blt_image(self):
        img = self.adjust_dict(self.imgset.prv, self.imglyr)
        self.blit_image(img)

        asyncio.create_task(self.async_draw_histogram())

    def adjust_all(self):
        imgtask = asyncio.run(self.async_blt_image())                 

    def adjust_key(self, key):
        self.imglyr[key].make_diff(self.imgset.prv, self)
        self.tmblyr[key].make_diff(self.imgset.tmb, self)
        self.adjust_all()
        return True
    
class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = 'escargot'

    def build(self): 
        widget = MainWidget()

        # testcode
        #widget.load_image("DSCF0090.raf")
        widget.load_image(os.getcwd() + "/DSCF0090.tif")
        
        return widget

if __name__ == '__main__':
    MainApp().run()



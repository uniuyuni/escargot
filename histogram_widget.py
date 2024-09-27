import numpy as np
import cv2
from kivy.app import App as KVApp
from kivy.uix.image import Image as KVImage
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.graphics import Color as KVColor, Rectangle as KVRectangle, Line as KVLine
from kivy.metrics import dp

import core

class HistogramWidget(KVImage):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_image(self, image_path):
        # 画像を読み込み、ヒストグラムを計算
        pixels = cv2.imread(image_path)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        pixels = pixels.astype(np.float32)/256.0
        self.draw_histogram(pixels)

    def draw_histogram(self, pixels):
        # RGB各チャンネルのヒストグラムを計算
        #r_hist, g_hist, b_hist = [np.histogram(pixels[..., i], bins=256, range=(0, 1.0))[0] for i in range(3)]
        r_hist, g_hist, b_hist = [cv2.calcHist([pixels], [i], None, [256], [0, 1.0]) for i in range(3)]
        
        # 輝度の計算
        #luminance = core.cvtToGrayColor(pixels)
        luminance = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        #luminance = 0.2989 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        #l_hist, _ = np.histogram(luminance, bins=256, range=(0, 1.0))
        l_hist = cv2.calcHist([luminance], [0], None, [256], [0, 1.0])

        # ヒストグラムを描画
        self.canvas.clear()
        with self.canvas:
            KVColor((0.8, 0.8, 0.8, 1))
            KVLine(rectangle=(self.pos[0], self.pos[1], dp(256), dp(100)), width=1)
        self.draw_histogram_bars(r_hist, (1, 0, 0, 0.8))
        self.draw_histogram_bars(g_hist, (0, 1, 0, 0.8))
        self.draw_histogram_bars(b_hist, (0, 0, 1, 0.8))
        self.draw_histogram_bars(l_hist, (0.8, 0.8, 0.8, 1))  # 輝度ヒストグラムをグレーで表示

    def draw_histogram_bars(self, histogram, color, offset_x=0, offset_y=0):
        max_value = np.max(histogram)
        bar_width = 2
        offset_x += self.pos[0]
        offset_y += self.pos[1]
        
        with self.canvas:
            KVColor(*color)
            for x, value in enumerate(histogram):
                height = (value / max_value) * 200  # ヒストグラムの高さを設定
                KVRectangle(pos=(x * bar_width + offset_x, offset_y), size=(bar_width, height))

class Histogram_WidgetApp(KVApp):
    def build(self):
        histogram = HistogramWidget()
        histogram.load_image("/Users/uniuyuni/PythonProjects/escargot/picture/DSCF0025.JPG")
        return histogram

if __name__ == '__main__':
    Histogram_WidgetApp().run()

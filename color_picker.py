from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.slider import MDSlider
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.gridlayout import MDGridLayout
from kivy.properties import ListProperty, NumericProperty
from kivy.graphics import Color, Ellipse, Quad
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.lang import Builder
from colorsys import rgb_to_hls, hls_to_rgb
import math

import param_slider

class CWColorButton(MDCard):
    background_color = ListProperty([1, 0, 0, 1])
    
    def on_color_select(self):
        app = MDApp.get_running_app()
        self.parent.parent.current_color = self.background_color

class CWColorPreview(MDCard):
    color = ListProperty([1, 0, 0, 1])

class CWColorWheel(MDBoxLayout):
    selected_color = ListProperty([1, 0, 0, 1])
    hue = NumericProperty(0)
    lightness = NumericProperty(0.5)
    saturation = NumericProperty(1)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self.draw_wheel)
        self.bind(pos=self.update_wheel, size=self.update_wheel)
    
    def draw_wheel(self, dt):
        self.wheel_radius = min(self.size[0], self.size[1]) / 2
        self.center_x = self.pos[0] + self.size[0] / 2
        self.center_y = self.pos[1] + self.size[1] / 2
        
        self.canvas.clear()
        with self.canvas:
            segments = 360  # 円周方向の分割数
            radial_steps = 50  # 半径方向のステップ数
            
            for r in range(radial_steps):
                inner_dist = r / radial_steps
                outer_dist = (r + 1) / radial_steps
                
                for angle in range(segments):
                    current_angle = math.radians(angle)
                    next_angle = math.radians((angle + 1) % segments)
                    
                    # 内側と外側の頂点座標を計算
                    inner_x1 = self.center_x + inner_dist * self.wheel_radius * math.cos(current_angle)
                    inner_y1 = self.center_y + inner_dist * self.wheel_radius * math.sin(current_angle)
                    inner_x2 = self.center_x + inner_dist * self.wheel_radius * math.cos(next_angle)
                    inner_y2 = self.center_y + inner_dist * self.wheel_radius * math.sin(next_angle)
                    outer_x1 = self.center_x + outer_dist * self.wheel_radius * math.cos(next_angle)
                    outer_y1 = self.center_y + outer_dist * self.wheel_radius * math.sin(next_angle)
                    outer_x2 = self.center_x + outer_dist * self.wheel_radius * math.cos(current_angle)
                    outer_y2 = self.center_y + outer_dist * self.wheel_radius * math.sin(current_angle)
                    
                    # 色の計算
                    hue = angle / segments
                    saturation = outer_dist
                    lightness = 0.5
                    r, g, b = hls_to_rgb(hue, lightness, saturation)
                    Color(r, g, b, 1)
                    
                    # 三角形を描画
                    Quad(points=[
                        inner_x1, inner_y1,  # 内側の1点
                        inner_x2, inner_y2,  # 内側の2点
                        outer_x1, outer_y1,  # 外側の1点
                        outer_x2, outer_y2   # 外側の次の点
                    ])
               
            # 選択位置のマーカー
            self.marker_color = Color(1, 1, 1, 1)
            self.marker_size = dp(10)
            self.marker = Ellipse(size=(self.marker_size, self.marker_size))
            self.update_marker()    
            
    def update_wheel(self, *args):
        Clock.schedule_once(self.draw_wheel)
    
    def update_marker(self):
        angle = self.hue * 2 * math.pi
        radius = self.saturation * self.wheel_radius
        x = self.center_x + radius * math.cos(angle)
        y = self.center_y + radius * math.sin(angle)
        self.marker.pos = (x - self.marker_size / 2, y - self.marker_size / 2)
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.update_color_from_touch(touch)
            return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self.update_color_from_touch(touch)
            return True
        return super().on_touch_move(touch)
    
    def update_color_from_touch(self, touch):
        dx = touch.pos[0] - self.center_x
        dy = touch.pos[1] - self.center_y
        
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance > self.wheel_radius:
            return  # ホイール外部のタッチは無視
        
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        
        self.hue = angle / (2 * math.pi)
        self.saturation = distance / self.wheel_radius
        
        r, g, b = hls_to_rgb(self.hue, self.lightness, self.saturation)
        self.selected_color = [r, g, b, 1]
        self.update_marker()

    def set_color_from_rgb(self, rgba):
        # RGBをHLSに変換
        r, g, b, a = rgba
        hue, lightness, saturation = rgb_to_hls(r, g, b)
        
        # プロパティを更新
        self.hue = hue
        self.lightness = lightness
        self.saturation = saturation
        
        # 選択色を更新
        self.selected_color = [r, g, b, 1]
        
        # マーカーの位置を更新
        self.update_marker()

class CWColorPicker(MDCard):
    current_color = ListProperty([1, 0, 0, 1])
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self.setup_ui)

    def setup_ui(self, dt):
        
        # パレットの設定
        default_colors = [
            [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
            [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
            [1, 1, 1, 1], [0, 0, 0, 1], [0.5, 0.5, 0.5, 1]
        ]
        
        for color in default_colors:
            btn = CWColorButton(background_color=color)
            self.ids.palette.add_widget(btn)
        
        self.ids.color_wheel.bind(selected_color=self.on_wheel_color)
        self.bind(current_color=self.on_current_color)

    def update_preview(self, instance, value):
        self.ids.preview.color = value

    def on_current_color(self, instance, value):
        r, g, b, _ = self.current_color
        self.ids.red_slider.set_slider_value(r * 255)
        self.ids.green_slider.set_slider_value(g * 255)
        self.ids.blue_slider.set_slider_value(b * 255)

        h, l, s = rgb_to_hls(r, g, b)
        self.ids.hue_slider.set_slider_value(h * 360)
        self.ids.lightness_slider.set_slider_value(l * 100)
        self.ids.saturation_slider.set_slider_value(s * 100)

        self.ids.color_wheel.set_color_from_rgb(self.current_color)
        self.update_preview(instance, value)
        self.update_value_label()

    def on_wheel_color(self, instance, value):
        self.current_color = value
    
    def on_slider_change_rgb(self):
        r = self.ids.red_slider.value / 255
        g = self.ids.green_slider.value / 255
        b = self.ids.blue_slider.value / 255
        self.current_color = [r, g, b, 1]

    def on_slider_change_hls(self):
        h = self.ids.hue_slider.value / 360
        l = self.ids.lightness_slider.value / 100
        s = self.ids.saturation_slider.value / 100
        r, g, b = hls_to_rgb(h, l, s)
        self.current_color = [r, g, b, 1]
    
    def update_value_label(self):
        r, g, b, _ = self.current_color
        self.ids.value_label.text = f"RGB: ({int(r*255)}, {int(g*255)}, {int(b*255)})"
        #h, l, s = rgb_to_hls(r, g, b)
        #self.ids.value_label.text = f"HLS: ({int(h*360)}°, {int(l*100)}%, {int(s*100)}%)"

class MainScreen(MDScreen):
    pass

class CustomColorPickerApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        Builder.load_file('color_picker.kv')
        screen = MainScreen()
        screen.add_widget(CWColorPicker(id='color_picker'))
        return screen

if __name__ == '__main__':
    CustomColorPickerApp().run()

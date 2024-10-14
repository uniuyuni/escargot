
from kivy.app import App as KVApp
from kivy.graphics import Color as KVColor, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Translate as KVTranslate
from kivy.properties import NumericProperty as KVNumericProperty, ListProperty as KVListProperty, BooleanProperty as KVBooleanProperty
from kivy.uix.relativelayout import RelativeLayout as KVRelativeLayout
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.label import Label as KVLabel
from kivy.metrics import dp

class CropEditor(KVFloatLayout):
    input_width = KVNumericProperty(dp(400))  # 元の画像の幅を指定するプロパティ
    input_height = KVNumericProperty(dp(300))  # 元の画像の高さを指定するプロパティ
    scale = KVNumericProperty(1.0)
    crop_pos = KVListProperty((0, 0))
    crop_size = KVListProperty((0, 0))
    corner_threshold = KVNumericProperty(dp(10))  # 四隅の操作ポイントの判定範囲を指定する変数
    minimum_rect = KVNumericProperty(dp(16))

    def __init__(self, **kwargs):
        super(CropEditor, self).__init__(**kwargs)
        self.corner_dragging = None

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate()
            KVColor(1, 1, 1, 1)
            self.white_line = KVLine(rectangle=(self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1]), width=2)
            KVColor(0, 0, 0, 1)
            self.black_line = KVLine(rectangle=(self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1]), width=1)
            KVPopMatrix()
        
        self.label = KVLabel(font_size=20, bold=True, halign='left')
        self.add_widget(self.label)

        self.bind(crop_pos=self.update_rect,
                  crop_size=self.update_rect,
                  input_width=self.update_crop_size,
                  input_height=self.update_crop_size,
                  scale=self.update_crop_size,
                  size=self.update_centering)

        # 初期設定の反映
        self.update_crop_size()

    def update_crop_size(self, *args):
        # 画像のサイズとスケールを考慮して矩形の範囲を設定
        scaled_width = self.input_width * self.scale
        scaled_height = self.input_height * self.scale

        # 矩形のサイズを設定
        self.crop_size[0] = scaled_width
        self.crop_size[1] = scaled_height

        # 中心にシフトするためのトランスレーションを設定
        self.update_centering()

    def update_rect(self, *args):
        self.white_line.rectangle = (self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1])
        self.black_line.rectangle = (self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1])
        self.label.x, self.label.y = self.crop_pos[0]-self.translate.x+dp(80), self.crop_pos[1]-self.translate.y-dp(40)
        self.label.text = str(int(self.crop_size[0]/self.scale)) + " x " + str(int(self.crop_size[1]/self.scale))

    def update_centering(self, *args):
        # 中心に移動するためのトランスレーションを設定
        self.translate.x = (self.width - self.crop_size[0]) / 2
        self.translate.y = (self.height - self.crop_size[1]) / 2

        self.update_rect()

    def on_touch_down(self, touch):
        self.corner_dragging = self.get_dragging_corner(touch)
        if self.corner_dragging is not None:
            return True
            
        return super(CropEditor, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.corner_dragging is not None:
            # 矩形のサイズを四隅のドラッグで変更
            self.resize_crop(touch)

        return super(CropEditor, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.corner_dragging = None

        return super(CropEditor, self).on_touch_up(touch)

    def get_dragging_corner(self, touch):
        x, y = touch.pos
        cx = self.crop_pos[0] + self.translate.x
        cy = self.crop_pos[1] + self.translate.y
        cw, ch = self.crop_size
        if abs(x - cx) < self.corner_threshold and abs(y - cy) < self.corner_threshold:
            return 'bottom_left'
        if abs(x - (cx + cw)) < self.corner_threshold and abs(y - cy) < self.corner_threshold:
            return 'bottom_right'
        if abs(x - cx) < self.corner_threshold and abs(y - (cy + ch)) < self.corner_threshold:
            return 'top_left'
        if abs(x - (cx + cw)) < self.corner_threshold and abs(y - (cy + ch)) < self.corner_threshold:
            return 'top_right'
        return None

    def resize_crop(self, touch):
        cx, cy = self.crop_pos
        cw, ch = self.crop_size
        if self.corner_dragging == 'bottom_left':
            new_cx = max(0, min(cx + cw - self.minimum_rect, touch.x - self.translate.x))
            new_cy = max(0, min(cy + ch - self.minimum_rect, touch.y - self.translate.y))
            self.crop_size = [cw + cx - new_cx, ch + cy - new_cy]
            self.crop_pos = [new_cx, new_cy]
            
        elif self.corner_dragging == 'bottom_right':
            new_cx = min(self.input_width*self.scale, max(self.minimum_rect, touch.x - cx - self.translate.x))
            new_cy = max(0, min(cy + ch - self.minimum_rect, touch.y - self.translate.y))
            self.crop_size = [new_cx, ch + cy - new_cy]
            self.crop_pos[1] = new_cy

        elif self.corner_dragging == 'top_left':
            new_cx = max(0, min(cx + cw - self.minimum_rect, touch.x - self.translate.x))
            new_cy = min(self.input_height*self.scale, max(self.minimum_rect, touch.y - cy - self.translate.y))
            self.crop_size = [cw + cx - new_cx, new_cy]
            self.crop_pos[0] = new_cx

        elif self.corner_dragging == 'top_right':
            new_cx = min(self.input_width*self.scale, max(self.minimum_rect, touch.x - cx - self.translate.x))
            new_cy = min(self.input_height*self.scale, max(self.minimum_rect, touch.y - cy - self.translate.y))
            self.crop_size = [new_cx, new_cy]

    def is_within_bounds(self, x, y):
        # 画像のスケーリングされた範囲内に四角形が収まるようにする
        return 0 <= x <= self.input_width - self.crop_size[0] and \
               0 <= y <= self.input_height - self.crop_size[1]

    def get_crop_info(self):
        # 上下反転させて返す
        crop_x = int(self.crop_pos[0] / self.scale)
        crop_y = int(self.input_height - (self.crop_pos[1]+self.crop_size[1]) / self.scale)
        crop_width = int(self.crop_size[0] / self.scale)
        crop_height = int(self.crop_size[1] / self.scale)
        return [crop_x, crop_y, crop_width, crop_height, self.scale]

class CropApp(KVApp):
    def build(self):
        root = KVRelativeLayout()
        # ここで縦横サイズとスケールを指定
        crop_editor = CropEditor(input_width=dp(800), input_height=dp(600), scale=1.0)
        crop_editor.pos = (dp(100), dp(100))
        root.add_widget(crop_editor)
        return root

if __name__ == '__main__':
    CropApp().run()

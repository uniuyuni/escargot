import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, PushMatrix, PopMatrix, Translate
from kivy.properties import NumericProperty, ListProperty, BooleanProperty
from kivy.uix.relativelayout import RelativeLayout

class CropWidget(Widget):
    width_input = NumericProperty(400)  # 元の画像の幅を指定するプロパティ
    height_input = NumericProperty(300)  # 元の画像の高さを指定するプロパティ
    scale = NumericProperty(1.0)
    crop_pos = ListProperty([0, 0])
    crop_size = ListProperty([0, 0])
    fixed_size = BooleanProperty(False)
    corner_threshold = NumericProperty(20)  # 四隅の操作ポイントの判定範囲を指定する変数

    def __init__(self, **kwargs):
        super(CropWidget, self).__init__(**kwargs)
        self.dragging = False
        self.resizing = False
        self.corner_dragging = None

        with self.canvas:
            PushMatrix()
            self.translate = Translate()
            Color(1, 1, 1, 1)
            self.white_line = Line(rectangle=(self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1]), width=2)
            Color(0, 0, 0, 1)
            self.black_line = Line(rectangle=(self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1]), width=1)
            PopMatrix()

        self.bind(crop_pos=self.update_rect,
                  crop_size=self.update_rect,
                  width_input=self.update_crop_size,
                  height_input=self.update_crop_size,
                  scale=self.update_crop_size,
                  size=self.update_centering)

        # 初期設定の反映
        self.update_crop_size()

    def update_crop_size(self, *args):
        # 画像のサイズとスケールを考慮して矩形の範囲を設定
        scaled_width = self.width_input * self.scale
        scaled_height = self.height_input * self.scale

        # 矩形のサイズを設定
        self.crop_size[0] = scaled_width
        self.crop_size[1] = scaled_height

        # 中心にシフトするためのトランスレーションを設定
        self.update_centering()

    def update_rect(self, *args):
        self.white_line.rectangle = (self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1])
        self.black_line.rectangle = (self.crop_pos[0], self.crop_pos[1], self.crop_size[0], self.crop_size[1])

    def update_centering(self, *args):
        # 中心に移動するためのトランスレーションを設定
        self.translate.x = (self.width - self.crop_size[0]) / 2
        self.translate.y = (self.height - self.crop_size[1]) / 2

        self.update_rect()

    def on_touch_down(self, touch):
        if self.is_touch_in_crop_area(touch):
            if not self.fixed_size:
                self.corner_dragging = self.get_dragging_corner(touch)
            if self.corner_dragging is None:
                self.dragging = True
            return True
        return super(CropWidget, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.dragging and not self.corner_dragging:
            # 矩形の位置をドラッグで移動
            new_x = touch.x - self.translate.x - self.crop_size[0] / 2
            new_y = touch.y - self.translate.y - self.crop_size[1] / 2
            if self.is_within_bounds(new_x, new_y):
                self.crop_pos = [new_x, new_y]
        elif self.corner_dragging is not None:
            # 矩形のサイズを四隅のドラッグで変更
            self.resize_crop(touch)
        return super(CropWidget, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.dragging = False
        self.corner_dragging = None
        return super(CropWidget, self).on_touch_up(touch)

    def is_touch_in_crop_area(self, touch):
        x, y = touch.pos
        cx = self.crop_pos[0] + self.translate.x
        cy = self.crop_pos[1] + self.translate.y
        cw, ch = self.crop_size
        return cx <= x <= cx + cw and cy <= y <= cy + ch

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
            new_cx = min(cx + cw - 16, touch.x - self.translate.x)
            new_cy = min(cy + ch - 16, touch.y - self.translate.y)
            self.crop_size = [cw + cx - new_cx, ch + cy - new_cy]
            self.crop_pos = [new_cx, new_cy]
            
        elif self.corner_dragging == 'bottom_right':
            new_cx = max(16, touch.x - cx - self.translate.x)
            new_cy = min(cy + ch - 16, touch.y - self.translate.y)
            self.crop_size = [new_cx, ch + cy - new_cy]
            self.crop_pos[1] = new_cy

        elif self.corner_dragging == 'top_left':
            new_cx = min(cx + cw - 16, touch.x - self.translate.x)
            new_cy = max(16, touch.y - cy - self.translate.y)
            self.crop_size = [cw + cx - new_cx, new_cy]
            self.crop_pos[0] = new_cx

        elif self.corner_dragging == 'top_right':
            new_cx = max(16, touch.x - cx - self.translate.x)
            new_cy = max(16, touch.y - cy - self.translate.y)
            self.crop_size = [new_cx, new_cy]

    def is_within_bounds(self, x, y):
        # 画像のスケーリングされた範囲内に四角形が収まるようにする
        return 0 <= x <= self.width_input - self.crop_size[0] and \
               0 <= y <= self.height_input - self.crop_size[1]

    def get_crop_area(self):
        # 上下反転させて返す
        crop_x = int(self.crop_pos[0] / self.scale)
        crop_y = int((self.height_input - self.crop_pos[1] - self.crop_size[1]) / self.scale)
        crop_width = int(self.crop_size[0] / self.scale)
        crop_height = int(self.crop_size[1] / self.scale)
        return [crop_x, crop_y, crop_width, crop_height]

class CropApp(App):
    def build(self):
        root = RelativeLayout()
        # ここで縦横サイズとスケールを指定
        crop_widget = CropWidget(width_input=800, height_input=600, scale=1.0)
        crop_widget.pos = [100, 100]
        root.add_widget(crop_widget)
        return root

if __name__ == '__main__':
    CropApp().run()

# mask_editor.py

import os
import numpy as np
import math
import cv2
from PIL import Image as PILImage, ImageDraw

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (
    NumericProperty, ObjectProperty, ListProperty,
    StringProperty, BooleanProperty
)
from kivy.graphics import (
    Color, Ellipse, Line, PushMatrix, PopMatrix, Rotate, Translate,
    ClearBuffers, ClearColor, Rectangle
)
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger
from functools import partial

# コントロールポイントのクラス
class ControlPoint(Widget):
    touching = BooleanProperty(False)
    is_center = BooleanProperty(False)  # 中心のコントロールポイントかどうか
    color = ListProperty([1, 1, 1])  # デフォルトの色
    ctrl_center = ListProperty([0, 0])
    type = ListProperty(['c', 0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = (20, 20)  # コントロールポイントのサイズ
        with self.canvas:
            PushMatrix()
            #self.translate = Translate()
            #self.rotate = Rotate(angle=0, origin=(0, 0))            
            self.color_instruction = Color(*self.color)
            self.circle = Ellipse(pos=self.pos, size=self.size)
            PopMatrix()
        self.bind(pos=self.update_graphics, color=self.update_color)

    def update_graphics(self, *args):
        self.circle.pos = self.pos #(self.x - self.width / 2, self.y - self.height / 2)

    def update_color(self, *args):
        self.color_instruction.rgb = self.color

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.touching = True
            return True
        return False

    def on_touch_move(self, touch):
        if self.touching:
            self.ctrl_center = [touch.x, touch.y]
            #self.center = [touch.x, touch.y]
            return True
        return False

    def on_touch_up(self, touch):
        if self.touching:
            self.touching = False
            return True
        return False

# マスクのベースクラス
class BaseMask(Widget):
    color = ListProperty([1, 0, 0, 0.5])  # デフォルトの半透明赤色
    selected = BooleanProperty(False)
    active = BooleanProperty(False)
    mask_texture = ObjectProperty(None)  # マスクのテクスチャ
    name = StringProperty("Mask")

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor  # MaskEditorのインスタンスへの参照
        self.control_points = []  # 標準のPythonリストで管理
        self.crop_info = editor.crop_info.copy()
        self.bind(active=self.on_active_changed)

    def on_active_changed(self, instance, value):
        if value:
            self.show_all_control_points()
        else:
            self.show_center_control_point_only()

    def show_all_control_points(self):
        self.opacity = 1
        for cp in self.control_points:
            cp.opacity = 1
            if cp.is_center:
                cp.color = [0, 0, 1]  # アクティブなマスクの中心点
            else:
                if cp.type[0] == 'r' or cp.type[0] == 's':
                    cp.color = [1, 1, 0]
                else:
                    cp.color = [1, 1, 1]  # 他のコントロールポイントは白色
        self.update_mask()

    def show_center_control_point_only(self):
        self.opacity = 0.2
        for cp in self.control_points:
            if cp.is_center:
                cp.opacity = 2
                cp.color = [1, 0, 0]  # 非アクティブなマスクの中心点は赤色
            else:
                cp.opacity = 0  # 非表示
        self.update_mask()

    def on_touch_down(self, touch):
        for cp in self.control_points:
            if cp.collide_point(*touch.pos):
                if cp.is_center:
                    self.editor.set_active_mask(self)
                    cp.on_touch_down(touch)
                    return True
                elif self.active:
                    cp.on_touch_down(touch)
                    return True
        return False

    def on_touch_move(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_move(touch)
                return True
        return False

    def on_touch_up(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_up(touch)
                return True
        return False

    def get_name(self):
        return self.name

    def serialize(self):
        pass

    def deserialize(self):
        pass

    def set_crop_info(self, crop_info):
        pass

    def update_mask(self, *args):
        pass  # 具体的な実装は各マスククラスで行う

    def get_mask_image(self):
        pass

    def draw_mask_to_fbo(self):
        pass  # 具体的な実装は各マスククラスで行う

# 円形グラデーションマスクのクラス
class CircularGradientMask(BaseMask):
    inner_radius_x = NumericProperty(0)
    inner_radius_y = NumericProperty(0)
    outer_radius_x = NumericProperty(0)
    outer_radius_y = NumericProperty(0)
    rotate_rad = NumericProperty(0)

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Circle"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            Color(*self.color)
            # 外側の円
            self.outer_line = Line(ellipse=(0, 0, 0, 0), width=2)
            # 内側の円
            self.inner_line = Line(ellipse=(0, 0, 0, 0), width=2)
            PopMatrix()

#        self.bind(inner_radius_x=self.update_mask, inner_radius_y=self.update_mask, 
#                  outer_radius_x=self.update_mask, outer_radius_y=self.update_mask,
#                  rotate_rad=self.update_mask)
        self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            self.center_x = touch.x
            self.center_y = touch.y
            self.inner_radius_x = 0
            self.inner_radius_y = 0
            self.outer_radius_x = 0
            self.outer_radius_y = 0
            return True
        else:
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.initializing:
            dx = touch.x - self.center_x
            dy = touch.y - self.center_y
            self.outer_radius_x = (dx**2 + dy**2) ** 0.5
            self.outer_radius_y = (dx**2 + dy**2) ** 0.5
            self.inner_radius_x = self.outer_radius_x * 0.7  # 内側の半径を仮設定
            self.inner_radius_y = self.outer_radius_y * 0.7  # 内側の半径を仮設定
            self.update_mask()
            return True
        else:
            return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 8つのコントロールポイントを作成
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        types  = ['x', 'r', 'y', 'r', 'x', 'r', 'y', 'r']
        self.control_points = []
        # 中心のコントロールポイント
        cp_center = ControlPoint()
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        for i, angle in enumerate(angles):
            # 内側のコントロールポイント
            cp_inner = ControlPoint()
            cp_inner.type = [types[i], angle]
            cp_inner.center = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            cp_inner.ctrl_center = cp_inner.center
            cp_inner.bind(ctrl_center=self.on_inner_control_point_move)
            self.control_points.append(cp_inner)
            self.add_widget(cp_inner)

            # 外側のコントロールポイント
            cp_outer = ControlPoint()
            cp_outer.type = [types[i], angle]
            cp_outer.center = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            cp_outer.ctrl_center = cp_outer.center
            cp_outer.bind(ctrl_center=self.on_outer_control_point_move)
            self.control_points.append(cp_outer)
            self.add_widget(cp_outer)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        center = self.editor.to_texture(*self.center)
        cx = center[0] / self.crop_info[4] + self.crop_info[0]
        cy = center[1] / self.crop_info[4] + self.crop_info[1]
        cx = cx - self.editor.image_size[0]/2
        cy = cy - self.editor.image_size[1]/2

        ix = self.inner_radius_x / self.crop_info[4]
        iy = self.inner_radius_y / self.crop_info[4]
        ox = self.outer_radius_x / self.crop_info[4]
        oy = self.outer_radius_y / self.crop_info[4]

        dict = {
            'center': [cx, cy],
            'inner_radius': [ix, iy],
            'outer_radius': [ox, oy],
            'rotate_rad': self.rotate_rad
        }
        return dict

    def deserialize(self, dict):
        cx, cy = dict['center']
        ix, iy = dict['inner_radius']
        ox, oy = dict['outer_radius']
        self.rotate_rad = dict['rotate_rad'] 

        cx = cx + self.editor.image_size[0]/2
        cy = cy + self.editor.image_size[1]/2
        cx = (cx - self.crop_info[0]) * self.crop_info[4] 
        cy = (cy - self.crop_info[1]) * self.crop_info[4]
        wx, wy = self.editor.to_window(*self.editor.pos)
        self.center = (cx+wx, cy+wy)
        self.inner_radius_x = ix * self.crop_info[4]
        self.inner_radius_y = iy * self.crop_info[4]
        self.outer_radius_x = ox * self.crop_info[4]
        self.outer_radius_y = oy * self.crop_info[4]

        # 描き直し
        self.update_control_points()
        self.update_mask()

    def set_crop_info(self, crop_info):
        # いったんグローバル座標に戻す
        if self.crop_info:
            cx, cy = self.editor.to_texture(*self.center)
            cx = cx / self.crop_info[4] + self.crop_info[0]
            cy = cy / self.crop_info[4] + self.crop_info[1]
            ix = self.inner_radius_x / self.crop_info[4]
            iy = self.inner_radius_y / self.crop_info[4]
            ox = self.outer_radius_x / self.crop_info[4]
            oy = self.outer_radius_y / self.crop_info[4]

            # 新しいローカル座標系に更新
            cx = (cx - crop_info[0]) * crop_info[4] 
            cy = (cy - crop_info[1]) * crop_info[4]
            wx, wy = self.editor.to_window(*self.editor.pos)
            self.center_x, self.center_y = cx+wx, cy+wy
            self.inner_radius_x = ix * crop_info[4]
            self.inner_radius_y = iy * crop_info[4]
            self.outer_radius_x = ox * crop_info[4]
            self.outer_radius_y = oy * crop_info[4]

            # 描き直し
            self.update_control_points()
            self.update_mask()

        # 新しい情報に更新
        self.crop_info = crop_info.copy()

    def rotate_ellipse(self, center, ellipse_angle, rotation_angle):

        # 楕円の中心 (cx, cy) を原点周りに rotation_angle だけ回転
        cx, cy = center
        new_cx = cx * math.cos(rotation_angle) - cy * math.sin(rotation_angle)
        new_cy = cx * math.sin(rotation_angle) + cy * math.cos(rotation_angle)
        
        # 楕円の回転角も rotation_angle を加算
        new_ellipse_angle = ellipse_angle + rotation_angle
        
        # 楕円の半径（内側・外側）は回転では変わらないため、そのまま返す
        return [new_cx, new_cy], new_ellipse_angle

    def set_center_rotate(self, rad):
        dict = self.serialize()
        dict['center'], dict['rotate_rad'] = self.rotate_ellipse(dict['center'], dict['rotate_rad'], rad)
        self.deserialize(dict)

    def calculate_point(self, radius_x, radius_y, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        dx = radius_x * np.cos(angle_rad)
        dy = radius_y * np.sin(angle_rad)
        new_r_x = dx * np.cos(-self.rotate_rad) + dy * np.sin(-self.rotate_rad)
        new_r_y = dy * np.cos(-self.rotate_rad) - dx * np.sin(-self.rotate_rad)
        return (float(self.center_x + new_r_x), float(self.center_y + new_r_y))

    def calculate_rotate(self, radius_x, radius_y, angle_deg, dx, dy):
        angle_rad = np.deg2rad(angle_deg)
        px = radius_x * np.cos(angle_rad)
        py = radius_y * np.sin(angle_rad)
        rotate_rad = np.arctan2(dy, dx)
        new_rad = rotate_rad-np.arctan2(py, px)
        return float(new_rad)

    def on_center_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            self.center_x += dx
            self.center_y += dy
            for cp in self.control_points:
                if cp != instance:
                    cp.center_x += dx
                    cp.center_y += dy
            self.update_control_points()
            self.update_mask()

    def update_ellipse(self, dx, dy):
        # 回転角の変化に応じて、半径を更新
        new_r_x = dx * np.cos(self.rotate_rad) + dy * np.sin(self.rotate_rad)
        new_r_y = dy * np.cos(self.rotate_rad) - dx * np.sin(self.rotate_rad)
        
        return (abs(float(new_r_x)), abs(float(new_r_y)))

    def on_outer_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            sx = self.inner_radius_x / self.outer_radius_x
            sy = self.inner_radius_y / self.outer_radius_y
            if instance.type[0] == 'x':
                self.outer_radius_x, _ = self.update_ellipse(dx, dy)
                self.inner_radius_x = self.outer_radius_x * sx
                self.outer_radius_x = max(10, max(self.outer_radius_x, self.inner_radius_x))
            elif instance.type[0] == 'y':
                _, self.outer_radius_y = self.update_ellipse(dx, dy)
                self.inner_radius_y = self.outer_radius_y * sy
                self.outer_radius_y = max(10, max(self.outer_radius_y, self.inner_radius_y))
            elif instance.type[0] == 'r':
                self.rotate_rad = self.calculate_rotate(self.outer_radius_x, self.outer_radius_y, instance.type[1], dx, dy)
            self.update_control_points()
            self.update_mask()

    def on_inner_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            sx = self.inner_radius_x / self.outer_radius_x
            sy = self.inner_radius_y / self.outer_radius_y
            if instance.type[0] == 'x':
                self.inner_radius_x, _ = self.update_ellipse(dx, dy)
                self.inner_radius_x = max(5, min(self.inner_radius_x, self.outer_radius_x-10))
                #self.inner_radius_y = self.outer_radius_y * sx
            elif instance.type[0] == 'y':
                _, self.inner_radius_y = self.update_ellipse(dx, dy)
                self.inner_radius_y = max(5, min(self.inner_radius_y, self.outer_radius_y-10))
                #self.inner_radius_x = self.outer_radius_x * sy
            elif instance.type[0] == 'r':
                self.rotate_rad = self.calculate_rotate(self.inner_radius_x, self.inner_radius_y, instance.type[1], dx, dy)
            self.update_control_points()
            self.update_mask()

    def update_control_points(self):
        # コントロールポイントの位置を更新
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        cp_center = self.control_points[0]
        cp_center.center = self.center
        index = 1  # 0は中心点
        for angle in angles:
            cp_inner = self.control_points[index]
            cp_inner.center_x, cp_inner.center_y = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            index += 1
            cp_outer = self.control_points[index]
            cp_outer.center_x, cp_outer.center_y = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            index += 1

    def update_mask(self, *args):
        #self.canvas.clear()
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            self.translate.x, self.translate.y = self.center_x, self.center_y
            self.rotate.angle = float(np.rad2deg(self.rotate_rad))
            self.outer_line.ellipse = (-self.outer_radius_x, -self.outer_radius_y, self.outer_radius_x*2, self.outer_radius_y*2)
            self.inner_line.ellipse = (-self.inner_radius_x, -self.inner_radius_y, self.inner_radius_x*2, self.inner_radius_y*2)
        
        self.draw_mask_to_fbo()

    def draw_mask_to_fbo(self):
        if not self.crop_info:
            Logger.warning(f"{self.__class__.__name__}: crop_infoが未設定。")
            return

        if self.active == True:

            # パラメータ設定
            image_size = (int(self.crop_info[2]), int(self.crop_info[3]))
            center = self.editor.to_texture(*self.center)
            inner_axes = (self.inner_radius_x, self.inner_radius_y)  # 内側の楕円の半径(x, y)
            outer_axes = (self.outer_radius_x, self.outer_radius_y)  # 外側の楕円の半径(x, y)

            # グラデーションを描画
            gradient_image = self.draw_elliptical_gradient(image_size, center, inner_axes, outer_axes, self.rotate_rad, (1, 0, 0, 1))

            # イメージを描画してもらう
            self.editor.draw_mask_image(gradient_image)


    def draw_elliptical_gradient(self, image_size, center, inner_axes, outer_axes, angle_rad, color):
        # 画像の初期化（黒背景、RGBA）
        image = np.zeros((image_size[1], image_size[0], 4), dtype=np.float32)

        # RGBの値を0-1に正規化
        r, g, b, a = color[0], color[1], color[2], color[3]

        # 回転角をラジアンに変換
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 画像の座標系を作成
        y_indices, x_indices = np.indices((image_size[1], image_size[0]))
        x_indices = x_indices - center[0]
        y_indices = y_indices - center[1]

        # 回転を適用して軸を調整
        x_rotated = x_indices * cos_angle + y_indices * sin_angle
        y_rotated = -x_indices * sin_angle + y_indices * cos_angle

        # 内側と外側の楕円の距離を計算
        inner_ellipse_distance = (x_rotated**2 / inner_axes[0]**2 + y_rotated**2 / inner_axes[1]**2)
        outer_ellipse_distance = (x_rotated**2 / outer_axes[0]**2 + y_rotated**2 / outer_axes[1]**2)

        # グラデーションの計算
        # 平方根を用いて変化を緩やかにする
        gradient = np.clip(np.sqrt((outer_ellipse_distance - 1) / (outer_ellipse_distance - inner_ellipse_distance + 1e-5)), 0, 1)

        # 内側の楕円を完全に白にする
        #gradient[inner_ellipse_distance < 1] = 1
        # 外側の楕円を完全に黒にする
        gradient[outer_ellipse_distance > 1] = 0

        # RGBA画像の生成
        image[..., 0] = r  # Rチャンネル
        image[..., 1] = g  # Gチャンネル
        image[..., 2] = b  # Bチャンネル
        image[..., 3] = gradient * a     # Aチャンネル

        return image


# GradientMask クラス
class GradientMask(BaseMask):
    start_point = ListProperty([0, 0])    # グラデーションの開始点
    end_point = ListProperty([0, 0])      # グラデーションの終点
    rotate_rad = NumericProperty(0)            # グラデーションの角度
    
    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Line"
        self.initializing = True  # 初期配置中かどうか
#        self.bind(start_point=self.update_mask, end_point=self.update_mask,
#                  rotate_rad=self.update_mask, center=self.update_mask)

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            Color(*self.color)
            self.start_line = Line(points=(0, 0, 0, 0), width=2)
            self.center_line = Line(points=(0, 0, 0, 0), width=2)
            self.end_line = Line(points=(0, 0, 0, 0), width=2)
            PopMatrix()

        self.update_mask()
    
    def on_touch_down(self, touch):
        if self.initializing:
            self.center = (touch.x, touch.y)
            self.start_point = [touch.x, touch.y]
            return True
        else:
            return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.initializing:
            self.end_point = [touch.x, touch.y]
            self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                           (self.start_point[1] + self.end_point[1]) / 2]
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = float(np.arctan2(dy, dx))
            self.update_mask()
            return True
        else:
            return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)
    
    def serialize(self):
        start_point = self.editor.to_texture(*self.start_point)
        sx = start_point[0] / self.crop_info[4] + self.crop_info[0]
        sy = start_point[1] / self.crop_info[4] + self.crop_info[1]
        sx = sx - self.editor.image_size[0]/2
        sy = sy - self.editor.image_size[1]/2

        end_point = self.editor.to_texture(*self.end_point)
        ex = end_point[0] / self.crop_info[4] + self.crop_info[0]
        ey = end_point[1] / self.crop_info[4] + self.crop_info[1]
        ex = ex - self.editor.image_size[0]/2
        ey = ey - self.editor.image_size[1]/2

        dict = {
            'start_point': [sx, sy],
            'end_point': [ex, ey],
            'rotate_rad': self.rotate_rad
        }
        return dict

    def deserialize(self, dict):
        wx, wy = self.editor.to_window(*self.editor.pos)
        self.rotate_rad = dict['rotate_rad']

        sx, sy = dict['start_point']
        sx = sx + self.editor.image_size[0]/2
        sy = sy + self.editor.image_size[1]/2
        sx = (sx - self.crop_info[0]) * self.crop_info[4] 
        sy = (sy - self.crop_info[1]) * self.crop_info[4]
        self.start_point = (sx+wx, sy+wy)

        ex, ey = dict['end_point']
        ex = ex + self.editor.image_size[0]/2
        ey = ey + self.editor.image_size[1]/2
        ex = (ex - self.crop_info[0]) * self.crop_info[4] 
        ey = (ey - self.crop_info[1]) * self.crop_info[4]
        self.end_point = (ex+wx, ey+wy)

        self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                       (self.start_point[1] + self.end_point[1]) / 2]
        
        # 描き直し
        self.update_control_points()
        self.update_mask()

    def set_crop_info(self, crop_info):
        # いったんグローバル座標に戻す
        if self.crop_info:
            cx, cy = self.editor.to_texture(*self.center)
            cx = cx / self.crop_info[4] + self.crop_info[0]
            cy = cy / self.crop_info[4] + self.crop_info[1]
            sx, sy = self.editor.to_texture(*self.start_point)
            sx = sx / self.crop_info[4] + self.crop_info[0]
            sy = sy / self.crop_info[4] + self.crop_info[1]
            ex, ey = self.editor.to_texture(*self.end_point)
            ex = ex / self.crop_info[4] + self.crop_info[0]
            ey = ey / self.crop_info[4] + self.crop_info[1]

            # 新しいローカル座標系に更新
            wx, wy = self.editor.to_window(*self.editor.pos)
            cx = (cx - crop_info[0]) * crop_info[4] 
            cy = (cy - crop_info[1]) * crop_info[4]
            self.center = (cx+wx, cy+wy)
            sx = (sx - crop_info[0]) * crop_info[4]
            sy = (sy - crop_info[1]) * crop_info[4]
            self.start_point = [sx+wx, sy+wy]
            ex = (ex - crop_info[0]) * crop_info[4]
            ey = (ey - crop_info[1]) * crop_info[4]
            self.end_point = [ex+wx, ey+wy]

            # 描き直し
            self.update_control_points()
            self.update_mask()

        # 新しい情報に更新
        self.crop_info = crop_info.copy()

    def rotate_point(self, pos, rotation_rad):

        # 楕円の中心 (cx, cy) を原点周りに rotation_angle だけ回転
        cx, cy = pos
        new_cx = cx * math.cos(rotation_rad) - cy * math.sin(rotation_rad)
        new_cy = cx * math.sin(rotation_rad) + cy * math.cos(rotation_rad)
        
        return [new_cx, new_cy]

    def set_center_rotate(self, rad):
        dict = self.serialize()
        dict['start_point'] = self.rotate_point(dict['start_point'], rad)
        dict['end_point'] = self.rotate_point(dict['end_point'], rad)
        dict['rotate_rad'] = dict['rotate_rad'] + rad
        self.deserialize(dict)

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint()
        cp_center.center = self.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)
    
        # グラデーションの開始点と終点のコントロールポイント
        cp_start = ControlPoint()
        cp_start.center = self.start_point
        cp_start.type = ['s', 0]
        cp_start.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_start)
        self.add_widget(cp_start)
    
        cp_end = ControlPoint()
        cp_end.center = self.end_point
        cp_end.type = ['e', 0]
        cp_end.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_end)
        self.add_widget(cp_end)
    
        if not self.active:
            self.show_center_control_point_only()
    
    def calculate_point(self, point, dir):
        r = np.sqrt((point[0]-self.center_x)**2+(point[1]-self.center_y)**2)
        dx = dir * r
        dy = 0.
        new_r_x = dx * np.cos(-self.rotate_rad) + dy * np.sin(-self.rotate_rad)
        new_r_y = dy * np.cos(-self.rotate_rad) - dx * np.sin(-self.rotate_rad)
        return (float(self.center_x + new_r_x), float(self.center_y + new_r_y))

    def calculate_line_init(self, point, dir):
        r = np.sqrt((point[0]-self.start_point[0])**2+(point[1]-self.start_point[1])**2)
        dx = dir * r
        dy = -self.editor.width
        new_dx1 = dx
        new_dy1 = dy
        dx = dir * r
        dy = self.editor.width
        new_dx2 = dx
        new_dy2 = dy
        return (float(new_dx1), float(new_dy1), float(new_dx2), float(new_dy2))

    def calculate_line(self, point, dir):
        r = np.sqrt((point[0]-self.center_x)**2+(point[1]-self.center_y)**2)
        dx = dir * r
        dy = -self.editor.width
        new_dx1 = dx
        new_dy1 = dy
        dx = dir * r
        dy = self.editor.width
        new_dx2 = dx
        new_dy2 = dy
        return (float(new_dx1), float(new_dy1), float(new_dx2), float(new_dy2))

    def on_center_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center[0]
            dy = instance.ctrl_center[1] - self.center[1]
            self.start_point = [self.start_point[0] + dx, self.start_point[1] + dy]
            self.end_point = [self.end_point[0] + dx, self.end_point[1] + dy]
            self.center = [self.center[0] + dx, self.center[1] + dy]
            for cp in self.control_points:
                if cp != instance:
                    cp.center_x += dx
                    cp.center_y += dy
            self.update_control_points()
            self.update_mask()
    
    def on_control_point_move(self, instance, value):
        if self.active:
            if instance == self.control_points[1]:
                self.start_point = [instance.ctrl_center[0], instance.ctrl_center[1]]
                dx = self.center_x - self.start_point[0]
                dy = self.center_y - self.start_point[1]
                self.end_point[0] = self.center_x + dx
                self.end_point[1] = self.center_y + dy
            elif instance == self.control_points[2]:
                self.end_point = [instance.ctrl_center[0], instance.ctrl_center[1]]
                dx = self.center_x - self.end_point[0]
                dy = self.center_y - self.end_point[1]
                self.start_point[0] = self.center_x + dx
                self.start_point[1] = self.center_y + dy
            # 再計算
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = float(np.arctan2(dy, dx))
            #self.width = np.hypot(dx, dy)
            self.update_control_points()
            self.update_mask()

    def update_control_points(self):
        # コントロールポイントの位置を更新
        cp_center = self.control_points[0]
        cp_center.center = self.center
        cp_start = self.control_points[1]
        cp_start.center = self.start_point
        cp_end = self.control_points[2]
        cp_end.center = self.end_point
    
    def update_mask(self, *args):
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            Logger.debug("GradientMask: image_sizeが未設定のため、マスクの更新をスキップ")
            return

        with self.canvas:
            self.rotate.angle = float(np.rad2deg(self.rotate_rad))
            if self.initializing:
                self.translate.x, self.translate.y = self.start_point[0], self.start_point[1]
                self.start_line.points = self.calculate_line_init(self.start_point, 0)
                self.center_line.points = self.calculate_line_init(self.center, +1)
                self.end_line.points = self.calculate_line_init(self.end_point, +1)
            else:
                self.translate.x, self.translate.y = self.center_x, self.center_y
                self.start_line.points = self.calculate_line(self.start_point, -1)
                self.center_line.points = self.calculate_line(self.center, 0)
                self.end_line.points = self.calculate_line(self.end_point, +1)

        self.draw_mask_to_fbo()
    
    def draw_mask_to_fbo(self):
        if not self.crop_info:
            Logger.warning(f"{self.__class__.__name__}: crop_infoが未設定。")
            return

        if self.active == True:
            
            # パラメータ設定
            image_size = (int(self.crop_info[2]), int(self.crop_info[3]))
            center = self.editor.to_texture(*self.center)
            start_point = self.editor.to_texture(*self.start_point)
            end_point = self.editor.to_texture(*self.end_point)

            # グラデーションを描画
            gradient_image = self.draw_gradient(image_size, center, start_point, end_point, self.rotate_rad, (1, 0, 0, 1))

            # イメージを描画してもらう
            self.editor.draw_mask_image(gradient_image)

    def draw_gradient(self, image_size, center, start_point, end_point, angle, color, smoothness=2):

        width, height = image_size
        img = np.zeros((height, width, 4), dtype=np.float32)  # 開始点前はすべて(0, 0, 0, 0)

        # ベクトル計算のための設定
        start_x, start_y = start_point
        end_x, end_y = end_point
        vec_start_end = np.array([end_x - start_x, end_y - start_y])
        length_start_end = np.linalg.norm(vec_start_end)
        if length_start_end == 0:
            return img  # 開始点と終了点が同じ場合はそのまま透明を返す

        # 正規化した方向ベクトル
        unit_vec_start_end = vec_start_end / length_start_end

        # ピクセルの座標
        y_coords, x_coords = np.indices((height, width))
        pixel_vectors = np.stack((x_coords - start_x, y_coords - start_y), axis=-1)

        # 各ピクセルの射影を計算（開始点からの距離をグラデーションの方向に投影）
        projection_lengths = np.dot(pixel_vectors, unit_vec_start_end)

        # 射影の割合tを計算し、範囲を制限
        t = np.clip(projection_lengths / length_start_end, 0, 1)

        # グラデーションの急激さを調整（逆数のべき乗による非線形変換）
        t = t ** (1 / smoothness)

        # tが0より大きい場所にのみ色を設定（開始点以前は透明）
        mask = projection_lengths >= 0
        color_array = np.array(color).reshape(1, 1, 4)
        img[mask] = t[mask, np.newaxis] * color_array

        return img

# 自由描画マスクのクラス
class FreeDrawMask(BaseMask):
    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.points = []
        self.line = None
        self.drawing = False
        self.initializing = True
        self.brush_size = 10  # ブラシサイズ
        self.brush_cursor = None  # ブラシカーソル

        with self.canvas:
            self.brush_color = Color(1, 1, 1, 0.5)
            self.brush_cursor = Ellipse(size=(self.brush_size, self.brush_size))

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.initializing:
                self.points = [touch.x, touch.y]
                with self.canvas:
                    Color(*self.color)
                    self.line = Line(points=self.points, width=self.brush_size)
                self.drawing = True
                self.update_brush_cursor(touch.x, touch.y)
                Logger.debug("FreeDrawMask: タッチ開始")
                return True
            else:
                return super().on_touch_down(touch)
        return False

    def on_touch_move(self, touch):
        if self.drawing:
            self.points.extend([touch.x, touch.y])
            self.line.points = self.points
            self.update_brush_cursor(touch.x, touch.y)
            Logger.debug(f"FreeDrawMask: タッチ移動 to {touch.pos}")
            return True
        return False

    def on_touch_up(self, touch):
        if self.drawing:
            self.drawing = False
            self.initializing = False
            self.create_control_point()
            self.editor.set_active_mask(self)
            self.brush_cursor.size = (0, 0)
            Logger.debug("FreeDrawMask: タッチ終了")
            return True
        return super().on_touch_up(touch)

    def create_control_point(self):
        # 自由描画マスクの中心点を計算
        xs = self.points[::2]
        ys = self.points[1::2]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        # 中心のコントロールポイント
        cp_center = ControlPoint()
        cp_center.center_x = center_x
        cp_center.center_y = center_y
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(center_x=self.on_center_control_point_move,
                       center_y=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()
        Logger.debug("FreeDrawMask: コントロールポイントを作成しました。")

    def on_center_control_point_move(self, instance, value):
        if self.active:
            avg_x = sum(self.points[::2]) / len(self.points[::2])
            avg_y = sum(self.points[1::2]) / len(self.points[1::2])
            dx = instance.center_x - avg_x
            dy = instance.center_y - avg_y
            self.points = [x + dx if i % 2 == 0 else x + dy for i, x in enumerate(self.points)]
            self.line.points = self.points
            self.update_mask()
            Logger.debug(f"FreeDrawMask: 中心移動 dx={dx}, dy={dy}")

    def update_brush_cursor(self, x, y):
        self.brush_cursor.pos = (x - self.brush_size / 2, y - self.brush_size / 2)
        self.brush_cursor.size = (self.brush_size, self.brush_size)
        Logger.debug(f"FreeDrawMask: ブラシカーソル更新 to ({x}, {y})")

    def update_mask(self, *args):
        # 自由描画マスクでは、描画済みのラインがそのまま残るので特に更新処理は不要
        pass

    def draw_mask_to_fbo(self):
        if not self.image_size or not self.scale_factor:
            Logger.warning(f"{self.__class__.__name__}: image_size または scale_factor が未設定。")
            return
        with self.fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            Color(1, 1, 1, 1)
            scaled_points = []
            for i in range(0, len(self.points), 2):
                x = self.points[i] * self.scale_factor[0]
                y = self.image_size[1] - self.points[i + 1] * self.scale_factor[1]
                scaled_points.extend([x, y])
            Line(points=scaled_points, width=self.brush_size * self.scale_factor[0])
        Logger.debug(f"{self.__class__.__name__}: FBOにマスクを描画しました。")

# マスクレイヤーの管理クラス
class MaskLayer(BoxLayout):
    mask = ObjectProperty(None)
    mask_name = StringProperty('')

    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = 30
        self.mask = mask
        self.mask_name = mask.get_name()

        btn_delete = Button(text='Del', size_hint=(0.4, 1))
        btn_delete.bind(on_press=self.delete_mask)

        label = Button(text=self.mask_name, size_hint=(0.8, 1), background_normal='', background_color=(0,0,0,1))
        label.bind(on_press=self.set_active)

        self.add_widget(label)
        self.add_widget(btn_delete)

    def delete_mask(self, instance):
        self.parent.remove_widget(self)
        self.mask.parent.remove_widget(self.mask)

    def set_active(self, instance):
        self.mask.editor.set_active_mask(self.mask)

# メインのエディタークラス
class MaskEditor2(FloatLayout):
    mask_layers = ListProperty([])
    active_mask = ObjectProperty(None, allownone=True)
    image_size = ListProperty([0, 0])  # 画像のサイズを保持
    crop_info = ListProperty([0, 0, 0, 0, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.image_widget = Image(allow_stretch=False, keep_ratio=True)
        self.image_widget.pos_hint = {"x":0, "top":1}
        self.add_widget(self.image_widget)

        self.mask_container = Widget()
        self.add_widget(self.mask_container)
        self.rectangle = None

        self.ui_layout = BoxLayout(orientation='vertical', size_hint=(0.2, 1))
        self.ui_layout.pos_hint = {"x":0, "top":1}
        self.add_widget(self.ui_layout)
        self.create_ui()
        self.current_mask_type = None

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        Logger.info("MaskEditor: 初期化完了")

    def imread(self, image_source, dt):
        # 画像の読み込みとサイズの取得
        if not os.path.isfile(image_source):
            Logger.error(f"MaskEditor: 画像ファイルが見つかりません: {image_source}")
            return False
        
        self.image_widget.source = image_source

        self.image_size = self.image_widget.texture.size
        if self.image_size[0] >= self.image_size[1]:
            scale = self.size[0] / self.image_size[0]
        else:
            scale = self.size[1] / self.image_size[1]
        self.crop_info = [0, 0, self.size[0], self.size[1], scale]

        # 既存のマスクに対するタッチイベントを処理
        for mask in reversed(self.mask_layers):
            mask.set_crop_info(self.crop_info)     

        return True   

    def create_ui(self):
        # マスクタイプ選択ボタン
        btn_circular = Button(text='Circle', size_hint=(1, 0.1))
        btn_circular.bind(on_press=self.select_circular_gradient_mask)
        self.ui_layout.add_widget(btn_circular)

        btn_gradient = Button(text='Line', size_hint=(1, 0.1))
        btn_gradient.bind(on_press=self.select_gradient_mask)
        self.ui_layout.add_widget(btn_gradient)

        btn_free_draw = Button(text='Draw', size_hint=(1, 0.1))
        btn_free_draw.bind(on_press=self.select_free_draw_mask)
        self.ui_layout.add_widget(btn_free_draw)

        # マスクレイヤー表示
        self.layer_list = BoxLayout(orientation='vertical', size_hint=(1, 0.7))
        self.ui_layout.add_widget(self.layer_list)

    def draw_mask_image(self, rgba):
        texture = Texture.create(size=(rgba.shape[1], rgba.shape[0]), colorfmt='rgba')
        texture.blit_buffer(rgba.tobytes(), colorfmt='rgba', bufferfmt='float')
        #texture.flip_vertical()
        if self.rectangle is not None:
            self.mask_container.canvas.before.remove(self.rectangle)
        with self.mask_container.canvas.before:
            pos = self.to_window(*self.pos)
            Color(1, 0, 0, 0.5)
            self.rectangle = Rectangle(texture=texture, pos=pos, size=self.size)

    def select_circular_gradient_mask(self, instance):
        self.current_mask_type = 'circular_gradient'

    def select_gradient_mask(self, instance):
        self.current_mask_type = 'gradient'

    def select_free_draw_mask(self, instance):
        self.current_mask_type = 'free_draw'

    def on_touch_down(self, touch):
        if self.current_mask_type:
            # 画像サイズがまだ設定されていない場合、マスクの作成をスキップ
            if self.image_size == [0, 0]:
                Logger.warning("MaskEditor: 画像がまだロードされていません。マスクを追加できません。")
                return False
            
            # マスク作成
            if self.current_mask_type == 'circular_gradient':
                mask = CircularGradientMask(editor=self)
            elif self.current_mask_type == 'gradient':
                mask = GradientMask(editor=self)
            elif self.current_mask_type == 'free_draw':
                mask = FreeDrawMask(editor=self)
            else:
                Logger.error(f"MaskEditor: 不明なマスクタイプ: {self.current_mask_type}")
                return False
            
            self.mask_container.add_widget(mask)
            self.mask_layers.append(mask)
            # レイヤーUIに追加
            layer = MaskLayer(mask=mask)
            self.layer_list.add_widget(layer)
            mask.on_touch_down(touch)
            self.current_mask_type = None

            Logger.debug(f"MaskEditor: マスク {mask.__class__.__name__} を追加しました。")
            return True

        else:
            # 既存のマスクに対するタッチイベントを処理
            for mask in reversed(self.mask_layers):
                if mask.on_touch_down(touch):
                    return True
            return super().on_touch_down(touch)

    def set_active_mask(self, mask):
        if self.active_mask and self.active_mask != mask:
            self.active_mask.active = False
            Logger.debug(f"MaskEditor: マスク {self.active_mask.__class__.__name__} を非アクティブにしました。")
        self.active_mask = mask
        self.active_mask.active = True
        Logger.debug(f"MaskEditor: マスク {self.active_mask.__class__.__name__} をアクティブにしました。")

    def to_texture(self, x, y):
        wpos = self.to_window(*self.pos)
        return (x-wpos[0], y-wpos[1])
    
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'z':
            if self.crop_info[4] == 1.0:
                if self.image_size[0] >= self.image_size[1]:
                    scale = self.size[0] / self.image_size[0]
                else:
                    scale = self.size[1] / self.image_size[1]
                self.crop_info[4] = scale
            else:
                self.crop_info[4] = 1.0
        elif keycode[1] == 'up':
            self.crop_info[1] -= 100
        elif keycode[1] == 'down':
            self.crop_info[1] += 100
        elif keycode[1] == 'left':
            self.crop_info[0] -= 100
        elif keycode[1] == 'right':
            self.crop_info[0] += 100
        elif keycode[1] == 't':
            for mask in reversed(self.mask_layers):
                dict = mask.serialize()        
                mask.deserialize(dict)
            return True

        elif keycode[1] == 'r':
            for mask in reversed(self.mask_layers):
                dict = mask.set_center_rotate(math.radians(1))        
            return True
        
        # 既存のマスクに対するタッチイベントを処理
        for mask in reversed(self.mask_layers):
            mask.set_crop_info(self.crop_info)        
    
        return True
    
# アプリケーションクラス
class MaskEditor2App(App):
    def build(self):
        # 画像ファイルのパスを正しく設定してください
        image_path = 'your_image.jpg'

        box0 = BoxLayout()
        box0.orientation = 'vertical'
        box0.size = (Window.width, Window.height)
        box0.size_hint = (None, None)

        box1 = FloatLayout()
        box1.size_hint_y = 2
        box0.add_widget(box1)
        box2 = BoxLayout()
        box2.orientation = 'horizontal'
        box2.size_hint_y = 6
        box0.add_widget(box2)
        box3 = FloatLayout()
        box3.size_hint_y = 2
        box0.add_widget(box3)

        box4 = FloatLayout()
        box4.size_hint_x = 2
        box2.add_widget(box4)
        editor = MaskEditor2()
        editor.size_hint_x = 6
        box2.add_widget(editor)
        box5 = FloatLayout()
        box5.size_hint_x = 2
        box2.add_widget(box5)

        Clock.schedule_once(partial(editor.imread, image_path), -1)

        return box0

if __name__ == '__main__':
    MaskEditor2App().run()

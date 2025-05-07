# mask_editor2.py

import os
import numpy as np
import math
import cv2

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (
    NumericProperty, ObjectProperty, ListProperty,
    StringProperty, BooleanProperty, Property
)
from kivy.graphics import (
    Color, Ellipse, Line, PushMatrix, PopMatrix, Rotate, Translate,
    ClearBuffers, ClearColor, Rectangle
)
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger
from functools import partial
import importlib

import core
import util
import effects
import export

MASKTYPE_CIRCULAR = 'circular'
MASKTYPE_GRADIENT = 'gradient'
MASKTYPE_FULL = 'full'
MASKTYPE_FREEDRAW = 'free_draw'
MASKTYPE_SEGMENT = 'segment'

# コントロールポイントのクラス
class ControlPoint(Widget):
    touching = BooleanProperty(False)
    is_center = BooleanProperty(False)  # 中心のコントロールポイントかどうか
    color = ListProperty([0, 0, 0])  # デフォルトの色
    ctrl_center = ListProperty([0, 0])
    type = ListProperty(['c', 0])

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        with self.canvas:
            PushMatrix()
            self.translate = Translate()
            #self.rotate = Rotate(angle=0, origin=(0, 0))            
            self.color_instruction = Color(*self.color)
            self.circle = Ellipse(pos=(-10, -10), size=(20, 20))
            PopMatrix()
        self.center = (0, 0)
        #self.update_graphics()
        self.bind(center=self.update_graphics, color=self.update_color)

    def update_graphics(self, *args):
        cx, cy = self.editor.tcg_to_window(*self.center)
        self.translate.x = cx
        self.translate.y = cy
        self.size = self.editor.world_to_tcg_scale(20, 20)

    def update_color(self, *args):
        self.color_instruction.rgb = self.color

    def on_touch_down(self, touch):
        self.touching = True
        return True

    def on_touch_move(self, touch):
        if self.touching:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.ctrl_center = [cx, cy]
            #self.cnter = (cx, cy)
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
    name = StringProperty("Mask")

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor  # MaskEditorのインスタンスへの参照
        self.control_points = []  # 標準のPythonリストで管理
        self.bind(active=self.on_active_changed)

        # エフェクトパラメータ保持
        self.effects = effects.create_effects()
        self.effects_param = {
            'color_temperature_switch': False,
        }

        self.is_draw_mask = True
        self.image_mask_cache = None
        self.image_mask_cache_hash = None

    def start(self):
        pass

    def end(self):
        pass

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
        self.is_draw_mask = True
        self.update_mask()

    def show_center_control_point_only(self):
        self.opacity = 0.2
        for cp in self.control_points:
            if cp.is_center:
                cp.opacity = 2
                cp.color = [1, 0, 0]  # 非アクティブなマスクの中心点は赤色
            else:
                cp.opacity = 0  # 非表示
        self.is_draw_mask = False
        self.update_mask()

    def on_touch_down(self, touch):
        for cp in self.control_points:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if cp.collide_point(cx, cy):
                if cp.is_center:
                    self.editor.set_active_mask(self)
                    cp.on_touch_down(touch)
                    self.is_draw_mask = True
                    return True
                elif self.active:
                    cp.on_touch_down(touch)
                    self.is_draw_mask = True
                    return True
        return False

    def on_touch_move(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_move(touch)
                self.is_draw_mask = True
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

    def update(self):
        if len(self.control_points) > 0:
            cp_center = self.control_points[0]
            #cp_center.property('ctrl_center').dispatch(cp_center)
            cp_center.ctrl_center[0] += float(np.finfo(np.float32).eps)
            cp_center.ctrl_center[0] -= float(np.finfo(np.float32).eps)

    def update_control_points(self):
        pass

    def draw_mask_to_fbo(self):
        if not self.editor.disp_info:
            Logger.warning(f"{self.__class__.__name__}: disp_infoが未設定。")
            return

        if self.active == True:
            mask_image = self.get_mask_image()

            # イメージを描画してもらう
            self.editor.draw_mask_image(mask_image)

    def draw_hls_mask(self, image):
        himg = self.draw_hue_mask(image)
        limg = self.draw_lum_mask(himg)
        simg = self.draw_sat_mask(limg)
        
        return simg
    
    def apply_mask_blur(self, image):
        ksize = max(0, self.effects_param.get('mask2_blur', 0)-1)
        img2 = core.gaussian_blur(image, (ksize, ksize))
        return img2

    def draw_hue_mask(self, image):
        if self.editor.crop_image_hls is not None:            
            himg = self.editor.crop_image_hls[..., 0]
            
            hmin = self.effects_param.get('mask2_hue_min', 0) / 255
            hmax = self.effects_param.get('mask2_hue_max', 359) / 359
            if (hmin != 0) or (1 != hmax):
                himg = np.where((himg < hmin) | (hmax < himg), 0, image)
            else:
                himg = image

            return himg
        
        return None

    def draw_lum_mask(self, image):
        if self.editor.crop_image_hls is not None:            
            limg = self.editor.crop_image_hls[..., 1]
            
            lmin = self.effects_param.get('mask2_lum_min', 0) / 255
            lmax = self.effects_param.get('mask2_lum_max', 255) / 255
            if (lmin != 0) or (1 != lmax):
                limg = np.where((limg < lmin) | (lmax < limg), 0, image)
            else:
                limg = image

            #pw = (image_size[0]-limg.shape[1])//2
            #ph = (image_size[1]-limg.shape[0])//2
            #limg = np.pad(limg,((ph, ph), (pw, pw)))

            return limg
        
        return None

    def draw_sat_mask(self, image):
        if self.editor.crop_image_hls is not None:            
            simg = self.editor.crop_image_hls[..., 2]
            
            smin = self.effects_param.get('mask2_sat_min', 0) / 255
            smax = self.effects_param.get('mask2_sat_max', 255) / 255
            if (smin != 0) or (1 != smax):
                simg = np.where((simg < smin) | (smax < simg), 0, image)
            else:
                simg = image

            return simg
        
        return None

# 円形グラデーションマスクのクラス
class CircularGradientMask(BaseMask):
    inner_radius_x = NumericProperty(0)
    inner_radius_y = NumericProperty(0)
    outer_radius_x = NumericProperty(0)
    outer_radius_y = NumericProperty(0)
    rotate_rad = NumericProperty(0)
    invert = NumericProperty(0)

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Circle"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            Color(*self.color)
            self.outer_line = Line(ellipse=(0, 0, 0, 0), width=2) # 外側の円
            self.inner_line = Line(ellipse=(0, 0, 0, 0), width=2) # 内側の円
            PopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.inner_radius_x = 0
            self.inner_radius_y = 0
            self.outer_radius_x = 0
            self.outer_radius_y = 0
            return True
        else:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if touch.is_double_tap and self.control_points[0].collide_point(cx, cy):
                self.invert = 1-self.invert
                self.update_control_points()
                self.update_mask()
                return True
            
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            dx = cx - self.center_x
            dy = cy - self.center_y
            self.outer_radius_x = ((dx**2 + dy**2) ** 0.5)
            self.outer_radius_y = ((dx**2 + dy**2) ** 0.5)
            self.inner_radius_x = self.outer_radius_x * 0.5  # 内側の半径を仮設定
            self.inner_radius_y = self.outer_radius_y * 0.5  # 内側の半径を仮設定
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
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        for i, angle in enumerate(angles):
            # 内側のコントロールポイント
            cp_inner = ControlPoint(self.editor)
            cp_inner.type = [types[i], angle]
            cp_inner.center = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            cp_inner.ctrl_center = cp_inner.center
            cp_inner.bind(ctrl_center=self.on_inner_control_point_move)
            self.control_points.append(cp_inner)
            self.add_widget(cp_inner)

            # 外側のコントロールポイント
            cp_outer = ControlPoint(self.editor)
            cp_outer.type = [types[i], angle]
            cp_outer.center = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            cp_outer.ctrl_center = cp_outer.center
            cp_outer.bind(ctrl_center=self.on_outer_control_point_move)
            self.control_points.append(cp_outer)
            self.add_widget(cp_outer)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = self.center
        ix = self.inner_radius_x
        iy = self.inner_radius_y
        ox = self.outer_radius_x
        oy = self.outer_radius_y

        param = export.delete_special_param(self.effects_param)
        dict = {
            'type': MASKTYPE_CIRCULAR,
            'name': self.name,
            'center': [cx, cy],
            'inner_radius': [ix, iy],
            'outer_radius': [ox, oy],
            'rotate_rad': self.rotate_rad,
            'invert': self.invert,
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']
        ix, iy = dict['inner_radius']
        ox, oy = dict['outer_radius']
        self.rotate_rad = dict['rotate_rad']
        self.invert = dict['invert']
        self.effects_param.update(dict['effects_param'])

        self.center = (cx, cy)
        self.inner_radius_x = ix
        self.inner_radius_y = iy
        self.outer_radius_x = ox
        self.outer_radius_y = oy

        self.create_control_points()
        #self.update_mask()
 
    def calculate_point(self, radius_x, radius_y, angle_deg):
        angle_rad = math.radians(angle_deg)
        radius_x = radius_x
        radius_y = radius_y
        dx = radius_x * math.cos(angle_rad)
        dy = radius_y * math.sin(angle_rad)
        new_r_x = dx * math.cos(-self.rotate_rad) - dy * math.sin(-self.rotate_rad)
        new_r_y = dx * math.sin(-self.rotate_rad) + dy * math.cos(-self.rotate_rad)
        return (self.center_x + new_r_x, self.center_y + new_r_y)

    def calculate_rotate(self, radius_x, radius_y, angle_deg, dx, dy):
        angle_rad = math.radians(angle_deg)
        px = radius_x * math.cos(angle_rad)
        py = radius_y * math.sin(angle_rad)
        rotate_rad = -math.atan2(dy, dx)
        new_rad = rotate_rad+math.atan2(py, px)
        return new_rad

    def update_ellipse(self, dx, dy):
        # 回転角の変化に応じて、半径を更新
        new_r_x = dx * math.cos(self.rotate_rad) - dy * math.sin(self.rotate_rad)
        new_r_y = dx * math.sin(self.rotate_rad) + dy * math.cos(self.rotate_rad)
        
        return (abs(new_r_x), abs(new_r_y))
    
    def on_center_control_point_move(self, instance, value):
        dx = instance.ctrl_center[0] - self.center_x
        dy = instance.ctrl_center[1] - self.center_y
        self.center = (self.center_x + dx, self.center_y + dy)
        for cp in self.control_points:
            if cp != instance:
                cp.center = (cp.center_x + dx, cp.center_y + dy)
        self.update_control_points()
        self.update_mask()
        self.editor.start_draw_image()

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
            self.editor.start_draw_image()

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
            self.editor.start_draw_image()

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

    def update_mask(self):
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:            
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
            self.rotate.angle = math.degrees(self.editor.get_rotate_rad(self.rotate_rad))
            ix, iy = self.editor.tcg_to_world_scale(self.inner_radius_x, self.inner_radius_y)
            self.inner_line.ellipse = (-ix, -iy, ix*2, iy*2)
            ox, oy = self.editor.tcg_to_world_scale(self.outer_radius_x, self.outer_radius_y)
            self.outer_line.ellipse = (-ox, -oy, ox*2, oy*2)
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        inner_axes = self.editor.tcg_to_world_scale(self.inner_radius_x, self.inner_radius_y)
        outer_axes = self.editor.tcg_to_world_scale(self.outer_radius_x, self.outer_radius_y)
        rotate_rad = self.editor.get_rotate_rad(self.rotate_rad)

        newhash = hash((self.editor.get_hash_items(), image_size, center, inner_axes, outer_axes, rotate_rad, self.invert))
        if self.image_mask_cache is None or self.image_mask_cache_hash != newhash:

            # グラデーションを描画
            gradient_image = self.draw_elliptical_gradient(image_size, center, inner_axes, outer_axes, rotate_rad, self.invert)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache

    def draw_elliptical_gradient(self, image_size, center, inner_axes, outer_axes, angle_rad, invert=0, smoothness=2.0):
    
        # 座標グリッドの作成
        y_indices, x_indices = np.indices((image_size[1], image_size[0]))
        x = x_indices - center[0]
        y = y_indices - center[1]

        # 回転角をラジアンに変換
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 座標の回転
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle

        # 内側と外側の楕円の値を計算
        e_inner = (x_rot / inner_axes[0])**2 + (y_rot / inner_axes[1])**2 - 1
        e_outer = (x_rot / outer_axes[0])**2 + (y_rot / outer_axes[1])**2 - 1

        # グラデーションの初期化
        gradient = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        # 内側の楕円内のピクセルを設定
        mask_inner = e_inner <= 0
        gradient[mask_inner] = 0.0

        # 外側の楕円の外側のピクセル
        gradient[e_outer >= 0] = 1.0

        # 内側と外側の楕円の間のピクセルに対してグラデーションを計算
        mask_between = (~mask_inner) & (e_outer <= 0)

        # グラデーション値の計算（線形補間）
        t = e_inner[mask_between] / (e_inner[mask_between] - e_outer[mask_between])
        t = np.clip(t, 0.0, 1.0)

        # スムーズネスを適用
        t = np.power(t, smoothness)

        # グラデーションを適用
        gradient[mask_between] = t

        # 反転オプションの適用
        if invert == 0:
            gradient = 1.0 - gradient

        return gradient
    
# GradientMask クラス
class GradientMask(BaseMask):
    start_point = ListProperty([0, 0])    # グラデーションの開始点
    end_point = ListProperty([0, 0])      # グラデーションの終点
    
    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Line"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            Color(*self.color)
            self.start_line = Line(points=(0, 0, 0, 0), width=2)
            self.center_line = Line(points=(0, 0, 0, 0), width=2)
            self.end_line = Line(points=(0, 0, 0, 0), width=2)
            PopMatrix()

        self.rotate_rad = 0
        #self.update_mask()
    
    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center = (cx, cy)
            self.start_point = [cx, cy]
            return True
        else:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if touch.is_double_tap and self.control_points[0].collide_point(cx, cy):
                end = self.end_point
                self.end_point = self.start_point
                self.start_point = end
                self.update_control_points()
                self.update_mask()
                return True
            
            return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.end_point = [cx, cy]
            self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                           (self.start_point[1] + self.end_point[1]) / 2]
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = math.atan2(dy, dx)
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
        sx, sy = self.start_point[0], self.start_point[1]
        ex, ey = self.end_point[0], self.end_point[1]

        param = export.delete_special_param(self.effects_param)
        dict = {
            'type': MASKTYPE_GRADIENT,
            'name': self.name,
            'start_point': [sx, sy],
            'end_point': [ex, ey],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        sx, sy = dict['start_point']
        ex, ey = dict['end_point']
        self.effects_param.update(dict['effects_param'])

        self.end_point = [ex, ey]
        self.start_point = [sx, sy]

        self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                       (self.start_point[1] + self.end_point[1]) / 2]
        
        self.create_control_points()
        #self.update_mask()

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = self.center
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)
    
        # グラデーションの開始点と終点のコントロールポイント
        cp_start = ControlPoint(self.editor)
        cp_start.center = self.start_point
        cp_start.ctrl_center = cp_start.center
        cp_start.type = ['s', 0]
        cp_start.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_start)
        self.add_widget(cp_start)
    
        cp_end = ControlPoint(self.editor)
        cp_end.center = self.end_point
        cp_end.ctrl_center = cp_end.center
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

    def on_center_control_point_move(self, instance, value):
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
        self.editor.start_draw_image()        
    
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
            self.rotate_rad = math.atan2(dy, dx)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()        

    def update_control_points(self):
        # コントロールポイントの位置を更新
        cp_center = self.control_points[0]
        cp_center.center = self.center
        cp_start = self.control_points[1]
        cp_start.center = self.start_point
        cp_end = self.control_points[2]
        cp_end.center = self.end_point
    
    def calculate_line(self, point1, point2, dir):
        p1x, p1y = self.editor.tcg_to_window(*point1)
        p2x, p2y = self.editor.tcg_to_window(*point2)
        r = math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
        dx = dir * r
        dy = -self.editor.width
        new_dx1 = dx
        new_dy1 = dy
        dx = dir * r
        dy = self.editor.width
        new_dx2 = dx
        new_dy2 = dy
        dx = p1x-p2x
        dy = p1y-p2y
        rad = 0 if dx == 0 else math.atan2(dy, dx)
        return (new_dx1, new_dy1, new_dx2, new_dy2), rad

    def update_mask(self):
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            if self.initializing:
                tx, ty = self.editor.tcg_to_window(*self.start_point)
                #self.line_color.rgba = self.color
                self.translate.x, self.translate.y = tx, ty
                self.start_line.points, _ = self.calculate_line(self.start_point, self.start_point, 0)
                self.center_line.points, _ = self.calculate_line(self.center, self.start_point, +1)
                self.end_line.points, rad = self.calculate_line(self.end_point, self.start_point, +1)
            else:
                tx, ty = self.editor.tcg_to_window(*self.center)
                self.translate.x, self.translate.y = tx, ty
                self.start_line.points, rad = self.calculate_line(self.start_point, self.center, -1)
                self.center_line.points, _ = self.calculate_line(self.center, self.center, 0)
                self.end_line.points, _ = self.calculate_line(self.end_point, self.center, +1)
            
            self.rotate.angle = math.degrees(rad)

        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()
    
    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        start_point = self.editor.tcg_to_texture(*self.start_point)
        end_point = self.editor.tcg_to_texture(*self.end_point)

        newhash = hash((self.editor.get_hash_items(), image_size, center, start_point, end_point))
        if self.image_mask_cache is None or self.image_mask_cache_hash != newhash:
            # グラデーションを描画
            gradient_image = self.draw_gradient(image_size, center, start_point, end_point)
            
            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache
    
    def draw_gradient(self, image_size, center, start_point, end_point, smoothness=1):

        width, height = image_size
        img = np.zeros((height, width), dtype=np.float32)  # 開始点前はすべて(0, 0, 0, 0)

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
        img[mask] = t[mask]

        return img #np.flipud(img)

# 全体マスクのクラス
class FullMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Full"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            PopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
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
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = self.center

        param = export.delete_special_param(self.effects_param)
        dict = {
            'type': MASKTYPE_FULL,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])

        self.center = (cx, cy)

        # 描き直し
        self.create_control_points()
        #self.update_mask()

    def on_center_control_point_move(self, instance, value):
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
        self.editor.start_draw_image()        

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)

        newhash = hash((self.editor.get_hash_items(), image_size, center))
        if self.image_mask_cache is None or self.image_mask_cache_hash != newhash:
            # 描画
            gradient_image = self.draw_full(image_size, center)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
        
        return self.image_mask_cache

    def draw_full(self, image_size, center):
        # 画像の初期化（黒背景、RGBA）
        image = np.ones((image_size[1], image_size[0]), dtype=np.float32)

        #image[...] = 1

        return image

# 自由描画マスクのクラス
class FreeDrawMask(BaseMask):

    class Line:
        def __init__(self, is_eracing=False, size=10, soft=1.5, **kwargs):
            self.size = size
            self.soft = soft
            self.points = []
            self.is_erasing = is_eracing

        def add_point(self, x, y):
            self.points.append((x, y))

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Draw"
        self.initializing = True

        self.lines = []  # 複数の線を保持
        self.current_line = None
        self.brush_size = 100

        with self.canvas:
            PushMatrix()
            self.translate = Translate(0, 0)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            self.brush_color = Color((0, 1, 1, 1))
            self.brush_cursor = Line(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            PopMatrix()

    def start(self):
        Window.bind(mouse_pos=self.on_mouse_pos)

    def end(self):
        self.brush_color.rgba = (0, 0, 0, 0)
        Window.unbind(mouse_pos=self.on_mouse_pos)

    def serialize(self):
        """マスクの状態をシリアライズ"""
        cx, cy = self.center
        param = export.delete_special_param(self.effects_param)
        dict = {
            'type': MASKTYPE_FREEDRAW,
            'name': self.name,
            'center': [cx, cy],
            'lines': self.lines,
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']
        self.lines = dict['lines']
        self.effects_param.update(dict['effects_param'])
        self.center = (cx, cy)
        self.create_control_points()

    def on_mouse_pos(self, window, pos):
        self.update_brush_cursor(pos[0], pos[1])

    def on_touch_down(self, touch):
        if touch.is_mouse_scrolling:
            if self.editor.collide_point(*touch.pos):
                # 描画中または消去中はブラシサイズを変更できない
                if self.current_line is None:
                    if touch.button == 'scrolldown':
                        self.brush_size = max(10, self.brush_size - 10)
                    elif touch.button == 'scrollup':
                        self.brush_size = min(100, self.brush_size + 10)
                        
                    self.update_brush_cursor(touch.pos[0], touch.pos[1])

                    return super().on_touch_down(touch)

        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)            

        # 右クリックで消去モード、左クリックで描画モード
        is_erasing = (touch.button == 'right')            
        cx, cy = self.editor.window_to_tcg(*touch.pos)
        self.current_line = FreeDrawMask.Line(is_erasing, self.brush_size)
        self.current_line.add_point(cx, cy)
        self.editor.set_active_mask(self)
        self.lines.append(self.current_line)

        self.update_mask()
        self.editor.start_draw_image()        
        
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.current_line is not None:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.current_line.add_point(cx, cy)

            self.update_mask()
            self.editor.start_draw_image()        

        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.current_line is not None:
            self.current_line = None
            # マスクを更新
            self.update_mask()
            self.editor.start_draw_image()        
        
        return super().on_touch_up(touch)

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

    def on_center_control_point_move(self, instance, value):
        dx = instance.ctrl_center[0] - self.center_x
        dy = instance.ctrl_center[1] - self.center_y
        self.center = (self.center_x + dx, self.center_y + dy)
        for cp in self.control_points:
            if cp != instance:
                cp.center = (cp.center_x + dx, cp.center_y + dy)
        self.update_control_points()
        self.update_mask()
        self.editor.start_draw_image()

    def update_brush_cursor(self, x, y):
        self.translate.x, self.translate.y = x - self.brush_size / 2, y - self.brush_size / 2
        self.brush_cursor.ellipse = (0, 0, self.brush_size, self.brush_size)
        #self.brush_color.rgba = (1, 1, 1, 0.5)

    def update_mask(self):
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            return
        
        self.rotate.angle = math.degrees(self.editor.get_rotate_rad(0))

        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):
            
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        nline = len(self.lines)
        npoint = 0
        for line in self.lines:
            npoint += len(line.points)

        newhash = hash((self.editor.get_hash_items(), image_size, nline, npoint))
        if self.image_mask_cache is None or self.image_mask_cache_hash != newhash:
             
            mask = self.draw_line(image_size, self.lines)

            # ルミナンスとマスクを作成
            mask = self.draw_hls_mask(mask)

            # マスクぼかし
            mask = self.apply_mask_blur(mask)

            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache
    
    def draw_line(self, image_size, lines):
        # 画像の初期化（透明背景）
        width, height = image_size
        image = np.zeros((height, width), dtype=np.float32)
        
        # 各線を描画
        for line in lines:
            # 点が0個の場合はスキップ
            if len(line.points) == 0:
                continue
                
            # 線の太さとソフトネス
            brush_size = line.size
            brush_radius = brush_size / 2
            brush_soft = line.soft
            is_erasing = line.is_erasing
            
            # 点が1つだけの場合は、その点にブラシを適用
            if len(line.points) == 1:
                p = self.editor.tcg_to_texture(*line.points[0])
                x, y = int(p[0]), int(p[1])
                
                # OpenCVを使用して円を描画
                center = (x, y)
                # ブラシの形状を作成（円形）
                brush_img = np.zeros((int(brush_size*2), int(brush_size*2)), dtype=np.float32)
                cv2.circle(brush_img, (int(brush_size), int(brush_size)), int(brush_radius), 1.0, -1)
                
                # ソフトエッジを適用
                if brush_soft > 1.0:
                    # ソフトエッジ効果を作成
                    dist_img = np.zeros_like(brush_img)
                    cv2.circle(dist_img, (int(brush_size), int(brush_size)), int(brush_radius), 1.0, -1)
                    dist_img = cv2.distanceTransform(dist_img.astype(np.uint8), cv2.DIST_L2, 3)
                    dist_img = dist_img / np.max(dist_img)
                    brush_img = np.power(dist_img, 1.0/brush_soft)
                
                # ブラシを画像に適用
                x_min = max(0, x - int(brush_size))
                y_min = max(0, y - int(brush_size))
                x_max = min(width, x + int(brush_size))
                y_max = min(height, y + int(brush_size))
                
                # x_minがx_maxより大きい、またはy_minがy_maxより大きい場合の修正
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                if y_min > y_max:
                    y_min, y_max = y_max, y_min
                
                # ブラシの対応部分を切り出し
                brush_x_min = max(0, int(brush_size) - x)
                brush_y_min = max(0, int(brush_size) - y)
                brush_x_max = brush_x_min + (x_max - x_min)
                brush_y_max = brush_y_min + (y_max - y_min)
                
                brush_part = brush_img[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
                
                if is_erasing:
                    # 消しゴムモード
                    image[y_min:y_max, x_min:x_max] = np.maximum(0, image[y_min:y_max, x_min:x_max] - brush_part)
                else:
                    # 描画モード
                    image[y_min:y_max, x_min:x_max] = np.minimum(1, image[y_min:y_max, x_min:x_max] + brush_part)
                continue
            
            # 複数点がある場合は、各点を結ぶ線を描画
            # 全ての点をテクスチャ座標に変換
            points = [self.editor.tcg_to_texture(*p) for p in line.points]
            
            # OpenCVのpolylines関数用に点を整形
            cv_points = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
            
            # 線の描画用のマスク画像
            line_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 太い線を描画
            cv2.polylines(line_mask, [cv_points], False, 255, int(brush_size))
            
            # 線の端点も円で描画して繋ぎ目を滑らかに
            for p in cv_points:
                cv2.circle(line_mask, (p[0], p[1]), int(brush_radius), 255, -1)
            
            # ブラシのソフトエッジを適用
            if brush_soft > 1.0:
                # 距離変換でソフトエッジを作成
                dist_img = cv2.distanceTransform((line_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
                max_dist = np.max(dist_img)
                if max_dist > 0:
                    dist_img = dist_img / max_dist
                    # ソフトネスを適用
                    soft_mask = np.power(dist_img, 1.0/brush_soft)
                    line_mask = (soft_mask * 255).astype(np.uint8)
            
            # 最終的なマスクを適用
            line_mask_float = line_mask.astype(np.float32) / 255.0
            
            if is_erasing:
                # 消しゴムモード
                image = np.maximum(0, image - line_mask_float)
            else:
                # 描画モード
                image = np.minimum(1, image + line_mask_float)
        
        return image

# セグメントマスクのクラス
class SegmentMask(BaseMask):
    __iopaint_plugins = None
    __iopaint_predict = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Segment"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            PopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
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
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = self.center

        param = export.delete_special_param(self.effects_param)
        dict = {
            'type': MASKTYPE_SEGMENT,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])

        self.center = (cx, cy)

        # 描き直し
        self.create_control_points()
        #self.update_mask()

    def on_center_control_point_move(self, instance, value):
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
        self.editor.start_draw_image()        

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_full_image(*self.center)

        newhash = hash((self.editor.get_hash_items(), image_size, center))
        if self.image_mask_cache is None or self.image_mask_cache_hash != newhash:
            # 描画
            gradient_image = self.draw_segment(image_size, center)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
            
        return self.image_mask_cache

    def draw_segment(self, image_size, center):
        if SegmentMask.__iopaint_plugins is None:
            SegmentMask.__iopaint_plugins = importlib.import_module('iopaint.plugins')
        if SegmentMask.__iopaint_predict is None:
            SegmentMask.__iopaint_predict = importlib.import_module('iopaint.predict')
        
        img = self.editor.full_image_rgb
        result = SegmentMask.__iopaint_predict.predict_plugin(img, SegmentMask.__iopaint_plugins.InteractiveSeg.name, click=center)
        result = ((result > 0) * 1).astype(np.float32)

        nw, nh, ox, oy = core.crop_size_and_offset_from_texture(self.editor.texture_size[0], self.editor.texture_size[1], self.editor.disp_info)
        cx, cy ,cw, ch, scale = self.editor.disp_info
        result = cv2.resize(result[cy:cy+ch, cx:cx+cw], (nw, nh))
        if scale < 1:
            result = np.pad(result, ((oy, self.editor.texture_size[0]-(oy+nh)), (ox, self.editor.texture_size[1]-(ox+nw))), constant_values=0)

        return result

# マスクレイヤーの管理クラス
class MaskLayer(BoxLayout):
    mask = ObjectProperty(None)
    mask_name = StringProperty('')

    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.root = None

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
        self.mask.editor._delete_mask(self.mask)

    def set_active(self, instance):
        self.mask.editor.set_active_mask(self.mask)

# メインのエディタークラス
class MaskEditor2(FloatLayout):
    mask_layers = ListProperty([])
    active_mask = ObjectProperty(None, allownone=True)
    image_size = ListProperty([0, 0])  # 画像のサイズを保持
    disp_info = Property((0, 0, 0, 0, 1))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.image_widget = Image(allow_stretch=False, keep_ratio=True)
        self.image_widget.pos_hint = {"x":0, "top":1}
        self.add_widget(self.image_widget)

        self.mask_container = Widget()
        self.add_widget(self.mask_container)
        self.rectangle = None

        self.ui_layout = BoxLayout(orientation='vertical', size_hint=(0.1, 1))
        self.ui_layout.pos_hint = {"x":0, "top":1}
        self.add_widget(self.ui_layout)
        self.create_ui()

        self.create_mask = None
        self.center_rotate_rad = 0
        self.orientation = (0, 0)
        self.margin = (0, 0)
        self.texture_size = (0, 0)
        #self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        #self._keyboard.bind(on_key_down=self._on_keyboard_down)

        Logger.info("MaskEditor: 初期化完了")

    # 終了処理
    def end(self):
        if self.active_mask is not None:
            self.active_mask.end()

    def imread(self, image_source, dt):
        # 画像の読み込みとサイズの取得
        if not os.path.isfile(image_source):
            Logger.error(f"MaskEditor: 画像ファイルが見つかりません: {image_source}")
            return False
        
        self.image_widget.source = image_source
        self.image_size = self.image_widget.texture.size
        self.image_widget.opacity = 1

        if self.image_size[0] >= self.image_size[1]:
            scale = self.size[0] / self.image_size[0]
        else:
            scale = self.size[1] / self.image_size[1]
        self.texture_size = (self.size[0], self.size[0])
        self.disp_info = (0, 0, self.size[0], self.size[1], scale)
        self.__set_image_info()
        return True
    
    def set_ref_image(self, crop_image, full_image):
        self.full_image_rgb = np.clip(full_image, 0, 1)
        self.crop_image_hls = cv2.cvtColor(np.clip(crop_image, 0, 1), cv2.COLOR_RGB2HLS_FULL)

    def set_texture_size(self, tx, ty):
        self.texture_size = (tx, ty)

    def set_image(self, size, disp_info):
        self.image_widget.source = None
        self.image_widget.opacity = 0

        self.image_size[0], self.image_size[1] = size
        self.disp_info = disp_info

        self.__set_image_info()
    
    def set_orientation(self, rotation, rotation2, flip):
        self.center_rotate_rad = math.radians(rotation)
        self.orientation = (math.radians(rotation2), flip)

    def get_hash_items(self):
        return (self.disp_info, self.center_rotate_rad, self.orientation)

    def __set_image_info(self):
        self.margin = ((self.size[0]-self.texture_size[0])/2, (self.size[1]-self.texture_size[1])/2)
        for mask in reversed(self.mask_layers):
            #pass    # 無限ループ対策
            effects.reeffect_all(mask.effects)
        
    def update(self):
        # 既存のマスクに対する更新を処理
        for mask in reversed(self.mask_layers):
            #pass    # 無限ループ対策
            mask.update()

    def serialize(self):
        list = []
        for mask in reversed(self.mask_layers):
            list.append(mask.serialize())
        if len(list) <= 0:
            return None

        dict = {
            'mask2': list,
        }
        return dict

    def deserialize(self, dict):
        list = dict['mask2']

        for dict in list:
            type = dict.get('type', None)
            mask = self._create_mask(type)
            mask.deserialize(dict)
            mask.update()

    def get_active_mask(self):
        if self.disabled == True:
            return None
        
        return self.active_mask
    
    def get_layers_list(self):
        return self.mask_layers
    
    def create_ui(self):
        # マスクタイプ選択ボタン
        btn_circular = Button(text='Circle', size_hint=(1, 0.05))
        btn_circular.bind(on_release=self.select_circular_gradient_mask)
        self.ui_layout.add_widget(btn_circular)

        btn_gradient = Button(text='Line', size_hint=(1, 0.05))
        btn_gradient.bind(on_release=self.select_gradient_mask)
        self.ui_layout.add_widget(btn_gradient)

        btn_full = Button(text='Full', size_hint=(1, 0.05))
        btn_full.bind(on_release=self.select_full_mask)
        self.ui_layout.add_widget(btn_full)

        btn_free_draw = Button(text='Draw', size_hint=(1, 0.05))
        btn_free_draw.bind(on_release=self.select_free_draw_mask)
        self.ui_layout.add_widget(btn_free_draw)

        btn_segment = Button(text='Segment', size_hint=(1, 0.05))
        btn_segment.bind(on_release=self.select_segment_mask)
        self.ui_layout.add_widget(btn_segment)

        # マスクレイヤー表示
        self.layer_list = BoxLayout(orientation='vertical', size_hint=(2, 0.7))
        self.ui_layout.add_widget(self.layer_list)

    def set_draw_mask(self, is_draw_mask):
        if is_draw_mask == False:
            if self.rectangle is not None:
                self.mask_container.canvas.before.remove(self.rectangle)
                self.rectangle = None
        mask = self.get_active_mask()
        if mask is not None:
            mask.is_draw_mask = is_draw_mask
            if is_draw_mask == True:
                mask.update()
    
    def start_draw_image(self):
        if self.root is not None:
            self.root.start_draw_image()

    def draw_mask_image(self, glayimg):
        if self.rectangle is not None:
            self.mask_container.canvas.before.remove(self.rectangle)
            self.rectangle = None

        if glayimg is not None:
            with self.mask_container.canvas.before:
                texture = Texture.create(size=(glayimg.shape[1], glayimg.shape[0]), colorfmt='luminance', bufferfmt='float')
                texture.blit_buffer(glayimg.tobytes(), colorfmt='luminance', bufferfmt='float')
                texture.flip_vertical()
                px, py = self.to_window(*self.pos)
                px, py = px+self.margin[0], py+self.margin[1]
                Color(1, 0, 0, 0.3)
                self.rectangle = Rectangle(texture=texture, pos=(px, py), size=self.texture_size)

        # cv2.imwrite('combined_mask.png', (glayimg*255).astype(np.uint8))

    def select_circular_gradient_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_CIRCULAR)

    def select_gradient_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_GRADIENT)

    def select_full_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_FULL)

    def select_free_draw_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_FREEDRAW)

    def select_segment_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_SEGMENT)

    def _create_start_new_mask(self, type):
        # 画像サイズがまだ設定されていない場合、マスクの作成をスキップ
        if self.image_size == [0, 0]:
            Logger.warning("MaskEditor: 画像がまだロードされていません。マスクを追加できません。")
            return
        
        self.ui_layout.disabled = True
        mask = self._create_mask(type)
        self.set_active_mask(None)
        self.create_mask = mask

    def _create_end_new_mask(self):
        self.ui_layout.disabled = False
        self.set_active_mask(self.create_mask)

    def on_touch_down(self, touch):
        if self.disabled == True:
            return False
        
        # 既存のマスクに対するタッチイベントを処理
        for mask in reversed(self.mask_layers):
            if mask.on_touch_down(touch):
                return True

        return super().on_touch_down(touch)
        
    def on_touch_up(self, touch):
        if self.disabled == True:
            return False
        
        result = super().on_touch_up(touch)

        # こっちを後でやらないとまだコントロールポイントが作られてない
        if self.create_mask is not None:
            self._create_end_new_mask()        
            self.create_mask = None
        
        return result

    def _create_mask(self, mask_type):
        # マスク作成
        if mask_type == MASKTYPE_CIRCULAR:
            mask = CircularGradientMask(editor=self)
        elif mask_type == MASKTYPE_GRADIENT:
            mask = GradientMask(editor=self)
        elif mask_type == MASKTYPE_FULL:
            mask = FullMask(editor=self)
        elif mask_type == MASKTYPE_FREEDRAW:
            mask = FreeDrawMask(editor=self)
        elif mask_type == MASKTYPE_SEGMENT:
            mask = SegmentMask(editor=self)
        else:
            Logger.error(f"MaskEditor: 不明なマスクタイプ: {self.current_mask_type}")
            return None
            
        self.mask_container.add_widget(mask)
        self.mask_layers.append(mask)

        # レイヤーUIに追加
        layer = MaskLayer(mask=mask)
        self.layer_list.add_widget(layer)
        if self.root is not None:
            self.root.set2widget_all(mask.effects, mask.effects_param)

        return mask

    def _delete_mask(self, mask):
        if len(self.mask_layers) <= 1:
            self.draw_mask_image(None)
            self.set_active_mask(None)
        else:
            i = self.mask_layers.index(mask)
            i = i+1 if i+1 < len(self.mask_layers) else i-1
            self.set_active_mask(self.mask_layers[i])
 
        self.mask_container.remove_widget(mask)
        self.mask_layers.remove(mask)

    def clear_mask(self):
        self.set_active_mask(None)
        self.draw_mask_image(None)
        self.mask_container.clear_widgets()
        self.mask_layers.clear()
        self.layer_list.clear_widgets()

    def set_active_mask(self, mask):
        if self.active_mask and self.active_mask != mask:
            self.active_mask.active = False
            self.active_mask.end()

        self.active_mask = mask
        if mask is not None:
            mask.active = True
            self.root.set2widget_all(mask.effects, mask.effects_param)
            mask.start()
            #mask.update()
        else:
            self.draw_mask_image(None)
            if self.root is not None:
                self.root.set2widget_all(None, None)
        self.start_draw_image()

    def get_rotate_rad(self, rotate_rad):
        rad, flip = self.orientation
        """
        if flip == 0:
            rad = rotate_rad + rad
        if flip == 1:
            rad = -rotate_rad + rad
        if flip == 2:
            rad = -rotate_rad + rad
        """
        return self.center_rotate_rad + rad + rotate_rad
    
    def world_to_tcg_scale(self, x, y):
        return (x / self.disp_info[4], y / self.disp_info[4])
    
    def tcg_to_world_scale(self, x, y):
        return (x * self.disp_info[4], y * self.disp_info[4])

    # ワールド座標からテクスチャのグローバル座標に
    def window_to_tcg(self, cx, cy):
        wx, wy = self.to_window(*self.pos)
        cx, cy = cx - wx, cy - wy
        cx, cy = cx - self.margin[0], cy - self.margin[1]
        cx, cy = cx, self.texture_size[1] - cy
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, self.disp_info)
        cx, cy = cx - offset_x, cy - offset_y
        cx, cy = cx / self.disp_info[4], cy / self.disp_info[4]
        cx, cy = cx + self.disp_info[0], cy + self.disp_info[1]
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = cx - imax, cy - imax
        cx, cy = self.center_rotate_invert(cx, cy, self.center_rotate_rad)
        return (cx, cy)

    def tcg_to_window(self, cx, cy):
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - self.disp_info[0], cy - self.disp_info[1]
        cx, cy = cx * self.disp_info[4], cy * self.disp_info[4]        
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, self.disp_info)
        cx, cy = cx + offset_x, cy + offset_y
        cx, cy = cx, self.texture_size[1] - cy
        cx, cy = cx + self.margin[0], cy + self.margin[1]
        wx, wy = self.to_window(*self.pos)
        cx, cy = cx + wx, cy + wy
        return (cx, cy)

    def tcg_to_texture(self, cx, cy):
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - self.disp_info[0], cy - self.disp_info[1]
        cx, cy = cx * self.disp_info[4], cy * self.disp_info[4]        
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, self.disp_info)
        cx, cy = cx + offset_x, cy + offset_y
        return (cx, cy)

    def tcg_to_full_image(self, cx, cy):
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        return (cx, cy)

    def apply_orientation(self, cx, cy):
        rad, flip = self.orientation

        if (flip & 1) == 1:
            cx = -cx
        if (flip & 2) == 2:
            cy = -cy

        return cx, cy, rad

    def center_rotate(self, cx, cy, rotation_rad):
        cx, cy, rad = self.apply_orientation(cx, cy)
        rad = rotation_rad + rad
        rad = -rad

        new_cx = cx * math.cos(rad) - cy * math.sin(rad)
        new_cy = cx * math.sin(rad) + cy * math.cos(rad)

        return (new_cx, new_cy)

    def center_rotate_invert(self, cx, cy, rotation_rad):
        rad, _ = self.orientation
        rad = rotation_rad + rad
        rad = -rad

        new_cx = cx * math.cos(rad) + cy * math.sin(rad)
        new_cy = -cx * math.sin(rad) + cy * math.cos(rad)

        new_cx, new_cy, _ = self.apply_orientation(new_cx, new_cy)

        return (new_cx, new_cy)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'z':
            if self.disp_info[4] == 1.0:
                self.disp_info[0], self.disp_info[1] = 0, 0
                if self.image_size[0] >= self.image_size[1]:
                    scale = self.size[0] / self.image_size[0]
                else:
                    scale = self.size[1] / self.image_size[1]
                self.disp_info[4] = scale
            else:
                self.disp_info[0], self.disp_info[1] = self.image_size[0]/2, self.image_size[1]/2
                self.disp_info[4] = 1.0
        elif keycode[1] == 'up':
            self.disp_info[1] -= 100
        elif keycode[1] == 'down':
            self.disp_info[1] += 100
        elif keycode[1] == 'left':
            self.disp_info[0] -= 100
        elif keycode[1] == 'right':
            self.disp_info[0] += 100
        elif keycode[1] == 't':
            for mask in reversed(self.mask_layers):
                dict = mask.serialize()        
                mask.deserialize(dict)
            return True

        elif keycode[1] == 'r':
            self.center_rotate_rad += math.radians(5)
            for mask in reversed(self.mask_layers):
                mask.update()
            return True

        elif keycode[1] == 'a':
            self.root.force_full_redraw()
            
            return True
        
        elif keycode[1] == 's':
            self.root.save_json()
            return True
        
        # 既存のマスクに対するタッチイベントを処理
        for mask in reversed(self.mask_layers):
            mask.update()  
    
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
        float = FloatLayout()
        float.size_hint_x = 6
        editor = MaskEditor2()
        editor.pos_hint = {'x': 0, 'top': 1}
        #editor.size_hint_x = 6
        float.add_widget(editor)
        box2.add_widget(float)
        box5 = FloatLayout()
        box5.size_hint_x = 2
        box2.add_widget(box5)

        Clock.schedule_once(partial(editor.imread, image_path), -1)

        return box0

if __name__ == '__main__':
    MaskEditor2App().run()

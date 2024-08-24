import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics import Rectangle
from kivy.properties import NumericProperty
from kivy.graphics.texture import Texture

import cv2
import numpy as np

class MaskEditor(Image):
    brush_size = NumericProperty(300)
    zoom = NumericProperty(1.0)

    def __init__(self, width=512*2, height=512*2, **kwargs):
        super(MaskEditor, self).__init__(**kwargs)
        self.canvas_width = width
        self.canvas_height = height
        self.drawing = False
        self.erasing = False
        self.clear_mask()
        self.canvas_texture = Texture.create(size=(self.canvas_width, self.canvas_height), colorfmt='rgba')
        self.canvas_texture.flip_vertical()
        self.update_canvas()
        self.bind(size=self.update_canvas, pos=self.update_canvas)

    def get_mask(self):
        return self.mask[:,:,3]
    
    def clear_mask(self):
        self.mask = np.zeros((self.canvas_height, self.canvas_width, 4), dtype=np.uint8)

    def update_canvas(self, *args):
        # Update canvas with the current mask
        self.canvas_texture.blit_buffer(self.mask.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.texture = self.canvas_texture
        with self.canvas:
            self.canvas.clear()
            zw = self.canvas_width * self.zoom
            zh = self.canvas_height * self.zoom
            Rectangle(texture=self.canvas_texture, pos=((self.size[0]-zw)/2, (self.size[1]-zh)/2), size=(zw, zh))

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            self.drawing = touch.button == 'left'
            self.erasing = touch.button == 'right'
            self.previous_pos = self.get_canvas_coordinates(touch.x, touch.y)
            self.paint(touch)
            return True
        return super(MaskEditor, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.drawing or self.erasing:
            self.paint(touch)
            return True
        return super(MaskEditor, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.drawing = False
        self.erasing = False
        return super(MaskEditor, self).on_touch_up(touch)

    def get_shift_pos(self):
        zw = self.canvas_width * self.zoom
        zh = self.canvas_height * self.zoom
        return int((self.size[0]-zw)/2), int((self.size[1]-zh)/2)

    def get_canvas_coordinates(self, x, y):
        # Convert screen coordinates to canvas coordinates
        sx, sy = self.get_shift_pos()
        cx = int((x -self.x -sx) / (self.canvas_width * self.zoom) * self.canvas_width)
        cy = int((self.size[1]-(y -self.y +sy)) / (self.canvas_height * self.zoom) * self.canvas_height)
        return cx, cy

    def paint(self, touch):
        x, y = self.get_canvas_coordinates(touch.x, touch.y)
        size = int(self.brush_size / self.canvas_width * self.canvas_width)
        radius = size // 2

        if self.drawing:
            self.draw_circle(x, y, radius, value=[255, 255, 255, 255])
            self.draw_line(self.previous_pos, (x, y), radius, value=[255, 255, 255, 255])
        elif self.erasing:
            self.draw_circle(x, y, radius, value=[0, 0, 0, 0])
            self.draw_line(self.previous_pos, (x, y), radius, value=[0, 0, 0, 0])

        self.previous_pos = (x, y)
        self.update_canvas()

    def draw_circle(self, cx, cy, radius, value):
        cv2.circle(self.mask, (cx, cy), radius, value, thickness=-1)
        """
        for y in range(-radius, radius):
            for x in range(-radius, radius):
                if x**2 + y**2 <= radius**2:
                    nx, ny = cx + x, cy + y
                    if 0 <= nx < self.canvas_width and 0 <= ny < self.canvas_height:
                        self.mask[ny, nx] = value
        """

    def draw_line(self, start, end, radius, value):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.draw_circle(x0, y0, radius, value)
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def on_mouse_scroll(self, window, pos, scroll):
        self.brush_size = max(1, self.brush_size + scroll)

    def load_mask(self, filepath):
        image = Image.open(filepath).convert('L')
        self.canvas_width, self.canvas_height = image.size
        self.mask = np.array(image, dtype=np.uint8)
        self.canvas_texture = Texture.create(size=(self.canvas_width, self.canvas_height), colorfmt='rgba')
        self.update_canvas()

    def save_mask(self, filepath):
        image = Image.fromarray(self.mask)
        image.save(filepath, format='PNG')

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        # MaskDrawingWidgetのインスタンスを作成
        mask_widget = MaskEditor(width=1024, height=1024)
        
        # 他のUI要素と一緒にレイアウトに追加
        layout.add_widget(mask_widget)
        
        return layout

if __name__ == '__main__':
    MyApp().run()

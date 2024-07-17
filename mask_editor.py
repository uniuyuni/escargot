# File: mask_editor.py

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.core.window import Window
from PIL import Image as PILImage
import numpy as np

class DrawingCanvas(Widget):
    def __init__(self, **kwargs):
        super(DrawingCanvas, self).__init__(**kwargs)
        self.brush_size = 10
        self.paint_mode = True
        self.last_touch_pos = None

    def on_touch_down(self, touch):
        with self.canvas:
            if self.paint_mode:
                Color(1,1,1,1)
            else:
                Color(0,0,0,1)
            d = self.brush_size
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
        self.last_touch_pos = touch.pos

    def on_touch_move(self, touch):
        with self.canvas:
            if self.paint_mode:
                Color(1,1,1,1)
            else:
                Color(0,0,0,1)
            d = self.brush_size
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            Line(points=[self.last_touch_pos[0], self.last_touch_pos[1], touch.x, touch.y], width=self.brush_size)

        self.last_touch_pos = touch.pos

    def change_brush_size(self, instance, value):
        self.brush_size = value

    def set_paint_mode(self, instance):
        self.paint_mode = True

    def set_erase_mode(self, instance):
        self.paint_mode = False

class MaskEditor(BoxLayout):
    def __init__(self, **kwargs):
        super(MaskEditor, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # Image canvas for drawing
        self.canvas_widget = Scatter(do_scale=False, do_translation=False, do_rotation=False, size_hint=(1, 1))
        self.drawing_widget = DrawingCanvas(size_hint=(None, None))
        self.canvas_widget.add_widget(self.drawing_widget)

        # Control buttons and sliders at the top
        self.controls = BoxLayout(size_hint_y=None, height='50dp')
        self.zoom_in_button = Button(text='Zoom In')
        self.zoom_in_button.bind(on_press=self.zoom_in)
        self.controls.add_widget(self.zoom_in_button)

        self.zoom_out_button = Button(text='Zoom Out')
        self.zoom_out_button.bind(on_press=self.zoom_out)
        self.controls.add_widget(self.zoom_out_button)

        self.brush_size_slider = Slider(min=1, max=50, value=10)
        self.brush_size_slider.bind(value=self.drawing_widget.change_brush_size)
        self.controls.add_widget(self.brush_size_slider)

        self.paint_button = Button(text='Paint')
        self.paint_button.bind(on_press=self.drawing_widget.set_paint_mode)
        self.controls.add_widget(self.paint_button)

        self.erase_button = Button(text='Erase')
        self.erase_button.bind(on_press=self.drawing_widget.set_erase_mode)
        self.controls.add_widget(self.erase_button)

        self.save_button = Button(text='Save Mask')
        self.save_button.bind(on_press=self.save_mask)
        self.controls.add_widget(self.save_button)

        self.add_widget(self.controls)
        self.add_widget(self.canvas_widget)

        self.mask_texture = None
        self.load_image('your_image.png')

        self.bind(size=self.update_canvas)

    def zoom_in(self, instance):
        self.canvas_widget.scale = min(2.0, self.canvas_widget.scale + 0.1)

    def zoom_out(self, instance):
        self.canvas_widget.scale = max(0.1, self.canvas_widget.scale - 0.1)

    def update_canvas(self, instance, value):
        self.drawing_widget.size = self.canvas_widget.size
        self.drawing_widget.pos = self.canvas_widget.pos
        pass

    def load_image(self, image_path):
        pil_image = PILImage.open(image_path)
        image_data = pil_image.tobytes()
        self.mask_texture = Texture.create(size=pil_image.size)
        self.mask_texture.blit_buffer(image_data, colorfmt='rgba')
        self.drawing_widget.size = pil_image.size
        self.canvas_widget.size = pil_image.size

    def save_mask(self, instance):
        pil_image = PILImage.frombytes('RGBA', self.drawing_widget.size, self.drawing_widget.texture.pixels)
        pil_image.save('mask.png')

class MaskEditorApp(App):
    def build(self):
        return MaskEditor()

if __name__ == '__main__':
    MaskEditorApp().run()
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, StringProperty, BooleanProperty

class ParamSlider(BoxLayout):
    text = StringProperty()
    min = NumericProperty(-100)
    max = NumericProperty(100)
    value = NumericProperty(0)
    step = NumericProperty(1)
    for_float = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super(ParamSlider, self).__init__(**kwargs)
    
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.change_label()
        self.change_slider()
        self.change_slider_value()
        self.init_value = self.value
    
    def change_label(self):
        self.ids['label'].text = self.text

    def change_slider(self):
        self.ids['slider'].min = self.min
        self.ids['slider'].max = self.max
        self.ids['slider'].value = self.value
        self.ids['slider'].step = self.step

    def change_slider_value(self):
        self.value = self.ids['slider'].value
        self.ids['input'].text = str(self.value)

    def change_input_text(self):
        try:
            if self.for_float:
                val = float(self.ids['input'].text)
            else:
                val = int(self.ids['input'].text)
        except ValueError:
            val = self.init_value
        val = min(self.max, max(self.min, val))
        self.ids['input'].text = str(val)
        self.value = val
        self.ids['slider'].value = self.value
    
    def press_button(self, step):
        self.value  = min(self.max, max(self.min, self.ids['slider'].value + step))
        self.ids['slider'].value = self.value
    
    def press_slider(self, touch):
        if touch.is_double_tap:
            self.ids['slider'].value = self.init_value


class WidgetApp(MDApp):
    def __init__(self, **kwargs):
        super(WidgetApp, self).__init__(**kwargs)
        
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = ParamSlider()

        return widget

if __name__ == '__main__':
    WidgetApp().run()


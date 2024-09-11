
from kivymd.app import MDApp
from kivymd.uix.widget import MDWidget
from kivy.core.window import Window

import curve
import param_slider
import viewer_widget
import spacer

class MainWidget(MDWidget):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adjust_lv0(self, layer):
        return True

    def adjust_lv1(self, layer):
        return True
    
    def adjust_lv2(self, layer):
        return True

   
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = 'escargot'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"

    def build(self): 
        widget = MainWidget()

        Window.size = (1024, 800)

        return widget


if __name__ == '__main__':
    MainApp().run()



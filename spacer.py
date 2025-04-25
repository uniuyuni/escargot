
from kivy.uix.widget import Widget as KVWidget
from kivy.properties import NumericProperty as KVNumericProperty

class Spacer(KVWidget):
    pass

class HSpacer(Spacer):
    ref_width = KVNumericProperty(8)

class VSpacer(Spacer):
    ref_height = KVNumericProperty(8)

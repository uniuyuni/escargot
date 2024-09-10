import os
import threading
import base64
import rawpy
import exiftool
import numpy as np
import cv2

from kivymd.app import MDApp
from kivy.metrics import dp
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivymd.uix.card import MDCard
from kivymd.uix.gridlayout import MDGridLayout
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty, DictProperty, ObjectProperty, BooleanProperty
from kivy.clock import mainthread

import AppKit
from Cocoa import NSDragOperationCopy

from spacer import HSpacer, VSpacer

class ThumbnailCard(MDCard):
    file_path = StringProperty()
    thumb_source = ObjectProperty()
    rating = NumericProperty(0)
    exif_data = DictProperty()
    grid_width = NumericProperty(dp(180))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (None, None)
        self.size = (self.grid_width, self.grid_width+dp(40))
        self.md_bg_color = [0.1, 0.1, 0.1, 1]
        self.radius = [5, 5, 5, 5]
        self.elevation = 2

        vbox = MDBoxLayout()
        vbox.orientation = 'vertical'
        self.add_widget(vbox)

        # サムネイル表示
        vbox.add_widget(HSpacer(height=dp(4)))
        hbox = MDBoxLayout(size_hint_y=7)
        hbox.orientation = 'horizontal'
        hbox.add_widget(VSpacer(width=dp(4)))
        self.image = Image(source='spinner.zip', allow_stretch=False)
        self.image.anim_delay = 0.01
        hbox.add_widget(self.image)
        hbox.add_widget(VSpacer(width=dp(4)))
        vbox.add_widget(HSpacer(height=dp(4)))
        vbox.add_widget(hbox)

        # ファイル名ラベル
        name = os.path.basename(self.file_path)
        self.label = Label(text=name, bold=True, size_hint_y=3)
        vbox.add_widget(self.label)

        self.bind(on_touch_down=self.on_card_touch_down)

    @mainthread
    def set_image(self, exif, thumb):
        self.exif_data = exif
        self.thumb_source = thumb
        self.texture = Texture.create(size=(thumb.shape[1], thumb.shape[0]), colorfmt='rgb', bufferfmt='ushort')
        self.texture.flip_vertical()
        self.texture.blit_buffer(thumb.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.image.source = ''
        self.image.size = (thumb.shape[1], thumb.shape[0])
        self.image.texture = self.texture

    def on_card_touch_down(self, instance, touch):
        if self.collide_point(*touch.pos):
            # macOSのペーストボードにファイルをセット
            self.start_dragging()

    def start_dragging(self):
        # macOSのNSPasteboardを使って、ドラッグ&ドロップ用の情報をペーストボードに配置
        pasteboard = AppKit.NSPasteboard.generalPasteboard()
        pasteboard.declareTypes_owner_([AppKit.NSFilenamesPboardType], None)
        pasteboard.setPropertyList_forType_([self.file_path], AppKit.NSFilenamesPboardType)

        # ドラッグ操作を開始
        print(f"Dragging: {self.file_path}")

class ViewerWidget(MDBoxLayout):
    last_selected = ObjectProperty()
    cols = NumericProperty(4)
    grid_width = NumericProperty(dp(180))
    thumb_width = NumericProperty(dp(160))
    do_scroll_x = BooleanProperty(True)
    do_scroll_y = BooleanProperty(False)

    supported_formats_rgb = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')
    supported_formats_raw = ('.cr2', '.nef', '.arw', '.dng', '.orf', '.raf', '.rw2', '.sr2', '.pef', '.raw')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cards = []
        self.selected_cards = set()
        
        Window.bind(on_key_down=self.on_key_down)

    def set_path(self, directory):
        self.cards = []
        file_path_dict = {}
        file_list = os.listdir(directory)
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            if self.is_supported_image(file_name):
                card = self.add_image_to_grid(file_path)
                self.cards.append(card)
                file_path_dict[file_path] = card
        self.cols = len(file_path_dict)

        self.load_images(file_path_dict)

    def load_images(self, file_path_dict):
        threading.Thread(target=self.load_images_thread, args=(file_path_dict, 16), daemon=True).start()

    def load_images_thread(self, file_path_dict, chunk_size):

        # 辞書のキーからファイルパスのリストを取得
        file_path_list = list(file_path_dict.keys())
        
        # ファイルパスを指定した分割数で処理
        for i in range(0, len(file_path_list), chunk_size):
            chunk = file_path_list[i:i + chunk_size]
            
            with exiftool.ExifToolHelper(common_args=['-b']) as et:
                exif_data_list = et.get_metadata(chunk)
        
            # ここで exif_data_list を処理する
            thumb_data_list = self.process_exif_data(chunk, exif_data_list)

            for i in range(len(chunk)):
                file_path = chunk[i]
                thumb = thumb_data_list[i]
                file_path_dict[file_path].set_image(exif_data_list[i], thumb)

        return exif_data_list

    def is_supported_image(self, file_name):
        return file_name.lower().endswith(ViewerWidget.supported_formats_rgb) or file_name.lower().endswith(ViewerWidget.supported_formats_raw)

    # @mainthread
    def add_image_to_grid(self, file_path):
        card = ThumbnailCard(file_path=file_path, grid_width=self.grid_width)
        card.bind(on_touch_down=self.on_select)
        self.ids['grid_layout'].add_widget(card)
        return card

    def process_exif_data(self, file_path_list, exif_data_list):
        thumb_data_list = []
        try:
            for i in range(len(file_path_list)):
                exif_data = exif_data_list[i]
                file_path = file_path_list[i]

                thumb_base64 = exif_data.get('ThumbnailImage')
                if thumb_base64 is not None:
                    image = np.fromstring(base64.b64decode(thumb_base64[7:]), dtype=np.uint8)
                    thumb = cv2.imdecode(image, 1)
                    thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                else:
                    if file_path.lower().endswith(ViewerWidget.supported_formats_raw):
                        with rawpy.imread(file_path) as raw:
                            thumb = raw.postprocess()
                    else:
                        thumb = cv2.imread(file_path)
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            
                thumb_size = self.calc_resize_image((thumb.shape[1], thumb.shape[0]), self.thumb_width)
                thumb = cv2.resize(thumb, thumb_size)
                orientation = exif_data.get('Orientation')
                if orientation is not None:
                    if orientation == 'Rotate 180':
                        thumb = cv2.rotate(thumb, cv2.ROTATE_180)
                    elif orientation == 'Rotate 270 CW':
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif orientation == 'Rotate 90 CW':
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                    elif orientation == 'Mirror horizontal':
                        thumb = cv2.flip(thumb, 1)
                    elif orientation == 'Mirror vertical':
                        thumb = cv2.flip(thumb, 0)
                    elif orientation == 'Mirror horizontal and rotate 270 CW':
                        thumb = cv2.flip(thumb, 1)
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif orientation == 'Mirror horizontal and rotate 90 CW':
                        thumb = cv2.flip(thumb, 1)
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                thumb = thumb.astype(np.float32)/256.0
                thumb_data_list.append(thumb)

            return thumb_data_list

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
        
    def active_card(self, card):
        card.md_bg_color = [0.8, 0.8, 0.8, 1]  # 選択色に変更
        self.selected_cards.add(card)

    def deactive_card(self, card):
        card.md_bg_color = [0.1, 0.1, 0.1, 1]  # デフォルト色に戻す
        self.selected_cards.remove(card)
        
    def toggle_card(self, card):
        if card in self.selected_cards:
            self.deactive_card(card)
        else:
            self.active_card(card)

    def clear_selection(self):
        for card in self.selected_cards:
            card.md_bg_color = [0.1, 0.1, 0.1, 1]  # デフォルト色に戻す
        self.selected_cards.clear()

    def on_select(self, instance, touch):
        if instance.collide_point(*touch.pos):
            if touch.is_mouse_scrolling or (touch.button == 'left'):
                if 'shift' in Window.modifiers and self.last_selected:

                    # 追加でないなら消去
                    if not( 'ctrl' in Window.modifiers or 'meta' in Window.modifiers ):
                        self.clear_selection()

                    # シフトキーで範囲選択
                    start_index = self.get_card_index(self.last_selected)
                    end_index = self.get_card_index(instance)
                    if start_index is not None and end_index is not None:
                        for i in range(min(start_index, end_index), max(start_index, end_index) + 1):
                            card = self.cards[i]
                            self.active_card(card)

                else:
                    # 単独選択またはCmd/Ctrlでのトグル
                    if 'ctrl' in Window.modifiers or 'meta' in Window.modifiers:
                        self.toggle_card(instance)
                    else:
                        # すべてのラベルを非選択にしてから選択
                        self.clear_selection()
                        self.toggle_card(instance)

                self.last_selected = instance


    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if (key == 97 and ('ctrl' in modifier or 'meta' in modifier)):  # Aキー
            self.clear_selection() # ２重登録禁止
            for card in self.cards:
                self.active_card(card)
            return True

    def get_card_index(self, card):
        if card in self.cards:
            return self.cards.index(card)
        return None
    
    def calc_resize_image(self, original_size, max_length):
        width, height = original_size

        if width > height:
            # 幅が長辺の場合
            scale_factor = max_length / width
        else:
            # 高さが長辺の場合
            scale_factor = max_length / height

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        return (new_width, new_height)

# テストアプリケーション
class Viewer_WidgetApp(MDApp):
    def build(self):
        viewer = ViewerWidget(grid_width=dp(120), thumb_width=dp(160))

        viewer.set_path("/Users/uniuyuni/PythonProjects/escargot/picture")  # 画像フォルダーのパスを指定

        return viewer

if __name__ == "__main__":
    Viewer_WidgetApp().run()

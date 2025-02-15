import os
import numpy as np
import pyautogui as pag
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.core.window import Window as KVWindow
from kivy.graphics.texture import Texture as KVTexture
from kivy.clock import Clock, mainthread
from kivy.metrics import Metrics, dp
from functools import partial
import re

import core
import imageset
import curve
import effects
import pipeline
import param_slider
import viewer_widget
import histogram_widget
import metainfo
import util
import mask_editor2
import color_picker
import macos
import film_simulation
import lens_simulator
import config
import export
from export_dialog import ExportDialog, ExportConfirmDialog


class MainWidget(MDBoxLayout):

    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)

        self.texture_height = 1024
        self.texture_width = 1024
        self.texture = None
        self.imgset = None
        self.click_x = 0
        self.click_y = 0        
        self.crop_image = None
        self.crop_info = None
        self.is_zoomed = False
        self.drag_start_point = None
        self.primary_param = {}
        self.primary_effects = effects.create_effects()
        self.apply_thread = None
        self.is_draw_image = False

        #self._keyboard = KVWindow.request_keyboard(self._keyboard_closed, self)
        #self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def on_kv_post(self, *args, **kwargs):
        super(MainWidget, self).on_kv_post(*args, **kwargs)

        self.ids['mask_editor2'].opacity = 0
        self.ids['mask_editor2'].disabled = True
        self.set_film_presets()
        self.set_lens_presets()
 
    def load_image(self, file_path, exif_data):
        self.imgset = imageset.ImageSet()
        self.imgset.load(file_path, exif_data, self.primary_param, self.start_draw_image_and_crop)

        self.texture = KVTexture.create(size=(self.texture_width, self.texture_height), colorfmt='rgb', bufferfmt='float')
        self.texture.flip_vertical()

        self.click_x = 0
        self.click_y = 0
        self.is_zoomed = False
        self.crop_image = None
        _, self.crop_info = core.crop_image(self.imgset.img, self.texture_width, self.texture_height, self.click_x, self.click_y, (0, 0), self.is_zoomed)
        self.start_draw_image()
    
    def start_draw_image_and_crop(self, imgset, offset=(0, 0)):
        if self.is_draw_image == False:
            if self.imgset == imgset:
                self.crop_image = None
                effects.reeffect_all(self.primary_effects)
                self.start_draw_image(offset)

    def empty_image(self):
        self.imgset = None
        self.texture = KVTexture.create(size=(self.texture_width, self.texture_height), colorfmt='rgb', bufferfmt='float')
        self.texture.flip_vertical()
        self.ids["preview"].texture = None

    # @mainthread
    def blit_image(self, img):
        self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.ids["preview"].texture = None # 更新のために必要
        self.ids["preview"].texture = self.texture

    def draw_histogram(self, img):
        self.ids["histogram"].draw_histogram(img)

    def draw_image(self, offset, dt):
        if self.imgset is not None:
            img, self.crop_image, self.crop_info = pipeline.process_pipeline(self.imgset.img, self.crop_info, offset, self.crop_image, self.is_zoomed, self.texture_width, self.texture_height, self.click_x, self.click_y, self.primary_effects, self.primary_param, self.ids['mask_editor2'])
            util.print_nan_inf(img)
            
            #self.ids['mask_editor2'].set_crop_image(self.crop_image)
            img = core.apply_gamma(img, 1.0/2.222)
            self.draw_histogram(img)
            img = np.clip(img, 0, 1)
            self.blit_image(img)
        self.is_draw_image = False

    def start_draw_image(self, offset=(0, 0)):
        if self.is_draw_image == False: #２重コール防止
            self.is_draw_image = True
            Clock.schedule_once(partial(self.draw_image, offset), -1)
        #self.apply_thread = threading.Thread(target=self.draw_image, daemon=True)
        #self.apply_thread.start()
    
    def apply_effects_lv(self, lv, effect):
        mask = self.ids['mask_editor2'].get_active_mask()
        if mask is None:
            self.primary_effects[lv][effect].set2param(self.primary_param, self)
        else:
            mask.effects[lv][effect].set2param(mask.effects_param, self)

        self.ids['mask_editor2'].set_draw_mask(lv == 3)
        self.start_draw_image()

    def set_effect_param(self, lv, effect, arg):
        mask = self.ids['mask_editor2'].get_active_mask()
        if mask is None:
            self.primary_effects[lv][effect].set2param2(self.primary_param, arg)
        else:
            mask.effects[lv][effect].set2param2(mask.effects_param, arg)

        #if lv == 0 and effect == 'rotation':
        #    self.ids['mask_editor2'].set_orientation(self.primary_param.get('rotation', 0), self.primary_param.get('rotation2', 0), self.primary_param.get('flip_mode', 0))

        self.ids['mask_editor2'].set_draw_mask(lv == 3)
        self.start_draw_image()
    
    def reset_param(self, param):
        param.clear()

    def set2widget_all(self, effects, param):
        if effects is None:
            effects = self.primary_effects
            param = self.primary_param
            
        for dict in effects:
            for l in dict.values():
                l.set2widget(self, param)
                l.reeffect()
    
    @mainthread
    def on_select(self, card):
        # 前の設定を保存
        if self.imgset is not None:
            export.save_json(self.imgset.file_path, self.primary_param, self.ids['mask_editor2'])

        self.reset_param(self.primary_param)
        self.ids['mask_editor2'].clear_mask()
        if card is not None:
            self.load_image(card.file_path, card.exif_data)
            export.load_json(card.file_path, self.primary_param, self.ids['mask_editor2'])
            self.set2widget_all(self.primary_effects, self.primary_param)
            self.set_exif_data(card.exif_data)
        else:
            self.empty_image()

    def on_image_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # ズーム操作
            if touch.is_double_tap == True:
                self.is_zoomed = not self.is_zoomed
                if self.is_zoomed == False:
                    self.click_x, self.click_y = 0, 0
                else:
                    # ウィンドウ座標からローカルイメージ座標に変換
                    self.click_x, self.click_y = util.to_texture(touch.pos, self.ids['preview'])

                self.crop_info = None
                #_, self.crop_info = core.crop_image(self.imgset.img, self.texture_width, self.texture_height, self.click_x, self.click_y, (0, 0), self.is_zoomed)
                effects.reeffect_all(self.primary_effects)
                self.start_draw_image_and_crop(self.imgset)

            # ドラッグ操作
            elif self.is_zoomed == True:
                self.drag_start_point = touch.pos

    def on_image_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            if self.is_zoomed == True:
                if self.drag_start_point != None:
                    offset_x = touch.pos[0] - self.drag_start_point[0]
                    offset_y = touch.pos[1] - self.drag_start_point[1]
                    offset_x = -offset_x

                    #_, self.crop_info = core.crop_image_info(self.imgset.img, self.crop_info, (offset_x, offset_y))
                    effects.reeffect_all(self.primary_effects)
                    self.start_draw_image_and_crop(self.imgset, (offset_x, offset_y))

                    self.drag_start_point = touch.pos
                
    def on_image_touch_up(self, touch):
        if self.is_zoomed == True:
            if self.drag_start_point != None:
                self.drag_start_point = None

    def on_select_press(self):
        macos.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self.handle_for_dir_selection).run()

    def on_export_press(self):
        if self.imgset is not None:
            export.save_json(self.imgset.file_path, self.primary_param, self.ids['mask_editor2'])

        dialog = ExportDialog(callback=self.handle_export_dialog)
        dialog.open()

    def handle_export_dialog(self, preset):
        # 保存先ファイルの存在チェック
        cards = self.ids['viewer'].get_selected_cards()
        isfile = False
        for x in cards:
            ex_path = self._make_export_path(x.file_path, preset)
            if os.path.isfile(ex_path):
                isfile = True
                break

        if isfile == True:
            dialog = ExportConfirmDialog(preset=preset, callback=self.handle_confirm_dialog)
            dialog.open()

        elif len(cards) > 0:
            self.handle_confirm_dialog('Overwrite', preset)            

    def handle_confirm_dialog(self, select, preset):
        if select in ['Overwrite', 'Rename']:
            cards = self.ids['viewer'].get_selected_cards()
            for x in cards:
                ex_path = self._make_export_path(x.file_path, preset)
                if select == 'Rename':
                    ex_path = self._find_not_duplicate_filename(ex_path)

                exfile = export.ExportFile(x.file_path, x.exif_data)
                exfile.write_to_file(ex_path, 85)

    def _make_export_path(seslf, path, preset):
        dirname, basename = os.path.split(path)
        basename_with_out_ext, ext = os.path.splitext(basename)
        if len(preset['output_path']) > 0:
            if preset['output_path'] != '/':
                ex_path = os.path.join(dirname, preset['output_path'], basename_with_out_ext) + preset['format']
            else:
                ex_path = os.path.join(preset['output_path'], basename_with_out_ext) + preset['format']
        else:
            ex_path = os.path.join(dirname, basename_with_out_ext) + preset['format']
        return ex_path

    def _find_not_duplicate_filename(self, path):
        addnum = -1
        while os.path.isfile(path) == True:
            path_with_out_ext, ext = os.path.splitext(path)
            path_with_out_ext = re.split('-[0-9]+$', path_with_out_ext)
            path = path_with_out_ext[0] + str(addnum) + ext
            addnum -= 1

        return path

    def delay_set_image(self, dt):
        self.ids['mask_editor2'].set_orientation(self.primary_param.get('rotation', 0), self.primary_param.get('rotation2', 0), self.primary_param.get('flip_mode', 0))
        self.ids['mask_editor2'].set_image(self.imgset.img, self.texture_width, self.texture_height, self.crop_info, -1)
        self.ids['mask_editor2'].update()

    def on_mask2_press(self, value):
        if value == "down":
            self.ids['mask_editor2'].opacity = 1
            self.ids['mask_editor2'].disabled = False
            Clock.schedule_once(self.delay_set_image, -1)   # editor2のサイズが未決定なので遅らせる
        else:
            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self.ids['mask_editor2'].set_active_mask(None)

    def handle_for_dir_selection(self, selection):
        if selection is not None:
            config.set_config('import_path', selection[0].decode())

    def on_lut_select_folder(self):
        macos.FileChooser(title="Select LUT Folder", mode="dir", filters=[("CUBE Files", "*.cube")], on_selection=self.handle_for_lut).run()

    def handle_for_lut(self, selection):
        if selection is not None:
            path = selection[0].decode()
            config.set_config('lut_path', path)

    def on_current_tab(self, current):
        if self.imgset is not None:
            self.apply_effects_lv(0, "crop")

    def set_lut_path(self, path):
        lut_values = ['None']
        effects.LUTEffect.file_pathes = { 'None': None, }

        file_list = os.listdir(path)
        file_list.sort()
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if file_name.lower().endswith(('.cube')):
                lut_values.append(file_name)
                effects.LUTEffect.file_pathes[file_name] = file_path
        self.ids['lut_spinner'].values = lut_values

    def set_film_presets(self):
        presets = ['None']

        for preset in film_simulation.FilmSimulation.FILM_PRESETS:
            presets.append(preset)

        self.ids['spinner_film_preset'].values = presets

    def set_lens_presets(self):
        presets = ['None']

        for preset in lens_simulator.LensSimulator.LENS_PRESETS:
            presets.append(preset)

        self.ids['spinner_lens_preset'].values = presets

    def set_exif_data(self, exif_data):
        self.ids['exif_file_name'].value = exif_data.get("FileName", "-")
        self.ids['exif_file_size'].value = exif_data.get("FileSize", "-")
        self.ids['exif_create_date'].value = exif_data.get("CreateDate", "-")
        self.ids['exif_image_size'].value = exif_data.get("ImageSize", "-")
        self.ids['exif_iso_speed'].value = str(exif_data.get("ISO", "-"))
        self.ids['exif_aperture'].value = str(exif_data.get("ApertureValue", "-"))
        self.ids['exif_shutter_speed'].value = exif_data.get("ShutterSpeedValue", "-")
        self.ids['exif_brightness'].value = str(exif_data.get("BrightnessValue", "-"))
        self.ids['exif_flash'].value = exif_data.get("Flash", "-")
        self.ids['exif_white_balance'].value = exif_data.get("WhiteBalance", "-")
        self.ids['exif_focal_length'].value = exif_data.get("FocalLength", "-")
        self.ids['exif_exposure_program'].value = exif_data.get("PictureMode", "-")
        self.ids['exif_make'].value = exif_data.get("Make", "-")
        self.ids['exif_model'].value = exif_data.get("Model", "-")
        self.ids['exif_lens_model'].value = exif_data.get("LensModel", "-")
        self.ids['exif_software'].value = exif_data.get("Software", "-")
        #self.ids['exif_'].value = exif_data.get("", "-")

    def get_scale(self):
        return self.crop_info[4]
    
    """
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'a':
            self._trigger_layout()

        return True
    """
    
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        
        self.title = "escargot"
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Blue'

    def build(self): 
        widget = MainWidget()

        KVWindow.size = (dp(600), dp(400))
        scr_w,scr_h = pag.size()
        KVWindow.left = (scr_w - dp(600)) // 2
        KVWindow.top = (scr_h - dp(400)) // 2

        # testcode
        #widget.load_image("DSCF0090.raf")
        #widget.load_image(os.getcwd() + "/picture/DSCF0002-small.tif")
        widget.ids['viewer'].set_path(os.getcwd() + "/picture")

        config.set_main_widget(widget)
        config.load_config()

        return widget

if __name__ == '__main__':
    """
    import imageio
    import imageio.v3 as iio

    imageio.plugins.freeimage.download()

    print(imageio.plugins.formats)

    metadata = iio.immeta(os.getcwd() + "/picture/DSCF0002.jpg")
    metadata = iio.immeta(os.getcwd() + "/picture/DSCF0002.raf", plugin='rawpy', exclude_applied=False)

    reader = imageio.get_reader(os.getcwd() + "/picture/DSCF0002.raf")
    metadata = reader.get_meta_data(0)
    """
    MainApp().run()



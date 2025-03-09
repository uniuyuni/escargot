
import os
import numpy as np
from wand.image import Image as WandImage
from datetime import datetime as dt
import json

from imageset import ImageSet
import effects
import pipeline
import mask_editor2
import color
import crop_editor
import config

SPECIAL_PARAM = [
    # for core.adjust_shape
    'original_img_size',
    'img_size',
    'crop_info',
    # for imageset._set_temperature
    'color_temerature_switch',
    'color_temperature_reset',
    'color_tint_reset',
    'color_Y',
    # for effects.CropEffect
    'crop_enable',
]

def delete_special_param(param):
    p = param.copy()

    for key in SPECIAL_PARAM:
        try:
            val = p[key]
            del p[key]
        except KeyError:
            pass
    
    return p

def copy_special_param(tar, src):
    for key in SPECIAL_PARAM:
        try:
            val = src[key]
            tar[key] = val
        except KeyError:
            pass

def serialize(param, mask_editor2):
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y/%m/%d')
    mask_dict = mask_editor2.serialize()

    # セーブしないパラメータを削除
    param = delete_special_param(param)

    # パラメータがないのでそもそもファイルを作らない
    if len(param) == 0 and (mask_dict is None or len(mask_dict) == 0):
        return None

    dict = {
        'make': "escargo",
        'date': tstr,
        'version': "0.4.1",
        'primary_param': param,
    }
    if mask_dict is not None:
        dict.update(mask_dict)

    return dict

def deserialize(dict, param, mask_editor2):
    param.update(dict['primary_param'])
    param['crop_info'] = crop_editor.CropEditor.convert_rect_to_info(param['crop_rect'], config.get_config('preview_size')/max(param['original_img_size']))

    mask_editor2.clear_mask()
    mask_dict = dict.get('mask2', None)
    if mask_dict is not None:
        mask_editor2.deserialize(dict)

def save_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        dict = serialize(param, mask_editor2)
        if dict is not None:
            with open(file_path, 'w') as f:
                json.dump(dict, f)

def load_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        try:
            with open(file_path, 'r') as f:
                dict = json.load(f)
                deserialize(dict, param, mask_editor2)
                return dict
            
        except FileNotFoundError as e:
            pass
    
    return None

class ExportFile():

    FORMAT = {
        '.JPG': 'JPEG',
        '.TIFF': 'TIFF',
        '.HEIC': 'HEIC',
    }

    def __init__(self, file_path, exif_data):
        self.file_path = str(file_path)
        self.exif_data = exif_data.copy()

        self.ex_path = None
        self.quality = 100
        self.color_space = "sRGB"
        self.imgset = None
        self.effects = effects.create_effects()
        self.param = {}
        self.mask_editor2 = mask_editor2.MaskEditor2()

    def write_to_file(self, ex_path, quality, resize_str, sharpen, color_space, exif_data):
        self.quality = quality
        self.ex_path = ex_path
        self.color_space = color_space
        self.imgset = ImageSet()
        result = self.imgset.load(self.file_path, self.exif_data, self.param)
        if result == False:
            return
        elif result == True:
            pass
        else:
            self.imgset.load(self.file_path, self.exif_data, self.param, result)
        #self.mask_editor2.set_orientation(self.param.get('rotation', 0), self.param.get('rotation2', 0), self.param.get('flip_mode', 0))
        self.mask_editor2.set_texture_size(self.imgset.img.shape[1], self.imgset.img.shape[0])
        self.mask_editor2.set_image(self.param['original_img_size'], self.param.get('crop_info', None))
        #self.mask_editor2.update()

        load_json(self.file_path, self.param, self.mask_editor2)
        img = pipeline.export_pipeline(self.imgset.img, self.effects, self.param, self.mask_editor2)
        img = color.xyz_to_rgb(img, self.color_space, True)
        img = np.clip(img, 0, 1)

        ex_ext = os.path.splitext(self.ex_path)[1]
        try:
            format = ExportFile.FORMAT[ex_ext]
        except KeyError:
            return
        
        ex_dir = os.path.dirname(self.ex_path)
        os.makedirs(ex_dir, exist_ok=True)

        with WandImage.from_array(img) as wi:
            wi.compression_quality = self.quality 
            wi.format = format
            wi.sharpen(sharpen)
            wi.transform(resize=resize_str)
            wi.save(filename=self.ex_path)

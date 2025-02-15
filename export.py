
import os
from wand.image import Image as WandImage
from datetime import datetime as dt
import json

from imageset import ImageSet
import effects
import pipeline
import mask_editor2

def serialize(param, mask_editor2):
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y/%m/%d')
    mask_dict = mask_editor2.serialize()

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
    mask_dict = dict.get('mask2', None)
    if mask_dict is not None:
        mask_editor2.deserialize(dict)

def save_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        with open(file_path, 'w') as f:
            dict = serialize(param, mask_editor2)
            json.dump(dict, f)

def load_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        try:
            with open(file_path, 'r') as f:
                dict = json.load(f)
                deserialize(dict, param, mask_editor2)
        except FileNotFoundError as e:
            pass

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
        self.imgset = None
        self.effects = effects.create_effects()
        self.param = {}
        self.mask_editor2 = mask_editor2.MaskEditor2()

    def write_to_file(self, ex_path, quality):
        self.quality = quality
        self.ex_path = ex_path
        self.imgset = ImageSet()
        self.imgset.load(self.file_path, self.exif_data, self.param, self._start_pipeline)

    def _start_pipeline(self, imgset):
        load_json(self.file_path, self.param, self.mask_editor2)
        img = pipeline.export_pipeline(imgset.img, self.effects, self.param, self.mask_editor2)

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
            wi.save(filename=self.ex_path)

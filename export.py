
import os
import numpy as np
from wand.image import Image as WandImage
import json
import exiftool
import colour

import core
import define
from imageset import ImageSet
import effects
import pipeline
import mask_editor2
import params
import effects
import config

safe_tags = [
    # EXIF（カメラと撮影設定）
    "EXIF:Make",
    "EXIF:Model",
    "EXIF:Software",
    "EXIF:ExposureTime",
    "EXIF:FNumber",
    "EXIF:ApertureValue",
    "EXIF:Aperture",
    "EXIF:ISO",
    "EXIF:ISOSpeedRatings",
    "EXIF:ShutterSpeedValue",
    "EXIF:ExposureProgram",
    "EXIF:ExposureCompensation",
    "EXIF:ExposureBiasValue",
    "EXIF:MeteringMode",
    "EXIF:Flash",
    "EXIF:FlashMode",
    "EXIF:WhiteBalance",
    "EXIF:FocalLength",
    "EXIF:FocalLengthIn35mmFormat",
    "EXIF:DigitalZoomRatio",
    "EXIF:LensModel",
    "EXIF:LensInfo",
    "EXIF:LensMake",
    "EXIF:LensSerialNumber",
    "EXIF:SceneCaptureType",
    "EXIF:Contrast",
    "EXIF:Saturation",
    "EXIF:Sharpness",
    "EXIF:SubjectDistance",
    "EXIF:SubjectDistanceRange",
    "EXIF:BrightnessValue",
    "EXIF:WhiteBalance",
    "EXIF:PictureMode",
    "EXIF:SubjectDistanceRange",
    
    # EXIF（基本情報）
    "EXIF:Artist",
    "EXIF:Copyright",
    "EXIF:ImageDescription",
    "EXIF:UserComment",
    "EXIF:XPTitle",
    "EXIF:XPComment",
    "EXIF:XPAuthor",
    "EXIF:XPKeywords",
    "EXIF:XPSubject",
    "EXIF:DocumentName",
    "EXIF:Orientation",
    #"EXIF:ImageWidth",
    #"EXIF:ImageHeight",
    #"EXIF:XResolution",
    #"EXIF:YResolution",
    #"EXIF:ResolutionUnit",
    
    # EXIF（日時情報）
    "EXIF:DateTimeOriginal",
    "EXIF:CreateDate",
    "EXIF:ModifyDate",
    "EXIF:DateTimeDigitized",
    
    # EXIF（GPS情報）
    "EXIF:GPSLatitude",
    "EXIF:GPSLongitude",
    "EXIF:GPSAltitude",
    "EXIF:GPSTimeStamp",
    "EXIF:GPSDateStamp",
    "EXIF:GPSProcessingMethod",
    "EXIF:GPSImgDirection",
    
    # IPTC（基本情報）
    "IPTC:ObjectName",
    "IPTC:Keywords",
    "IPTC:Caption-Abstract",
    "IPTC:Writer-Editor",
    "IPTC:Headline",
    "IPTC:SpecialInstructions",
    "IPTC:Byline",
    "IPTC:BylineTitle",
    "IPTC:Credit",
    "IPTC:Source",
    "IPTC:CopyrightNotice",
    "IPTC:Contact",
    
    # XMP（基本情報）
    "XMP:Title",
    "XMP:Description",
    "XMP:Creator",
    "XMP:Rights",
    "XMP:Subject",
    "XMP:Label",
    "XMP:Rating",
    "XMP:CreateDate",
    "XMP:ModifyDate"
]

def make_safe_metadata(exif_data):
    safe_metadata = {}
    for tag in safe_tags:
        group, field = tag.split(':')
        
        # タグが存在する場合のみ追加
        if field in exif_data:
            safe_metadata[field] = exif_data[field]
    return safe_metadata
  
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
        self.icc_profile = "sRGB"
        self.imgset = None
        self.effects = effects.create_effects()
        self.param = {}
        self.mask_editor2 = mask_editor2.MaskEditor2()

    def write_to_file(self, ex_path, quality, resize_str, sharpen, icc_profile, exifsw, dithering):
        self.quality = quality
        self.ex_path = ex_path
        self.icc_profile = icc_profile
        self.imgset = ImageSet()
        result = self.imgset.preload(self.file_path, self.exif_data, self.param)
        self.imgset.load(result, self.file_path, self.exif_data, self.param)

        #self.mask_editor2.set_orientation(self.param.get('rotation', 0), self.param.get('rotation2', 0), self.param.get('flip_mode', 0))
        self.imgset.img.shape[1], self.imgset.img.shape[0]
        self.mask_editor2.set_texture_size(self.imgset.img.shape[1], self.imgset.img.shape[0])
        self.mask_editor2.set_primary_param(self.param, params.get_disp_info(self.param))
        self.mask_editor2.set_ref_image(self.imgset.img, self.imgset.img)
        #self.mask_editor2.update()

        params.load_json(self.file_path, self.param, self.mask_editor2)
        img = pipeline.export_pipeline(self.imgset.img, self.effects, self.param, self.mask_editor2)

        img = colour.RGB_to_RGB(img, 'ProPhoto RGB', core.ICC_PROFILE_TO_COLOR_SPACE[self.icc_profile], config.get_config('cat'),
                                apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32)
        img = np.clip(img, 0, 1)

        ex_ext = os.path.splitext(self.ex_path)[1]
        try:
            format = ExportFile.FORMAT[ex_ext]
        except KeyError:
            return

        if dithering:
            if format == 'JPEG':
                img = core.jjn_dither_uint8(img)
            elif format == 'TIFF':
                img = core.jjn_dither_uint16(img)
            elif format == 'HEIC':
                img = core.jjn_dither_uint16(img)

        ex_dir = os.path.dirname(self.ex_path)
        os.makedirs(ex_dir, exist_ok=True)

        # ファイル書き込み
        with WandImage.from_array(img) as wi:
            wi.compression_quality = self.quality 
            wi.format = format
            wi.sharpen(sharpen)
            wi.transform(resize=resize_str)
            wi.save(filename=self.ex_path)

        # Exif書き込み
        if exifsw:
            with exiftool.ExifToolHelper(common_args=['-P', '-overwrite_original']) as et:
                safe_metadata = make_safe_metadata(self.exif_data)
                safe_metadata["Software"] = define.APPNAME + " " + define.VERSION
                result = et.set_tags(self.ex_path, tags=safe_metadata)
                print(result)

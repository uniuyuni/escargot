import cv2
import numpy as np
import core
import rawpy
import math
import base64
import logging
import os
import threading
from dcp_profile import DCPReader, DCPProcessor
#import skimage
#import maxim_util

import util
import viewer_widget

class ImageSet:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.file_path = None
        self.src = None     # 元画像 uint16
        self.img = None     # 加工元画像 float32
        self.tmb = None     # 縮小画像 float32
        self.prv = None     # 加工画像 float32

    def __get_image_size(self, exif_data):
        top, left = exif_data.get("RawImageCropTopLeft", "0 0").split()
        top, left = int(top), int(left)

        width, height = exif_data.get("RawImageCroppedSize", "0x0").split('x')
        width, height = int(width), int(height)
        if width == 0 and height == 0:
            width, height = exif_data.get("ImageSize", "0x0").split('x')
            width, height = int(width), int(height)
            if width == 0 and height == 0:
                raise AttributeError("Not Find image size data")
            
        return (top, left, width, height)
    
    def __set_image_size(self, exif_data, top, left, width, height):
        if exif_data.get("RawImageCropTopLeft", None) is not None:
            exif_data["RawImageCropTopLeft"] = str(top) + " " + str(left)

        if exif_data.get("RawImageCroppedSize", None) is not None:
            exif_data["RawImageCroppedSize"] = str(width) + "x" + str(height)

        exif_data["ImageSize"] = str(width) + "x" + str(height)

    def __set_temperature(self, param, temp, tint, Y):
        param['color_temperature_reset'] = temp
        param['color_temperature'] = param['color_temperature_reset']
        param['color_tint_reset'] = tint
        param['color_tint'] = param['color_tint_reset']
        param['color_Y'] = Y

    def __load_raw(self, file_path, exif_data, param, callback):
        try:
            # RAWで読み込んでみる
            with rawpy.imread(file_path) as raw:

                self.src = raw.postprocess( output_color=rawpy.ColorSpace.raw,
                                            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                            output_bps=16,
                                            no_auto_scale=False,
                                            use_camera_wb=False,
                                            user_wb = [1.0, 1.0, 1.0, 0.0],
                                            gamma=(1.0, 0.0),
                                            four_color_rgb=True,
                                            #user_black=0,
                                            no_auto_bright=True,
                                            highlight_mode=5,
                                            auto_bright_thr=0.0005)

                self.img = self.src

                # float32へ
                self.img = self.img.astype(np.float32)/65535.0

                # レンズ補正
                self.img = core.modify_lens(self.img, exif_data)

                # 回転情報取得
                rad, flip = util.split_orientation(util.str_to_orientation(exif_data.get("Orientation", "")))

                # クロップとexifデータの回転
                top, left, width, height = self.__get_image_size(exif_data)
                if rad < 0.0:
                    top, left = left, top
                    width, height = height, width
                    self.__set_image_size(exif_data, top, left, width, height)
                self.img = self.img[top:top+height, left:left+width]

                # プロファイルを適用
                dcp_path = os.getcwd() + "/dcp/Fujifilm X-Pro3 Adobe Standard velvia.dcp"
                reader = DCPReader(dcp_path)
                profile = reader.read()
                processor = DCPProcessor(profile)
                self.img = processor.process(self.img, illuminant='1', use_look_table=True).astype(np.float32)

                # コントラスト補正
                #self.img = skimage.exposure.equalize_adapthist(self.img, clip_limit=0.03)
                
                # 明るさ補正
                source_ev, _ = core.calculate_ev_from_image(core.normalize_image(self.img))
                Av = exif_data.get('ApertureValue', 1.0)
                _, Tv = exif_data.get('ShutterSpeedValue', "1/100").split('/')
                Tv = float(_) / float(Tv)
                Ev = math.log2((Av**2)/Tv)
                Sv = math.log2(exif_data.get('ISO', 100)/100.0)
                Ev = Ev + Sv
                self.img = self.img * core.calc_exposure(self.img, core.calculate_correction_value(source_ev, Ev))

                # ホワイトバランス定義
                wb = raw.camera_whitebalance
                wb = np.array([wb[0], wb[1], wb[2]]).astype(np.float32)/1024.0
                wb[1] = np.sqrt(wb[1])
                self.img[:,:,1] *= wb[1]
                temp, tint, Y, = core.invert_RGB2TempTint(wb, 5000.0)
                self.__set_temperature(param, temp, tint, Y)

                param['img_size'] = [self.img.shape[1], self.img.shape[0]]
                if self.img.shape[1] != width or self.img.shape[0] != height:
                    logging.error("ImageSize is not ndarray.shape")

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
            return False
        
        callback(self)
        return True

    def __load_rgb(self, file_path, exif_data, param, callback):
        # RGB画像で読み込んでみる
        self.src = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if self.src is not None:
            if self.src.dtype == np.uint8:
                self.src = self.src.astype(np.uint16)
                self.src = self.src*256
            self.img = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB).astype(np.float32)/65535.0
            self.img = core.apply_gamma(self.img, 2.222)

            temp, tint, Y, = core.invert_RGB2TempTint((1.0, 1.0, 1.0), 5000.0)
            self.__set_temperature(param, temp, tint, Y)
            
            param['img_size'] = [self.img.shape[1], self.img.shape[0]]
            top, left, width, height = self.__get_image_size(exif_data)
            if self.img.shape[1] != width or self.img.shape[0] != height:
                logging.error("ImageSize is not ndarray.shape")
        else:
            logging.warning("file is not supported " + file_path)
            return False
        
        callback(self)
        return True
    
    def __load_thumb(self, exif_data, param):
        thumb_base64 = exif_data.get('ThumbnailImage')
        if thumb_base64 is not None:
            thumb = np.frombuffer(base64.b64decode(thumb_base64[7:]), dtype=np.uint8)
            thumb = cv2.imdecode(thumb, 1)
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumb = thumb.astype(np.float32)/255
            thumb = core.apply_gamma(thumb, 2.222)
            temp, tint, Y, = core.invert_RGB2TempTint((1.0, 1.0, 1.0), 5000.0)
            self.__set_temperature(param, temp, tint, Y)
            self.img = thumb

            top, left, width, height = self.__get_image_size(exif_data)
            self.img = cv2.resize(self.img, dsize=(width, height))
            param['img_size'] = [width, height]
    
    def load(self, file_path, exif_data, param, callback):
        if file_path.lower().endswith(viewer_widget.supported_formats_raw):
            self.__load_thumb(exif_data, param)
            thread = threading.Thread(target=self.__load_raw, args=[file_path, exif_data, param, callback], daemon=True)
            thread.start()

        elif file_path.lower().endswith(viewer_widget.supported_formats_rgb):
            self.__load_thumb(exif_data, param)
            thread = threading.Thread(target=self.__load_rgb, args=[file_path, exif_data, param, callback], daemon=True)
            thread.start()
            
        else:
            logging.warning("file is not supported " + file_path)
            return False
            
        self.file_path = file_path
        logging.info("loading file " + file_path)

        return True

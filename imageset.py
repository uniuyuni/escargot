import cv2
import numpy as np
import core
import rawpy
import math
import base64
import logging
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
    
    def load(self, file_path, exif_data, param):
        if file_path.lower().endswith(viewer_widget.supported_formats_raw):

            try:
                # RAWで読み込んでみる
                with rawpy.imread(file_path) as raw:

                    self.src = raw.postprocess( output_color=rawpy.ColorSpace.sRGB,
                                                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                                output_bps=16,
                                                no_auto_scale=False,
                                                use_camera_wb=False,
                                                user_wb = [1.0, 1.0, 1.0, 0.0],
                                                gamma=(1.0, 0.0),
                                                four_color_rgb=True,
                                                #user_black=0,
                                                no_auto_bright=True,
                                                highlight_mode=3,
                                                auto_bright_thr=0.0005)

                    # トーンカーブ適用
                    self.img = raw.tone_curve[self.src]

                    # 回転情報取得
                    rad, flip = util.split_orientation(util.str_to_orientation(exif_data.get("Orientation", "")))

                    # クロップ
                    top, left = exif_data.get("RawImageCropTopLeft", "0 0").split()
                    top, left = int(top), int(left)
                    width, height = exif_data.get("RawImageCroppedSize", "0x0").split('x')
                    width, height = int(width), int(height)
                    if rad < 0.0:
                        top, left = left, top
                        exif_data["RawImageCropTopLeft"] = str(top) + " " + str(left)

                        width, height = height, width
                        exif_data["RawImageCroppedSize"] = str(width) + "x" + str(height)
                        exif_data["ImageSize"] = str(width) + "x" + str(height)
                    self.img = self.img[top:top+height, left:left+width]

                    # float32へ
                    self.img = self.img.astype(np.float32)/65535.0

                    # 明るさ補正
                    source_ev, _ = core.calculate_ev_from_image(core.normalize_image(self.img))
                    Av = exif_data.get('ApertureValue', 1.0)
                    _, Tv = exif_data.get('ShutterSpeedValue', "1/100").split('/')
                    Tv = float(_) / float(Tv)
                    Ev = math.log2((Av**2)/Tv)
                    Sv = math.log2(exif_data.get('ISO', 100)/100.0)
                    Ev = Ev + Sv
                    self.img = self.img * core.adjust_exposure(self.img, core.calculate_correction_value(source_ev, Ev))

                    # ホワイトバランス定義
                    wb = raw.camera_whitebalance
                    wb = np.array([wb[0], wb[1], wb[2]]).astype(np.float32)/1024.0
                    wb[1] = np.sqrt(wb[1])
                    self.img[:,:,1] *= wb[1]
                    temp, tint, Y, = core.invert_RGB2TempTint(wb, 5000.0)
                    param['color_temperature_reset'] = temp
                    param['color_temperature'] = param['color_temperature_reset']
                    param['color_tint_reset'] = tint
                    param['color_tint'] = param['color_tint_reset']
                    param['color_Y'] = Y

                    #self.img = core.restore_saturated_colors(self.img)

                    # ヒストグラムマッチング
                    thumb_base64 = exif_data.get('ThumbnailImage')
                    thumb_base64 = None
                    if thumb_base64 is not None:
                        thumb = np.frombuffer(base64.b64decode(thumb_base64[7:]), dtype=np.uint8)
                        thumb = cv2.imdecode(thumb, 1)
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                        thumb = thumb.astype(np.float32) / 255.0
                        thumb = core.apply_gamma(thumb, 2.222)
                        temp = self.img * wb

                        #temp = skimage.exposure.match_histograms(temp, thumb, channel_axis=-1)

                        temp = temp / wb
                        self.img = temp

                    """
                    # 平均輝度を計算
                    mean_brightness = np.mean(self.img)
                    target_brightness = 0.5  # 目標の輝度 (0.0 - 1.0 の範囲)
                    #param['after_exposure'] = (target_brightness / mean_brightness)
                    self.img = self.img * (target_brightness / mean_brightness)
                    # ハイライトの圧縮
                    self.img = self.img / (1 + self.img)

                    # 最大値を2.0にスケーリング
                    self.img = self.img * (2.0 / self.img.max())

                    # クリッピング
                    #self.img = np.clip(self.img, 0, 2.0)

                    # ホワイトレベルをコントラスト調整にへ渡す
                    param['raw_white_level'] = raw.white_level
                    """

                    if np.any(self.img < 0.0) or np.any(self.img > 2.0):
                        print("outofrange", self.img)
                        self.img = np.clip(self.img, 0, 2)

            except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
                logging.warning("file is not supported " + file_path)
                return False

        elif file_path.lower().endswith(viewer_widget.supported_formats_rgb):

            # RGB画像で読み込んでみる
            self.src = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.src is not None:
                if type(self.src[0][0][0]) is np.uint8:
                    self.src = self.src.astype(np.uint16)
                    self.src = self.src*256
                self.img = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB).astype(np.float32)/65535.0
                self.img = core.apply_gamma(self.img, 2.222)
            else:
                logging.warning("file is not supported " + file_path)
                return False
            
        else:
            logging.warning("file is not supported " + file_path)
            return False
            
        param['img_size'] = [self.img.shape[1], self.img.shape[0]]

        #self.tmb = cv2.resize(self.img, dsize=core.calc_resize_image((self.img.shape[1], self.img.shape[0]), 256))

        self.file_path = file_path

        logging.info("load file " + file_path)

        return True

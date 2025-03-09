
import cv2
import numpy as np
import core
import rawpy
import math
import base64
import logging
import os
import threading
from wand.image import Image as WandImage

#from dcp_profile import DCPReader, DCPProcessor
import config
import util
import viewer_widget
import color


class ImageSet:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.file_path = None
        self.img = None     # 加工元画像 float32
        self.param = None
        self.loaded_json = None

    def _set_temperature(self, param, temp, tint, Y):
        param['color_temperature_reset'] = temp
        param['color_temperature'] = param['color_temperature_reset']
        param['color_tint_reset'] = tint
        param['color_tint'] = param['color_tint_reset']
        param['color_Y'] = Y

    def _black(self, in_img, black_level):
        in_img[in_img < black_level] = black_level
        out_img = in_img - black_level
        return out_img
    

    def _recover_saturated_colors(self, image):
        """
        画像内の飽和したRGB値を他のチャンネルの比率を使って復元する
        
        Parameters:
        image: numpy.ndarray
            Shape が (height, width, 3) の RGB画像データ。値は0-1の範囲を想定。
            
        Returns:
        numpy.ndarray:
            復元された画像データ
        """
        # 入力画像のコピーを作成
        recovered = image.copy()
        
        # 飽和しているピクセルを見つける（どれかのチャンネルが1.0以上）
        saturated_pixels = np.any(image >= 1.0, axis=2)
        
        # 飽和したピクセルに対して処理を行う
        for y, x in np.argwhere(saturated_pixels):
            pixel = image[y, x]
            
            # 飽和していないチャンネルのインデックスを取得
            unsaturated_idx = np.where(pixel < 1.0)[0]
            
            if len(unsaturated_idx) >= 2:
                # 飽和していない2つのチャンネル間の比率を計算
                ratio = pixel[unsaturated_idx[0]] / pixel[unsaturated_idx[1]]
                
                # 飽和したチャンネルの値を推定
                saturated_idx = np.where(pixel >= 1.0)[0]
                for idx in saturated_idx:
                    # 飽和していないチャンネルの最大値を基準に、比率を使って推定
                    max_unsaturated = max(pixel[unsaturated_idx])
                    estimated_value = max_unsaturated * (1 + ratio)
                    recovered[y, x, idx] = estimated_value
                    
        return recovered
    

    def _load_raw_preview(self, file_path, exif_data, param):
        try:
            # RAWで読み込んでみる
            raw = rawpy.imread(file_path)

            # プレビューを読む
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                # JPEGフォーマットの場合
                # バイナリデータをNumPy配列に変換
                img_array = np.frombuffer(thumb.data, dtype=np.uint8)
                # OpenCVで画像としてデコード
                img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_array= cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                # BITMAPフォーマットの場合
                img_array = thumb.data                    
            else:
                raise ValueError(f"Unsupported thumbnail format: {thumb.format}")

            # クロップとexifデータの回転
            rad, flip = util.split_orientation(util.str_to_orientation(exif_data.get("Orientation", "")))
            top, left, width, height = core.get_exif_image_size(exif_data)
            if rad < 0.0:
                top, left = left, top
                width, height = height, width
                core.set_exif_image_size(exif_data, top, left, width, height)
            img_array = img_array[top:top+height, left:left+width]
            if "Orientation" in exif_data:
                del exif_data["Orientation"]

            # RGB画像初期設定
            img_array = img_array.astype(np.float32)/255.0
            img_array = color.rgb_to_xyz(img_array, "sRGB", True)
            temp, tint, Y, = core.invert_RGB2TempTint((1.0, 1.0, 1.0), 5000.0)
            self._set_temperature(param, temp, tint, Y)

            # RAW画像のサイズに合わせてリサイズ
            _, _, width, height = core.get_exif_image_size(exif_data)
            img_array = cv2.resize(img_array, (width, height))

            # イメージサイズを設定し、正方形にする
            img_array = core.adjust_shape(img_array, param)

            self.img = img_array

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
            return False
        
        return raw
                             
    def _load_raw(self, raw, file_path, exif_data, param):
        try:
            img_array = raw.postprocess(output_color=rawpy.ColorSpace.Adobe,
                                        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                        output_bps=16,
                                        no_auto_scale=False,
                                        use_camera_wb=True,
                                        user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 1.0),
                                        four_color_rgb=True,
                                        #user_black=0,
                                        no_auto_bright=True,
                                        highlight_mode=5,
                                        auto_bright_thr=0.0005)
            """
            # ブラックレベル補正
            raw_image = self.__black(raw.raw_image_visible, raw.black_level_per_channel[0])

            # float32へ
            raw_image = raw_image.astype(np.float32) / ((1<<exif_data.get('BitsPerSample', 14))-1)
            """

            # float32へ
            img_array = img_array.astype(np.float32)/65535.0
            
            # 色空間変換
            img_array= color.rgb_to_xyz(img_array, "Adobe RGB")
            #img_array = color.d50_to_d65(img_array)

            # 飽和ピクセル復元
            wb = raw.camera_whitebalance
            wb = np.array([wb[0], wb[1], wb[2]]).astype(np.float32)/1024.0
            _, Tv = exif_data.get('ShutterSpeedValue', "1/100").split('/')
            Tv = float(_) / float(Tv)
            ISO = exif_data.get('ISO', 100)                
            #img_array = self._recover_saturated_colors(img_array)
            #img_array = core.recover_saturated_pixels(img_array, Tv, ISO, wb)
            
            # プロファイルを適用
            #dcp_path = os.getcwd() + "/dcp/Fujifilm X-Pro3 Adobe Standard velvia.dcp"
            #reader = DCPReader(dcp_path)
            #profile = reader.read()
            #processor = DCPProcessor(profile)
            #img_array = processor.process(img_array, illuminant='1', use_look_table=True).astype(np.float32)
            
            # ホワイトバランス定義
            #wb = raw.camera_whitebalance
            #wb = np.array([wb[0], wb[1], wb[2]]).astype(np.float32)/1024.0
            wb[1] = np.sqrt(wb[1])
            img_array /= wb
            temp, tint, Y, = core.invert_RGB2TempTint(wb, 5000.0)
            self._set_temperature(param, temp, tint, Y)

            # 明るさ補正
            if config.get_config('raw_auto_exposure') == True:
                source_ev, _ = core.calculate_ev_from_image(core.normalize_image(img_array))
                Av = exif_data.get('ApertureValue', 1.0)
                #_, Tv = exif_data.get('ShutterSpeedValue', "1/100").split('/')
                #Tv = float(_) / float(Tv)
                Ev = math.log2((Av**2)/Tv)
                #Sv = math.log2(exif_data.get('ISO', 100)/100.0)
                Sv = math.log2(ISO/100.0)
                Ev = Ev + Sv
                img_array = img_array * core.calc_exposure(img_array, core.calculate_correction_value(source_ev, Ev))

            # イメージサイズを設定し、正方形にする
            img_array = core.adjust_shape(img_array, param)
            
            self.img = img_array

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
            return False
        
        finally:
            raw.close()

        
        return True

    def _load_rgb(self, file_path, exif_data, param):
        # RGB画像で読み込んでみる
        with WandImage(filename=file_path) as img:
            img_array = np.array(img)
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.uint16)
                img_array = img_array*256
            img_array = img_array.astype(np.float32)/65535.0
            img_array = color.rgb_to_xyz(img_array, "sRGB", True)

            # 画像からホワイトバランスパラメータ取得
            temp, tint, Y, = core.invert_RGB2TempTint((1.0, 1.0, 1.0), 5000.0)
            self._set_temperature(param, temp, tint, Y)
            
            top, left, width, height = core.get_exif_image_size(exif_data)
            if img_array.shape[1] != width or img_array.shape[0] != height:
                logging.error("ImageSize is not ndarray.shape")
            img_array = core.adjust_shape(img_array, param)
            
        #else:
        #    logging.warning("file is not supported " + file_path)
        #    return False
        
        self.img = img_array
        
        return True
    
    def _load_thumb(self, exif_data, param):
        thumb_base64 = exif_data.get('ThumbnailImage')
        if thumb_base64 is not None:
            thumb = np.frombuffer(base64.b64decode(thumb_base64[7:]), dtype=np.uint8)
            thumb = cv2.imdecode(thumb, 1)
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumb = thumb.astype(np.float32)/255
            thumb = color.rgb_to_xyz(thumb, "sRGB", True)
            temp, tint, Y, = core.invert_RGB2TempTint((1.0, 1.0, 1.0), 5000.0)
            self._set_temperature(param, temp, tint, Y)

            top, left, width, height = core.get_exif_image_size(exif_data)
            img_array = cv2.resize(img_array, dsize=(width, height))
            param['img_size'] = [width, height]
            param['original_img_size'] = [width, height]

            self.img = thumb

    def load(self, file_path, exif_data, param, raw=None):
        self.file_path = file_path
        self.param = param

        if file_path.lower().endswith(viewer_widget.supported_formats_raw):
            #self._load_thumb(exif_data, param)
            if raw is None:
                return self._load_raw_preview(file_path, exif_data, param)            
            else:
                return self._load_raw(raw, file_path, exif_data, param)

            #thread = threading.Thread(target=self._load_raw, args=[file_path, exif_data, param], daemon=True)
            #thread.start()

        elif file_path.lower().endswith(viewer_widget.supported_formats_rgb):
            #self._load_thumb(exif_data, param)
            return self._load_rgb(file_path, exif_data, param)
            #thread = threading.Thread(target=self._load_rgb, args=[file_path, exif_data, param], daemon=True)
            #thread.start()
            
        logging.warning("file is not supported " + file_path)
        return False

import cv2
import numpy as np
import core
import rawpy
#import maxim_util

class ImageSet:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.src = None     # 元画像 uint16
        self.img = None     # 加工元画像 float32
        self.tmb = None     # 縮小画像 float32
        self.prv = None     # 加工画像 float32
    
    def load(self, file_path, exif_data, param):        
        try:
            # RAWで読み込んでみる
            raw = rawpy.imread(file_path)
            black_level = raw.black_level_per_channel
            white_level = raw.camera_white_level_per_channel

            self.src = raw.postprocess( output_color=rawpy.ColorSpace.sRGB,
                                        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                        output_bps=16,
                                        no_auto_scale=False,
                                        use_camera_wb=False,
                                        user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 0.0),
                                        four_color_rgb=True,
                                        #user_black=0,
                                        no_auto_bright=False,
                                        auto_bright_thr=0.0005)

            #self.img = raw.tone_curve[self.src]
            #self.img = self.img.astype(np.float32)/65535.0

            # クロップ
            top, left = exif_data.get("RawImageCropTopLeft", "0 0").split()
            top, left = int(top), int(left)
            width, height = exif_data.get("RawImageCroppedSize", "0x0").split('x')
            width, height = int(width), int(height)
            self.img = self.src[top:top+height, left:left+width]

            # float32へ
            self.img = self.img.astype(np.float32)/65535.0

            # ホワイトバランス定義
            wb = raw.camera_whitebalance
            wb = np.array([wb[0], wb[1], wb[2]])/1024.0
            wb[1] = np.sqrt(wb[1])
            self.img[:,:,1] *= wb[1]
            #self.img = core.adjust_exposure(self.img, 1.6)
            temp, tint, Y, = core.convert_RGB2TempTint(wb)
            #wb2 = core.convert_TempTint2RGB(temp, tint, Y)
            param['color_temperature'] = float(temp)
            param['color_temperature_reset'] = float(temp)
            param['color_tint'] = float(-tint)
            param['color_tint_reset'] = float(-tint)
            param['color_Y'] = float(Y)

            # 明るさ補正
            #bv = exif_data.get("BrightnessValue", "0")
            #self.img = self.img * core.adjust_exposure(self.img, -bv)
            self.img = self.img / (1-(raw.white_level/65535))

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            # RGB画像で読み込んでみる
            self.src = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.src is not None:
                if type(self.src[0][0][0]) is np.uint8:
                    self.src = self.src.astype(np.uint16)
                    self.src = self.src*256
                self.img = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB).astype(np.float32)/65535.0
                self.img = core.apply_gamma(self.img, 2.222)
            else:
                print("file is not supported")
                return False
            
        param['img_size'] = self.img.shape

        self.tmb = cv2.resize(self.img, dsize=core.calc_resize_image((self.img.shape[0], self.img.shape[1]), 256))

        #self.crop_image(0, 0, self.img.shape[0], self.img.shape[1], False)

        return True

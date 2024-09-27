import cv2
import numpy as np
import core
import rawpy
#import maxim_util

class ImageSet:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.src = None     # 元画像 BGR uint16
        self.tmb = None     # 縮小画像 float32
        self.tmb_msk = None # 縮小画像用マスク float32
        self.img = None     # 加工元画像 float32
        self.img_msk = None # 加工元画像用マスク float32
        self.img_afn = None # アフィン変換後画像 float32
        self.prv = None     # 加工画像 float32
        self.prv_msk = None # 加工画像用マスク float32
        self.prv_crop_info = None
    
    def load(self, file_path, param):        
        try:
            # RAWで読み込んでみる
            raw = rawpy.imread(file_path)
            black_level = raw.black_level_per_channel
            white_level = raw.camera_white_level_per_channel

            self.src = raw.postprocess( output_color=rawpy.ColorSpace.sRGB,
                                        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                        output_bps=16,
                                        no_auto_scale = False,
                                        use_camera_wb=False,
                                        user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 0.0),
                                        four_color_rgb=True,
                                        #user_black=0,
                                        no_auto_bright=False,
                                        auto_bright_thr=0.0005)

            #self.img = raw.tone_curve[self.src]
            #self.img = self.img.astype(np.float32)/65535.0

            self.img = self.src.astype(np.float32)/65535.0

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

            #self.img /= np.min(wb)

            #self.img = maxim_util.predict(self.img)
    
            #processor = dcp.DCPProcessor(os.getcwd() + "/escargot/Fujifilm X-T5 Adobe Standard.dcp")
            #self.img = processor.process_raw(self.img)

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

        self.prv = self.img

        return True

    def crop_image(self, x, y, w, h, is_zoomed):
        img2, self.prv_crop_info = core.crop_image(self.img, w, h, x, y, is_zoomed)
        self.prv = img2

        return img2
    
    def crop_image2(self, offset):
        img2, self.prv_crop_info = core.crop_image2(self.img, self.prv_crop_info, offset)
        self.prv = img2

        return img2

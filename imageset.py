import cv2
import numpy as np
import core
import rawpy
import os
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
    
    def load(self, filename, maskname):        
        try:
            # RAWで読み込んでみる
            raw = rawpy.imread(filename)
            self.src = raw.postprocess(output_color=rawpy.ColorSpace.raw, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, output_bps=16, use_camera_wb=True, gamma=(1.0, 1.0), four_color_rgb=True, no_auto_bright=True)

            #self.img = raw.tone_curve[self.src]
            #self.img = self.img.astype(np.float32)/65535.0

            self.img = self.src.astype(np.float32)/65535.0

            #self.img = maxim_util.predict(self.img)
    
            #processor = dcp.DCPProcessor(os.getcwd() + "/escargot/Fujifilm X-T5 Adobe Standard.dcp")
            #self.img = processor.process_raw(self.img)

            #lut3d = colour.read_LUT(os.getcwd() + "/escargot/XT5_FLog2_FGamut_to_FLog2_BT.709_33grid_V.1.00.cube", 'Resolve Cube')
            #self.img = lut3d.apply(self.img)

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            # RGB画像で読み込んでみる
            self.src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if self.src is not None:
                if type(self.src[0][0][0]) is np.uint8:
                    self.src = self.src.astype(np.uint16)
                    self.src = self.src*256
                self.img = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB).astype(np.float32)/65535.0
                self.img = core.apply_gamma(self.img, 2.222)
            else:
                print("file is not supported")
                return False

        self.tmb = cv2.resize(self.img, dsize=(1024, 1024)).astype(np.float32)

        #self.img = core.apply_gamma(self.img, 2.222)
        #self.tmb = core.apply_gamma(self.tmb, 2.222)

        self.prv = self.img

        self.img_msk = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)
        if self.img_msk is not None:
            self.tmb_msk = cv2.resize(self.img_msk, dsize=(1024,1024))
            self.tmb_msk = self.tmb_msk.astype(np.float32)/255.0

            self.img_msk = self.img_msk.astype(np.float32)/255.0

            self.prv_msk = self.img_msk
        
        return True

    def make_clip(self, scale, x, y, w, h):
        img2, msk2 = core.make_clip(self.img, self.img_msk, scale, x, y, w, h)
        self.prv = img2
        self.prv_msk = msk2

        return img2, msk2

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, BinaryIO
import struct
import os
from dataclasses import dataclass
from enum import IntEnum
import io
import cv2

class TiffTag(IntEnum):
    # DCP関連のTIFFタグ
    CALIBRATION_ILLUMINANT1 = 50778
    CALIBRATION_ILLUMINANT2 = 50779
    COLOR_MATRIX1 = 50721
    COLOR_MATRIX2 = 50722
    FORWARD_MATRIX1 = 50964
    FORWARD_MATRIX2 = 50965
    CAMERA_CALIBRATION1 = 50723
    CAMERA_CALIBRATION2 = 50724
    PROFILE_LOOK_TABLE = 50982
    PROFILE_LOOK_TABLE_DIMS = 50983
    PROFILE_HUE_SAT_MAP_DIMS = 50937
    PROFILE_HUE_SAT_MAP1 = 50938
    PROFILE_HUE_SAT_MAP2 = 50939
    PROFILE_TONE_CURVE = 50940
    
    # 基本的なTIFFタグ
    IMAGE_WIDTH = 256
    IMAGE_LENGTH = 257
    BITS_PER_SAMPLE = 258
    COMPRESSION = 259
    PHOTOMETRIC_INTERPRETATION = 262
    STRIP_OFFSETS = 273
    SAMPLES_PER_PIXEL = 277
    ROWS_PER_STRIP = 278
    STRIP_BYTE_COUNTS = 279
    PLANAR_CONFIGURATION = 284

class TiffDataType(IntEnum):
    BYTE = 1
    ASCII = 2
    SHORT = 3
    LONG = 4
    RATIONAL = 5
    SBYTE = 6
    UNDEFINED = 7
    SSHORT = 8
    SLONG = 9
    SRATIONAL = 10
    FLOAT = 11
    DOUBLE = 12

@dataclass
class TiffIFDEntry:
    tag: int
    type: int
    count: int
    value_offset: int
    
@dataclass
class DCPIlluminant:
    temperature: float
    xy: Tuple[float, float]

class TiffReader:
    def __init__(self, file: BinaryIO):
        self.file = file
        self.is_little_endian = True
        self.ifd_offset = 0
        self._read_header()
    
    def _read_header(self):
        """TIFFヘッダーを読み込む"""
        byte_order = self.file.read(2)
        if byte_order == b'II':
            self.is_little_endian = True
        elif byte_order == b'MM':
            self.is_little_endian = False
        else:
            raise ValueError("Invalid TIFF byte order")
        
        magic = self._read_short()
        if magic != 0x4352:
            raise ValueError("Invalid TIFF magic number")
            
        self.ifd_offset = self._read_long()
    
    def _read_short(self) -> int:
        """16bit整数を読み込む"""
        data = self.file.read(2)
        if self.is_little_endian:
            return struct.unpack('<H', data)[0]
        return struct.unpack('>H', data)[0]
    
    def _read_long(self) -> int:
        """32bit整数を読み込む"""
        data = self.file.read(4)
        if self.is_little_endian:
            return struct.unpack('<L', data)[0]
        return struct.unpack('>L', data)[0]
    
    def _read_rational(self) -> float:
        """有理数を読み込む"""
        numerator = self._read_long()
        denominator = self._read_long()
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def _read_srational(self) -> float:
        """符号付き有理数を読み込む"""
        numerator = struct.unpack('<l' if self.is_little_endian else '>l',
                                self.file.read(4))[0]
        denominator = struct.unpack('<l' if self.is_little_endian else '>l',
                                  self.file.read(4))[0]
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def _read_float(self) -> float:
        """32bit浮動小数点を読み込む"""
        data = self.file.read(4)
        if self.is_little_endian:
            return struct.unpack('<f', data)[0]
        return struct.unpack('>f', data)[0]
    
    def _read_double(self) -> float:
        """64bit浮動小数点を読み込む"""
        data = self.file.read(8)
        if self.is_little_endian:
            return struct.unpack('<d', data)[0]
        return struct.unpack('>d', data)[0]

class DCPReader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tiff_reader = None
        self.profile = DCPProfile(
            color_matrices={},
            forward_matrices={},
            camera_calibrations={},
            illuminants={},
            forward_tone_curve=None,
            rgb_tables=None,
            hue_sat_map_dims=None,
            hue_sat_maps={},
            look_table=None,
            look_table_dims=None
        )
    
    def read(self):
        """DCPファイルを読み込む"""
        with open(self.filepath, 'rb') as f:
            self.tiff_reader = TiffReader(f)
            self._read_ifd()
        return self.profile
    
    def _read_ifd(self):
        """Image File Directoryを読み込む"""
        self.tiff_reader.file.seek(self.tiff_reader.ifd_offset)
        entry_count = self.tiff_reader._read_short()
        
        for _ in range(entry_count):
            entry = self._read_ifd_entry()
            self._process_ifd_entry(entry)
    
    def _read_ifd_entry(self) -> TiffIFDEntry:
        """IFDエントリーを読み込む"""
        tag = self.tiff_reader._read_short()
        type_ = self.tiff_reader._read_short()
        count = self.tiff_reader._read_long()
        value_offset = self.tiff_reader._read_long()
        
        return TiffIFDEntry(tag, type_, count, value_offset)
    
    def _process_ifd_entry(self, entry: TiffIFDEntry):
        """IFDエントリーを処理する"""
        if entry.tag == TiffTag.COLOR_MATRIX1:
            self.profile.color_matrices['1'] = self._read_matrix(entry)
        elif entry.tag == TiffTag.COLOR_MATRIX2:
            self.profile.color_matrices['2'] = self._read_matrix(entry)
        elif entry.tag == TiffTag.FORWARD_MATRIX1:
            self.profile.forward_matrices['1'] = self._read_matrix(entry)
        elif entry.tag == TiffTag.FORWARD_MATRIX2:
            self.profile.forward_matrices['2'] = self._read_matrix(entry)
        elif entry.tag == TiffTag.CAMERA_CALIBRATION1:
            self.profile.camera_calibrations['1'] = self._read_matrix(entry)
        elif entry.tag == TiffTag.CAMERA_CALIBRATION2:
            self.profile.camera_calibrations['2'] = self._read_matrix(entry)
        elif entry.tag == TiffTag.CALIBRATION_ILLUMINANT1:
            self.profile.illuminants['1'] = self._read_illuminant(entry)
        elif entry.tag == TiffTag.CALIBRATION_ILLUMINANT2:
            self.profile.illuminants['2'] = self._read_illuminant(entry)
        elif entry.tag == TiffTag.PROFILE_TONE_CURVE:
            self.profile.forward_tone_curve = self._read_tone_curve(entry)
        elif entry.tag == TiffTag.PROFILE_HUE_SAT_MAP_DIMS:
            self.profile.hue_sat_map_dims = self._read_hue_sat_map_dims(entry)
        elif entry.tag == TiffTag.PROFILE_HUE_SAT_MAP1:
            self.profile.hue_sat_maps['1'] = self._read_hue_sat_map(entry)
        elif entry.tag == TiffTag.PROFILE_HUE_SAT_MAP2:
            self.profile.hue_sat_maps['2'] = self._read_hue_sat_map(entry)
        elif entry.tag == TiffTag.PROFILE_LOOK_TABLE:
            self.profile.look_table = self._read_look_table(entry)
        elif entry.tag == TiffTag.PROFILE_LOOK_TABLE_DIMS:
            self.profile.look_table_dims = self._read_look_table_dims(entry)

    def _read_illuminant(self, entry: TiffIFDEntry):
        return self.tiff_reader._read_short()
    
    def _read_matrix(self, entry: TiffIFDEntry) -> np.ndarray:
        """行列データを読み込む"""
        if entry.type not in [TiffDataType.SRATIONAL, TiffDataType.RATIONAL]:
            raise ValueError(f"Invalid matrix data type: {entry.type}")
            
        current_pos = self.tiff_reader.file.tell()
        self.tiff_reader.file.seek(entry.value_offset)
        
        matrix = []
        for _ in range(entry.count):
            if entry.type == TiffDataType.SRATIONAL:
                value = self.tiff_reader._read_srational()
            else:
                value = self.tiff_reader._read_rational()
            matrix.append(value)
            
        self.tiff_reader.file.seek(current_pos)
        
        # 3x3 または 3x4 行列に変換
        if len(matrix) == 9:
            return np.array(matrix).reshape(3, 3)
        elif len(matrix) == 12:
            return np.array(matrix).reshape(3, 4)
        else:
            raise ValueError(f"Invalid matrix size: {len(matrix)}")

    def _read_tone_curve(self, entry: TiffIFDEntry) -> List[Tuple[float, float]]:
        """トーンカーブデータを読み込む"""
        current_pos = self.tiff_reader.file.tell()
        self.tiff_reader.file.seek(entry.value_offset)
        
        points = []
        point_count = entry.count // 2
        for _ in range(point_count):
            x = self.tiff_reader._read_float()
            y = self.tiff_reader._read_float()
            points.append((x, y))
            
        self.tiff_reader.file.seek(current_pos)
        return points

    def _read_hue_sat_map_dims(self, entry: TiffIFDEntry) -> Tuple[int, int, int]:
        """色相・彩度マップの次元を読み込む"""
        current_pos = self.tiff_reader.file.tell()
        self.tiff_reader.file.seek(entry.value_offset)
        
        hue_divisions = self.tiff_reader._read_long()
        sat_divisions = self.tiff_reader._read_long()
        val_divisions = self.tiff_reader._read_long()
        
        self.tiff_reader.file.seek(current_pos)
        return (hue_divisions, sat_divisions, val_divisions)

    def _read_hue_sat_map(self, entry: TiffIFDEntry) -> np.ndarray:
        """色相・彩度マップを読み込む"""
        if self.profile.hue_sat_map_dims is None:
            raise ValueError("Hue/Sat map dimensions must be read first")
            
        current_pos = self.tiff_reader.file.tell()
        self.tiff_reader.file.seek(entry.value_offset)
        
        h, s, v = self.profile.hue_sat_map_dims
        map_data = np.zeros((h, s, v, 3), dtype=np.float32)
        
        for i in range(h):
            for j in range(s):
                for k in range(v):
                    for c in range(3):
                        map_data[i,j,k,c] = self.tiff_reader._read_float()
        
        self.tiff_reader.file.seek(current_pos)
        return map_data

    def _read_look_table_dims(self, entry: TiffIFDEntry) -> Tuple[int, int, int]:
        """ルックテーブルの次元を読み込む"""
        current_pos = self.tiff_reader.file.tell()
        self.tiff_reader.file.seek(entry.value_offset)
        
        dim1 = self.tiff_reader._read_long()
        dim2 = self.tiff_reader._read_long()
        dim3 = self.tiff_reader._read_long()
        
        self.tiff_reader.file.seek(current_pos)
        return (dim1, dim2, dim3)

    def _read_look_table(self, entry: TiffIFDEntry) -> np.ndarray:
        """ルックテーブルを読み込む"""
        if self.profile.look_table_dims is None:
            raise ValueError("Look table dimensions must be read first")
            
        current_pos = self.tiff_reader.file.tell()
        self.tiff_reader.file.seek(entry.value_offset)
        
        d1, d2, d3 = self.profile.look_table_dims
        table_data = np.zeros((d1, d2, d3, 3), dtype=np.float32)
        
        for i in range(d1):
            for j in range(d2):
                for k in range(d3):
                    for c in range(3):
                        table_data[i,j,k,c] = self.tiff_reader._read_float()
        
        self.tiff_reader.file.seek(current_pos)
        return table_data
    

@dataclass
class DCPProfile:
    """DCPプロファイルデータ構造"""
    color_matrices: Dict[str, np.ndarray]
    forward_matrices: Dict[str, np.ndarray]
    camera_calibrations: Dict[str, np.ndarray]
    illuminants: Dict[str, DCPIlluminant]
    forward_tone_curve: Optional[List[Tuple[float, float]]]
    rgb_tables: Optional[List[List[float]]]
    hue_sat_map_dims: Optional[Tuple[int, int, int]]
    hue_sat_maps: Dict[str, np.ndarray]
    look_table: Optional[np.ndarray]
    look_table_dims: Optional[Tuple[int, int, int]]

class DCPProcessor:
    """DCPプロファイル適用プロセッサー"""
    
    def __init__(self, profile: DCPProfile):
        self.profile = profile
        
    def process(self, 
                image: np.ndarray, 
                illuminant: str = '1',
                use_look_table: bool = True) -> np.ndarray:
        """
        画像にDCPプロファイルを適用
        
        Parameters:
            image: 入力画像 (float32, 0-1範囲, shape=(H,W,3))
            illuminant: 使用する光源プロファイル ('1' or '2')
            use_look_table: ルックテーブルを適用するかどうか
            
        Returns:
            処理済み画像
        """
        if image.dtype != np.float32 or not (0 <= image.min() and image.max() <= 1):
            raise ValueError("Image must be float32 in 0-1 range")
        
        result = image.copy()
        
        """
        # 1. カメラキャリブレーション行列の適用
        if illuminant in self.profile.camera_calibrations:
            result = self._apply_matrix(result, 
                                      self.profile.camera_calibrations[illuminant])
        
        # 2. カラーマトリクスの適用
        if illuminant in self.profile.color_matrices:
            result = self._apply_matrix(result, 
                                      self.profile.color_matrices[illuminant])
        
        # 3. フォワードマトリクスの適用
        if illuminant in self.profile.forward_matrices:
            result = self._apply_matrix(result, 
                                      self.profile.forward_matrices[illuminant])
        """
        # 4. 色相・彩度マップの適用
        if illuminant in self.profile.hue_sat_maps:
            result = self._apply_hue_sat_map(result, 
                                           self.profile.hue_sat_maps[illuminant])
        
        # 5. トーンカーブの適用
        if self.profile.forward_tone_curve is not None:
            result = self._apply_tone_curve(result)
        
        # 6. ルックテーブルの適用
        if use_look_table and self.profile.look_table is not None:
            result = self._apply_look_table(result)
        
        return result
    
    def _apply_matrix(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """行列変換を適用"""
        result = np.zeros_like(image)
        
        if matrix.shape[1] == 4:  # 3x4 行列
            result[..., 0] = (matrix[0,0] * image[..., 0] + 
                            matrix[0,1] * image[..., 1] + 
                            matrix[0,2] * image[..., 2] + 
                            matrix[0,3])
            result[..., 1] = (matrix[1,0] * image[..., 0] + 
                            matrix[1,1] * image[..., 1] + 
                            matrix[1,2] * image[..., 2] + 
                            matrix[1,3])
            result[..., 2] = (matrix[2,0] * image[..., 0] + 
                            matrix[2,1] * image[..., 1] + 
                            matrix[2,2] * image[..., 2] + 
                            matrix[2,3])
        else:  # 3x3 行列
            result = np.dot(image, matrix.T)
            
        return result
    
    def _apply_hue_sat_map(self, 
                          image: np.ndarray, 
                          hue_sat_map: np.ndarray) -> np.ndarray:
        """色相・彩度マップを適用"""
        # RGB → HSV 変換
        hsv = self._rgb_to_hsv(image)
        
        h, s, v = self.profile.hue_sat_map_dims
        result = np.zeros_like(image)
        
        # 各ピクセルに対して補間処理
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                h_val = hsv[y,x,0] * (h - 1)
                s_val = hsv[y,x,1] * (s - 1)
                v_val = hsv[y,x,2] * (v - 1)
                
                # 補間のためのインデックスを計算
                h_idx = int(h_val)
                s_idx = int(s_val)
                v_idx = int(v_val)
                
                h_frac = h_val - h_idx
                s_frac = s_val - s_idx
                v_frac = v_val - v_idx
                
                # トリリニア補間
                result[y,x] = self._trilinear_interpolate(
                    hue_sat_map, 
                    h_idx, s_idx, v_idx,
                    h_frac, s_frac, v_frac
                )
        
        return result
    
    def _trilinear_interpolate(self,
                             table: np.ndarray,
                             h: int, s: int, v: int,
                             h_frac: float, s_frac: float, v_frac: float) -> np.ndarray:
        """3次元テーブルのトリリニア補間"""
        h1 = min(h + 1, table.shape[0] - 1)
        s1 = min(s + 1, table.shape[1] - 1)
        v1 = min(v + 1, table.shape[2] - 1)
        
        # 8つの頂点での値を取得
        c000 = table[h, s, v]
        c001 = table[h, s, v1]
        c010 = table[h, s1, v]
        c011 = table[h, s1, v1]
        c100 = table[h1, s, v]
        c101 = table[h1, s, v1]
        c110 = table[h1, s1, v]
        c111 = table[h1, s1, v1]
        
        # トリリニア補間を実行
        c00 = c000 * (1 - h_frac) + c100 * h_frac
        c01 = c001 * (1 - h_frac) + c101 * h_frac
        c10 = c010 * (1 - h_frac) + c110 * h_frac
        c11 = c011 * (1 - h_frac) + c111 * h_frac
        
        c0 = c00 * (1 - s_frac) + c10 * s_frac
        c1 = c01 * (1 - s_frac) + c11 * s_frac
        
        return c0 * (1 - v_frac) + c1 * v_frac
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """RGB → HSV変換"""
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        deltac = maxc - minc
        s = np.where(maxc != 0, deltac / maxc, 0)
        
        h = np.zeros_like(deltac)
        h_mask = deltac != 0
        
        rc = np.where(h_mask, (maxc - r) / deltac, 0)
        gc = np.where(h_mask, (maxc - g) / deltac, 0)
        bc = np.where(h_mask, (maxc - b) / deltac, 0)
        
        h = np.where(r == maxc, bc - gc, h)
        h = np.where(g == maxc, 2.0 + rc - bc, h)
        h = np.where(b == maxc, 4.0 + gc - rc, h)
        
        h = (h / 6.0) % 1.0
        
        return np.dstack((h, s, v))

    def _apply_tone_curve(self, image: np.ndarray) -> np.ndarray:
        """トーンカーブを適用"""
        # トーンカーブの制御点をnumpy配列に変換
        points = np.array(self.profile.forward_tone_curve)
        x = points[:, 0]
        y = points[:, 1]
        
        # チャンネルごとに処理
        result = np.zeros_like(image)
        for c in range(3):
            channel = image[..., c].flatten()
            # numpy.interp で補間
            result[..., c] = np.interp(channel, x, y).reshape(image.shape[:2])
        
        return result
    
    def _apply_look_table(self, image: np.ndarray) -> np.ndarray:
        """ルックテーブルを適用"""          
        d1, d2, d3 = self.profile.look_table_dims
        result = np.zeros_like(image)
        
        # 各ピクセルに対して3次元補間
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # 入力値をテーブルのインデックスに変換
                r_val = image[y,x,0] * (d1 - 1)
                g_val = image[y,x,1] * (d2 - 1)
                b_val = image[y,x,2] * (d3 - 1)
                
                # インデックスと補間係数を計算
                r_idx = int(r_val)
                g_idx = int(g_val)
                b_idx = int(b_val)
                
                r_frac = r_val - r_idx
                g_frac = g_val - g_idx
                b_frac = b_val - b_idx
                
                # トリリニア補間で出力値を計算
                result[y,x] = self._trilinear_interpolate(
                    self.profile.look_table,
                    r_idx, g_idx, b_idx,
                    r_frac, g_frac, b_frac
                )
        
        return result


if __name__ == '__main__':
    # DCPファイルを読み込む
    dcp_path = os.getcwd() + "/dcp/Fujifilm X-T5 Adobe Standard.dcp"
    reader = DCPReader(dcp_path)
    profile = reader.read()

    # 画像を読み込む（例：OpenCVやPillowを使用）
    # 画像は0-1範囲のfloat32に正規化する必要がある
    image = cv2.imread(os.getcwd() + "/picture/DSCF0002.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    image = np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )

    # プロファイルを適用
    processor = DCPProcessor(profile)
    result = processor.process(image, illuminant='1', use_look_table=True)

    # sRGBガンマ補正を適用
    result = np.where(
        result <= 0.0031308,
        result * 12.92,
        1.055 * (result ** (1/2.4)) - 0.055
    )
    result = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imshow("predict", result)
    cv2.waitKey(0)

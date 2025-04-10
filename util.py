
import math
import numpy as np

def to_texture(pos, widget):
    # ウィンドウ座標からローカルイメージ座標に変換
    local_x, local_y = widget.to_widget(*pos)
    local_x = local_x - widget.pos[0]
    local_y = local_y - widget.pos[1]

    # ローカル座標をテクスチャ座標に変換
    tex_y = widget.height-local_y
    tex_x = local_x - (widget.width - widget.texture_size[0])/2
    tex_y = tex_y - (widget.height - widget.texture_size[1])/2

    return (tex_x, tex_y)

def str_to_orientation(str):
    if str == "Horizontal (normal)":
        orientation = 1
    if str == "Mirror horizontal":
        orientation = 2
    if str == "Rotate 180":
        orientation = 3
    if str  == "Mirror vertical":
        orientation = 4
    if str == "Mirror horizontal and rotate 90 CW":
        orientation = 5
    if str == "Rotate 90 CW":
        orientation = 6
    if str == "Mirror horizontal and rotate 270 CW":
        orientation = 7
    if str == "Rotate 270 CW":
        orientation = 8
    else:
        orientation = 1

    return orientation

def split_orientation(orientation):
    rad, flip = 0, 0
    if orientation == 1:
        rad, flip = 0, 0
        print("Horizontal (normal)")
    elif orientation == 2:
        rad, flip = 0, 1
        print("Mirror horizontal")
    elif orientation == 3:
        rad, flip = math.radians(180), 0
        print("Rotate 180")
    elif orientation == 4:
        rad, flip = 0, 2
        print("Mirror vertical")
    elif orientation == 5:
        rad, flip = math.radians(-90), 1
        print("Mirror horizontal and rotate 90 CW")
    elif orientation == 6:
        rad, flip = math.radians(-90), 0
        print("Rotate 90 CW")
    elif orientation == 7:
        rad, flip = math.radians(-270), 1
        print("Mirror horizontal and rotate 270 CW")
    elif orientation == 8:
        rad, flip = math.radians(-270), 0
        print("Rotate 270 CW")

    return rad, flip

def make_orientation(rotation, flip):
    """
    回転角と反転情報からEXIFオリエンテーションタグの値を生成する関数

    Args:
        rotation (int): 回転角（0, 90, 180, 270のいずれか）
        flip_horizontal (bool, optional): 水平方向の反転
        flip_vertical (bool, optional): 垂直方向の反転

    Returns:
        int: 対応するEXIFオリエンテーションタグの値
    """

    # 回転と反転の組み合わせをマッピング
    orientation_reverse_map = {
        (0, False, False): 1,
        (0, True, False): 2,
        (180, False, False): 3,
        (0, True, True): 3,
        (0, False, True): 4,
        (270, True, False): 5,
        (270, False, False): 6,
        (270, False, True): 7,
        (180, True, False): 4,
        (90, True, False): 7,
        (90, False, False): 8,
        (0, True, True): 3,
        (180, False, True): 8,
        (180, True, True): 5,
        (90, True, True): 8,
        (90, False, True): 7,
        (270, True, True): 6,
    }

    # 入力値のバリデーション
    if rotation not in [0, 90, 180, 270]:
        raise ValueError(f"無効な回転角: {rotation}")
    flip_horizontal = (flip & 1) == 1
    flip_vertical = (flip & 2) == 2

    orientation = orientation_reverse_map.get((rotation, flip_horizontal, flip_vertical))
    
    if orientation is None:
        raise ValueError(f"サポートされていない回転・反転の組み合わせ")

    return orientation

def print_nan_inf(img):
    result = np.isnan(img)
    count = result.sum()
    print("NaN=", count)
    result = np.isinf(img)
    count = result.sum()
    print("inf=", count)

def tone_map(x, threshold=1.0):
    return np.where(x > threshold, np.log1p(x - threshold) + threshold, x)

def soft_clip(x, threshold=1.0):
    return threshold * np.tanh(x / threshold)

def convert_to_float32(img):
    """
    画像のデータ型をfloat32に変換する関数

    Args:
        img (numpy.ndarray): 変換する画像データ

    Returns:
        numpy.ndarray: float32の画像データ
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
    elif img.dtype == np.uint16:
        img = img.astype(np.float32)/65535
    elif img.dtype == np.uint32:
        img = img.astype(np.float32)/4294967295
    elif img.dtype == np.uint64:
        img = img.astype(np.float32)/18446744073709551615
    elif img.dtype == np.uint128:
        img = img.astype(np.float32)/340282366920938463463374607431768211455
    elif img.dtype == np.int8:
        img = img.astype(np.float32)/127
    elif img.dtype == np.int16:
        img = img.astype(np.float32)/32767
    elif img.dtype == np.int32:
        img = img.astype(np.float32)/2147483647
    elif img.dtype == np.int64:
        img = img.astype(np.float32)/9223372036854775807
    elif img.dtype == np.int128:
        img = img.astype(np.float32)/170141183460469231731687303715884105727
    elif img.dtype == np.int256:
        img = img.astype(np.float32)/1152921504606846976
    elif img.dtype == np.float16:
        img = img.astype(np.float32)
    elif img.dtype == np.float32:
        pass
    elif img.dtype == np.float64:
        img = img.astype(np.float32)
    elif img.dtype == np.float128:
        img = img.astype(np.float32)
    else:
        raise ValueError(f"サポートされていないデータ型: {img.dtype}")

    return img

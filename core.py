import cv2
import numpy as np
import math
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import splprep, splev

import dehazing.dehaze
import sigmoid
import dehazing


# 画像の読み込み
def imgread(filename):
    # filename: ファイル名
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img


# マスクイメージの読み込み
def mskread(filename):
    # ファイル名
    msk = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    msk = msk.astype(np.float32)/255.0

    return msk

# RGBからグレイスケールへの変換
def cvtToGrayColor(rgb):
    # 変換元画像 RGB
    gry = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    return gry

# ガンマ補正
def apply_gamma(img, gamma):
    # img: 変換元画像 RGB
    # gamma: ガンマ値
    #apply_img = 65535.0*(img/65535.0)**gamma
    apply_img = img**gamma

    return apply_img

def convert_Kelvin2RGB(colour_temperature):
    """
    Converts from K to RGB, algorithm courtesy of 
    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    """
    #range check
    if colour_temperature < 1000: 
        colour_temperature = 1000
    elif colour_temperature > 40000:
        colour_temperature = 40000
    
    tmp_internal = colour_temperature / 100.0
    
    # red 
    if tmp_internal <= 66:
        red = 255
    else:
        tmp_red = 329.698727446 * math.pow(tmp_internal - 60, -0.1332047592)
        if tmp_red < 0:
            red = 0
        elif tmp_red > 255:
            red = 255
        else:
            red = tmp_red
    
    # green
    if tmp_internal <=66:
        tmp_green = 99.4708025861 * math.log(tmp_internal) - 161.1195681661
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    else:
        tmp_green = 288.1221695283 * math.pow(tmp_internal - 60, -0.0755148492)
        if tmp_green < 0:
            green = 0
        elif tmp_green > 255:
            green = 255
        else:
            green = tmp_green
    
    # blue
    if tmp_internal >=66:
        blue = 255
    elif tmp_internal <= 19:
        blue = 0
    else:
        tmp_blue = 138.5177312231 * math.log(tmp_internal - 10) - 305.0447927307
        if tmp_blue < 0:
            blue = 0
        elif tmp_blue > 255:
            blue = 255
        else:
            blue = tmp_blue
    
    return red/255.0, green/255.0, blue/255.0

def convert_RGB2Kelvin(red, green, blue):
    # Wide RGB D65 https://gist.github.com/popcorn245/30afa0f98eea1c2fd34d
    X = red * 0.649926 + green * 0.103455 + blue * 0.197109
    Y = red * 0.234327 + green * 0.743075 + blue * 0.022598
    Z = red * 0.000000 + green * 0.053077 + blue * 1.035763

    # CIEXYZ D65 https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
    # X = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    # Y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    # Z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 + 6253.80338 * math.e**(-n / 0.92159) +
           28.70599 * math.e**(-n / 0.20039) +
           0.00004 * math.e**(-n / 0.07125))
    n = (x - 0.3356) / (y - 0.1691) if CCT > 50000 else n
    CCT = 36284.48953 + 0.00228 * math.e**(-n / 0.07861) + (
        5.4535 * 10**-36) * math.e**(-n / 0.01543) if CCT > 50000 else CCT
    return CCT

# ローパスフィルタ
def lowpass_filter(img, r):
    lpf = cv2.GaussianBlur(img, ksize=(r, r), sigmaX=0.0)

    return lpf

# ハイパスフィルタ
def highpass_filter(img, r):
    """
    gry = cvtToGrayColor(img)
    fft_img = np.fft.fft2(gry)
    shift_fft_img = np.fft.fftshift(fft_img)

    h,w = shift_fft_img.shape
    cy = int(h/2)
    cx = int(w/2)
    filter = np.ones(shift_fft_img.shape, dtype=np.float32)
    #cv2.circle(filter,(cx,cy),r,0.0,thickness=-1)
    cv2.rectangle(filter, (cx-r, cy-r), (cx+r, cy+r), 0.5, thickness=-1, lineType=cv2.LINE_AA)
    filter = cv2.GaussianBlur(filter, ksize=(31, 31), sigmaX=0.0)
    shift_fft_img *= filter
    
    fds = np.fft.ifftshift(shift_fft_img)
    ds = np.fft.ifft2(fds).astype(np.float32)
    #ds = shift_fft_img.astype(np.float32)
    dss = np.stack([ds, ds, ds], axis=2)

    return dss
    """

    hpf = img - cv2.GaussianBlur(img, ksize=(r, r), sigmaX=0.0)+0.5

    return hpf

# オーバーレイ合成
def blend_overlay(base, over):
    result = np.zeros(base.shape, dtype=np.float32)
    darker = base < 0.5
    base_inv = 1.0-base
    over_inv = 1.0-over
    result[darker] = base[darker]*over[darker] *2
    #result[~darker] = (base[~darker]+over[~darker] - base[~darker]*over[~darker])*2-1
    result[~darker] = 1 - base_inv[~darker]*over_inv[~darker] *2
    
    return result

# 露出補正
def adjust_exposure(img, ev):
    # img: 変換元画像
    # ev: 補正値 -5.0〜5.0

    img2 = np.clip(img*(2.0**ev), 0, 1)

    return img2


# コントラスト補正
def adjust_contrast(img, cf):
    # img: 変換元画像
    # cf: コントラストファクター -100.0〜100.0

    c = 0.5   # 中心値
    f = cf/100.0*5.0  #-5.0〜5.0に変換

    if f == 0.0:
        adjust_img = img.copy()
    elif f > 0.0:
        adjust_img = sigmoid.scaled_sigmoid(img, f, c)
    else:
        adjust_img = sigmoid.scaled_inverse_sigmoid(img, f, c)

    return adjust_img


# 画像の明るさを制御点を元に補正する
def apply_curve(image, control_points, control_values, return_spline=False):    
    # image: 入力画像 HLS(float32)のLだけ
    # control_points : ピクセル値の制御点 list of float32 
    # control_values : 各制御点に対する補正値 list of float32
    # return_spline : Trueの場合、スプライン補間のオブジェクトを返す bool

    # エルミート補間
    cs = PchipInterpolator(control_points, control_values)
    
    # 画像データに補正を適用
    corrected_image = cs(image)

    # 値をuint16の範囲内にクリッピング
    # corrected_image = np.clip(corrected_image, 0, 65535).astype(np.float32)

    if return_spline:
        return corrected_image, cs
    else:
        return corrected_image

# 黒レベル補正
def adjust_shadow(img, black):
    f = -black/100.0*5.0

    if f == 0.0:
        adjust_img = img.copy()
    elif f > 0.0:
        adjust_img = sigmoid.scaled_sigmoid(img, f, 0.90)
    else:
        adjust_img = sigmoid.scaled_inverse_sigmoid(img, f, 0.90)

    return adjust_img

# 白レベル補正
def adjust_hilight(img, white):
    f = white/100.0*5.0

    if f == 0.0:
        adjust_img = img.copy()
    elif f > 0.0:
        adjust_img = sigmoid.scaled_sigmoid(img, f, 0.1)
    else:
        adjust_img = sigmoid.scaled_inverse_sigmoid(img, f, 0.1)

    return adjust_img

# レベル補正
def apply_level_adjustment(image, black_level, white_level, midtone_level):
    # image: 変換元イメージ
    # black_level: 黒レベル 0〜255
    # white_level: 白レベル 0〜255
    # midtone_level: 中間色レベル 0〜255

    # 16ビット画像の最大値
    max_val = 255

    # 指定された0-255のレベルを0-65535にスケーリング
    scale_factor = max_val / 255.0
    black_level_scaled = black_level * scale_factor
    white_level_scaled = white_level * scale_factor
    midtone_factor = midtone_level / 128.0

    # ルックアップテーブル (LUT) の作成
    lut = np.linspace(0, max_val, max_val+1, dtype=np.float32)  # Liner space creation

    # Pre-calculate constants
    range_inv = 1.0 / (white_level_scaled - black_level_scaled)
    #lut = np.clip((lut - black_level_scaled) * range_inv, 0, 1)  # Scale and clip
    lut = (lut - black_level_scaled) * range_inv  # Scale
    lut = np.power(lut, midtone_factor) * max_val  # Apply midtone factor and scale
    #lut = np.clip(lut, 0, max_val).astype(np.uint16)  # Final clip and type conversion
    lut = np.clip(lut, 0, max_val)  # Final clip and type conversion
  
    # 画像全体にルックアップテーブルを適用
    adjusted_image = lut[np.clip(image*max_val, 0, max_val).astype(np.uint16)]
    adjusted_image = (adjusted_image/max_val).astype(np.float32)

    return adjusted_image

# 彩度補正と自然な彩度補正
def adjust_saturation(s, sat, vib):
    # s: HLS画像 (彩度チャネル)
    # sat: 彩度の変更値
    # vib: 自然な彩度の変更値

    # 彩度変更値と自然な彩度変更値を計算
    sat = 1.0 + sat

    # 自然な彩度調整
    if vib == 0.0:
        final_s = s

    elif vib > 0.0:
        # 通常の計算
        vib = vib**2
        final_s = np.log(1.0 + vib * s) / np.log(1.0 + vib)
    else:
        # 逆関数を使用
        vib = vib**2
        final_s = (np.exp(s * np.log(1.0 + vib)) - 1.0) / vib

    # 彩度を適用
    final_s = final_s * sat

    return final_s

def apply_dehaze(img, dehaze):
    img2 = dehazing.dehaze.dehaze(img)
    
    img2 = dehaze * img2 + (1.0 - dehaze) * img
    return img2

# スプラインカーブの適用
def apply_point_list(img, point_list):
    # ソートとリスト内包表記をtogetherly処理
    point_list = sorted((pl[0], pl[1]) for pl in point_list)
    
    # unzip and convert to numpy arrays
    x, y = map(np.array, zip(*point_list))
    
    tck, u = splprep([x, y], k=min(3, len(x)-1), s=0)
    unew = np.linspace(0, 1.0, 1000, dtype=np.float32)
    out = splev(unew, tck)
    #out[1] = np.clip(out[1], 0, self.height)
    
    return apply_spline(img, out)

def apply_spline(img, spline):
    # img: 適用イメージ RGB
    # spline: スプライン

    if spline is None:
        return img

    x, y = spline

    # Generate the tone curve mapping
    lut = np.interp(np.arange(65536), x*65535, y*65535).astype(np.float32)

    # Apply the tone curve mapping to the image
    img2 = lut[(img*65535).astype(np.uint16)]
    img2 = img2/65535.0

    return img2

def __adjust_hls(hls_img, mask, adjust):
    hls = hls_img.copy()
    hls[mask, 0] = hls_img[mask, 0] + adjust[0]*1.8
    hls[mask, 1] = np.clip(hls_img[mask, 1] * (2.0**adjust[1]), 0, 1.0)
    hls[mask, 2] = np.clip(hls_img[mask, 2] * (2.0**adjust[2]), 0, 1.0)
    return hls

def adjust_hls_red(hls_img, red_adjust):
    hue_img = hls_img[:, :, 0]

    # 赤
    red_mask = ((hue_img >= 0) & (hue_img < 22.5)) | ((hue_img >= 337.5) & (hue_img < 360))
    hls_img = __adjust_hls(hls_img, red_mask, red_adjust)

    return hls_img

def adjust_hls_orange(hls_img, orange_adjust):
    hue_img = hls_img[:, :, 0]

    # オレンジ
    orange_mask = (hue_img >= 22.5) & (hue_img < 45)
    hls_img = __adjust_hls(hls_img, orange_mask, orange_adjust)

    return hls_img

def adjust_hls_yellow(hls_img, yellow_adjust):
    hue_img = hls_img[:, :, 0]

    # 黄色
    yellow_mask = (hue_img >= 45) & (hue_img < 75)
    hls_img = __adjust_hls(hls_img, yellow_mask, yellow_adjust)

    return hls_img

def adjust_hls_green(hls_img, green_adjust):
    hue_img = hls_img[:, :, 0]

    # 緑
    green_mask = (hue_img >= 75) & (hue_img < 150)
    hls_img = __adjust_hls(hls_img, green_mask, green_adjust)

    return hls_img

def adjust_hls_cyan(hls_img, cyan_adjust):
    hue_img = hls_img[:, :, 0]

    # シアン
    cyan_mask = (hue_img >= 150) & (hue_img < 210)
    hls_img = __adjust_hls(hls_img, cyan_mask, cyan_adjust)

    return hls_img

def adjust_hls_blue(hls_img, blue_adjust):
    hue_img = hls_img[:, :, 0]

    # 青
    blue_mask = (hue_img >= 210) & (hue_img < 270)
    hls_img = __adjust_hls(hls_img, blue_mask, blue_adjust)

    return hls_img

def adjust_hls_purple(hls_img, purple_adjust):
    hue_img = hls_img[:, :, 0]

    # 紫
    purple_mask = (hue_img >= 270) & (hue_img < 300)
    hls_img = __adjust_hls(hls_img, purple_mask, purple_adjust)

    return hls_img

def adjust_hls_magenta(hls_img, magenta_adjust):
    hue_img = hls_img[:, :, 0]

    # マゼンタ
    magenta_mask = (hue_img >= 300) & (hue_img < 337.5)
    hls_img = __adjust_hls(hls_img, magenta_mask, magenta_adjust)

    return hls_img

def adjust_density(hls_img, intensity):
    hls = np.zeros_like(hls_img)

    # 濃さ
    intensity = -intensity
    hls[:, :, 0] = hls_img[:, :, 0]
    hls[:, :, 1] = np.clip(hls_img[:, :, 1] * (1 + intensity / 200.0), 0, 1)
    hls[:, :, 2] = np.clip(hls_img[:, :, 2] * (1 - intensity / 100.0), 0, 1)

    return hls

def adjust_clear_color(rgb_img, intensity):

    # 清書色、濁色
    if intensity >= 0:
        rgb = np.clip(rgb_img * (1 + intensity / 100.0), 0, 1)
    else:
        gray = cvtToGrayColor(rgb_img)
        factor = -intensity / 200.0
        rgb = rgb_img * (1 - factor) + gray[..., np.newaxis] * factor
        rgb = rgb * (1 - factor * 0.3)
        rgb = np.clip(rgb, 0, 1)

    return rgb

def adjust_histogram(img, center, direction, intensity):
    """
    Adjust the histogram of a float64 image with 65536 levels.
    
    Parameters:
    img (numpy.ndarray): Input image as a float64 numpy array with values in the range [0, 65535].
    center (float): Desired histogram center (0-65535).
    direction (int): Shift direction (-1 for left, 1 for right).
    intensity (float): Shift intensity.
    
    Returns:
    numpy.ndarray: Adjusted image.
    """
    # Normalize the input image to the range [0, 1]
    img_normalized = img / 65535.0
    
    # Compute the histogram
    hist, bin_edges = np.histogram(img_normalized, bins=65536, range=(0, 1))
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    
    # Create a lookup table (LUT) based on the CDF and desired center shift
    lut = np.arange(65536, dtype=np.float32)
    adjustment = direction * intensity * (center / 65535.0 - lut / 65535.0)
    lut = np.clip(lut + adjustment * 65535, 0, 65535)

    # Apply the LUT to the normalized image
    adjusted_img_normalized = lut[(img_normalized * 65535).astype(np.uint16)] / 65535.0
    
    # Rescale the image back to the original range [0, 65535]
    adjusted_img = (adjusted_img_normalized * 65535).astype(np.float32)
    
    return adjusted_img

def make_clip(img, scale, x, y, w, h):
        
    img2 = cv2.resize(img, None, fx=scale, fy=scale)

    xx = int(x * scale)
    yy = int(y * scale)
    px = xx+w
    if px > img2.shape[1]:
        px = img2.shape[1]
    py = yy+h
    if py > img2.shape[0]:
        py = img2.shape[0]

    img3 = img2[yy:py, xx:px]
        
    return img3

# マスクイメージの適用
def apply_mask(img1, img2, msk=None):
    # img1: 元イメージ RGB
    # img2: 変更後イメージ RGB
    # msk: マスク

    if msk is not None:
        img = msk[:, :, np.newaxis] * img2 + (1.0 - msk[:, :, np.newaxis]) * img1

    return img

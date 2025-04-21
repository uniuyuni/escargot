import sys
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jit
from functools import partial
import colour
import lensfunpy
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import splprep, splev
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter
import logging
import math

import config
import sigmoid
import dng_sdk
import crop_editor
#from scipyjax import interpolate

jax.config.update("jax_platform_name", "METAL")

def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_image = (image_data - min_val) / (max_val - min_val)
    return normalized_image

def calculate_ev_from_image(image_data):
    if image_data.size == 0:
        raise ValueError("画像データが空です。")

    average_value = np.mean(image_data)

    # ここで基準を明確に設定
    # 例えば、EV0が0.5に相当する場合
    ev = np.log2(0.5 / average_value)  # 0.5を基準

    return ev, average_value

def calculate_correction_value(ev_histogram, ev_setting):
    correction_value = ev_setting - ev_histogram
    
    # 補正値を適切にクリッピング
    if correction_value > 4:  # 過剰補正を避ける
        correction_value = 4
    elif correction_value < -4:  # 過剰補正を避ける
        correction_value = -4

    return correction_value

# RGBからグレイスケールへの変換
@jit
def __cvtToGrayColor(rgb):
    # 変換元画像 RGB
    #gry = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gry = jnp.dot(rgb, jnp.array([0.2989, 0.5870, 0.1140]))

    return gry

def cvtToGrayColor(rgb):
    array = __cvtToGrayColor(rgb)
    array.block_until_ready()

    return np.array(array)

def convert_RGB2TempTint(rgb):

    xyz = colour.RGB_to_XYZ(rgb, 'sRGB')

    xy = colour.XYZ_to_xy(xyz)

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.set_xy_coord(xy)

    return float(dng.fTemperature), float(dng.fTint), float(xyz[1])

def __invert_temp_tint(temp, tint, ref_temp):

    # 色温度の反転
    mired_temp = 1e6 / temp
    mired_ref = 1e6 / ref_temp
    inverted_temp = 1e6 / (mired_ref - (mired_temp - mired_ref) + sys.float_info.min)

    # ティントの反転
    inverted_tint = -tint

    return inverted_temp, inverted_tint

def invert_RGB2TempTint(rgb, ref_temp=5000.0):
    temp, tint, Y = convert_RGB2TempTint(rgb)

    invert_temp, invert_tint = __invert_temp_tint(temp, tint, ref_temp)

    return invert_temp, invert_tint, Y


def convert_TempTint2RGB(temp, tint, Y):

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.fTemperature = temp
    dng.fTint = tint

    xy = dng.get_xy_coord()

    xyz = colour.xy_to_XYZ(xy)
    xyz *= Y

    rgb = colour.XYZ_to_RGB(xyz, 'sRGB')

    return rgb.astype(np.float32)

def invert_TempTint2RGB(temp, tint, Y, reference_temp=5000.0):

    inverted_temp, inverted_tint = __invert_temp_tint(temp, tint, reference_temp)
    
    # DNG SDKの関数を使用して元のRGB値を取得
    r, g, b = convert_TempTint2RGB(inverted_temp, inverted_tint, Y)

    return [r, g, b]

def calc_resize_image(original_size, max_length):
    width, height = original_size

    if width > height:
        # 幅が長辺の場合
        scale_factor = max_length / width
    else:
        # 高さが長辺の場合
        scale_factor = max_length / height

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return (new_width, new_height)

def rotation(img, angle, flip_mode=0):
    # 元の画像の高さと幅を取得
    height = img.shape[0]
    width = img.shape[1]
    
    # 回転の中心点を計算
    center = (int(width/2), int(height/2))
    
    # 回転行列を計算（スケール付き）
    trans = cv2.getRotationMatrix2D(center, angle, 1)

    # 回転後画像サイズ    
    size = max(width, height)
    
    # 変換行列に平行移動を追加
    trans[0, 2] += (size / 2) - center[0]
    trans[1, 2] += (size / 2) - center[1]

    # フリップモードの処理
    # bit0が1なら左右反転、bit1が1なら上下反転
    img_affine = img
    if flip_mode & 1:  # 左右反転
        img_affine = cv2.flip(img_affine, 1)
    
    if flip_mode & 2:  # 上下反転
        img_affine = cv2.flip(img_affine, 0)

    # 回転と中心補正を同時に行う
    img_affine = cv2.warpAffine(img_affine, trans, (size, size), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))

    return img_affine

@partial(jit, static_argnums=(0, 1, 2))
def __create_gaussian_kernel(width: int, height: int, sigma: float):
    # kernelの中心と1個横のセルの値の比
    if sigma == 0.0:
        sigma = 0.3*(max(width, height)/2 - 1) + 0.8

    r = jnp.exp(-1 / (2 * sigma**2))

    # kernelの中心index
    mw = width // 2
    mh = height // 2

    # kernelの中心を原点とした座標が入った配列
    xs = jnp.arange(-mw, mw + 1)
    ys = jnp.arange(-mh, mh + 1)
    x, y = jnp.meshgrid(xs, ys)

    # rを絶対値の2乗分累乗すると、ベースとなる配列が求まる
    g = r**(x**2 + y**2)

    # 正規化
    return g / np.sum(g)

@partial(jit, static_argnums=(1, 2))
def __gaussian_blur(src, ksize=(3, 3), sigma=0.0):
    kw, kh = ksize
    kernel = __create_gaussian_kernel(kw, kh, sigma)

    if src.ndim >= 3:
        dest = []
        for i in range(src.shape[2]):
            dest.append(jscipy.signal.convolve2d(src[:, :, i], kernel, mode='same'))

        dest = jnp.stack(dest, -1)
    else:
        dest = jscipy.signal.convolve2d(src, kernel, mode='same')

    return dest

def gaussian_blur(src, ksize=(3, 3), sigma=0.0):
    array = __gaussian_blur(src, ksize, sigma)
    array.block_until_ready()

    return np.array(array)


def lensblur_filter(image, radius):
    # カーネルを生成
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    
    # カーネルに円を描く
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    kernel /= np.sum(kernel)

    # レンズブラーを適用
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

@partial(jit, static_argnums=1)
def __lucy_richardson_gauss(srcf, iteration):

    # 出力用の画像を初期化
    destf = srcf.copy()

    for i in range(iteration):
        # ガウスぼかしを適用してぼけをシミュレーション
        bdest = __gaussian_blur(destf, ksize=(9, 9), sigma=0)

        # 元画像とぼけた画像の比を計算
        bdest = bdest + jnp.finfo(jnp.float32).eps
        ratio = jnp.divide(srcf, bdest)

        # 誤差の分配のために再びガウスぼかしを適用
        ratio_blur = __gaussian_blur(ratio, ksize=(9, 9), sigma=0)

        # 元の出力画像に誤差を乗算
        destf = jnp.multiply(destf, ratio_blur)
    
    return destf

def lucy_richardson_gauss(srcf, iteration):
    array = __lucy_richardson_gauss(srcf, iteration)
    array.block_until_ready()

    return np.array(array)

def tone_mapping(x, exposure=1.0):
    # Reinhard トーンマッピング
    return x / (x + exposure)

def highlight_compress(image):

    return tone_mapping(image)

def correct_overexposed_areas(image_rgb: np.ndarray,
                            threshold_low=0.94,
                            threshold_high=1.0,
                            correction_color=(0.94, 0.94, 0.96),
                            blur_sigma=15.0) -> np.ndarray:
    """
    float32形式のRGB画像（0-1）の白飛び部分を自然に補正する関数
    
    Parameters:
    -----------
    image_rgb : np.ndarray
        入力画像（float32形式、RGB、値域0-1）
        shape: (height, width, 3)
    threshold_low : float
        補正を開始する明るさの閾値（デフォルト: 0.94）
    threshold_high : float
        完全な白とみなす閾値（デフォルト: 1.0）
    correction_color : tuple
        補正に使用する色（デフォルト: わずかに青みがかった白）
    blur_sigma : float
        ガウシアンブラーのシグマ値（デフォルト: 15.0）
        大きい値ほど広い範囲でブレンドされる
    """
    # 各チャンネルの輝度に基づいてマスクを作成
    # 各ピクセルの最小値を使用することで、より慎重に白領域を検出
    pixel_brightness = np.min(image_rgb, axis=2)
    
    # グラデーショナルなマスクを作成
    correction_mask = np.clip((pixel_brightness - threshold_low) / 
                            (threshold_high - threshold_low), 0, 1)
    
    # マスクを大きめのシグマ値でぼかす
    correction_mask = gaussian_filter(correction_mask, sigma=blur_sigma)
    
    # さらになめらかな補正のため、マスクを累乗して非線形に
    correction_mask = correction_mask ** 1.5  # 補正の急激な変化を抑える
    
    # 補正色のマップを作成
    correction = np.zeros_like(image_rgb, dtype=np.float32)
    for i in range(3):
        correction[:,:,i] = correction_color[i]
    
    # 補正を適用
    correction_mask = np.expand_dims(correction_mask, axis=2)
    
    # 元の画像と補正色をブレンド
    result = image_rgb * (1 - correction_mask) + correction * correction_mask
    
    return result

# ローパスフィルタ
def lowpass_filter(img, r):
    lpf = gaussian_blur(img, ksize=(r, r), sigma=0.0)

    return lpf

# ハイパスフィルタ
def highpass_filter(img, r):
    hpf = img - gaussian_blur(img, ksize=(r, r), sigma=0.0)+0.5

    return hpf    

# オーバーレイ合成
def blend_overlay(base, over):
    result = np.zeros(base.shape, dtype=np.float32)
    darker = base < 0.5
    base_inv = 1.0-base
    over_inv = 1.0-over
    result[darker] = base[darker] * over[darker] * 2
    #result[~darker] = (base[~darker]+over[~darker] - base[~darker]*over[~darker])*2-1
    result[~darker] = 1 - base_inv[~darker] * over_inv[~darker] * 2
    
    return result

# スクリーン合成
def blend_screen(base, over):
    result = 1 - (1.0-base)*(1-over)

    return result

def modify_lens(img, exif_data, is_cm=True, is_sd=True, is_gd=True):
    #logging.info(exif_data['Make'], exif_data['Model'])
    #logging.info(exif_data['LensMake'], exif_data['LensModel'])
    #logging.info(exif_data['FocalLength'], exif_data['ApertureValue'])

    db = lensfunpy.Database()
    cam = db.find_cameras(exif_data['Make'], exif_data['Model'], loose_search=True)
    lens = db.find_lenses(cam[0], exif_data['LensMake'], exif_data['LensModel'], loose_search=True)

    height, width = img.shape[0], img.shape[1]
    mod = lensfunpy.Modifier(lens[0], cam[0].crop_factor, width, height)
    mod.initialize(float(exif_data['FocalLength'][0:-3]), exif_data['ApertureValue'], pixel_format=np.float32)

    modimg = img
    if is_cm == True:
        modimg = img.copy()
        did_apply = mod.apply_color_modification(modimg)
        logging.info("Apply Color Modification is Failed")

    if is_sd == True:
        undist_coords = mod.apply_subpixel_distortion()
        modimg[..., 0] = cv2.remap(modimg[..., 0], undist_coords[..., 0, :], None, cv2.INTER_LANCZOS4)
        modimg[..., 1] = cv2.remap(modimg[..., 1], undist_coords[..., 1, :], None, cv2.INTER_LANCZOS4)
        modimg[..., 2] = cv2.remap(modimg[..., 2], undist_coords[..., 2, :], None, cv2.INTER_LANCZOS4)

    if is_gd == True:
        undist_coords = mod.apply_geometry_distortion()
        modimg = cv2.remap(modimg, undist_coords, None, cv2.INTER_LANCZOS4)

    return np.clip(modimg, 0, 1)


# 露出補正
def calc_exposure(img, ev):
    # img: 変換元画像
    # ev: 補正値 -4.0〜4.0

    #img2 = img*(2.0**ev)

    return (2.0**ev)

# コントラスト補正
def adjust_contrast(img, cf, c):
    # img: 変換元画像
    # cf: コントラストファクター -100.0〜100.0
    # c: 中心値 0〜1.0
    """
    f = cf / 100.0 * 10.0  #-10.0〜10.0に変換

    if f == 0.0:
        adjust_img = img.copy()
    elif f >= 0.0:
        mm = max(1.0, np.max(img))
        adjust_img = sigmoid.scaled_sigmoid(img/mm, f, c/mm)*mm
    else:
        mm = max(1.0, np.max(img))
        adjust_img = sigmoid.scaled_inverse_sigmoid(img/mm, -f, c/mm)*mm
        
    return adjust_img
    """
    adjust = adjust_tone(img, cf, -cf)

    return adjust

# 画像の明るさを制御点を元に補正する
def apply_curve(image, control_points, control_values):    
    # image: 入力画像 HLS(float32)のLだけ
    # control_points : ピクセル値の制御点 list of float32 
    # control_values : 各制御点に対する補正値 list of float32

    # エルミート補間
    cs = PchipInterpolator(control_points, control_values)    
    corrected_image = cs(image)

    #corrected_image = interpolate(control_points, control_values, image)
    #corrected_image.block_until_ready()
    #return np.array(corrected_image).astype(np.float32)

    return corrected_image.astype(np.float32)

# レベル補正
def apply_level_adjustment(image, black_level, midtone_level, white_level):
    # image: 変換元イメージ
    # black_level: 黒レベル 0〜255
    # white_level: 白レベル 0〜255
    # midtone_level: 中間色レベル 0〜255

    # 16ビット画像の最大値
    max_val = 65535
    black_level *= 256
    white_level *= 256
    midtone_level *= 256

    # midtone_level を 1.0 を基準としてスケーリング (128が基準のため)
    midtone_factor = midtone_level / 32768.0

    # ルックアップテーブル (LUT) の作成 (0〜65535)
    lut = np.linspace(0, max_val, max_val + 1, dtype=np.float32)  # Liner space creation

    # Pre-calculate constants
    range_inv = 1.0 / (white_level - black_level)
    
    # LUT のスケーリングとクリッピング
    lut = np.clip((lut - black_level) * range_inv, 0, 1)  # Scale and clip
    lut = np.power(lut, midtone_factor) * max_val  # Apply midtone factor and scale
    lut = np.clip(lut, 0, max_val).astype(np.uint16)  # Final clip and type conversion
    
    # 画像全体にルックアップテーブルを適用
    adjusted_image = lut[np.clip(image*max_val, 0, max_val).astype(np.uint16)]
    adjusted_image = adjusted_image/max_val

    return adjusted_image

# 彩度補正と自然な彩度補正
@partial(jit, static_argnums=(1, 2))
def __calc_saturation(s, sat, vib):

    # 彩度変更値と自然な彩度変更値を計算
    if sat >= 0:
        sat = 1.0 + sat/100.0
    else:
        sat = 1.0 + sat/100.0
    vib /= 50.0

    # 計算の破綻を防止（元データは壊さない）
    s = jnp.clip(s, 0.0, 1.0)

    # 自然な彩度調整
    if vib == 0.0:
        final_s = s

    elif vib > 0.0:
        # 通常の計算
        vib = vib**2
        final_s = jnp.log(1.0 + vib * s) / jnp.log(1.0 + vib)
    else:
        # 逆関数を使用
        vib = vib**2
        final_s = (jnp.exp(s * jnp.log(1.0 + vib)) - 1.0) / vib

    # 彩度を適用
    final_s = final_s * sat

    return final_s

def calc_saturation(s, sat, vib):
    array = __calc_saturation(s, sat, vib)
    array.block_until_ready()

    return np.array(array)


def calc_point_list_to_lut(point_list, max_value=1.0):
    """
    スプライン補間を使った基本的なLUT生成関数
    
    Parameters:
    -----------
    point_list : list of tuples
        (x, y)形式のコントロールポイントのリスト
    max_value : float
        LUTが対応する最大値（デフォルト1.0）
        
    Returns:
    --------
    ndarray
        65536エントリーのLUT
    """
    # ポイントをソート
    point_list = sorted((pl[0], pl[1]) for pl in point_list)
    
    # ポイントからx, y配列を取得
    x, y = zip(*point_list)
    x, y = np.array(x), np.array(y)
    
    # 3点以上ある場合はスプライン補間を使用
    if len(x) >= 3:
        # スプライン補間のパラメータ（次数は点の数-1か3の小さい方）
        k = min(3, len(x) - 1)
        # スプライン補間の計算
        tck, u = splprep([x, y], k=k, s=0)
        
        # [0, 1]の範囲で細かい点を生成
        fine_u = np.linspace(0, 1, 1000)
        fine_points = splev(fine_u, tck)
        
        # 生成された点を取得
        fine_x, fine_y = fine_points
        
        # この点を使って通常の線形補間でLUTを生成
        lut_size = 65536
        input_range = np.linspace(0, max_value, lut_size)
        
        # max_valueを超える部分は直線で外挿
        mask_in_range = input_range <= max(fine_x)
        lut = np.zeros(lut_size, dtype=np.float32)
        
        # 範囲内は補間
        lut[mask_in_range] = np.interp(
            input_range[mask_in_range], 
            fine_x, 
            fine_y
        )
        
        # 範囲外は直線外挿（最後の2点から傾きを計算）
        if np.any(~mask_in_range):
            # 最後の2点から傾きを計算
            last_idx = len(fine_x) - 1
            second_last_idx = last_idx - 1
            slope = (fine_y[last_idx] - fine_y[second_last_idx]) / (fine_x[last_idx] - fine_x[second_last_idx])
            
            # 直線外挿
            x_out = input_range[~mask_in_range]
            y_last = fine_y[last_idx]
            x_last = fine_x[last_idx]
            lut[~mask_in_range] = y_last + slope * (x_out - x_last)
    else:
        # 点が少ない場合は単純な線形補間
        lut_size = 65536
        input_range = np.linspace(0, max_value, lut_size)
        lut = np.interp(input_range, x, y).astype(np.float32)
    
    return lut

def apply_lut(img, lut, max_value=1.0):
    """
    画像にLUTを適用する関数
    max_value: LUTが対応する最大値（デフォルト1.0）
    """
    # スケーリングしてLUTのインデックスに変換
    scale_factor = 65535 / max_value
    lut_indices = np.clip(np.round(img * scale_factor), 0, 65535).astype(np.uint16)

    # LUTを適用
    result = np.take(lut, lut_indices)
    
    return result

def apply_lut_mul(img, lut, max_value=1.0):

    # スケーリングしてLUTのインデックスに変換
    scale_factor = 65535 / max_value
    lut_indices = np.clip(np.round(img * scale_factor), 0, 65535).astype(np.uint16)
    
    # LUTを適用
    result = np.take(lut, lut_indices) * img
    
    return result

def apply_lut_add(img, lut, max_value=1.0):

    # スケーリングしてLUTのインデックスに変換
    scale_factor = 65535 / max_value
    lut_indices = np.clip(np.round(img * scale_factor), 0, 65535).astype(np.uint16)
    
    # LUTを適用
    result = np.take(lut, lut_indices) + img
    
    return result

def __adjust_hls(hls_img, hue_condition, adjust):
    """
    HLS色空間での色調整をJAXで実装（jnp.whereを使用）
    
    Args:
        hls_img: HLS形式の画像配列
        hue_condition: 色相に対する条件式
        adjust: 調整値の配列 [色相, 明度, 彩度]
    """
    # 各チャンネルごとにwhere演算で値を更新
    h = jnp.where(hue_condition[..., None], 
                  hls_img[..., 0:1] + adjust[0]*1.8,
                  hls_img[..., 0:1])
    
    l = jnp.where(hue_condition[..., None],
                  hls_img[..., 1:2] * (2.0**adjust[1]),
                  hls_img[..., 1:2])
    
    s = jnp.where(hue_condition[..., None],
                  hls_img[..., 2:3] * (2.0**adjust[2]),
                  hls_img[..., 2:3])
    
    # 結果を結合
    return jnp.concatenate([h, l, s], axis=-1)

def adjust_hls_red(hls_img, red_adjust):
    """
    赤色領域の調整をJAXで実装
    
    Args:
        hls_img: HLS形式の画像配列
        red_adjust: 赤色の調整値 [色相, 明度, 彩度]
    """
    hue_img = hls_img[..., 0]
    
    # 赤色の条件式
    red_condition = jnp.logical_or(
        jnp.logical_and(hue_img >= 0, hue_img < 22.5),
        jnp.logical_and(hue_img >= 337.5, hue_img < 360)
    )
    
    return __adjust_hls(hls_img, red_condition, red_adjust)


def adjust_hls_orange(hls_img, orange_adjust):
    hue_img = hls_img[:, :, 0]

    # オレンジ
    orange_condition = jnp.logical_and(hue_img >= 22.5, hue_img < 45)
    hls_img = __adjust_hls(hls_img, orange_condition, orange_adjust)

    return hls_img

def adjust_hls_yellow(hls_img, yellow_adjust):
    hue_img = hls_img[:, :, 0]

    # 黄色
    yellow_condition = jnp.logical_and(hue_img >= 45, hue_img < 75)
    hls_img = __adjust_hls(hls_img, yellow_condition, yellow_adjust)

    return hls_img

def adjust_hls_green(hls_img, green_adjust):
    hue_img = hls_img[:, :, 0]

    # 緑
    green_condition = jnp.logical_and(hue_img >= 75, hue_img < 150)
    hls_img = __adjust_hls(hls_img, green_condition, green_adjust)

    return hls_img

def adjust_hls_cyan(hls_img, cyan_adjust):
    hue_img = hls_img[:, :, 0]

    # シアン
    cyan_condition = jnp.logical_and(hue_img >= 150, hue_img < 210)
    hls_img = __adjust_hls(hls_img, cyan_condition, cyan_adjust)

    return hls_img

def adjust_hls_blue(hls_img, blue_adjust):
    hue_img = hls_img[:, :, 0]

    # 青
    blue_condition = jnp.logical_and(hue_img >= 210, hue_img < 270)
    hls_img = __adjust_hls(hls_img, blue_condition, blue_adjust)

    return hls_img

def adjust_hls_purple(hls_img, purple_adjust):
    hue_img = hls_img[:, :, 0]

    # 紫
    purple_condition = jnp.logical_and(hue_img >= 270, hue_img < 300)
    hls_img = __adjust_hls(hls_img, purple_condition, purple_adjust)

    return hls_img

def adjust_hls_magenta(hls_img, magenta_adjust):
    hue_img = hls_img[:, :, 0]

    # マゼンタ
    magenta_condition = jnp.logical_and(hue_img >= 300, hue_img < 337.5)
    hls_img = __adjust_hls(hls_img, magenta_condition, magenta_adjust)

    return hls_img


@partial(jit, static_argnums=(3,))
def create_smooth_weight(hue_img, center, width, transition=15.0):
    """
    滑らかな重み付け関数を作成（GPU対応版）
    
    Args:
        hue_img: 色相画像
        center: 中心となる色相値
        width: 色相の範囲の幅
        transition: 境界のぼかし幅（大きいほど滑らかに）
    
    Returns:
        0〜1の間の重み値の配列
    """
    # 中心からの距離を計算
    half_width = width / 2.0
    
    # 中心から上下half_width離れた点を境界とする
    lower = (center - half_width) % 360
    upper = (center + half_width) % 360
    
    # GPU対応：条件分岐を使わず、マスクと重みを計算
    
    # 通常ケース用の計算（lower < upper）
    # コア領域（完全に含まれる領域）のマスク
    normal_core = (hue_img >= lower) & (hue_img <= upper)
    
    # 下側の遷移領域のマスクと重み
    normal_lower_mask = (hue_img >= (lower - transition)) & (hue_img < lower)
    normal_lower_weight = (hue_img - (lower - transition)) / transition
    
    # 上側の遷移領域のマスクと重み
    normal_upper_mask = (hue_img > upper) & (hue_img <= (upper + transition))
    normal_upper_weight = ((upper + transition) - hue_img) / transition
    
    # 赤色など0/360度をまたぐケース用の計算（lower > upper）
    # コア領域が2つに分かれる
    wrap_core = (hue_img >= lower) | (hue_img <= upper)
    
    # 下側の遷移領域のマスクと重み
    wrap_lower_mask = (hue_img >= ((lower - transition) % 360)) & (hue_img < lower)
    wrap_lower_weight = (hue_img - ((lower - transition) % 360)) / transition
    
    # 上側の遷移領域のマスクと重み
    wrap_upper_mask = (hue_img > upper) & (hue_img <= ((upper + transition) % 360))
    wrap_upper_weight = (((upper + transition) % 360) - hue_img) / transition
    
    # 通常ケースと0/360度をまたぐケースの選択
    # lower < upperの場合、use_normal=1.0、それ以外は0.0
    use_normal = jnp.where(lower < upper, 1.0, 0.0)
    
    # コア領域のマスク
    core_mask = normal_core * use_normal + wrap_core * (1.0 - use_normal)
    
    # 下側遷移領域のマスクと重み
    lower_mask = normal_lower_mask * use_normal + wrap_lower_mask * (1.0 - use_normal)
    lower_weight = normal_lower_weight * normal_lower_mask * use_normal + \
                   wrap_lower_weight * wrap_lower_mask * (1.0 - use_normal)
    
    # 上側遷移領域のマスクと重み
    upper_mask = normal_upper_mask * use_normal + wrap_upper_mask * (1.0 - use_normal)
    upper_weight = normal_upper_weight * normal_upper_mask * use_normal + \
                   wrap_upper_weight * wrap_upper_mask * (1.0 - use_normal)
    
    # 最終的な重み計算
    weight = core_mask * 1.0 + lower_weight + upper_weight
    
    # 0〜1の範囲に収める
    return jnp.clip(weight, 0.0, 1.0)

@jit
def adjust_hls_smooth(hls_img, weight, adjust):
    """
    HLS色空間での色調整を滑らかな重みを使って実装（GPU対応版）
    
    Args:
        hls_img: HLS形式の画像配列
        weight: 各ピクセルの調整重み（0〜1）
        adjust: 調整値の配列 [色相, 明度, 彩度]
    """
    # 3次元の重みを計算（チャンネル数に合わせる）
    weight_3d = weight[..., jnp.newaxis]
    
    # 各チャンネルの抽出
    h = hls_img[..., 0:1]  # 色相
    l = hls_img[..., 1:2]  # 明度
    s = hls_img[..., 2:3]  # 彩度
    
    # 彩度が低い部分（グレースケールに近い）で色相変更を抑制
    # 彩度が低いほど、色相の変化が目立たなくなるため、彩度に比例して色相調整を適用
    saturation_factor = s  # 彩度がそのまま係数になる（0〜1）
    
    # 極端な明度（非常に暗いor明るい）でも色相/彩度変更を抑制
    # 明度0.5を中心にした二次関数で、L=0またはL=1で0になる係数
    luminance_factor = 1.0 - 4.0 * (l - 0.5) ** 2
    luminance_factor = jnp.clip(luminance_factor, 0.0, 1.0)
    
    # 彩度と明度の両方の要素を考慮した最終的な保護係数
    # この係数が低いほど、ハイライトやシャドウ、無彩色部分が保護される
    protection_factor = saturation_factor * luminance_factor
    
    # 各チャンネルの調整（保護係数を適用）
    # 色相調整：保護係数と重みの両方を適用
    h_adjusted = h + adjust[0] * 1.8 * weight_3d * protection_factor
    
    # 明度調整：明度の調整は通常通り適用
    l_adjusted = l * (1.0 + (2.0**adjust[1] - 1.0) * weight_3d * jnp.sqrt(luminance_factor))
    
    # 彩度調整：保護係数に反比例して抑制
    # これにより、元々彩度が低い部分では彩度上昇が抑えられる
    s_adjusted = s * (1.0 + (2.0**adjust[2] - 1.0) * weight_3d * jnp.sqrt(protection_factor))
    
    # 結果を結合
    return jnp.concatenate([h_adjusted, l_adjusted, s_adjusted], axis=-1)

# 各色の調整関数（JITで事前コンパイル）
@partial(jit, static_argnums=(2,))
def adjust_hls_red_smooth(hls_img, red_adjust, transition=15.0):
    """
    赤色領域の調整を滑らかに実装（GPU対応版）
    
    Args:
        hls_img: HLS形式の画像配列
        red_adjust: 赤色の調整値 [色相, 明度, 彩度]
        transition: 境界のぼかし幅（度数）
    """
    hue_img = hls_img[..., 0]
    
    # 赤色は0度を中心とし、幅45度の範囲
    red_weight = create_smooth_weight(hue_img, 0, 45, transition)
    
    return adjust_hls_smooth(hls_img, red_weight, jnp.array(red_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_orange_smooth(hls_img, orange_adjust, transition=15.0):
    """
    オレンジ色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # オレンジは33.75度を中心とし、幅22.5度の範囲
    orange_weight = create_smooth_weight(hue_img, 33.75, 22.5, transition)
    
    return adjust_hls_smooth(hls_img, orange_weight, jnp.array(orange_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_yellow_smooth(hls_img, yellow_adjust, transition=15.0):
    """
    黄色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # 黄色は60度を中心とし、幅30度の範囲
    yellow_weight = create_smooth_weight(hue_img, 60, 30, transition)
    
    return adjust_hls_smooth(hls_img, yellow_weight, jnp.array(yellow_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_green_smooth(hls_img, green_adjust, transition=15.0):
    """
    緑色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # 緑は112.5度を中心とし、幅75度の範囲
    green_weight = create_smooth_weight(hue_img, 112.5, 75, transition)
    
    return adjust_hls_smooth(hls_img, green_weight, jnp.array(green_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_cyan_smooth(hls_img, cyan_adjust, transition=15.0):
    """
    シアン色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # シアンは180度を中心とし、幅60度の範囲
    cyan_weight = create_smooth_weight(hue_img, 180, 60, transition)
    
    return adjust_hls_smooth(hls_img, cyan_weight, jnp.array(cyan_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_blue_smooth(hls_img, blue_adjust, transition=15.0):
    """
    青色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # 青は240度を中心とし、幅60度の範囲
    blue_weight = create_smooth_weight(hue_img, 240, 60, transition)
    
    return adjust_hls_smooth(hls_img, blue_weight, jnp.array(blue_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_purple_smooth(hls_img, purple_adjust, transition=15.0):
    """
    紫色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # 紫は285度を中心とし、幅30度の範囲
    purple_weight = create_smooth_weight(hue_img, 285, 30, transition)
    
    return adjust_hls_smooth(hls_img, purple_weight, jnp.array(purple_adjust))

@partial(jit, static_argnums=(2,))
def adjust_hls_magenta_smooth(hls_img, magenta_adjust, transition=15.0):
    """
    マゼンタ色領域の調整を滑らかに実装（GPU対応版）
    """
    hue_img = hls_img[..., 0]
    
    # マゼンタは318.75度を中心とし、幅37.5度の範囲
    magenta_weight = create_smooth_weight(hue_img, 318.75, 37.5, transition)
    
    return adjust_hls_smooth(hls_img, magenta_weight, jnp.array(magenta_adjust))


# マスクイメージの適用
@jit
def __apply_mask(img1, msk, img2):

    if msk is not None:
        _msk = msk[:, :, jnp.newaxis]
        img = img1 * _msk + img2 * (1.0 - _msk)

    return img

def apply_mask(img1, msk, img2):
    array = __apply_mask(img1, msk, img2)
    array.block_until_ready()

    return np.array(array)

def apply_vignette(image, intensity=0, radius_percent=100, crop_info=None, original_size=None):
    """
    float32形式の画像に周辺光量落ち効果を適用する関数（クロップと拡大に対応）
    
    Parameters:
    image: numpy.ndarray - 入力画像（float32形式、値域は0-1）
    intensity: int - 光量落ちの強度 (-100 から 100)
        負の値: 周辺を暗くする
        正の値: 周辺を明るくする
    radius_percent: float - 効果の及ぶ半径（画像の対角線の長さに対する割合、1-100）
    crop_info: list or None - クロップ情報 [x, y, w, h, scale]
        x: 切り出し開始x座標（元画像での位置）
        y: 切り出し開始y座標（元画像での位置）
        w: 切り出し幅
        h: 切り出し高さ
        scale: 拡大率
    original_size: tuple or None - 元画像のサイズ (width, height)
        crop_infoが指定された場合は必須
    
    Returns:
    numpy.ndarray - 効果を適用した画像（float32形式、値域は0-1）
    """
    
    # 入力値の検証
    if image.dtype != np.float32:
        raise ValueError("入力画像はfloat32形式である必要があります")
    
    if crop_info is not None:
        if len(crop_info) != 5:
            raise ValueError("crop_infoは[x, y, w, h, scale]の形式である必要があります")
        if original_size is None:
            raise ValueError("crop_infoが指定された場合、original_sizeは必須です")
    
    intensity = np.clip(intensity, -100, 100)
    radius_percent = np.clip(radius_percent, 1, 100)
    
    # 現在の画像サイズを取得
    current_rows, current_cols = image.shape[:2]
    
    if crop_info is None:
        # クロップ情報がない場合は現在の画像中心を使用
        center_x = current_cols / 2
        center_y = current_rows / 2
        max_radius = np.sqrt(current_rows**2 + current_cols**2) / 2
    else:
        x, y, w, h, scale = crop_info
        original_w, original_h = original_size
        
        # 元画像の中心座標
        original_center_x = original_w / 2
        original_center_y = original_h / 2
        
        # クロップ領域の中心座標（元画像での座標）
        crop_center_x = x + w / 2
        crop_center_y = y + h / 2
        
        # 中心からのオフセットを計算し、スケールを適用
        offset_x = (original_center_x - crop_center_x) * scale
        offset_y = (original_center_y - crop_center_y) * scale
        
        # 現在の画像での中心座標
        center_x = current_cols / 2 + offset_x
        center_y = current_rows / 2 + offset_y
        
        # 元画像の対角線長さから最大半径を計算し、スケールを適用
        max_radius = np.sqrt(original_w**2 + original_h**2) / 2 * scale
    
    # 実際に使用する半径を計算
    radius = max_radius * (radius_percent / 100)
    
    # マスクを作成
    Y, X = np.ogrid[:current_rows, :current_cols]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # 正規化された距離マップを作成（0-1の範囲）
    mask = dist_from_center / radius
    mask = np.clip(mask, 0, 1)
    
    # intensity値に基づいてマスクを調整
    if intensity < 0:
        # 周辺を暗くする
        mask = mask * (-intensity / 100)
        mask = 1 - mask
    else:
        # 周辺を明るくする
        mask = mask * (intensity / 100)
    
    # 画像が3チャンネル（カラー）の場合は、マスクを3次元に拡張
    if len(image.shape) == 3:
        mask = np.dstack([mask] * 3)
    
    # float32形式を維持したまま効果を適用
    if intensity < 0:
        result = image * mask
    else:
        # 明るくする場合
        bright_increase = (1 - image) * mask
        result = image + bright_increase
    
    # float32形式を維持
    return result.astype(np.float32)

# テクスチャサイズとクロップ情報から、新しい描画サイズと余白の大きさを得る
def crop_size_and_offset_from_texture(texture_width, texture_height, crop_info):

    # アスペクト比を計算
    crop_aspect = crop_info[2] / crop_info[3]
    texture_aspect = texture_width / texture_height

    if crop_aspect > texture_aspect:
        # 画像が横長の場合
        new_width = texture_width
        new_height = int(texture_width / crop_aspect)
    else:
        # 画像が縦長の場合
        new_width = int(texture_height * crop_aspect)
        new_height = texture_height

    # 中央に配置するためのオフセットを計算
    offset_x = (texture_width - new_width) // 2
    offset_y = (texture_height - new_height) // 2

    return (new_width, new_height, offset_x, offset_y)

def crop_image(image, crop_info, texture_width, texture_height, click_x, click_y, offset, is_zoomed):

    # 画像のサイズを取得
    image_height, image_width = image.shape[:2]

    new_width, new_height, offset_x, offset_y = crop_size_and_offset_from_texture(texture_width, texture_height, crop_info)

    # スケールを求める
    if crop_info[2] >= crop_info[3]:
        scale = texture_width/crop_info[2]
    else:
        scale = texture_height/crop_info[3]

    if not is_zoomed:
        # リサイズ
        cx, cy, cw, ch, _ = crop_info
        resized_img = cv2.resize(image[cy:cy+ch, cx:cx+cw], (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # リサイズした画像を中央に配置
        result = np.pad(resized_img, ((offset_y, texture_height-(offset_y+new_height)), (offset_x, texture_width-(offset_x+new_width)), (0, 0)), constant_values=0)

        # 再設定
        crop_info = (cx, cy, cw, ch, scale)

    else:
        # クリック位置を元の画像の座標系に変換
        click_x = click_x - offset_x
        click_y = click_y - offset_y
        click_image_x = click_x / scale
        click_image_y = click_y / scale

        # 切り抜き範囲を計算
        crop_width = int(texture_width)
        crop_height = int(texture_height)

        if offset == (0, 0):
            # クリック位置を中心にする
            crop_x = crop_info[0] + click_image_x - crop_width // 2
            crop_y = crop_info[1] + click_image_y - crop_height // 2
        else:
            # スクロール
            crop_x = crop_info[0]
            crop_y = crop_info[1]

        # クロップ
        result, crop_info = crop_image_info(image, (crop_x, crop_y, crop_width, crop_height, 1.0), offset)
    
    return result, crop_info


def crop_image_info(image, crop_info, offset=(0, 0)):
    
    # 情報取得
    image_height, image_width = image.shape[:2]
    crop_x, crop_y, crop_width, crop_height, scale = crop_info

    # オフセット適用
    crop_x = int(crop_x + offset[0])
    crop_y = int(crop_y + offset[1])

    # 画像の範囲外にならないように調整
    crop_x = max(0, min(crop_x, image_width - crop_width))
    crop_y = max(0, min(crop_y, image_height - crop_height))

    # 画像を切り抜く
    #cropped_img = jax.lax.slice(image, (crop_y, crop_x, 0), (crop_y+crop_height, crop_x+crop_width, 3))
    cropped_img = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    return cropped_img, (crop_x, crop_y, crop_width, crop_height, scale)


def get_multiple_mask_bbox(mask):
    """
    マスク画像から複数の独立した領域それぞれのバウンディングボックスを計算する
    
    Args:
        mask マスク画像（2次元のnumpy配列）
        
    Returns:
        各領域の(x_min, y_min, x_max, y_max)座標のリスト
            空のマスクの場合は空リストを返す
    """
    # マスクが空かチェック
    if not np.any(mask > 0):
        return []
    
    # 連結成分のラベリングを実行
    labeled_array, num_features = label(mask > 0)
    
    bboxes = []
    # 各ラベルについてバウンディングボックスを計算
    for label_id in range(1, num_features + 1):
        # 現在のラベルのマスクを作成
        current_mask = labeled_array == label_id
        
        # 行と列それぞれについて、マスクが存在する座標を取得
        rows = np.any(current_mask, axis=1)
        cols = np.any(current_mask, axis=0)
        
        # 最小と最大の座標を取得
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        bboxes.append((int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)))
    
    return bboxes

# EXIFから画像の初期設定を設定する
def set_image_param(param, exif_data):
    _, _, width, height = get_exif_image_size(exif_data)

    # イメージサイズをパラメータに入れる
    param['original_img_size'] = (width, height)
    param['img_size'] = (width, height)
    param['crop_rect'] = param.get('crop_rect', crop_editor.CropEditor.get_initial_crop_rect(width, height))
    param['crop_info'] = crop_editor.CropEditor.convert_rect_to_info(param['crop_rect'], config.get_config('preview_size')/max(param['original_img_size']))

    return (width, height)

# 上下または左右の余白を追加
def adjust_shape(img, param):
    imax = max(img.shape[1], img.shape[0])

    # イメージを正方形にする
    offset_y = (imax-img.shape[0])//2
    offset_x = (imax-img.shape[1])//2
    img = np.pad(img, ((offset_y, imax-(offset_y+img.shape[0])), (offset_x, imax-(offset_x+img.shape[1])), (0, 0)), constant_values=0)

    return img

def adjust_tone(img, highlights=0, shadows=0, midtone=0, white_level=0, black_level=0):
    """
    Lightroom風のシャドウ、ハイライト、白レベル、黒レベル調整を行う関数。
    
    Parameters:
        img (np.ndarray): 入力画像 (float32, RGB)
        shadows (float): シャドウの調整 (-100～100)
        highlights (float): ハイライトの調整 (-100～100)
        midtone (float): 中間調の調整 (-100～100)
        white_level (float): 白レベルの調整 (-100～100)
        black_level (float): 黒レベルの調整 (-100～100)
    
    Returns:
        np.ndarray: 調整後の画像（クリッピングなし）
    """

    # 中間調の調整
    midtone_scale = 4
    if midtone > 0:
        C = midtone / 100 * midtone_scale
        img = np.log(1 + img * C) / math.log(1 + C)

    elif midtone < 0:
        C = -midtone / 100 * midtone_scale
        img = (np.exp(img * math.log(1 + C)) - 1) / C

    # シャドウ（暗部）の調整
    if shadows > 0:
        factor = shadows / 100
        influence = np.exp(-5 * img)
        img = img * (1 + factor * influence)
    
    elif shadows < 0:
        factor = shadows / 100
        influence = np.exp(-5 * img)
        min_val = img * 0.1;  # 最小でも元の値の10%は維持
        raw_result = img * (1 + factor * influence)
        img = np.maximum(raw_result, min_val)
    
    # ハイライト（明部）の調整
    highlight_scale = 4
    if highlights > 0:
        factor = highlights / 100 * highlight_scale
        max_val = np.max(img)
        base = img / max_val  # 0-1に正規化
        expansion = 1 + factor * (np.log1p(np.log1p(base)) / math.log1p(math.log1p(max(max_val, 2))))
        img = img * expansion

    elif highlights < 0:
        """
        max_val = np.max(img)
        n = np.round((np.log2(0.5 / max_val) + np.log2(1.0 / max_val)) / 2)
        img = img * (2 ** n)
        """
        factor = -highlights / 100
        max_val = np.max(img)
        target = np.log1p(np.log1p(img)) / math.log1p(math.log1p(max(max_val, 2)))
        img = img * (1-factor) + target * factor

    # 黒レベル（全体の暗い部分の引き下げ）
    black_level_const = -1
    if black_level > 0:
        factor = black_level / 100
        influence = np.exp(black_level_const * img)
        img = img * (1 + factor * influence)
    
    elif black_level < 0:
        factor = black_level / 100
        influence = np.exp(black_level_const * img)
        min_val = img * 0.05;  # 最小でも元の値の5%は維持
        raw_result = img * (1 + factor * influence)
        img = np.maximum(raw_result, min_val)

    # 白レベル（全体の明るい部分の引き上げ）
    white_level_scale = 4
    if white_level > 0:
        factor = white_level / 100 * white_level_scale
        max_val = np.max(img)
        base = img / max_val  # 0-1に正規化
        expansion = 1 + factor * (np.log1p(np.log1p(np.log1p(base))) / math.log1p(math.log1p(math.log1p(max(max_val, 2)))))
        img = img * expansion

    elif white_level < 0:
        factor = -white_level / 100
        max_val = np.max(img)
        target = np.log1p(np.log1p(np.log1p(img))) / math.log1p(math.log1p(math.log1p(max(max_val, 2))))
        img = img * (1-factor) + target * factor

    return img  # クリッピングなし

def recover_saturated_pixels(rgb_data: np.ndarray, 
                           exposure_time: float,
                           iso: float,
                           wb_gains: list) -> np.ndarray:
    """
    飽和画素を修復する関数
    
    Parameters:
    -----------
    rgb_data : np.ndarray
        Shape (H, W, 3) のRGB画像データ (float32)
    exposure_time : float
        露出時間（秒）
    iso : float
        ISO感度
    wb_gains : list
    
    Returns:
    --------
    np.ndarray
        修復されたRGB画像データ
    """
    # 入力チェック
    assert rgb_data.dtype == np.float32
    assert rgb_data.ndim == 3 and rgb_data.shape[2] == 3
    
    # 定数
    SATURATION_THRESHOLD = 1.0
    MAX_SCALING = 1.5  # 最大スケーリング係数
    
    # 出力用配列の準備
    recovered = rgb_data.copy()
    
    # 飽和マスクの作成（チャンネルごと）
    saturated_mask = rgb_data >= SATURATION_THRESHOLD
    
    # 各チャンネルのゲインを配列化
    gains = np.array([wb_gains[0], wb_gains[1], wb_gains[2]])
    
    # チャンネルごとの最大理論値を計算
    theoretical_max = exposure_time * iso * gains
    theoretical_max = theoretical_max / np.max(theoretical_max)  # 正規化
    
    # 飽和ピクセルの位置を特定
    saturated_pixels = np.any(saturated_mask, axis=2)
    y_sat, x_sat = np.where(saturated_pixels)
    
    for y, x in zip(y_sat, x_sat):
        pixel_values = rgb_data[y, x]
        is_saturated = saturated_mask[y, x]
        
        # 飽和していないチャンネルの最大値を基準に補正
        unsaturated_channels = ~is_saturated
        if np.any(unsaturated_channels):
            # 飽和していないチャンネルの最大値を見つける
            max_unsaturated = np.max(pixel_values[unsaturated_channels] / 
                                   gains[unsaturated_channels])
            
            # 飽和したチャンネルの補正
            for ch in range(3):
                if is_saturated[ch]:
                    # 理論値に基づいて補正（必ず元の値より大きくなる）
                    scale = theoretical_max[ch] / theoretical_max[np.argmax(pixel_values[unsaturated_channels])]
                    corrected_value = max_unsaturated * gains[ch] * scale
                    
                    # 元の値より小さくならないように補正
                    recovered[y, x, ch] = max(
                        pixel_values[ch],
                        min(corrected_value * MAX_SCALING, MAX_SCALING)  # 上限を設定
                    )
        else:
            # すべてのチャンネルが飽和している場合
            # 理論値の比率で補正（最大値で正規化）
            max_value = np.max(pixel_values)
            for ch in range(3):
                scale = theoretical_max[ch] / np.max(theoretical_max)
                recovered[y, x, ch] = max(pixel_values[ch], max_value * scale)
    
    return recovered


def get_exif_image_size(exif_data):
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

def set_exif_image_size(exif_data, top, left, width, height):
    if exif_data.get("RawImageCropTopLeft", None) is not None:
        exif_data["RawImageCropTopLeft"] = str(top) + " " + str(left)

    if exif_data.get("RawImageCroppedSize", None) is not None:
        exif_data["RawImageCroppedSize"] = str(width) + "x" + str(height)

    exif_data["ImageSize"] = str(width) + "x" + str(height)
    

def _estimate_depth_map(img, params=(0.121779, 0.959710, -0.780245), sigma=0.5):
    """
    色線形変換先行法（Color Attenuation Prior）を使用して深度マップを推定
    
    img: 入力画像（0-1の範囲のfloat32、BGR形式）
    params: 線形モデルの係数 (β0, β1, β2)
    sigma: ガウシアンフィルタのシグマ値
    
    Zhu らの論文 "Fast Single Image Haze Removal Using Color Attenuation Prior" に基づく
    """
    # RGB画像をHSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # 彩度と明度の差分を計算
    diff = s - v
    
    # 線形モデルを使用して深度を推定: d = β0 + β1 * v + β2 * s
    beta0, beta1, beta2 = params
    depth = beta0 + beta1 * v + beta2 * s
    
    # フィルタリングで深度マップを滑らかにする
    depth = cv2.GaussianBlur(depth, (0, 0), sigma)
    
    # 正規化（0-1の範囲に変換）
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)
    
    return depth

def _estimate_atmospheric_light(img, depth_map, top_percent=0.001):
    """
    大気光を推定（最も深い点の上位N%を使用）
    
    img: 入力画像（BGR形式）
    depth_map: 深度マップ（値が大きいほど霧が濃い）
    top_percent: 使用する上位のピクセルの割合
    """
    # 画像サイズと上位N%のピクセル数を計算
    h, w = depth_map.shape
    size = h * w
    num_pixels = int(size * top_percent)
    
    # 深度マップに基づいてピクセルをソート
    indices = np.argsort(depth_map.flatten())[-num_pixels:]
    depth_pixels = np.zeros((size), dtype=bool)
    depth_pixels[indices] = True
    depth_pixels = depth_pixels.reshape(depth_map.shape)
    
    # 最も深いN%のピクセルから大気光を計算
    A = np.zeros(3, dtype=np.float32)
    for i in range(3):
        A[i] = np.mean(img[:,:,i][depth_pixels])
    
    return A

def _estimate_transmission(depth_map, strength=0.5, lower_bound=0.1):
    """
    深度マップから透過率を推定
    
    depth_map: 深度マップ（0-1の範囲）
    strength: 霞除去の強さ（0-1の範囲）
    lower_bound: 透過率の最小値
    """
    # 深度マップから透過率を計算 (t = e^(-β*d))
    beta = 1.0 * strength  # 散乱係数を強さパラメータに関連付け
    transmission = np.exp(-beta * depth_map)
    
    # 下限値を設定
    transmission = np.maximum(transmission, lower_bound)
    
    return transmission

def dehaze_image(img, strength=0.5):
    """
    色線形変換先行法を使用した霞除去・霧追加
    
    img: 入力画像（0-1の範囲のfloat32、BGR形式）
    strength: 霞除去（正の値）または霧追加（負の値）の強さ、-1から1の範囲
    """
    
    # 深度マップの推定
    depth_map = _estimate_depth_map(img)
    
    # 大気光の推定
    A = _estimate_atmospheric_light(img, depth_map)
    
    # 強さに基づいて調整
    if strength >= 0:
        # 霞除去モード
        effective_strength = strength
        # 透過率の推定
        transmission = _estimate_transmission(depth_map, effective_strength)

        # 霞補正された画像の計算（大気散乱モデル）
        result = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            result[:,:,i] = (img[:,:,i] - A[i]) / np.maximum(transmission, 0.1) + A[i]
    
    else:
        # ===== ヘイズ追加処理（霞を増やす）=====
        # Simple Atmospheric Scattering Modelを使用
        
        haze_strength = -strength  # 強度を正の値に変換
        
        # 大気光の色（純粋な白）
        atmospheric_light = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # RGB形式で白
        
        # 画像サイズを取得
        h, w = img.shape[:2]
        
        # 強度に応じて透過量を滑らかに調整
        # 強度0では透過量1.0（霞なし）
        # 強度-1では最小透過量（最大霞）
        # 二次関数的なイージングを適用して滑らかな遷移を実現
        min_trans = 0.4  # 最小透過量（最大霞）
        
        # 二次関数で滑らかな遷移を作成
        # haze_strength=0→transmission=1.0（元画像）
        # haze_strength=1→transmission=min_trans（最大霞）
        transmission_value = 1.0 - (1.0 - min_trans) * (haze_strength * haze_strength)
        
        # 均一な透過量で霞を生成
        transmission = np.ones((h, w), dtype=np.float32) * transmission_value
        
        # 透過量マップを3チャンネルに拡張
        transmission = np.stack([transmission] * 3, axis=2)
        
        # 散乱モデルによる霞の合成
        result = img * transmission + atmospheric_light * (1 - transmission)

    return result


def bicubic_kernel_vectorized(x, a=-0.5):
    """
    バイキュービック補間カーネル関数（ベクトル化版）
    
    Parameters:
    -----------
    x : numpy.ndarray
        距離値
    a : float
        バイキュービックパラメータ (デフォルト: -0.5)
        
    Returns:
    --------
    numpy.ndarray
        カーネルの重み
    """
    x = np.abs(x)
    result = np.zeros_like(x, dtype=np.float32)
    
    # |x| <= 1 の場合
    mask1 = x <= 1
    result[mask1] = ((a + 2) * x[mask1]**3 - (a + 3) * x[mask1]**2 + 1)
    
    # 1 < |x| < 2 の場合
    mask2 = np.logical_and(x > 1, x < 2)
    result[mask2] = (a * x[mask2]**3 - 5 * a * x[mask2]**2 + 8 * a * x[mask2] - 4 * a)
    
    return result

def resize_bicubic_vectorized(image, target_height, target_width):
    """
    ベクトル化されたバイキュービック補間による画像リサイズ
    
    Parameters:
    -----------
    image : numpy.ndarray
        入力画像、float32型のNumPy配列 (height, width) または (height, width, channels)
    target_height : int
        目標の高さ
    target_width : int
        目標の幅
        
    Returns:
    --------
    numpy.ndarray
        リサイズされた画像、float32型のNumPy配列
    """
    # float32型に変換
    image = image.astype(np.float32)
    
    # 画像の寸法を取得
    original_height, original_width = image.shape[:2]
    
    # グレースケール画像の処理
    is_grayscale = len(image.shape) == 2
    if is_grayscale:
        # 処理を統一するためチャンネル次元を追加
        image = image[:, :, np.newaxis]
    
    channels = image.shape[2]
    
    # スケーリング係数の計算
    if target_height > 1:
        scale_y = float(original_height - 1) / (target_height - 1)
    else:
        scale_y = 0.0
        
    if target_width > 1:
        scale_x = float(original_width - 1) / (target_width - 1)
    else:
        scale_x = 0.0
    
    # 出力ピクセル座標のグリッドを作成
    y_coords = np.arange(target_height, dtype=np.float32)
    x_coords = np.arange(target_width, dtype=np.float32)
    
    # 元画像の対応座標を計算
    y_coords = y_coords * scale_y
    x_coords = x_coords * scale_x
    
    # 整数部分と小数部分に分解
    y_floor = np.floor(y_coords).astype(np.int32)
    x_floor = np.floor(x_coords).astype(np.int32)
    
    y_frac = y_coords - y_floor
    x_frac = x_coords - x_floor
    
    # 範囲外アクセスを防ぐためのインデックス調整
    y_indices = np.zeros((target_height, 4), dtype=np.int32)
    x_indices = np.zeros((target_width, 4), dtype=np.int32)
    
    for i in range(-1, 3):
        y_idx = np.clip(y_floor + i, 0, original_height - 1)
        x_idx = np.clip(x_floor + i, 0, original_width - 1)
        y_indices[:, i+1] = y_idx
        x_indices[:, i+1] = x_idx
    
    # バイキュービックカーネルの重みを計算
    y_weights = np.zeros((target_height, 4), dtype=np.float32)
    x_weights = np.zeros((target_width, 4), dtype=np.float32)
    
    for i in range(-1, 3):
        y_dist = y_frac[:, np.newaxis] - i
        x_dist = x_frac[:, np.newaxis] - i
        y_weights[:, i+1] = bicubic_kernel_vectorized(y_dist).reshape(-1)
        x_weights[:, i+1] = bicubic_kernel_vectorized(x_dist).reshape(-1)
    
    # 出力画像の初期化
    output = np.zeros((target_height, target_width, channels), dtype=np.float32)
    
    # バイキュービック補間を適用
    for c in range(channels):
        for i in range(4):
            for j in range(4):
                # 元画像から必要な位置のピクセルを取得
                pixel_values = image[y_indices[:, i][:, np.newaxis], x_indices[:, j][np.newaxis, :], c]
                
                # 重みの行列積を計算
                weights = y_weights[:, i][:, np.newaxis] * x_weights[:, j][np.newaxis, :]
                
                # 重み付きピクセル値を出力に加算
                output[:, :, c] += pixel_values * weights
    
    # 元がグレースケールの場合、チャンネル次元を削除
    if is_grayscale:
        output = output[:, :, 0]
    
    return output

def resize_bicubic_fully_vectorized(image, target_height, target_width):
    """
    完全ベクトル化されたバイキュービック補間による画像リサイズ
    大きな画像でのメモリ使用量に注意
    
    Parameters:
    -----------
    image : numpy.ndarray
        入力画像、float32型のNumPy配列 (height, width) または (height, width, channels)
    target_height : int
        目標の高さ
    target_width : int
        目標の幅
        
    Returns:
    --------
    numpy.ndarray
        リサイズされた画像、float32型のNumPy配列
    """
    # float32型に変換
    image = image.astype(np.float32)
    
    # 画像の寸法を取得
    original_height, original_width = image.shape[:2]
    
    # グレースケール画像の処理
    is_grayscale = len(image.shape) == 2
    if is_grayscale:
        # 処理を統一するためチャンネル次元を追加
        image = image[:, :, np.newaxis]
    
    channels = image.shape[2]
    
    # スケーリング係数の計算
    if target_height > 1:
        scale_y = float(original_height - 1) / (target_height - 1)
    else:
        scale_y = 0.0
        
    if target_width > 1:
        scale_x = float(original_width - 1) / (target_width - 1)
    else:
        scale_x = 0.0
    
    # 出力ピクセル座標のグリッドを作成
    y_grid, x_grid = np.meshgrid(np.arange(target_height, dtype=np.float32),
                                 np.arange(target_width, dtype=np.float32),
                                 indexing='ij')
    
    # 元画像の対応座標を計算
    y_original = y_grid * scale_y
    x_original = x_grid * scale_x
    
    # 整数部分と小数部分に分解
    y_floor = np.floor(y_original).astype(np.int32)
    x_floor = np.floor(x_original).astype(np.int32)
    
    y_frac = y_original - y_floor
    x_frac = x_original - x_floor
    
    # 出力画像の初期化
    output = np.zeros((target_height, target_width, channels), dtype=np.float32)
    
    # 各チャンネルに対して補間を計算
    for c in range(channels):
        channel_sum = np.zeros((target_height, target_width), dtype=np.float32)
        
        # 16個の近傍点に対して重みを計算し適用
        for i in range(-1, 3):
            # y方向のインデックスとカーネル重み
            y_idx = np.clip(y_floor + i, 0, original_height - 1)
            y_dist = np.abs(y_frac - i)
            y_weight = bicubic_kernel_vectorized(y_dist)
            
            for j in range(-1, 3):
                # x方向のインデックスとカーネル重み
                x_idx = np.clip(x_floor + j, 0, original_width - 1)
                x_dist = np.abs(x_frac - j)
                x_weight = bicubic_kernel_vectorized(x_dist)
                
                # 重みの計算（ブロードキャスト）
                weight = y_weight * x_weight
                
                # 対応するピクセル値を取得
                pixel_value = image[y_idx, x_idx, c]
                
                # 重み付きピクセル値を加算
                channel_sum += pixel_value * weight
        
        output[:, :, c] = channel_sum
    
    # 元がグレースケールの場合、チャンネル次元を削除
    if is_grayscale:
        output = output[:, :, 0]
    
    return output


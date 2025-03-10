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
    コントロールポイントからLUTを生成する関数（垂直な傾きに対応）
    max_value: LUTが対応する最大値
    """
    # ソートとリスト内包表記をtogetherly処理
    point_list = sorted((pl[0], pl[1]) for pl in point_list)
    
    # unzip and convert to numpy arrays
    x, y = map(np.array, zip(*point_list))
    
    # スプライン補間の計算
    tck, u = splprep([x, y], k=min(3, len(x)-1), s=0)
    unew = np.linspace(0, 1.0, 1024*3, dtype=np.float32)
    out = splev(unew, tck)
    
    # Generate the tone curve mapping for 0-1 range
    x, y = out
    
    # 最後の点での傾きを計算
    last_point = point_list[-1]
    second_last_point = point_list[-2]
    dx = last_point[0] - second_last_point[0]
    dy = last_point[1] - second_last_point[1]
    
    # 傾きの計算（垂直な場合の処理を含む）
    if abs(dx) < 1e-6:  # 実質的に垂直な場合
        is_vertical = True
        y_at_1 = last_point[1]
        vertical_x = last_point[0]  # 垂直線のx座標
    else:
        is_vertical = False
        slope = dy / dx
        y_at_1 = np.interp(1.0, x, y)
    
    # 拡張されたレンジでLUTを生成
    lut_size = 65536
    extended_range = np.linspace(0, max_value, lut_size)
    
    # 結果を格納する配列
    lut = np.zeros(lut_size, dtype=np.float32)
    
    if is_vertical:
        # 垂直な場合の処理
        # vertical_xより左側は通常の補間
        mask_before_vertical = extended_range <= vertical_x
        lut[mask_before_vertical] = np.interp(
            extended_range[mask_before_vertical],
            x,
            y,
            left=y[0]
        ).astype(np.float32)
        
        # vertical_xより右側は最大値
        lut[~mask_before_vertical] = y_at_1
    else:
        # 通常の処理（非垂直の場合）
        mask_below_1 = extended_range <= 1.0
        
        # 1以下の値には通常の補間を適用
        lut[mask_below_1] = np.interp(
            extended_range[mask_below_1],
            x,
            y,
            left=y[0]
        ).astype(np.float32)
        
        if max_value > 1.0:
            x_above_1 = extended_range[~mask_below_1]
            
            if abs(slope) > 10:  # 急な傾き（ただし垂直でない）の場合
                # 対数関数ベースの外挿
                log_scale = np.log(slope + 1)
                rel_x = x_above_1 - 1.0
                lut[~mask_below_1] = y_at_1 + log_scale * np.log1p(rel_x * slope)
            else:  # 緩やかな傾きの場合
                # 有理関数による外挿
                scale_factor = min(5.0, max(1.0, abs(slope)))
                a = y_at_1 * scale_factor
                b = scale_factor - 1
                lut[~mask_below_1] = y_at_1 + (a * (x_above_1 - 1.0)) / (b + (x_above_1 - 1.0))
    
    # 最終的な値の範囲を確認し、必要に応じて調整
    if np.any(np.isnan(lut)) or np.any(np.isinf(lut)):
        problematic_mask = np.isnan(lut) | np.isinf(lut)
        if is_vertical:
            lut[problematic_mask] = y_at_1
        else:
            lut[problematic_mask] = y_at_1 + slope * (extended_range[problematic_mask] - 1.0)
    
    return lut

def apply_lut(img, lut, max_value=1.0):
    """
    画像にLUTを適用する関数
    max_value: LUTが対応する最大値（デフォルト1.0）
    """
    # スケーリングしてLUTのインデックスに変換
    lut_indices = ((img * 65535) / max_value).clip(0, 65535).astype(np.uint16)
    
    # LUTを適用
    result = lut[lut_indices]
    
    return result

def apply_lut_mul(img, lut, max_value=1.0):

    # スケーリングしてLUTのインデックスに変換
    lut_indices = ((img * 65535) / max_value).clip(0, 65535).astype(np.uint16)
    
    # LUTを適用
    result = lut[lut_indices] * img
    
    return result

def apply_lut_add(img, lut, max_value=1.0):

    # スケーリングしてLUTのインデックスに変換
    lut_indices = ((img * 65535) / max_value).clip(0, 65535).astype(np.uint16)
    
    # LUTを適用
    result = lut[lut_indices] + img
    
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
    hue = hls_img[..., 0]
    
    # 赤色の条件式
    red_condition = jnp.logical_or(
        jnp.logical_and(hue >= 0, hue < 22.5),
        jnp.logical_and(hue >= 337.5, hue < 360)
    )
    
    return __adjust_hls(hls_img, red_condition, red_adjust)


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

        # クリック位置を中心にする
        crop_x = crop_info[0] + click_image_x - crop_width // 2
        crop_y = crop_info[1] + click_image_y - crop_height // 2

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
        
        bboxes.append((x_min, y_min, x_max-x_min, y_max-y_min))
    
    return bboxes

def adjust_shape(img, param):
    imax = max(img.shape[1], img.shape[0])

    # イメージサイズをパラメータに入れる
    param['original_img_size'] = (img.shape[1], img.shape[0])
    param['img_size'] = (img.shape[1], img.shape[0])
    param['crop_rect'] = param.get('crop_rect', crop_editor.CropEditor.get_initial_crop_rect(img.shape[1], img.shape[0]))
    param['crop_info'] = crop_editor.CropEditor.convert_rect_to_info(param['crop_rect'], config.get_config('preview_size')/max(param['original_img_size']))

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

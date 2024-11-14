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

import sigmoid
import dng_sdk

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

# 色飽和の復元
def restore_saturated_colors(image):
    # Detect saturated channels
    threshold = 0.98  # threshold to detect saturation in 8-bit images
    mask_r = (image[:, :, 2] >= threshold)
    mask_g = (image[:, :, 1] >= threshold)
    mask_b = (image[:, :, 0] >= threshold)
    
    # Create a combined mask for pixels with one or two channels saturated
    saturation_mask = (mask_r + mask_g + mask_b) >= 1

    # Restore values using neighboring pixel information
    restored_image = image.copy()
    for channel in range(3):
        mask = (image[:, :, channel] >= threshold)
        restored_image[:, :, channel][mask] = np.median(image[:, :, channel][~saturation_mask])
    
    return restored_image

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

# ガンマ補正
def apply_gamma(img, gamma):

    return img**gamma

def convert_RGB2TempTint(rgb):

    xyz = colour.RGB_to_XYZ(rgb, 'sRGB')

    xy = colour.XYZ_to_xy(xyz)

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.set_xy_coord(xy)

    return float(dng.fTemperature), float(dng.fTint), float(xyz[1])

def __invert_temp_tint(temp, tint, ref_temp=6500.0):

    # 色温度の反転
    mired_temp = 1e6 / temp
    mired_ref = 1e6 / ref_temp
    inverted_temp = 1e6 / (mired_ref - (mired_temp - mired_ref) + sys.float_info.min)

    # ティントの反転
    inverted_tint = -tint

    return inverted_temp, inverted_tint

def invert_RGB2TempTint(rgb, ref_temp=6500.0):
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

def invert_TempTint2RGB(temp, tint, Y, reference_temp):

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

def rotation(img, angle):
    
    # 変換後の画像高さを定義
    height = img.shape[0]
    # 変換後の画像幅を定義
    width = img.shape[1]
    # 回転の軸を指定:今回は中心
    center = (int(width/2), int(height/2))
    # scaleを指定
    scale = 1
    
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotate_affine = cv2.warpAffine(img, trans, (width, height), flags=cv2.INTER_CUBIC)

    return img_rotate_affine

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
    print(kernel)

    dest = []
    for i in range(src.shape[2]):
        dest.append(jscipy.signal.convolve2d(src[:, :, i], kernel, mode='same'))

    dest = jnp.stack(dest, -1)

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

def modify_lens(img, exif_data):
    db = lensfunpy.Database()
    print(exif_data['Make'], exif_data['Model'])
    print(exif_data['LensMake'], exif_data['LensModel'])
    print(exif_data['FocalLength'], exif_data['ApertureValue'])

    cam = db.find_cameras(exif_data['Make'], exif_data['Model'], loose_search=True)
    print(cam)
    lens = db.find_lenses(cam[0], exif_data['LensMake'], exif_data['LensModel'], loose_search=True)
    print(lens)

    height, width = img.shape[0], img.shape[1]
    mod = lensfunpy.Modifier(lens[0], cam[0].crop_factor, width, height)
    mod.initialize(float(exif_data['FocalLength'][0:-3]), exif_data['ApertureValue'], pixel_format=np.float32)
    
    modimg = img.copy()
    did_apply = mod.apply_color_modification(modimg)

    undist_coords = mod.apply_subpixel_distortion()
    modimg[..., 0] = cv2.remap(modimg[..., 0], undist_coords[..., 0, :], None, cv2.INTER_LANCZOS4)
    modimg[..., 1] = cv2.remap(modimg[..., 1], undist_coords[..., 1, :], None, cv2.INTER_LANCZOS4)
    modimg[..., 2] = cv2.remap(modimg[..., 2], undist_coords[..., 2, :], None, cv2.INTER_LANCZOS4)

    undist_coords = mod.apply_geometry_distortion()
    modimg = cv2.remap(modimg, undist_coords, None, cv2.INTER_LANCZOS4)

    return modimg


# 露出補正
def adjust_exposure(img, ev):
    # img: 変換元画像
    # ev: 補正値 -4.0〜4.0

    #img2 = img*(2.0**ev)

    return (2.0**ev)


# コントラスト補正
def adjust_contrast(img, cf, c):
    # img: 変換元画像
    # cf: コントラストファクター -100.0〜100.0

    #c = 0.5   # 中心値
    f = cf/100.0*10.0  #-10.0〜10.0に変換

    if f == 0.0:
        adjust_img = img.copy()
    elif f >= 0.0:
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

def adjust_shadow(img, black):
    f = -black/100.0*5.0

    if f == 0.0:
        adjust_img = img.copy()
    elif f > 0.0:
        adjust_img = sigmoid.scaled_sigmoid(img, f, 0.80)
    else:
        adjust_img = sigmoid.scaled_inverse_sigmoid(img, f, 0.80)

    return adjust_img

def adjust_highlight(img, white):
    f = white/100.0*5.0

    if f == 0.0:
        adjust_img = img.copy()
    elif f > 0.0:
        adjust_img = sigmoid.scaled_sigmoid(img, f, 0.2)
    else:
        adjust_img = sigmoid.scaled_inverse_sigmoid(img, f, 0.2)

    return adjust_img

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
def __adjust_saturation(s, sat, vib):

    # 彩度変更値と自然な彩度変更値を計算
    if sat >= 0:
        sat = 1.0 + sat/100.0
    else:
        sat = 1.0 + sat/100.0
    vib /= 50.0

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

def adjust_saturation(s, sat, vib):
    array = __adjust_saturation(s, sat, vib)
    array.block_until_ready()

    return np.array(array)


# スプラインカーブの適用
def calc_point_list_to_lut(img, point_list):
    # ソートとリスト内包表記をtogetherly処理
    point_list = sorted((pl[0], pl[1]) for pl in point_list)
    
    # unzip and convert to numpy arrays
    x, y = map(np.array, zip(*point_list))
    
    # スプライン補間の計算
    tck, u = splprep([x, y], k=min(3, len(x)-1), s=0)
    unew = np.linspace(0, 1.0, 1024, dtype=np.float32)
    out = splev(unew, tck)
    
    # Generate the tone curve mapping
    x, y = out
    
    # 拡張されたレンジでLUTを生成（0〜2の範囲をカバー）
    extended_range = np.linspace(0, 2.0, 131072)  # 2倍のレンジ
    lut = np.interp(extended_range, x, y, left=y[0], right=y[-1]).astype(np.float32)
    
    return lut

def apply_lut(img, lut):
    if lut is None:
        return img
    
    # 入力画像の値を適切にクリップ
    img_clipped = np.clip(img, 0, 2.0)
    
    # スケーリングしてLUTのインデックスに変換
    lut_indices = (img_clipped * 65535).astype(np.uint16)
    
    # LUTを適用
    result = lut[lut_indices]
    
    return result

def __adjust_hls(hls_img, mask, adjust):
    hls = np.copy(hls_img)
    hls[mask, 0] = hls_img[mask, 0] + adjust[0]*1.8
    hls[mask, 1] = hls_img[mask, 1] * (2.0**adjust[1])
    hls[mask, 2] = hls_img[mask, 2] * (2.0**adjust[2])
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

def adjust_shadow_highlight(image, highlight_adjustment=0, shadow_adjustment=0):
    # 調整パラメータを [-1, 1] の範囲にスケーリング
    highlight_adjustment = np.clip(highlight_adjustment / 300, -1, 1)
    shadow_adjustment = np.clip(shadow_adjustment / 300, -1, 1)
    
    # ハイライト補正：山を押しつぶすようにトーンを変化
    def highlight_function(x):
        center, spread = 0.65, 0.22  # ハイライトの中心と広がり
        compression_effect = 1 - highlight_adjustment * np.exp(-((x - center) / spread) ** 2)
        adjusted_x = x * compression_effect + highlight_adjustment * np.exp(-((x - center) / (spread * 2)) ** 2)
        return np.where((x == 2.0) | (x == 0.0), x, adjusted_x)  # 完全な黒と白には影響を与えない

    # シャドウ補正：山を押しつぶすようにトーンを変化
    def shadow_function(x):
        center, spread = 0.35, 0.22  # シャドウの中心と広がり
        compression_effect = 1 + shadow_adjustment * np.exp(-((x - center) / spread) ** 2)
        adjusted_x = x * compression_effect + shadow_adjustment * np.exp(-((x - center) / (spread * 2)) ** 2)
        return np.where((x == 2.0) | (x == 0.0), x, adjusted_x)  # 完全な黒と白には影響を与えない
    
    # ハイライトとシャドウの範囲ごとに補正を適用
    highlights_adjusted = highlight_function(image)
    shadows_adjusted = shadow_function(image)
    
    # 両方の補正を平均して自然なトーンに調整
    adjusted_image = (highlights_adjusted + shadows_adjusted) / 2
    
    # 補正範囲を [0, 2] に制限
    return np.clip(adjusted_image, 0, 2)


# マスクイメージの適用
@jit
def __apply_mask(img1, msk, img2):

    if msk is not None:
        img = msk[:, :, jnp.newaxis] * img1 + (1.0 - msk[:, :, jnp.newaxis]) * img2

    return img

def apply_mask(img1, msk, img2):
    array = __apply_mask(img1, msk, img2)
    array.block_until_ready()

    return np.array(array)


#@partial(jit, static_argnums=(1,2,3,4,5,6))
def crop_image(image, texture_width, texture_height, click_x, click_y, offset, is_zoomed):
    # 画像のサイズを取得
    image_height, image_width = image.shape[:2]

    # アスペクト比を計算
    image_aspect = image_width / image_height
    texture_aspect = texture_width / texture_height

    if image_aspect > texture_aspect:
        # 画像が横長の場合
        new_width = texture_width
        new_height = int(texture_width / image_aspect)
    else:
        # 画像が縦長の場合
        new_width = int(texture_height * image_aspect)
        new_height = texture_height

    # 中央に配置するためのオフセットを計算
    offset_x = (texture_width - new_width) // 2
    offset_y = (texture_height - new_height) // 2

    # スケールを求める
    if image_width >= image_height:
        scale = texture_width/image_width
    else:
        scale = texture_height/image_height

    if not is_zoomed:
        # リサイズ
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        #resized_img = jax.image.resize(image, (new_height, new_width, 3), method="lanczos3")

        # 背景を作成（透明な黒）
        #result = jnp.zeros((texture_height, texture_width, 3), dtype=np.float32)

        # リサイズした画像を中央に配置
        #result[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_img
        result = np.pad(resized_img, ((offset_y, texture_height-(offset_y+new_height)), (offset_x, texture_width-(offset_x+new_width)), (0, 0)))

        crop_info = [0, 0, image_width, image_height, scale]

    else:
        # クリック位置を元の画像の座標系に変換
        click_x = click_x - offset_x
        click_y = click_y - offset_y
        click_image_x = int(click_x / scale)
        click_image_y = int(click_y / scale)

        # 切り抜き範囲を計算
        crop_width = int(texture_width)
        crop_height = int(texture_height)

        # クリック位置を中心にする
        crop_x = click_image_x - crop_width // 2
        crop_y = click_image_y - crop_height // 2

        # クロップ
        result, crop_info = crop_image_info(image, [crop_x, crop_y, crop_width, crop_height, 1.0], offset)
    
    return result, crop_info

def crop_image_info(image, crop_info, offset=(0, 0)):
    
    # 情報取得
    image_height, image_width = image.shape[:2]
    crop_x, crop_y, crop_width, crop_height, scale = crop_info

    # オフセット適用
    crop_x += int(offset[0])
    crop_y += int(offset[1])

    # 画像の範囲外にならないように調整
    crop_x = max(0, min(crop_x, image_width - crop_width))
    crop_y = max(0, min(crop_y, image_height - crop_height))

    # 画像を切り抜く
    cropped_img = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    return cropped_img, [crop_x, crop_y, crop_width, crop_height, scale]

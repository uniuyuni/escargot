import cv2
import numpy as np
import colour
import lensfunpy
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import splprep, splev

import sigmoid
import dng_sdk

# RGBからグレイスケールへの変換
def cvtToGrayColor(rgb):
    # 変換元画像 RGB
    gry = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # gry = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    return gry

# ガンマ補正
def apply_gamma(img, gamma):

    return img**gamma

def convert_RGB2TempTint(rgb):

    xyz = colour.RGB_to_XYZ(rgb, 'sRGB')

    xy = colour.XYZ_to_xy(xyz)

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.set_xy_coord(xy)

    return dng.fTemperature, dng.fTint, xyz[1]

def convert_TempTint2RGB(temp, tint, Y):

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.fTemperature = temp
    dng.fTint = tint

    xy = dng.get_xy_coord()

    xyz = colour.xy_to_XYZ(xy)
    xyz *= Y

    rgb = colour.XYZ_to_RGB(xyz, 'sRGB')

    return rgb

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


def lucy_richardson_gauss(srcf, iteration):

    # 出力用の画像を初期化
    destf = srcf.copy()

    for i in range(iteration):
        # ガウスぼかしを適用してぼけをシミュレーション
        bdest = cv2.GaussianBlur(destf, ksize=(9, 9), sigmaX=0)

        # 元画像とぼけた画像の比を計算
        ratio = np.divide(srcf, bdest, where=(bdest!=0))

        # 誤差の分配のために再びガウスぼかしを適用
        ratio_blur = cv2.GaussianBlur(ratio, ksize=(9, 9), sigmaX=0)

        # 元の出力画像に誤差を乗算
        destf = cv2.multiply(destf, ratio_blur)
    
    return destf

# ローパスフィルタ
def lowpass_filter(img, r):
    lpf = cv2.GaussianBlur(img, ksize=(r, r), sigmaX=0.0)

    return lpf

# ハイパスフィルタ
def highpass_filter(img, r):
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
def adjust_contrast(img, cf):
    # img: 変換元画像
    # cf: コントラストファクター -100.0〜100.0

    c = 0.5   # 中心値
    f = cf/100.0*10.0  #-10.0〜10.0に変換

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

def adjust_shadow(img, black):
    f = -black/100.0*5.0

    if f == 0.0:
        adjust_img = img.copy()
    elif f > 0.0:
        adjust_img = sigmoid.scaled_sigmoid(img, f, 0.90)
    else:
        adjust_img = sigmoid.scaled_inverse_sigmoid(img, f, 0.90)

    return adjust_img

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
def adjust_saturation(s, sat, vib):

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
        final_s = np.log(1.0 + vib * s) / np.log(1.0 + vib)
    else:
        # 逆関数を使用
        vib = vib**2
        final_s = (np.exp(s * np.log(1.0 + vib)) - 1.0) / vib

    # 彩度を適用
    final_s = final_s * sat

    return final_s

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
    lut = np.interp(np.arange(65536*2), x*65535, y*65535).astype(np.float32) #0.~2.

    # Apply the tone curve mapping to the image
    img2 = lut[(img*65535).astype(np.uint16)]
    img2 = img2/65535

    return img2

def __adjust_hls(hls_img, mask, adjust):
    hls = hls_img.copy()
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

def adjust_density(hls_img, intensity):
    hls = np.zeros_like(hls_img)

    # 濃さ
    intensity = -intensity
    hls[:, :, 0] = hls_img[:, :, 0]
    hls[:, :, 1] = adjust_shadow(hls_img[:, :, 1], intensity/2)
    hls[:, :, 1] = adjust_hilight(hls[:, :, 1], intensity/2)
    hls[:, :, 2] = hls_img[:, :, 2] * (1.0 - intensity / 100.0)

    return hls

def adjust_clear_color(rgb_img, intensity):

    # 清書色、濁色
    if intensity >= 0:
        rgb = np.clip(rgb_img * (1 + intensity / 100.0), 0.0, 1.0)
    else:
        gray = cvtToGrayColor(rgb_img)
        factor = -intensity / 200.0
        rgb = rgb_img * (1 - factor) + gray[..., np.newaxis] * factor
        rgb = rgb * (1 - factor * 0.3)
        rgb = np.clip(rgb, 0.0, 1.0)

    return rgb
           
# マスクイメージの適用
def apply_mask(img1, msk, img2):

    if msk is not None:
        img = msk[:, :, np.newaxis] * img1 + (1.0 - msk[:, :, np.newaxis]) * img2

    return img

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
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # 背景を作成（透明な黒）
        result = np.zeros((texture_height, texture_width, 3), dtype=np.float32)

        # リサイズした画像を中央に配置
        result[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_img
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

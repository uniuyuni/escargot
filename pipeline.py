
import math
import cv2
import numpy as np

import core

def process_pipeline(img, crop_info, offset, crop_image, is_zoomed, texture_width, texture_height, click_x, click_y, primary_effects, primary_param, mask_editor):

    # 背景レイヤー
    img0, reset = pipeline_lv0(img, primary_effects, primary_param)
    if crop_image is None or reset == True:
        if is_zoomed:
            imgc, crop_info2 = core.crop_image_info(img0, crop_info, offset)
        else:
            imgc, crop_info2 = core.crop_image(img0, texture_width, texture_height, click_x, click_y, offset, is_zoomed)
        mask_editor.set_image(img0, texture_width, texture_height, crop_info2, math.radians(primary_param.get('rotation', 0)), -1)
    else:
        imgc = crop_image
        crop_info2 = crop_info

    # 並列処理
    #split_img = []
    #split_img.extend(np.vsplit(imgc, 4))
    #height = 1024//4
    #for i, img in enumerate(split_img):
    #    split_img[i] = MainWidget._process_pipeline2(img, height*i, height, primary_effects, primary_param, mask_editor)
    #result_img = joblib.Parallel(n_jobs=-1, require='sharedmem')(joblib.delayed(MainWidget._process_pipeline2)(img, height*i, height, primary_effects, primary_param, mask_editor) for i, img in enumerate(split_img))
    #img2 = np.vstack(result_img)        
    img2 = _process_pipeline2(imgc, 0, 1024, primary_effects, primary_param, mask_editor)

    return img2, imgc, crop_info2

def _process_pipeline2(imgc, slice_y, slice_h, primary_effects, primary_param, mask_editor):
    img1 = pipeline_lv1(imgc, primary_effects, primary_param)
    img2 = pipeline_lv2(img1, primary_effects, primary_param)
    img3 = pipeline_lv3(img2, primary_effects, primary_param)

    # マスクレイヤー
    img2 = img3
    mask_list = mask_editor.get_layers_list()
    for mask in mask_list:
        img1 = pipeline_lv1(img2, mask.effects, mask.effects_param)
        img2 = pipeline_lv2(img1, mask.effects, mask.effects_param)

        img2 = core.apply_mask(img2, mask.get_mask_image()[slice_y:slice_y+slice_h, :], img3)

    return img2

def pipeline_lv0(img, effects, param):
    lv0 = effects[0]
    lv1reset = False

    rgb = img
    for i, n in enumerate(lv0):
        if lv1reset == True:
            lv0[n].reeffect()
            
        pre_diff = lv0[n].diff
        diff = lv0[n].make_diff(rgb, param)
        if diff is not None:
            rgb = lv0[n].apply_diff(rgb)

        if pre_diff is not diff:
            lv1reset = True

    if lv1reset == True:
        for v in effects[1].values():
            v.reeffect()
        for v in effects[2].values():
            v.reeffect()

    return rgb, lv1reset

def pipeline_lv1(img, effects, param):
    lv1 = effects[1]
    lv2reset = False

    rgb = img.copy()
    for i, n in enumerate(lv1):
        if lv2reset == True:
            lv1[n].reeffect()
            
        pre_diff = lv1[n].diff
        diff = lv1[n].make_diff(rgb, param)
        if diff is not None:
            rgb = diff

        if pre_diff is not diff:
            lv2reset = True
            
    if lv2reset == True:
        for l in effects[2].values():
            l.reeffect()

    return rgb


def pipeline_lv2(img, effects, param):
    lv2 = effects[2]

    rgb = img.copy()

    diff = lv2['color_temperature'].make_diff(rgb, param)
    if diff is not None: rgb = lv2['color_temperature'].apply_diff(rgb)

    # 以降HLS
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)

    # Lのみ
    #hls_l = hls[:, :, 1]
    #hls2_l = hls_l.copy()

    # HLS
    hls2 = hls.copy()
    diff = lv2['hls_red'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_red'].apply_diff(hls2)
    diff = lv2['hls_orange'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_orange'].apply_diff(hls2)
    diff = lv2['hls_yellow'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_yellow'].apply_diff(hls2)
    diff = lv2['hls_green'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_green'].apply_diff(hls2)
    diff = lv2['hls_cyan'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_cyan'].apply_diff(hls2)
    diff = lv2['hls_blue'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_blue'].apply_diff(hls2)
    diff = lv2['hls_purple'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_purple'].apply_diff(hls2)
    diff = lv2['hls_magenta'].make_diff(hls, param)
    if diff is not None: hls2 = lv2['hls_magenta'].apply_diff(hls2)
    hls = np.array(hls2)

    # Hのみ
    hls_h = hls[:, :, 0]
    hls2_h = hls_h.copy()
    diff = lv2['HuevsHue'].make_diff(hls_h, param)
    if diff is not None: hls2_h = lv2['HuevsHue'].apply_diff(hls2_h)
    hls[:, :, 0] = hls2_h

    #　Lのみ
    hls_l = hls[:, :, 1]
    hls2_l = hls_l.copy()
    diff = lv2['HuevsLum'].make_diff(hls_l, param)
    if diff is not None: hls2_l = lv2['HuevsLum'].apply_diff(hls2_l)
    diff = lv2['LumvsLum'].make_diff(hls_l, param)
    if diff is not None: hls2_l = lv2['LumvsLum'].apply_diff(hls2_l)
    diff = lv2['SatvsLum'].make_diff(hls_l, param)
    if diff is not None: hls2_l = lv2['SatvsLum'].apply_diff(hls2_l)
    hls[:, :, 1] = hls2_l

    # Sのみ
    hls_s = hls[:, :, 2]
    hls2_s = hls_s.copy()
    diff = lv2['HuevsSat'].make_diff(hls_s, param)
    if diff is not None: hls2_s = lv2['HuevsSat'].apply_diff(hls2_s)
    diff = lv2['LumvsSat'].make_diff(hls_s, param)
    if diff is not None: hls2_s = lv2['LumvsSat'].apply_diff(hls2_s)
    diff = lv2['SatvsSat'].make_diff(hls_s, param)
    if diff is not None: hls2_s = lv2['SatvsSat'].apply_diff(hls2_s)
    diff = lv2['saturation'].make_diff(hls2_s, param)

    if diff is not None: hls2_s = lv2['saturation'].apply_diff(hls2_s)
    hls[:, :, 2] = hls2_s

    # 合成
    #hls[:,:,1] = np.clip(hls[:,:,1], 0, 1.0)
    #hls[:,:,2] = np.clip(hls[:,:,2], 0, 1.0)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)

    # RGB
    diff = lv2['color_correct'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['dehaze'].make_diff(rgb, param)
    if diff is not None: rgb += diff

    rgb2 = rgb.copy()
    diff = lv2['tonecurve'].make_diff(rgb, param)
    if diff is not None: rgb2 = lv2['tonecurve'].apply_diff(rgb2)
    diff = lv2['tonecurve_red'].make_diff(rgb, param)
    if diff is not None: rgb2[:,:,0] = lv2['tonecurve_red'].apply_diff(rgb2[:,:,0])
    diff = lv2['tonecurve_green'].make_diff(rgb, param)
    if diff is not None: rgb2[:,:,1] = lv2['tonecurve_green'].apply_diff(rgb2[:,:,1])
    diff = lv2['tonecurve_blue'].make_diff(rgb, param)
    if diff is not None: rgb2[:,:,2] = lv2['tonecurve_blue'].apply_diff(rgb2[:,:,2])
    diff = lv2['grading1'].make_diff(rgb, param)
    if diff is not None: rgb2 = lv2['grading1'].apply_diff(rgb2)
    diff = lv2['grading2'].make_diff(rgb, param)
    if diff is not None: rgb2 = lv2['grading2'].apply_diff(rgb2)
    rgb = rgb2

    diff = lv2['exposure'].make_diff(rgb, param)
    if diff is not None: rgb = lv2['exposure'].apply_diff(rgb)
    diff = lv2['contrast'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['microcontrast'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['midtone'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['tone'].make_diff(rgb, param)
    if diff is not None: rgb += diff
    diff = lv2['level'].make_diff(rgb, param)
    if diff is not None: rgb += diff

    diff = lv2['lut'].make_diff(rgb, param)
    if diff is not None: rgb += diff

    #rgb = np.clip(rgb, 0, 1.0)
    return rgb

def pipeline_lv3(rgb, effects, param):
    lv3 = effects[3]

    for i, n in enumerate(lv3):            
        diff = lv3[n].make_diff(rgb, param)
        if diff is not None:
            rgb = lv3[n].apply_diff(rgb)

    return rgb


import os
import numpy as np
import torch
import cv2
import time
import splitimage

# 実行パス違い吸収
_cwd = os.getcwd()
if 'SCUNet' in os.getcwd():
    from models.network_scunet import SCUNet
else:
    from .models.network_scunet import SCUNet
    _cwd = os.path.join(_cwd, "SCUNet")

def setup_model(model_path=os.path.join(_cwd, "model_zoo/scunet_color_real_gan.pth"), device='cpu'):
    """モデルを初期化してロードする"""
    model = SCUNet(in_nc=3, config=[4]*7, dim=64)  # カラー入力用
    model.load_state_dict(torch.load(model_path))
    model = model.eval().to(device)
    return model

def denoise_image(model, np_image, device='cpu'):
    """
    numpy画像（float32, [0,1]範囲）をデノイズ
    入力: (H,W,3)のnumpy配列
    出力: (H,W,3)のnumpy配列
    """
    # 前処理
    tensor_img = torch.from_numpy(np_image.transpose(2,0,1))  # CHWに変換
    tensor_img = tensor_img.unsqueeze(0).float().to(device)  # バッチ次元追加

    # 推論
    with torch.no_grad():
        denoised = model(tensor_img)

    # 後処理
    denoised_np = denoised.squeeze().cpu().numpy().transpose(1,2,0)
    return np.clip(denoised_np, 0, 1).astype(np.float32)

def denoise_image_helper(model, np_image, device='cpu'):

    split_images, split_info = splitimage.split_image_with_overlap(np_image, 1024, 1024, 32)

    denoised_images = []
    for i, image in enumerate(split_images):
        print(f"SCUNet Predict {i+1} / {len(split_images)}")
        denoised_images.append(denoise_image(model, image, device))

    result = splitimage.combine_image_with_overlap(denoised_images, split_info)
    print("Completed with SCUNet")

    return result

if __name__ == "__main__":
    # 入力画像読み込み（例：512x512 RGB画像）
    input_rgb = cv2.imread("DSCF6765-small.jpg", cv2.IMREAD_COLOR_RGB).astype(np.float32) / 255.0
        
    # モデルセットアップ
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print( "Device set " + device)
    model = setup_model(device=device)
    
    # 時間計測
    starttime = time.time()

    # 実行
    result = denoise_image_helper(model, input_rgb, device)

    print(f"実行時間: {time.time()-starttime:.6f}秒")    
    
    # 結果表示
    view_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("result.jpg", view_bgr)
    cv2.imshow("Result", view_bgr)
    cv2.waitKey(-1)

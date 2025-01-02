import numpy as np
from scipy.interpolate import interp1d
import cv2

class FilmSimulation:
    """フィルムシミュレーションクラス"""
    
    # 実際のフィルムをベースにしたプリセット
    FILM_PRESETS = {
        "portra400": {
            "tone_curves": {
                "r": [(0, 0.05), (0.25, 0.27), (0.5, 0.55), (0.75, 0.82), (1, 0.95)],
                "g": [(0, 0.05), (0.25, 0.28), (0.5, 0.54), (0.75, 0.81), (1, 0.94)],
                "b": [(0, 0.06), (0.25, 0.26), (0.5, 0.53), (0.75, 0.80), (1, 0.93)]
            },
            "color_balance": [1.05, 1.0, 0.95],
            "saturation": 1.1,
            "contrast": 1.15,
            "grain": {"intensity": 0.35, "size": 1.5},
            "shadows": {"lift": 0.05, "gamma": 1.1},
            "highlights": {"compression": 0.8, "rolloff": 0.7}
        },
        "ektar100": {
            "tone_curves": {
                "r": [(0, 0.02), (0.25, 0.28), (0.5, 0.57), (0.75, 0.85), (1, 0.98)],
                "g": [(0, 0.02), (0.25, 0.27), (0.5, 0.56), (0.75, 0.84), (1, 0.97)],
                "b": [(0, 0.03), (0.25, 0.26), (0.5, 0.55), (0.75, 0.83), (1, 0.96)]
            },
            "color_balance": [1.1, 1.0, 0.9],
            "saturation": 1.3,
            "contrast": 1.25,
            "grain": {"intensity": 0.2, "size": 1.2},
            "shadows": {"lift": 0.02, "gamma": 1.2},
            "highlights": {"compression": 0.9, "rolloff": 0.8}
        },
        "velvia50": {
            "tone_curves": {
                "r": [(0, 0.02), (0.25, 0.29), (0.5, 0.58), (0.75, 0.86), (1, 0.99)],
                "g": [(0, 0.02), (0.25, 0.28), (0.5, 0.57), (0.75, 0.85), (1, 0.98)],
                "b": [(0, 0.03), (0.25, 0.27), (0.5, 0.56), (0.75, 0.84), (1, 0.97)]
            },
            "color_balance": [1.15, 1.05, 1.0],
            "saturation": 1.5,
            "contrast": 1.4,
            "grain": {"intensity": 0.15, "size": 1.0},
            "shadows": {"lift": 0.02, "gamma": 1.3},
            "highlights": {"compression": 0.95, "rolloff": 0.85}
        },
        "provia100f": {
            "tone_curves": {
                "r": [(0, 0.03), (0.25, 0.26), (0.5, 0.54), (0.75, 0.82), (1, 0.96)],
                "g": [(0, 0.03), (0.25, 0.26), (0.5, 0.53), (0.75, 0.81), (1, 0.95)],
                "b": [(0, 0.04), (0.25, 0.25), (0.5, 0.52), (0.75, 0.80), (1, 0.94)]
            },
            "color_balance": [1.0, 1.0, 1.05],
            "saturation": 1.2,
            "contrast": 1.2,
            "grain": {"intensity": 0.25, "size": 1.3},
            "shadows": {"lift": 0.03, "gamma": 1.15},
            "highlights": {"compression": 0.85, "rolloff": 0.75}
        },
        "tmax400": {
            "tone_curves": {
                "r": [(0, 0.04), (0.25, 0.27), (0.5, 0.52), (0.75, 0.80), (1, 0.94)],
                "g": [(0, 0.04), (0.25, 0.27), (0.5, 0.52), (0.75, 0.80), (1, 0.94)],
                "b": [(0, 0.04), (0.25, 0.27), (0.5, 0.52), (0.75, 0.80), (1, 0.94)]
            },
            "color_balance": [1.0, 1.0, 1.0],
            "saturation": 0,
            "contrast": 1.3,
            "grain": {"intensity": 0.45, "size": 1.8},
            "shadows": {"lift": 0.04, "gamma": 1.2},
            "highlights": {"compression": 0.8, "rolloff": 0.7}
        },
        "cinestill800t": {
            "tone_curves": {
                "r": [(0, 0.05), (0.25, 0.25), (0.5, 0.52), (0.75, 0.81), (1, 0.95)],
                "g": [(0, 0.06), (0.25, 0.26), (0.5, 0.53), (0.75, 0.82), (1, 0.96)],
                "b": [(0, 0.07), (0.25, 0.28), (0.5, 0.55), (0.75, 0.84), (1, 0.98)]
            },
            "color_balance": [0.9, 1.0, 1.2],
            "saturation": 1.1,
            "contrast": 1.15,
            "grain": {"intensity": 0.4, "size": 1.6},
            "shadows": {"lift": 0.05, "gamma": 1.1},
            "highlights": {"compression": 0.75, "rolloff": 0.65}
        },
        "fujicolor_pro400h": {
            "tone_curves": {
                "r": [(0, 0.04), (0.25, 0.26), (0.5, 0.53), (0.75, 0.81), (1, 0.95)],
                "g": [(0, 0.04), (0.25, 0.27), (0.5, 0.54), (0.75, 0.82), (1, 0.96)],
                "b": [(0, 0.05), (0.25, 0.25), (0.5, 0.52), (0.75, 0.80), (1, 0.94)]
            },
            "color_balance": [1.02, 1.0, 0.98],
            "saturation": 1.15,
            "contrast": 1.12,
            "grain": {"intensity": 0.35, "size": 1.5},
            "shadows": {"lift": 0.04, "gamma": 1.1},
            "highlights": {"compression": 0.82, "rolloff": 0.72}
        },
        "portra160": {
            "tone_curves": {
                "r": [(0, 0.03), (0.25, 0.26), (0.5, 0.54), (0.75, 0.81), (1, 0.94)],
                "g": [(0, 0.03), (0.25, 0.27), (0.5, 0.55), (0.75, 0.82), (1, 0.95)],
                "b": [(0, 0.04), (0.25, 0.25), (0.5, 0.53), (0.75, 0.80), (1, 0.93)]
            },
            "color_balance": [1.03, 1.0, 0.97],
            "saturation": 1.05,
            "contrast": 1.1,
            "grain": {"intensity": 0.25, "size": 1.3},
            "shadows": {"lift": 0.03, "gamma": 1.05},
            "highlights": {"compression": 0.85, "rolloff": 0.75}
        },
        "acros100": {
            "tone_curves": {
                "r": [(0, 0.02), (0.25, 0.25), (0.5, 0.53), (0.75, 0.82), (1, 0.96)],
                "g": [(0, 0.02), (0.25, 0.25), (0.5, 0.53), (0.75, 0.82), (1, 0.96)],
                "b": [(0, 0.02), (0.25, 0.25), (0.5, 0.53), (0.75, 0.82), (1, 0.96)]
            },
            "color_balance": [1.0, 1.0, 1.0],
            "saturation": 0,
            "contrast": 1.25,
            "grain": {"intensity": 0.2, "size": 1.2},
            "shadows": {"lift": 0.02, "gamma": 1.15},
            "highlights": {"compression": 0.9, "rolloff": 0.8}
        },
        "ektachrome100": {
            "tone_curves": {
                "r": [(0, 0.03), (0.25, 0.27), (0.5, 0.55), (0.75, 0.83), (1, 0.97)],
                "g": [(0, 0.03), (0.25, 0.26), (0.5, 0.54), (0.75, 0.82), (1, 0.96)],
                "b": [(0, 0.04), (0.25, 0.25), (0.5, 0.53), (0.75, 0.81), (1, 0.95)]
            },
            "color_balance": [1.05, 1.0, 1.02],
            "saturation": 1.2,
            "contrast": 1.18,
            "grain": {"intensity": 0.22, "size": 1.25},
            "shadows": {"lift": 0.03, "gamma": 1.12},
            "highlights": {"compression": 0.88, "rolloff": 0.78}
        }
    }
    
    def __init__(self, film_type):
        """初期化"""
        if film_type not in self.FILM_PRESETS:
            raise ValueError(f"Unknown film type: {film_type}")
        self.preset = self.FILM_PRESETS[film_type]
        self._create_tone_curves()

    def _create_tone_curves(self):
        """トーンカーブの補間関数を作成"""
        curves = self.preset["tone_curves"]
        #x_points = [p[0] for p in curves["r"]]
        
        self.curves = {
            "r": interp1d([p[0] for p in curves["r"]], [p[1] for p in curves["r"]], kind='cubic'),
            "g": interp1d([p[0] for p in curves["g"]], [p[1] for p in curves["g"]], kind='cubic'),
            "b": interp1d([p[0] for p in curves["b"]], [p[1] for p in curves["b"]], kind='cubic')
        }

    def _apply_tone_curves(self, image):
        """トーンカーブを適用"""
        result = np.zeros_like(image)
        for i, channel in enumerate("rgb"):
            valid_mask = (image[:,:,i] >= 0) & (image[:,:,i] <= 1)
            result[:,:,i][valid_mask] = self.curves[channel](image[:,:,i][valid_mask])
        return result

    def _generate_grain_pattern(self, shape, size):
        """
        よりフィルムライクなグレインパターンを生成
        粒状性を持つモノクロノイズを作成し、各色チャンネルで相関を持たせる
        """
        h, w = shape[:2]
        # ベースとなるモノクロノイズを生成（大きなグレイン）
        base_h, base_w = int(h/size), int(w/size)
        base_noise = np.random.normal(0, 1, (base_h, base_w))
        
        # ガウシアンフィルタでノイズを滑らかに
        base_noise = cv2.GaussianBlur(base_noise, (0, 0), 0.5)
        
        # 画像サイズにリサイズ
        base_noise = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 微細なディテールを追加
        fine_noise = np.random.normal(0, 0.2, (h, w))
        combined_noise = base_noise + fine_noise
        
        # 各チャンネルで相関のあるノイズを生成
        noise_r = combined_noise + np.random.normal(0, 0.1, (h, w))
        noise_g = combined_noise + np.random.normal(0, 0.1, (h, w))
        noise_b = combined_noise + np.random.normal(0, 0.1, (h, w))
        
        return np.dstack((noise_r, noise_g, noise_b))

    def _apply_grain(self, image):
        """改良版グレイン適用処理"""
        grain_params = self.preset["grain"]
        
        # グレインパターン生成
        noise = self._generate_grain_pattern(image.shape, grain_params["size"])
        
        # 輝度に応じたグレインの強度調整
        luminance = np.mean(image, axis=2, keepdims=True)
        grain_strength = (1 - luminance) * 0.7 + 0.3  # シャドウでより強く、ハイライトでも若干残す
        
        # グレインの適用
        grain_intensity = grain_params["intensity"] * 0.3  # 全体的な強度を抑える
        noise_scaled = noise * grain_strength * grain_intensity
        
        # ガンマ補正空間でグレインを適用
        gamma = 2.2
        image_linear = np.power(image, gamma)
        result = image_linear + noise_scaled
        result = np.power(np.clip(result, 0, 1), 1/gamma)
        
        return result

    def _apply_color_balance(self, image):
        """カラーバランスを適用"""
        return image * np.array(self.preset["color_balance"])

    def _adjust_shadows_highlights(self, image):
        """シャドウとハイライトの調整"""
        shadows = self.preset["shadows"]
        highlights = self.preset["highlights"]
        
        # シャドウの調整
        shadow_mask = 1 - image
        lifted = image + (shadow_mask ** 2) * shadows["lift"]
        gamma_adjusted = np.power(lifted, 1/shadows["gamma"])
        
        # ハイライトの調整
        highlight_mask = image
        compressed = gamma_adjusted * (1 - (highlight_mask ** 2) * (1 - highlights["compression"]))
        rolloff = compressed * (1 - (highlight_mask ** 3) * (1 - highlights["rolloff"]))
        
        return rolloff

    def _adjust_saturation(self, image):
        """彩度の調整"""
        if self.preset["saturation"] == 0:
            # モノクロ変換
            luminance = np.sum(image * [0.2989, 0.5870, 0.1140], axis=2, keepdims=True)
            return np.repeat(luminance, 3, axis=2)
        
        # RGB→HLS変換
        hls = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HLS_FULL)
        
        # 彩度の調整
        hls[:,:,2] *= self.preset["saturation"]
        #hls[:,:,2] = np.clip(hls[:,:,2], 0, 1)
        
        # HLS→RGB変換
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
        return rgb

    def _adjust_contrast(self, image):
        """コントラストの調整"""
        mean = np.mean(image)
        return mean + (image - mean) * self.preset["contrast"]

    def _apply_vignette(self, image):
        """周辺光量落ちを適用"""
        h, w = image.shape[:2]
        center_x, center_y = w/2, h/2
        
        # 楕円形の距離マップを作成
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(
            ((x - center_x)/(w/2.2))**2 + 
            ((y - center_y)/(h/2.2))**2
        )
        
        # 滑らかな減衰カーブを作成
        vignette = np.clip(1 - dist_from_center, 0, 1)
        vignette = np.power(vignette, 2)  # より自然な減衰カーブに
        
        # 光量落ちの強度を調整
        vignette = vignette * 0.2 + 0.8  # 20%の光量落ち
        
        # 適用
        result = image * vignette[:,:,np.newaxis]
        
        return result

    def process(self, image):
        """
        画像処理を実行（更新版）
        """
        
        # 処理の適用
        img = self._apply_tone_curves(image)
        img = self._apply_color_balance(img)
        img = self._adjust_shadows_highlights(img)
        img = self._adjust_contrast(img)
        img = self._adjust_saturation(img)
        #img = self._apply_vignette(img)  # 周辺光量落ちを追加
        #img = self._apply_grain(img)  # 改良版グレイン
        
        return img.astype(np.float32)


def apply_film_simulation(image, film_type):
    """
    便利な関数：画像にフィルムシミュレーションを適用
    Args:
        image: numpy.ndarray (RGB形式、0-255の値)
        film_type: str (フィルムの種類)
    Returns:
        numpy.ndarray: 処理後の画像
    """
    simulator = FilmSimulation(film_type)
    return simulator.process(image)

# 使用例：
"""
import cv2

# 画像を読み込み
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# フィルムシミュレーションを適用
processed = apply_film_simulation(image, 'portra400')

# 結果を保存
cv2.imwrite('output.jpg', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
"""

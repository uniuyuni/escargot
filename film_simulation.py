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
        },
        "kodachrome64": {
            "tone_curves": {
                "r": [(0, 0.02), (0.25, 0.28), (0.5, 0.57), (0.75, 0.85), (1, 0.98)],
                "g": [(0, 0.02), (0.25, 0.26), (0.5, 0.55), (0.75, 0.83), (1, 0.96)],
                "b": [(0, 0.03), (0.25, 0.25), (0.5, 0.54), (0.75, 0.82), (1, 0.95)]
            },
            "color_balance": [1.1, 0.95, 0.9],
            "saturation": 1.4,
            "contrast": 1.35,
            "grain": {"intensity": 0.2, "size": 1.1},
            "shadows": {"lift": 0.02, "gamma": 1.25},
            "highlights": {"compression": 0.92, "rolloff": 0.85},
            "vignette": {"strength": 0.15, "feather": 0.85}
        },

        "agfa_vista400": {
            "tone_curves": {
                "r": [(0, 0.04), (0.25, 0.27), (0.5, 0.54), (0.75, 0.82), (1, 0.96)],
                "g": [(0, 0.04), (0.25, 0.28), (0.5, 0.55), (0.75, 0.83), (1, 0.97)],
                "b": [(0, 0.05), (0.25, 0.26), (0.5, 0.53), (0.75, 0.81), (1, 0.95)]
            },
            "color_balance": [1.02, 1.0, 1.05],
            "saturation": 1.2,
            "contrast": 1.15,
            "grain": {"intensity": 0.35, "size": 1.4},
            "shadows": {"lift": 0.04, "gamma": 1.1},
            "highlights": {"compression": 0.8, "rolloff": 0.75},
            "vignette": {"strength": 0.18, "feather": 0.8}
        },

        "fuji_neopan1600": {
            "tone_curves": {
                "r": [(0, 0.05), (0.25, 0.28), (0.5, 0.53), (0.75, 0.81), (1, 0.95)],
                "g": [(0, 0.05), (0.25, 0.28), (0.5, 0.53), (0.75, 0.81), (1, 0.95)],
                "b": [(0, 0.05), (0.25, 0.28), (0.5, 0.53), (0.75, 0.81), (1, 0.95)]
            },
            "color_balance": [1.0, 1.0, 1.0],
            "saturation": 0,
            "contrast": 1.4,
            "grain": {"intensity": 0.5, "size": 1.8},
            "shadows": {"lift": 0.05, "gamma": 1.2},
            "highlights": {"compression": 0.75, "rolloff": 0.7},
            "vignette": {"strength": 0.25, "feather": 0.75}
        },

        "lomo_color100": {
            "tone_curves": {
                "r": [(0, 0.05), (0.25, 0.3), (0.5, 0.58), (0.75, 0.85), (1, 0.97)],
                "g": [(0, 0.04), (0.25, 0.28), (0.5, 0.56), (0.75, 0.84), (1, 0.96)],
                "b": [(0, 0.06), (0.25, 0.29), (0.5, 0.57), (0.75, 0.83), (1, 0.95)]
            },
            "color_balance": [1.15, 0.95, 1.1],
            "saturation": 1.45,
            "contrast": 1.35,
            "grain": {"intensity": 0.4, "size": 1.6},
            "shadows": {"lift": 0.06, "gamma": 1.15},
            "highlights": {"compression": 0.7, "rolloff": 0.65},
            "vignette": {"strength": 0.35, "feather": 0.7}
        },

        "ilford_hp5_plus": {
            "tone_curves": {
                "r": [(0, 0.03), (0.25, 0.27), (0.5, 0.52), (0.75, 0.8), (1, 0.96)],
                "g": [(0, 0.03), (0.25, 0.27), (0.5, 0.52), (0.75, 0.8), (1, 0.96)],
                "b": [(0, 0.03), (0.25, 0.27), (0.5, 0.52), (0.75, 0.8), (1, 0.96)]
            },
            "color_balance": [1.0, 1.0, 1.0],
            "saturation": 0,
            "contrast": 1.25,
            "grain": {"intensity": 0.4, "size": 1.5},
            "shadows": {"lift": 0.03, "gamma": 1.18},
            "highlights": {"compression": 0.82, "rolloff": 0.78},
            "vignette": {"strength": 0.2, "feather": 0.8}
        },

        "kodak_gold200": {
            "tone_curves": {
                "r": [(0, 0.03), (0.25, 0.27), (0.5, 0.55), (0.75, 0.83), (1, 0.97)],
                "g": [(0, 0.03), (0.25, 0.26), (0.5, 0.54), (0.75, 0.82), (1, 0.96)],
                "b": [(0, 0.04), (0.25, 0.25), (0.5, 0.53), (0.75, 0.81), (1, 0.95)]
            },
            "color_balance": [1.05, 0.98, 0.95],
            "saturation": 1.25,
            "contrast": 1.2,
            "grain": {"intensity": 0.3, "size": 1.4},
            "shadows": {"lift": 0.03, "gamma": 1.15},
            "highlights": {"compression": 0.85, "rolloff": 0.8},
            "vignette": {"strength": 0.15, "feather": 0.85}
        },

        "fuji_natura1600": {
            "tone_curves": {
                "r": [(0, 0.04), (0.25, 0.28), (0.5, 0.55), (0.75, 0.82), (1, 0.96)],
                "g": [(0, 0.05), (0.25, 0.29), (0.5, 0.56), (0.75, 0.83), (1, 0.97)],
                "b": [(0, 0.06), (0.25, 0.3), (0.5, 0.57), (0.75, 0.84), (1, 0.98)]
            },
            "color_balance": [0.95, 1.0, 1.15],
            "saturation": 1.15,
            "contrast": 1.2,
            "grain": {"intensity": 0.45, "size": 1.7},
            "shadows": {"lift": 0.05, "gamma": 1.1},
            "highlights": {"compression": 0.75, "rolloff": 0.7},
            "vignette": {"strength": 0.22, "feather": 0.78}
        },

        "agfa_ultra50": {
            "tone_curves": {
                "r": [(0, 0.02), (0.25, 0.26), (0.5, 0.54), (0.75, 0.83), (1, 0.98)],
                "g": [(0, 0.02), (0.25, 0.25), (0.5, 0.53), (0.75, 0.82), (1, 0.97)],
                "b": [(0, 0.03), (0.25, 0.24), (0.5, 0.52), (0.75, 0.81), (1, 0.96)]
            },
            "color_balance": [1.08, 1.0, 0.92],
            "saturation": 1.3,
            "contrast": 1.3,
            "grain": {"intensity": 0.15, "size": 1.2},
            "shadows": {"lift": 0.02, "gamma": 1.2},
            "highlights": {"compression": 0.9, "rolloff": 0.85},
            "vignette": {"strength": 0.15, "feather": 0.9}
        },

        "fujichrome_velvia100": {
            "tone_curves": {
                "r": [(0, 0.02), (0.25, 0.28), (0.5, 0.57), (0.75, 0.85), (1, 0.99)],
                "g": [(0, 0.02), (0.25, 0.27), (0.5, 0.56), (0.75, 0.84), (1, 0.98)],
                "b": [(0, 0.03), (0.25, 0.26), (0.5, 0.55), (0.75, 0.83), (1, 0.97)]
            },
            "color_balance": [1.12, 1.02, 1.0],
            "saturation": 1.6,
            "contrast": 1.45,
            "grain": {"intensity": 0.18, "size": 1.1},
            "shadows": {"lift": 0.02, "gamma": 1.25},
            "highlights": {"compression": 0.92, "rolloff": 0.88},
            "vignette": {"strength": 0.18, "feather": 0.85}
        },

        "ilford_delta3200": {
            "tone_curves": {
                "r": [(0, 0.05), (0.25, 0.29), (0.5, 0.54), (0.75, 0.82), (1, 0.95)],
                "g": [(0, 0.05), (0.25, 0.29), (0.5, 0.54), (0.75, 0.82), (1, 0.95)],
                "b": [(0, 0.05), (0.25, 0.29), (0.5, 0.54), (0.75, 0.82), (1, 0.95)]
            },
            "color_balance": [1.0, 1.0, 1.0],
            "saturation": 0,
            "contrast": 1.35,
            "grain": {"intensity": 0.55, "size": 1.9},
            "shadows": {"lift": 0.05, "gamma": 1.15},
            "highlights": {"compression": 0.7, "rolloff": 0.65},
            "vignette": {"strength": 0.25, "feather": 0.75}
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
        image = np.clip(image, 0, 1)    # 色破綻防止
        for i, channel in enumerate("rgb"):
            #valid_mask = (image[:,:,i] >= 0) & (image[:,:,i] <= 1)
            #result[:,:,i][valid_mask] = self.curves[channel](image[:,:,i][valid_mask])
            result[:,:,i] = self.curves[channel](image[:,:,i])
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

import numpy as np
from scipy.interpolate import CubicSpline
import json
from scipy.ndimage import gaussian_filter

class AdvancedFilmSimulation:
    def __init__(self, params_file=None):
        """
        パラメータファイルから設定を読み込むか、デフォルトパラメータを使用
        
        Parameters:
        params_file (str): JSONパラメータファイルのパス
        """
        self.params = self._load_default_params()
        if params_file:
            self._load_params_from_file(params_file)

    def _load_default_params(self):
        """デフォルトのフィルムシミュレーションパラメータを定義"""
        # Note: この実装ではパラメータは外部JSONから読み込むことを想定
        return {}

    def _load_params_from_file(self, params_file):
        """JSONファイルからパラメータを読み込む"""
        with open(params_file, 'r') as f:
            loaded_params = json.load(f)
            self.params.update(loaded_params)

    def _create_tone_curve(self, points_x, points_y):
        """トーンカーブを生成"""
        spline = CubicSpline(points_x, points_y, bc_type='natural')
        x = np.linspace(0, 1, 1024)
        return spline(x)

    def _rgb_to_hsv_vectorized(self, image):
        """RGB画像をHSV空間に変換（高速化版）"""
        reshape_needed = False
        if len(image.shape) == 3:
            original_shape = image.shape
            image = image.reshape(-1, 3)
            reshape_needed = True
        
        hsv = np.zeros_like(image)
        
        v = np.max(image, axis=1)
        delta = np.ptp(image, axis=1)
        s = np.where(v != 0, delta/v, 0)
        
        h = np.zeros(len(image))
        
        # R最大の場合
        mask_r = (v == image[:, 0]) & (delta != 0)
        h[mask_r] = 60 * ((image[mask_r, 1] - image[mask_r, 2]) / delta[mask_r])
        
        # G最大の場合
        mask_g = (v == image[:, 1]) & (delta != 0)
        h[mask_g] = 60 * (2 + (image[mask_g, 2] - image[mask_g, 0]) / delta[mask_g])
        
        # B最大の場合
        mask_b = (v == image[:, 2]) & (delta != 0)
        h[mask_b] = 60 * (4 + (image[mask_b, 0] - image[mask_b, 1]) / delta[mask_b])
        
        h = (h + 360) % 360
        
        hsv[:, 0] = h / 360
        hsv[:, 1] = s
        hsv[:, 2] = v
        
        if reshape_needed:
            hsv = hsv.reshape(original_shape)
        
        return hsv

    def _hsv_to_rgb_vectorized(self, hsv):
        """HSV画像をRGB空間に変換（高速化版）"""
        reshape_needed = False
        if len(hsv.shape) == 3:
            original_shape = hsv.shape
            hsv = hsv.reshape(-1, 3)
            reshape_needed = True
        
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h = h * 360
        
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        rgb = np.zeros_like(hsv)
        
        # 色相に基づいてRGB値を計算
        mask = (h >= 0) & (h < 60)
        rgb[mask] = np.column_stack([c[mask], x[mask], np.zeros_like(c[mask])])
        
        mask = (h >= 60) & (h < 120)
        rgb[mask] = np.column_stack([x[mask], c[mask], np.zeros_like(c[mask])])
        
        mask = (h >= 120) & (h < 180)
        rgb[mask] = np.column_stack([np.zeros_like(c[mask]), c[mask], x[mask]])
        
        mask = (h >= 180) & (h < 240)
        rgb[mask] = np.column_stack([np.zeros_like(c[mask]), x[mask], c[mask]])
        
        mask = (h >= 240) & (h < 300)
        rgb[mask] = np.column_stack([x[mask], np.zeros_like(c[mask]), c[mask]])
        
        mask = (h >= 300) & (h < 360)
        rgb[mask] = np.column_stack([c[mask], np.zeros_like(c[mask]), x[mask]])
        
        rgb += m[:, np.newaxis]
        
        if reshape_needed:
            rgb = rgb.reshape(original_shape)
        
        return rgb

    def _apply_hue_dependent_adjustments(self, image, hue_curves):
        """色相依存の調整を適用（色相シフト機能付き）"""
        hsv = self._rgb_to_hsv_vectorized(image)
        
        # 色相角度（0-360）
        hue_angles = hsv[..., 0] * 360
        
        # 各色相範囲での調整
        hue_ranges = {
            'red': (345, 15),
            'yellow': (15, 75),
            'green': (75, 165),
            'cyan': (165, 195),
            'blue': (195, 285),
            'magenta': (285, 345)
        }
        
        # 調整値を初期化
        sat_adjustment = np.zeros_like(hue_angles)
        lum_adjustment = np.zeros_like(hue_angles)
        hue_adjustment = np.zeros_like(hue_angles)
        
        # 各色相範囲に対して調整を適用
        for color, (start, end) in hue_ranges.items():
            if start > end:  # 赤の場合の特別処理
                mask = (hue_angles >= start) | (hue_angles < end)
            else:
                mask = (hue_angles >= start) & (hue_angles < end)
            
            # マスク領域の強度を計算
            center_angle = (start + end) / 2
            if start > end:  # 赤の場合
                center_angle = (start + end + 360) / 2
                if center_angle >= 360:
                    center_angle -= 360
            
            # 距離計算を改善（360度の循環を考慮）
            angle_diff = np.minimum(
                np.abs(hue_angles[mask] - center_angle),
                360 - np.abs(hue_angles[mask] - center_angle)
            )
            intensity = np.clip((90 - angle_diff) / 90, 0, 1)
            
            # 彩度の調整
            sat_curve = np.array(hue_curves['saturation'][color])
            sat_adjustment[mask] += np.interp(hsv[mask, 1], 
                                            np.linspace(0, 1, len(sat_curve)), 
                                            sat_curve) * intensity
            
            # 輝度の調整
            lum_curve = np.array(hue_curves['luminance'][color])
            lum_adjustment[mask] += np.interp(hsv[mask, 2], 
                                            np.linspace(0, 1, len(lum_curve)), 
                                            lum_curve) * intensity
            
            # 色相シフトの調整
            if 'hue_shift' in hue_curves:
                hue_shift_curve = np.array(hue_curves['hue_shift'][color])
                hue_shift = np.interp(hsv[mask, 0], 
                                    np.linspace(0, 1, len(hue_shift_curve)), 
                                    hue_shift_curve)
                hue_adjustment[mask] += hue_shift * intensity
        
        # 調整を適用
        hsv[..., 0] = np.mod(hsv[..., 0] + hue_adjustment / 360.0, 1.0)
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_adjustment, 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] + lum_adjustment, 0, 1)
        
        return self._hsv_to_rgb_vectorized(hsv)

    def _add_film_grain(self, image, intensity, size):
        """フィルムグレインを追加（輝度依存）"""
        # 輝度依存のグレイン強度
        luminance = np.mean(image, axis=2)
        grain_intensity = intensity * (1 - luminance) * 1.5
        
        # 色チャンネルごとに相関のあるノイズを生成
        noise_base = np.random.normal(0, 1, image.shape[:2])
        noise_base = gaussian_filter(noise_base, sigma=size)
        
        noise = np.zeros_like(image)
        for i in range(3):
            noise[..., i] = noise_base * np.random.normal(0.8, 0.2)
            
        noise *= grain_intensity[..., np.newaxis]
        return np.clip(image + noise, 0, 1)

    def apply_simulation(self, image, simulation_name):
        """指定されたシミュレーションを適用"""
        if simulation_name not in self.params:
            raise ValueError(f"Unknown simulation: {simulation_name}")
        
        params = self.params[simulation_name]
        
        # baseパラメータがある場合は、ベースのシミュレーションパラメータを継承
        if 'base' in params:
            base_params = self.params[params['base']].copy()
            base_params.update(params)
            params = base_params
        
        # トーンカーブの適用
        curve = self._create_tone_curve(
            params['tone_curve']['x'],
            params['tone_curve']['y']
        )
        result = np.interp(image, np.linspace(0, 1, len(curve)), curve)
        
        # 色相依存の調整を適用
        if 'hue_dependent_curves' in params:
            result = self._apply_hue_dependent_adjustments(
                result,
                params['hue_dependent_curves']
            )
        
        # 色変換行列の適用
        if 'color_matrix' in params:
            matrix = np.array(params['color_matrix'])
            result = np.dot(result.reshape(-1, 3), matrix).reshape(result.shape)
        
        # フィルムグレインの追加
        if 'grain' in params:
            result = self._add_film_grain(
                result,
                params['grain']['intensity'],
                params['grain']['size']
            )
        
        return np.clip(result, 0, 1)

    def add_simulation(self, name, params):
        """新しいシミュレーションパラメータを追加"""
        required_keys = {'tone_curve', 'color_matrix'}
        if not all(key in params for key in required_keys):
            raise ValueError(f"Missing required parameters: {required_keys}")
        
        self.params[name] = params

    def save_params(self, filename):
        """パラメータをJSONファイルに保存"""
        with open(filename, 'w') as f:
            json.dump(self.params, f, indent=4)

    def load_params(self, filename):
        """JSONファイルからパラメータを読み込み"""
        self._load_params_from_file(filename)

# 使用例
def process_raw_image(raw_image, simulation_name, params_file=None):
    """
    RAW画像にフィルムシミュレーションを適用
    
    Parameters:
    raw_image: numpy.ndarray
        デモザイク済みのRAW画像データ (float32, 範囲: 0-1)
    simulation_name: str
        適用するシミュレーションの名前
    params_file: str, optional
        カスタムパラメータファイルのパス
    
    Returns:
    numpy.ndarray
        処理済み画像 (float32, 範囲: 0-1)
    """
    simulator = AdvancedFilmSimulation(params_file)
    return simulator.apply_simulation(raw_image, simulation_name)

# 使用例
if __name__ == "__main__":
    # パラメータファイルの読み込み
    simulator = AdvancedFilmSimulation("film_params.json")
    
    # RAW画像の読み込み（例）
    raw_image = np.load("raw_image.npy")  # float32, 範囲: 0-1
    
    # シミュレーションの適用
    result = simulator.apply_simulation(raw_image, "classic_chrome")
    
    # 結果の保存（例）
    np.save("processed_image.npy", result)

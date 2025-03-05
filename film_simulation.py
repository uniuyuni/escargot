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

    def get_presets(self):
        return self.params

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
        hsv[..., 1] = hsv[..., 1] + sat_adjustment
        hsv[..., 2] = hsv[..., 2] + lum_adjustment
        
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
        return image + noise

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
        
        return result.astype(np.float32)

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

simulator = AdvancedFilmSimulation("film_presets.json")

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


import numpy as np
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

@jit
def __naive_sigmoid(x, gain, mid):    
    return 1.0 / (1.0 + jnp.exp((mid - x) * gain))

def naive_sigmoid(x, gain, mid):
    return np.array((__naive_sigmoid(x, gain, mid)).block_until_ready())

@jit
def __scaled_sigmoid(x, gain, mid):
    min = __naive_sigmoid(0.0, gain, mid)
    max = __naive_sigmoid(1.0, gain, mid)
    s = __naive_sigmoid(x, gain, mid)
    return (s - min) / (max - min)

def scaled_sigmoid(x, gain, mid):
    return np.array((__scaled_sigmoid(x, gain, mid)).block_until_ready())

@jit
def __naive_inverse_sigmoid(x, gain, mid):
    min = __naive_sigmoid(0.0, gain, mid)
    max = __naive_sigmoid(1.0, gain, mid)
    #s = __naive_sigmoid(jnp.clip(x, 1e-7, 1 - 1e-7), gain, mid)
    a = (max - min) * x + min
    return jnp.log(1.0 / a - 1.0)

def naive_inverse_sigmoid(x, gain, mid):
    return np.array((__naive_inverse_sigmoid(x, gain, mid)).block_until_ready())

@jit
def __scaled_inverse_sigmoid(x, gain, mid):
    min = __naive_inverse_sigmoid(0.0, gain, mid)
    max = __naive_inverse_sigmoid(1.0, gain, mid)
    s = __naive_inverse_sigmoid(x, gain, mid)
    return ((s - min) / (max - min))

def scaled_inverse_sigmoid(x, gain, mid):
    return np.array((__scaled_inverse_sigmoid(x, gain, mid)).block_until_ready())



def calculate_steepness(center):
    """中心位置に基づいて勾配係数を計算"""
    # 中心からの距離に基づいて勾配を調整
    distance_from_center = center - 0.5
    
    if distance_from_center >= 0:
        # 中心からの距離に応じて勾配を調整
        # 端に近づくほど大きな値を返す
        return 8 / (1 - distance_from_center)
    
    return 8 + distance_from_center * 8

def sigmoid(x, center):
    """シグモイド関数の計算"""
    x = np.clip(x, 0, 1)
    
    k = calculate_steepness(center)
    
    # 中心を基準とした相対位置の計算
    #relative_x = (x - center) / (1 - center)
    
    # 基本のシグモイド関数
    if center < 0.5:
        # 左寄りの場合
        #s = 1 / (1 + np.exp(-k * (x / center - 1)))
        s = 1 / (1 + np.exp(-k * (x - 0.5) * 2))
    elif center > 0.5:
        # 右寄りの場合
        s = 1 / (1 + np.exp(-k * ((x - center) / (1 - center))))
    else:
        # 中央の場合
        s = 1 / (1 + np.exp(-k * (x - 0.5) * 2))
    
    # 0と1の間に正規化
    result = (s - 1 / (1 + np.exp(k))) / (1 / (1 + np.exp(-k)) - 1 / (1 + np.exp(k)))
    
    return result

def plot_sigmoid(center):
    """シグモイド関数のプロット"""
    x = np.linspace(0, 1, 1000)
    y = [sigmoid(xi, center) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Sigmoid curve')
    plt.plot(center, sigmoid(center, center), 'ro', label='Center point')
    plt.axvline(x=center, color='gray', linestyle='--', alpha=0.5)
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Sigmoid function (center = {center:.2f})')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    
    # 勾配係数などの情報を表示
    k = calculate_steepness(center)
    center_y = sigmoid(center, center)
    info = f'Center: ({center:.3f}, {center_y:.3f})\nSteepness: {k:.3f}'
    plt.text(0.02, 0.98, info, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()

if __name__ == '__main__':
    # 使用例
    centers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for center in centers:
        plot_sigmoid(center)

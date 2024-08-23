#ライブラリのインポート
import numpy as np

#線形補間と連続性を生み出す子たち
def fade(t):return 6*t**5-15*t**4+10*t**3
def lerp(a,b,t):return a+fade(t)*(b-a)

#本体
def perlin(r,seed=np.random.randint(0,100)):
    np.random.seed(seed)

    ri = np.floor(r).astype(int) #整数部、インデックスとして使用
    ri[0] -= ri[0].min()         #
    ri[1] -= ri[1].min()         #インデックスとして使用するための準備
    rf = np.array(r) % 1         #小数部
    g = 2 * np.random.rand(ri[0].max()+2,ri[1].max()+2,2) - 1 #格子点の勾配
    e = np.array([[[[0,0],[0,1],[1,0],[1,1]]]])                       #四隅
    er = (np.array([rf]).transpose(2,3,0,1) - e).reshape(r.shape[1],r.shape[2],4,1,2) #四隅の各点から見た位置ベクトル
    gr = np.r_["3,4,0",g[ri[0],ri[1]],g[ri[0],ri[1]+1],g[ri[0]+1,ri[1]],g[ri[0]+1,ri[1]+1]].transpose(0,1,3,2).reshape(r.shape[1],r.shape[2],4,2,1) #おなじみファンシーソートで四隅の勾配をまとめて内積計算できる形に加工
    p = (er@gr).reshape(r.shape[1],r.shape[2],4).transpose(2,0,1) #全点まとめて勾配との内積計算

    return lerp(lerp(p[0],p[2],rf[0]),lerp(p[1],p[3],rf[0]),rf[1]) #補間して返す

def make_perlin_noise(w, h, n):
    y = np.zeros((h,w))
    for i in range(n):         #パーリンノイズは周波数を変えて何枚か重ねると本領を発揮するらしいのでループして加算（好きなだけどうぞ）
        x = np.linspace(0,8*i,w)
        r = np.array(np.meshgrid(x,x))
        y += perlin(r)                  #meshgridの形(2,N,N)で渡す

    y = y.astype(np.float32)
    return y
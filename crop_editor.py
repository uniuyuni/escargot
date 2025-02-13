
import numpy as np
from kivy.app import App as KVApp
from kivy.graphics import Color as KVColor, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Translate as KVTranslate, Rotate as KVRotate
from kivy.properties import NumericProperty as KVNumericProperty, ListProperty as KVListProperty, BooleanProperty as KVBooleanProperty
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.label import Label as KVLabel
from kivy.metrics import dp
from kivy.clock import Clock

def is_point_inside_rotated_rectangle(point_x, point_y, rect_width, rect_height, angle_degrees):
    """
    点が回転した四角形の内部にあるかどうかを判定する
    
    Parameters:
    point_x, point_y: 判定する点の座標
    rect_width, rect_height: 四角形のサイズ
    angle_degrees: 回転角度（度数法）
    
    Returns:
    bool: 点が四角形の内部にある場合True
    """
    # 度数法からラジアンに変換
    angle_rad = np.radians(angle_degrees)
    
    # 四角形の中心を原点とした場合の頂点座標
    half_width = rect_width / 2
    half_height = rect_height / 2
    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    
    # 回転行列
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 頂点を回転
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # 点が四角形の内部にあるかどうかを判定
    # 各辺に対して点が同じ側にあるかをチェック
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    point = np.array([point_x, point_y])
    signs = []
    for i in range(4):
        j = (i + 1) % 4
        signs.append(cross_product(rotated_corners[i], rotated_corners[j], point))
    
    # すべての符号が同じ（すべて正かすべて負）なら内部
    return all(sign > 0 for sign in signs) or all(sign <= 0 for sign in signs)

def find_nearest_point_on_edge(point_x, point_y, edge_start, edge_end):
    """
    線分上の最近接点を求める
    """
    edge_vector = edge_end - edge_start
    point_vector = np.array([point_x, point_y]) - edge_start
    
    # 線分の長さの二乗
    edge_length_sq = np.sum(edge_vector ** 2)
    
    # 線分上での位置（0から1の間）
    t = max(0, min(1, np.dot(point_vector, edge_vector) / edge_length_sq))
    
    # 最近接点
    return edge_start + t * edge_vector

def line_intersection(line1_start, line1_end, line2_start, line2_end):
    """
    2本の線分の交点を計算する関数
    
    Args:
        line1_start (tuple): 線分1の開始点 (x, y)
        line1_end (tuple): 線分1の終了点 (x, y)
        line2_start (tuple): 線分2の開始点 (x, y)
        line2_end (tuple): 線分2の終了点 (x, y)
    
    Returns:
        tuple or None: 交点の座標、または交差しない場合はNone
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    
    # 線分の傾きと切片を計算
    def line_equation(x1, y1, x2, y2):
        if x1 == x2:
            return float('inf'), x1  # 垂直線の場合
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    
    m1, b1 = line_equation(x1, y1, x2, y2)
    m2, b2 = line_equation(x3, y3, x4, y4)
    
    # 平行線の場合
    if m1 == m2:
        return None
    
    # 垂直線の処理
    if m1 == float('inf'):
        x = x1
        y = m2 * x + b2
    elif m2 == float('inf'):
        x = x3
        y = m1 * x + b1
    else:
        # 交点の計算
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    
    # 線分の範囲内かチェック
    def is_in_segment(x, y, x1, y1, x2, y2):
        return (min(x1, x2) <= x <= max(x1, x2) and 
                min(y1, y2) <= y <= max(y1, y2))
    
    if (is_in_segment(x, y, x1, y1, x2, y2) and 
        is_in_segment(x, y, x3, y3, x4, y4)):
        return (x, y)
    
    return None

def rotate_and_correct_point(point_x, point_y, old_px, old_py, rect_width, rect_height, angle_degrees):
    """
    四角形を中心で回転させた時、与えられた点が四角形の内部に収まるように補正する
    回転後の斜めの辺を正確に考慮する
    
    Parameters:
    point_x, point_y: 補正する点の座標
    old_px, old_py: 移動する前の座標
    rect_width, rect_height: 四角形のサイズ
    angle_degrees: 回転角度（度数法）
    
    Returns:
    tuple: 補正後の (x, y) 座標
    """
    px = point_x - rect_width/2
    py = point_y - rect_height/2

    # 点が既に内部にある場合は補正不要
    if is_point_inside_rotated_rectangle(px, py, rect_width, rect_height, angle_degrees):
        return point_x, point_y

    # 移動してない場合は補正不要
    #if point_x == old_px and point_y == old_py:
    #    return point_x, point_y

    # 回転後の頂点を計算
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    half_width = rect_width / 2
    half_height = rect_height / 2
    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # 移動線分と辺の交点を求める
    ox = old_px - half_width
    oy = old_py - half_height
    for i in range(4):
        j = (i + 1) % 4
        lipoint = line_intersection((px, py), (ox, oy), rotated_corners[i], rotated_corners[j])

        if lipoint is not None:
            return lipoint[0]+rect_width/2, lipoint[1]+rect_height/2

    # 各辺に対して最近接点を求め、その中で最も近いものを選択
    point = np.array([px, py])
    min_distance = float('inf')
    nearest_point = None
    for i in range(4):
        j = (i + 1) % 4
        edge_nearest = find_nearest_point_on_edge(
            px, py,
            rotated_corners[i],
            rotated_corners[j]
        )
        
        distance = np.sum((edge_nearest - point) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = edge_nearest
    
    return nearest_point[0]+rect_width/2, nearest_point[1]+rect_height/2
    

class CropEditor(KVFloatLayout):
    input_width = KVNumericProperty(dp(400))  # 元の画像の幅を指定するプロパティ
    input_height = KVNumericProperty(dp(300))  # 元の画像の高さを指定するプロパティ
    input_angle = KVNumericProperty(0)
    scale = KVNumericProperty(1.0)
    crop_rect = KVListProperty([0, 0, 0, 0])  # [左上x, 左上y, 右下x, 右下y]
    corner_threshold = KVNumericProperty(dp(10))  # 四隅の操作ポイントの判定範囲を指定する変数
    minimum_rect = KVNumericProperty(dp(16))
    aspect_ratio = KVNumericProperty(0)

    def __init__(self, **kwargs):
        super(CropEditor, self).__init__(**kwargs)
        self.corner_dragging = None

        scaled_width = self.input_width * self.scale
        scaled_height = self.input_height * self.scale

        self.set_crop_rect(self.crop_rect)

        with self.canvas:
            KVPushMatrix()
            self.input_translate = KVTranslate(scaled_width/2, scaled_height/2)
            self.input_rotate = KVRotate(angle=self.input_angle)
            KVColor(0.5, 0.5, 0.5, 1)
            self.input_line = KVLine(rectangle=(-scaled_width/2, -scaled_height/2, scaled_width, scaled_height), width=1)
            KVPopMatrix()

            KVPushMatrix()
            self.translate = KVTranslate()
            KVColor(1, 1, 1, 1)
            self.white_line = KVLine(rectangle=(0, 0, 0, 0), width=2)
            KVColor(0, 0, 0, 1)
            self.black_line = KVLine(rectangle=(0, 0, 0, 0), width=1)
            KVPopMatrix()
        
        self.label = KVLabel(font_size=20, bold=True, halign='left')
        self.add_widget(self.label)

        self.bind(crop_rect=self.update_rect,
                  input_width=self.update_crop_size,
                  input_height=self.update_crop_size,
                  scale=self.update_crop_size,
                  size=self.update_centering,
                  aspect_ratio=self.update_crop_size,
                  input_angle=self.update_crop_size)

        Clock.schedule_once(self.create_ui, -1)

    def create_ui(self, dt):
        self.pos = self.parent.pos
        
        # 初期設定の反映
        self.update_crop_size()

    def set_crop_rect(self, rect):
        scaled_width = self.input_width * self.scale
        scaled_height = self.input_height * self.scale

        # 矩形のサイズを設定 (初期値は画像のサイズと同じ)
        if rect == [0, 0, 0, 0]:
            self.crop_rect = [0, 0, scaled_width, scaled_height]
        else:
            #上下反転
            maxsize = max(self.input_width, self.input_height)
            padw = (maxsize - self.input_width) / 2
            padh = (maxsize - self.input_height) / 2
            x1, y1 = rect[0] - padw, rect[1] - padh
            x2, y2 = rect[2] - padw, rect[3] - padh
            #crop_width = (self.crop_rect[3] - self.crop_rect[1])
            #self.crop_rect[1] = self.input_height - (self.crop_rect[1] + crop_width)
            #self.crop_rect[3] = self.crop_rect[1] + crop_width
            height = y2 - y1
            y1 = self.input_height - (y1 + height)
            y2 = y1 + height
            x1, y1, x2, y2 = x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale
            self.crop_rect = [x1, y1, x2, y2]

    def update_crop_size(self, *args):

        # 縦横比補正
        self.corner_dragging = None
        self.__resize_crop(None)

        # 中心にシフトするためのトランスレーションを設定
        self.update_centering()

    def update_rect(self, *args):
        x1, y1, x2, y2 = self.crop_rect
        width = x2 - x1
        height = y2 - y1
        self.white_line.rectangle = (x1, y1, width, height)
        self.black_line.rectangle = (x1, y1, width, height)
        self.label.x, self.label.y = int(self.pos[0] + width/2), int(self.pos[1] + height/2)
        self.label.text = str(int(width/self.scale)) + " x " + str(int(height/self.scale))

    def update_centering(self, *args):
        # 中心に移動するためのトランスレーションを設定
        inwidth = self.input_width * self.scale
        inheight = self.input_height * self.scale
        self.translate.x = self.pos[0] + (self.width - inwidth) / 2
        self.translate.y = self.pos[1] + (self.height - inheight) / 2
        self.input_translate.x = self.translate.x + inwidth / 2
        self.input_translate.y = self.translate.y + inheight / 2
        self.input_rotate.angle = self.input_angle

        self.update_rect()

    def on_touch_down(self, touch):
        self.corner_dragging = self.__get_dragging_corner(touch)
        if self.corner_dragging is not None:
            return True
            
        return super(CropEditor, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.corner_dragging is not None:
            # 矩形のサイズを四隅のドラッグで変更
            self.__resize_crop(touch)

        return super(CropEditor, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.corner_dragging = None

        return super(CropEditor, self).on_touch_up(touch)

    def __get_dragging_corner(self, touch):
        x, y = touch.pos
        x1, y1, x2, y2 = self.crop_rect
        cx1 = x1 + self.translate.x
        cy1 = y1 + self.translate.y
        cx2 = x2 + self.translate.x
        cy2 = y2 + self.translate.y

        if abs(x - cx1) < self.corner_threshold and abs(y - cy1) < self.corner_threshold:
            return 'top_left'
        if abs(x - cx2) < self.corner_threshold and abs(y - cy1) < self.corner_threshold:
            return 'top_right'
        if abs(x - cx1) < self.corner_threshold and abs(y - cy2) < self.corner_threshold:
            return 'bottom_left'
        if abs(x - cx2) < self.corner_threshold and abs(y - cy2) < self.corner_threshold:
            return 'bottom_right'
        return None

    def __resize_crop(self, touch):
        x1, y1, x2, y2 = self.crop_rect
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        if self.corner_dragging == 'top_left':
            new_x1 = max(0, min(x2 - self.minimum_rect, touch.x - self.translate.x))
            new_y1 = max(0, min(y2 - self.minimum_rect, touch.y - self.translate.y))
            
        elif self.corner_dragging == 'top_right':
            new_x2 = min(0, max(x1 + self.minimum_rect, touch.x - self.translate.x))
            new_y1 = max(0, min(y2 - self.minimum_rect, touch.y - self.translate.y))

        elif self.corner_dragging == 'bottom_left':
            new_x1 = max(0, min(x2 - self.minimum_rect, touch.x - self.translate.x))
            new_y2 = min(0, max(y1 + self.minimum_rect, touch.y - self.translate.y))

        elif self.corner_dragging == 'bottom_right':
            new_x2 = min(0, max(x1 + self.minimum_rect, touch.x - self.translate.x))
            new_y2 = min(0, max(y1 + self.minimum_rect, touch.y - self.translate.y))

        # 縦横比を考慮
        if self.aspect_ratio > 0:
            width = new_x2 - new_x1
            height = new_y2 - new_y1
            if self.corner_dragging in ['top_left', 'top_right']:
                new_y1 = new_y2 - width / self.aspect_ratio
            elif self.corner_dragging in ['bottom_left', 'bottom_right']:
                new_y2 = new_y1 + width / self.aspect_ratio                                                          
            else:
                if width > height:
                    new_width = height * self.aspect_ratio
                    new_height = height
                else:
                    new_width = width
                    new_height = width / self.aspect_ratio
                center_x = (new_x1 + new_x2) / 2
                center_y = (new_y1 + new_y2) / 2
                new_x1 = center_x - new_width / 2
                new_x2 = center_x + new_width / 2
                new_y1 = center_y - new_height / 2
                new_y2 = center_y + new_height / 2

        # クリップ先の画像の中に収める
        new_x1, new_y1 = rotate_and_correct_point(new_x1, new_y1, x1, y1, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
        new_x2, new_y1 = rotate_and_correct_point(new_x2, new_y1, x2, y1, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
        new_x1, new_y2 = rotate_and_correct_point(new_x1, new_y2, x1, y2, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
        new_x2, new_y2 = rotate_and_correct_point(new_x2, new_y2, x2, y2, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)

        self.crop_rect = [new_x1, new_y1, new_x2, new_y2]

    @staticmethod
    def get_initial_crop_info(input_width, input_height, scale):
        # 上下反転させて返す
        x1, y1, x2, y2 = 0, 0, input_width, input_height
        maxsize = max(input_width, input_height)
        padw = (maxsize - input_width) / 2 
        padh = (maxsize - input_height) / 2
        crop_x = int(x1 + padw)
        crop_y = int(y1 + padh)
        crop_width = int(x2 - x1)
        crop_height = int(y2 - y1)
        return [crop_x, crop_y, crop_width, crop_height, scale]
    
    def get_crop_info(self):
        # 上下反転させて返す、パディングも付与
        x1, y1, x2, y2 = self.crop_rect
        maxsize = max(self.input_width, self.input_height)
        padw = (maxsize - self.input_width) / 2 
        padh = (maxsize - self.input_height) / 2
        crop_x = int(x1 / self.scale + padw)
        crop_y = int(self.input_height - (y1 + (y2-y1)) / self.scale + padh)
        crop_width = int((x2 - x1) / self.scale)
        crop_height = int((y2 - y1) / self.scale)
        return [crop_x, crop_y, crop_width, crop_height, self.scale]

class CropApp(KVApp):

    def build(self):
        root = KVFloatLayout()
        # ここで縦横サイズとスケールを指定
        crop_editor = CropEditor(input_width=dp(800), input_height=dp(600), input_angle=90, scale=1.0, aspect_ratio=0)
        crop_editor.pos = (dp(100), dp(100))
        root.add_widget(crop_editor)
        return root

if __name__ == '__main__':
    CropApp().run()
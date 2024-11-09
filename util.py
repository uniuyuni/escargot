
import math

def to_texture(pos, widget):
    # ウィンドウ座標からローカルイメージ座標に変換
    local_x, local_y = widget.to_widget(*pos)
    local_x = local_x - widget.pos[0]
    local_y = local_y - widget.pos[1]

    # ローカル座標をテクスチャ座標に変換
    tex_y = widget.height-local_y
    tex_x = local_x - (widget.width - widget.texture_size[0])/2
    tex_y = tex_y - (widget.height - widget.texture_size[1])/2

    return (tex_x, tex_y)

def str_to_orientation(str):
    if str == "Horizontal (normal)":
        orientation = 1
    if str == "Mirror horizontal":
        orientation = 2
    if str == "Rotate 180":
        orientation = 3
    if str  == "Mirror vertical":
        orientation = 4
    if str == "Mirror horizontal and rotate 90 CW":
        orientation = 5
    if str == "Rotate 90 CW":
        orientation = 6
    if str == "Mirror horizontal and rotate 270 CW":
        orientation = 7
    if str == "Rotate 270 CW":
        orientation = 8
    else:
        orientation = 1

    return orientation

def split_orientation(orientation):
    rad, flip = 0, 0
    if orientation == 1:
        rad, flip = 0, 0
        print("Horizontal (normal)")
    elif orientation == 2:
        rad, flip = 0, 1
        print("Mirror horizontal")
    elif orientation == 3:
        rad, flip = math.radians(180), 0
        print("Rotate 180")
    elif orientation == 4:
        rad, flip = 0, 2
        print("Mirror vertical")
    elif orientation == 5:
        rad, flip = math.radians(-90), 1
        print("Mirror horizontal and rotate 90 CW")
    elif orientation == 6:
        rad, flip = math.radians(-90), 0
        print("Rotate 90 CW")
    elif orientation == 7:
        rad, flip = math.radians(-270), 1
        print("Mirror horizontal and rotate 270 CW")
    elif orientation == 8:
        rad, flip = math.radians(-270), 0
        print("Rotate 270 CW")

    return rad, flip

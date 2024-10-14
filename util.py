
def to_texture(pos, widget):
    # ウィンドウ座標からローカルイメージ座標に変換
    local_x, local_y = widget.to_widget(*widget.to_window(*pos))

    # ローカル座標をテクスチャ座標に変換
    tex_y = widget.height-local_y
    tex_x = local_x - (widget.width - widget.texture_size[0])/2
    tex_y = tex_y - (widget.height - widget.texture_size[1])/2

    return (tex_x, tex_y)

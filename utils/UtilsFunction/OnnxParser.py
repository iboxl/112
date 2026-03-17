import os, re, onnx
from onnx import shape_inference

# ---------- 工具函数 ---------- #
def natural_key(text):
    """字符块保持字符串；数字块转 int，实现“自然排序”"""
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', text)]

def build_shape_dict(onnx_graph):
    """收集 value_info 里所有张量的静态 shape"""
    shape_dict = {}
    def _add(vi):
        dims = [d.dim_value if d.HasField("dim_value") else None
                for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    for vi in list(onnx_graph.value_info) + list(onnx_graph.input) + list(onnx_graph.output):
        _add(vi)
    return shape_dict

def loopdims_from_conv_node(node, shape_dict, initializer_dict):
    """提取单个 Conv 节点的 loopDim"""
    W = initializer_dict[node.input[1]]          # 权重：K,Cg,R,S
    Kg, Cg, R, S = W.dims
    group   = next((a.i   for a in node.attribute if a.name == "group"),   1)
    strides = next((a.ints for a in node.attribute if a.name == "strides"), [1, 1])
    pads    = next((a.ints for a in node.attribute if a.name == "pads"),    [0, 0, 0, 0])
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    """ 分组卷积的输入输出通道处理，将分组卷积视为 Group 个完全独立的、重复的子任务 """
    """ 为了方便与Zigzag对齐 将输出通道直接按组切分 不与pytorch的表示方法保持一致 """
    C = Cg
    K = Kg // group
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    X, Y = node.input[0], node.output[0]
    in_shape  = shape_dict.get(X, [None]*4)
    out_shape = shape_dict.get(Y, [None]*4)
    B, H, W0 = in_shape[0],  in_shape[2], in_shape[3]
    _, P, Q  = out_shape[0], out_shape[2], out_shape[3]
    return {
        'R': R, 'S': S, 'C': C, 'K': K,
        'P': P, 'Q': Q, 'G': group, 'B': B,
        'H': H, 'W': W0, 'Stride': strides[0], 'Padding': pads[0]
    }

def safe_load_onnx(fp):
    """
    加载 ONNX 模型，但 **不** 去尝试打开 external data 文件；
    只取网络结构与张量形状，足够我们解析 loopDim。
    """
    try:
        # ONNX >=1.12 推荐 load_model，旧版可 fallback 到 load
        if hasattr(onnx, "load_model"):
            model = onnx.load_model(fp, load_external_data=False)
        else:
            model = onnx.load(fp, load_external_data=False)
    except TypeError:
        # 极旧版本没有 load_external_data 参数，退回普通 load
        model = onnx.load(fp)
    return model


# ---------- 单文件解析 ---------- #
def parse_single_onnx(fp: str):
    """返回该 .onnx 文件中所有 Conv 节点的 layer_names、loopdims"""
    model = shape_inference.infer_shapes(safe_load_onnx(fp))
    g     = model.graph
    shapes = build_shape_dict(g)
    inits  = {i.name: i for i in g.initializer}
    layer_names, loopdims = [], []
    conv_idx = 0
    for node in g.node:
        if node.op_type != "Conv":
            continue
        dims = loopdims_from_conv_node(node, shapes, inits)
        lname = f"Conv_{conv_idx}_{dims['R']}_{dims['S']}_{dims['P']}_" \
                f"{dims['Q']}_{dims['C']}_{dims['K']}_{dims['G']}"
        layer_names.append(lname)
        loopdims.append(dims)
        conv_idx += 1
    if not layer_names:
        raise RuntimeError(f"{fp} 中未找到 Conv 节点")
    return layer_names, loopdims

# ---------- 总入口 ---------- #
def extract_loopdims(path: str):
    """
    Parameters
    ----------
    path : str
        .onnx 文件路径 或者 目录路径

    Returns
    -------
    names : list[str]
        - 文件模式 Conv 层名字列表
        - 目录模式：每个 .onnx 文件（去掉扩展名）的自然排序列表
    loopdims : list[dict]
        每个名字对应的 loopDim 字典
    """
    if path.lower().endswith(".onnx"):           # ------- 单文件 -------
        return parse_single_onnx(path)

    # ---------------------------- 目录 ---------------------------- #
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    fnames = sorted([f for f in os.listdir(path) if f.endswith(".onnx")],
                    key=natural_key)
    if not fnames:
        raise RuntimeError(f"目录 {path} 中没有 .onnx 文件")

    names, loopdims = [], []
    for f in fnames:
        full = os.path.join(path, f)
        # 假设每个文件只含一层 Conv；如有多层可改为取第 0 个
        _, dims_list = parse_single_onnx(full)
        names.append(os.path.splitext(f)[0])   # 模型名（去 .onnx）
        loopdims.append(dims_list[0])          # 该文件第一层 Conv 的数据
    return names, loopdims

# ---------- 示例调用 ---------- #
if __name__ == "__main__":
    p1 = "Evaluation/Zigzag_imc/zigzag-imc/zigzag/inputs/examples/workload/resnet18.onnx"     # 任意单文件
    p2 = "model/Resnet18"         # 目录

    names1, dims1 = extract_loopdims(p1)
    print("单个模型: ")
    for n, d in zip(names1, dims1):
        print(n)
        print(d)

    names2, dims2 = extract_loopdims(p2)
    print("\n目录: ")
    for n, d in zip(names2, dims2):
        print(n)
        print(d)



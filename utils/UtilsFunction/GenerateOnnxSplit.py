# this file is prepared for project 511
# Created by iboxl

import os
import onnx
from onnx import helper, TensorProto, shape_inference
from utils.UtilsFunction.ToolFunction import prepare_save_dir


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
import os
import onnx
from onnx import helper, TensorProto, shape_inference
from typing import Sequence, List, Union


def _get_dim_values(dims: Sequence[onnx.TensorShapeProto.Dimension]) -> List[Union[int, str]]:
    """Return concrete dimension values (int) or 'x' if unknown."""
    values: List[Union[int, str]] = []
    for d in dims:
        if d.HasField("dim_value"):
            values.append(d.dim_value)
        else:
            values.append("x")
    return values


def _dims_to_str(dims: Sequence[Union[int, str]]) -> str:
    return "_".join(str(d) for d in dims)


def split_conv_layers(
    input_model_path: str,
    output_dir: str,
    infer_shapes: bool = True,
) -> None:
    """Split each **Conv** layer in an ONNX model into a standalone ONNX file and
    name the file using the convention::

        Conv_<idx>_<R>_<S>_<P>_<Q>_<C>_<K>_<G>.onnx

    where
        * ``idx`` — discovery index (1‑based)
        * ``R, S`` — kernel height/width
        * ``P, Q`` — output feature‑map height/width
        * ``C`` — number of input channels
        * ``K`` — number of output channels (filters)
        * ``G`` — number of groups

    Parameters
    ----------
    input_model_path : str
        Path to the original ONNX model.
    output_dir : str
        Directory in which the split models will be saved.  Created if absent.
    infer_shapes : bool, default=True
        Run ONNX shape inference on each generated model for richer meta‑data.
    """

    if not os.path.isfile(input_model_path):
        raise FileNotFoundError(f"Input model '{input_model_path}' not found.")

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # Generate a file explaining the naming conventions.
    # =========================================================
    readme_path = os.path.join(outputdir, "Conv_idx_R_S_P_Q_C_K_G.txt")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write("Filename Format:\n")
        f.write("Conv_<idx>_<R>_<S>_<P>_<Q>_<C>_<K>_<G>.onnx\n\n")
        f.write("Legend:\n")
        f.write("  idx : Layer Index (discovery order)\n")
        f.write("  R   : Kernel Height\n")
        f.write("  S   : Kernel Width\n")
        f.write("  P   : Output Height\n")
        f.write("  Q   : Output Width\n")
        f.write("  C   : Input Channels (per group)\n")
        f.write("  K   : Output Channels\n")
        f.write("  G   : Groups\n")

    model = safe_load_onnx(input_model_path)
    graph = model.graph

    # Convenience maps -------------------------------------------------------
    initializer_map = {init.name: init for init in graph.initializer}
    value_info_map = {
        vi.name: vi for vi in list(graph.input) + list(graph.output) + list(graph.value_info)
    }

    conv_count = 0
    for node in graph.node:
        if node.op_type != "Conv":
            continue

        conv_count += 1

        # Build a *copy* of the Conv node ------------------------------------
        conv_node = helper.make_node(
            op_type="Conv",
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name or f"Conv_{conv_count}",
            **{attr.name: helper.get_attribute_value(attr) for attr in node.attribute},
        )

        # Minimal graph housing the single Conv ------------------------------
        new_graph = helper.make_graph(
            nodes=[conv_node],
            name=f"ConvGraph_{conv_count}",
            inputs=[],
            outputs=[],
            initializer=[],
        )

        # ----- Copy weight / bias initialisers ------------------------------
        for w_name in node.input[1:]:  # weights (and optional bias)
            if w_name in initializer_map:
                new_graph.initializer.append(initializer_map[w_name])

        # ----- Graph input (activation) -------------------------------------
        data_in = node.input[0]
        if data_in in value_info_map:
            new_graph.input.append(value_info_map[data_in])
        else:
            new_graph.input.append(
                helper.make_tensor_value_info(data_in, TensorProto.FLOAT, None)
            )

        # ----- Graph output --------------------------------------------------
        out_name = node.output[0]
        if out_name in value_info_map:
            new_graph.output.append(value_info_map[out_name])
        else:
            new_graph.output.append(
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)
            )

        new_model = helper.make_model(new_graph, producer_name="split_conv_layers")
        if infer_shapes:
            new_model = shape_inference.infer_shapes(new_model)

        # --------------------------------------------------------------------
        #  Derive R, S, C, K from weights  (weight layout = [K, C, R, S])
        #  Derive P, Q from output activation (layout = [N, K, P, Q])
        # --------------------------------------------------------------------
        w_name = node.input[1]
        if w_name not in initializer_map:
            raise RuntimeError(
                f"Weight tensor '{w_name}' for Conv node '{conv_node.name}' not found in initializers."
            )
        weight_dims = initializer_map[w_name].dims  # (K, C, R, S)
        if len(weight_dims) != 4:
            raise RuntimeError("Expected 4‑D weight tensor for Conv")
        K_ori, C, R, S = weight_dims

        # --------------------------------------------------------------------
        #  Derive group_attr <G>  
        # --------------------------------------------------------------------
        group_attr = next((a for a in node.attribute if a.name == "group"), None)
        G = helper.get_attribute_value(group_attr) if group_attr else 1

        """ 为了方便与Zigzag对齐 将输出通道直接按组切分 不与pytorch的表示方法保持一致 """
        K = K_ori // G

        # Output dims; may contain unknowns if shape inference failed ----------
        out_vi = next((vi for vi in new_model.graph.output if vi.name == out_name), None)
        if out_vi is not None and out_vi.type.HasField("tensor_type"):
            out_shape_dims = _get_dim_values(out_vi.type.tensor_type.shape.dim)
            if len(out_shape_dims) == 4:
                P, Q = out_shape_dims[2], out_shape_dims[3]
            else:
                P = Q = "x"
        else:
            P = Q = "x"

        file_name = f"Conv-{conv_count}_{R}_{S}_{P}_{Q}_{C}_{K}_{G}.onnx"
        out_path = os.path.join(output_dir, file_name)
        onnx.save(new_model, out_path)

    if conv_count == 0:
        raise RuntimeError("The supplied model contains no Conv nodes.")

    print(f"Extracted {conv_count} Conv layers to '{output_dir}'.")


if __name__ == "__main__":

    inputdir = "model/googlenet.onnx"
    outputdir = 'model/GoogleNet'
    prepare_save_dir(outputdir)


    split_conv_layers(inputdir, outputdir)
    
# python -m utils.UtilsFunction.GenerateOnnxSplit

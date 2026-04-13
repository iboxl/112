# CIMLoop 适配器: 工作负载匹配、mapper 调用、结果转换
# 模式参考: Evaluation/Zigzag_imc/CompatibleZigzag.py

import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path

import yaml
from baseline.types import BaselineLayer
from utils.CIMLoopUtils import (
    cimloop_cache_path,
    cimloop_models_root,
    cimloop_output_root,
    ensure_cimloop_available,
    ensure_cimloop_submodule,
)

# MIREDO model name → CIMLoop workload directory name
_MODEL_NAME_MAP = {
    "resnet18": "resnet18",
    "alexnet": "alexnet",
    "vgg19bn": "vgg16",
    "mobilenetV2": "mobilenet_v3",
    "resnet50": None,
    "EfficientNet-B0": None,
    "googlenet": None,
    "SingleLayer": None,
}

# 内存缓存 (session 级)
_WORKLOAD_CACHE = {}   # model -> list[dict]
_RESULT_CACHE = {}     # (model, layer_tag, macro, objective) -> dict

_DEFAULT_HISTOGRAMS = {
    "Inputs": [0, 0, 0, 3, 1, 0, 0],
    "Weights": [0, 1, 3, 4, 3, 1, 0],
    "Outputs": [0, 0, 0, 3, 1, 0, 0],
}

_TL_DIM_TO_MIREDO = {
    "R": "R",
    "S": "S",
    "P": "P",
    "Q": "Q",
    "C": "C",
    "M": "K",
    "G": "G",
    "N": "B",
}

_OPERAND_RELEVANCE = {
    "I": {"R", "S", "P", "Q", "C", "G", "B"},
    "W": {"R", "S", "C", "K", "G"},
    "O": {"P", "Q", "K", "G", "B"},
}

_OP_ORDER = ("I", "W", "O")
_MIREDO_RECTANGULAR_DIMS = {"R", "S", "P", "Q", "C", "M", "G"}


# ======================================================================
# 工作负载解析与匹配
# ======================================================================

def _parse_instance_from_yaml(path: Path) -> dict:
    """从 CIMLoop workload YAML 中提取 problem.instance 字典。

    YAML 文件含 Jinja2 模板语法, 不能直接 yaml.safe_load,
    但 instance 行格式固定: instance: {C: 3, M: 64, ...}
    用正则提取。
    """
    text = path.read_text()
    m = re.search(r"instance:\s*\{([^}]+)\}", text)
    if m is None:
        return {}
    pairs = {}
    for token in m.group(1).split(","):
        token = token.strip()
        if ":" not in token:
            continue
        key, val = token.split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            pairs[key] = int(val)
        except ValueError:
            pass  # 跳过非整数值 (如 ENCODED_INPUT_BITS)
    # 应用 problem_base 默认值
    pairs.setdefault("G", 1)
    pairs.setdefault("HStride", 1)
    pairs.setdefault("WStride", 1)
    return pairs


def load_cimloop_workloads(cimloop_model: str) -> list:
    """加载指定 CIMLoop 模型的所有 workload 层, 返回 [{path, index, instance}, ...]。"""
    if cimloop_model in _WORKLOAD_CACHE:
        return _WORKLOAD_CACHE[cimloop_model]

    workloads_dir = cimloop_models_root() / "workloads" / cimloop_model
    if not workloads_dir.is_dir():
        _WORKLOAD_CACHE[cimloop_model] = []
        return []

    layers = []
    for yaml_file in sorted(workloads_dir.glob("*.yaml")):
        if yaml_file.name.startswith("problem") or yaml_file.name.startswith("default"):
            continue
        instance = _parse_instance_from_yaml(yaml_file)
        if instance:
            layers.append({
                "path": yaml_file,
                "index": yaml_file.stem,  # "00", "01", ...
                "instance": instance,
            })
    _WORKLOAD_CACHE[cimloop_model] = layers
    return layers


def _dims_match(miredo_loopdim: dict, cimloop_instance: dict) -> bool:
    """比较 MIREDO loopdim 与 CIMLoop instance 是否匹配。

    维度映射: MIREDO K ↔ CIMLoop M, 其余相同。
    匹配条件: C, M/K, R, S, P, Q 完全一致 (忽略 G 和 stride)。
    """
    try:
        return (
            miredo_loopdim["C"] == cimloop_instance.get("C", 0)
            and miredo_loopdim["K"] == cimloop_instance.get("M", 0)
            and miredo_loopdim["R"] == cimloop_instance.get("R", 1)
            and miredo_loopdim["S"] == cimloop_instance.get("S", 1)
            and miredo_loopdim["P"] == cimloop_instance.get("P", 0)
            and miredo_loopdim["Q"] == cimloop_instance.get("Q", 0)
        )
    except KeyError:
        return False


def find_matching_workload(model_name: str, loopdim: dict) -> dict | None:
    """在 CIMLoop 预构建 workload 中查找与 MIREDO layer 维度匹配的条目。

    Returns: {"cimloop_model": str, "layer_index": str, "path": Path, "approximate": bool}
             或 None (无匹配)。
    """
    cimloop_model = _MODEL_NAME_MAP.get(model_name)
    if cimloop_model is None:
        return None

    approximate = (cimloop_model != model_name)
    workloads = load_cimloop_workloads(cimloop_model)

    for wl in workloads:
        if _dims_match(loopdim, wl["instance"]):
            return {
                "cimloop_model": cimloop_model,
                "layer_index": wl["index"],
                "path": wl["path"],
                "approximate": approximate,
            }
    return None


# ======================================================================
# Mapper 调用
# ======================================================================

_OBJECTIVE_MAP = {
    "latency": "delay",
    "energy": "energy",
    "edp": "edp",
}


def _layer_tag(loopdim: dict) -> str:
    """生成缓存用 layer 标识字符串。"""
    keys = ["R", "S", "P", "Q", "C", "K", "G", "Stride"]
    return "_".join(str(loopdim.get(k, 0)) for k in keys)


def _cache_tag(
    loopdim: dict,
    workload_path: str | Path | None,
    rectangular_compatible: bool = False,
) -> str:
    tag = _layer_tag(loopdim)
    if workload_path is not None:
        tag = f"onnx_{tag}"
    return f"{tag}_{'rectangular' if rectangular_compatible else 'native'}"


def _problem_shape():
    return {
        "coefficients": [
            {"name": "Wstride", "default": 1},
            {"name": "Hstride", "default": 1},
            {"name": "Wdilation", "default": 1},
            {"name": "Hdilation", "default": 1},
        ],
        "data_spaces": [
            {
                "name": "Weights",
                "projection": [[["C"]], [["M"]], [["R"]], [["S"]], [["Y"]], [["G"]]],
            },
            {
                "name": "Inputs",
                "projection": [
                    [["N"]],
                    [["C"]],
                    [["R", "Wdilation"], ["P", "Wstride"]],
                    [["S", "Hdilation"], ["Q", "Hstride"]],
                    [["X"]],
                    [["G"]],
                ],
            },
            {
                "name": "Outputs",
                "projection": [[["N"]], [["M"]], [["Q"]], [["P"]], [["Z"]], [["G"]]],
                "read_write": True,
            },
        ],
        "dimensions": ["C", "M", "R", "S", "N", "P", "Q", "X", "Y", "Z", "G"],
    }


def export_cimloop_workload(loopdim: dict, model_name: str, layer_name: str, output_root=None) -> Path:
    """Export a single ONNX-derived workload for CiMLoop without using vendored workload geometry."""
    base = Path(output_root).expanduser().resolve() if output_root is not None else cimloop_output_root()
    workload_dir = base / "cimloop_generated" / "workloads" / model_name
    workload_dir.mkdir(parents=True, exist_ok=True)
    path = workload_dir / f"{layer_name}_{_layer_tag(loopdim)}.yaml"

    stride = int(loopdim.get("Stride", 1))
    instance = {
        "N": "BATCH_SIZE",
        "X": "ENCODED_INPUT_BITS",
        "C": int(loopdim["C"]),
        "H": int(loopdim.get("H", 1)),
        "W": int(loopdim.get("W", 1)),
        "G": int(loopdim.get("G", 1)),
        "Y": "ENCODED_WEIGHT_BITS",
        "R": int(loopdim["R"]),
        "S": int(loopdim["S"]),
        "Hdilation": int(loopdim.get("Dilation", 1)),
        "Hstride": stride,
        "Wdilation": int(loopdim.get("Dilation", 1)),
        "Wstride": stride,
        "Z": "ENCODED_OUTPUT_BITS",
        "M": int(loopdim["K"]),
        "P": int(loopdim["P"]),
        "Q": int(loopdim["Q"]),
    }
    obj = {
        "problem": {
            "version": 0.4,
            "instance": instance,
            "shape": _problem_shape(),
            "name": layer_name,
            "dnn_name": model_name,
            "notes": "Generated from MIREDO ONNX loopdim for fair baseline input.",
            "histograms": _DEFAULT_HISTOGRAMS,
        }
    }
    text = yaml.safe_dump(obj, sort_keys=False)
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def run_cimloop_mapper(
    cimloop_model: str,
    layer_index: str | None,
    loopdim: dict,
    workload_path: str | Path | None = None,
    macro_name: str = "raella_isca_2023",
    objective: str = "latency",
    timeout: int = 600,
    output_root=None,
    rectangular_compatible: bool = False,
) -> dict:
    """运行 CIMLoop timeloop-mapper 求解单层, 返回原始结果字典。

    Returns: {"cycles": int, "energy_pj": float, "mapping": str, "computes": int}
    """
    ensure_cimloop_available()
    ensure_cimloop_submodule()
    objective = objective.lower()

    # 检查缓存
    tag = _cache_tag(loopdim, workload_path, rectangular_compatible)
    cache_key = (
        cimloop_model,
        tag,
        macro_name,
        objective,
        str(workload_path),
        rectangular_compatible,
    )
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key]

    pickle_path = cimloop_cache_path(macro_name, cimloop_model, tag, objective)
    if pickle_path.is_file():
        with open(pickle_path, "rb") as f:
            result = pickle.load(f)
        result.setdefault("rectangular_compatible", rectangular_compatible)
        result.setdefault("mapspace_template", "ruby" if rectangular_compatible else "uber")
        _RESULT_CACHE[cache_key] = result
        return result

    # 调用 CIMLoop API
    result = _invoke_mapper(
        cimloop_model=cimloop_model,
        layer_index=layer_index,
        macro_name=macro_name,
        objective=objective,
        timeout=timeout,
        workload_path=workload_path,
        output_root=output_root,
        layer_tag=tag,
        rectangular_compatible=rectangular_compatible,
    )

    # 持久化缓存
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(result, f)
    _RESULT_CACHE[cache_key] = result
    return result


def _invoke_mapper(
    cimloop_model: str,
    layer_index: str | None,
    macro_name: str,
    objective: str,
    timeout: int,
    workload_path: str | Path | None,
    output_root,
    layer_tag: str,
    rectangular_compatible: bool,
) -> dict:
    """通过 timeloopfe 构建 spec, 用 tl.call_mapper() 调用匹配版本的 timeloop-mapper。"""
    from utils.CIMLoopUtils import cimloop_runtime_env, ensure_cimloop_scripts_on_path
    ensure_cimloop_scripts_on_path()

    import pytimeloop.timeloopfe.v4 as tl
    from processors import ArrayProcessor

    models_dir = str(cimloop_models_root())
    top_yaml = os.path.join(models_dir, "top.yaml.jinja2")

    jinja_parse_data = {
        "macro": macro_name,
        "iso": macro_name,
        "system": "ws_dummy_buffer_many_macro",
    }
    if workload_path is not None:
        jinja_parse_data["layer"] = str(Path(workload_path).expanduser().resolve())
    else:
        jinja_parse_data["dnn"] = cimloop_model
        jinja_parse_data["layer"] = layer_index

    spec = tl.Specification.from_yaml_files(
        top_yaml,
        processors=[ArrayProcessor],
        jinja_parse_data=jinja_parse_data,
    )
    mapspace_template = "ruby" if rectangular_compatible else "uber"
    spec.mapspace["template"] = mapspace_template
    spec.mapspace.template = mapspace_template

    # 设置优化目标
    tl_objective = _OBJECTIVE_MAP.get(objective.lower(), "edp")
    spec.mapper.optimization_metrics = [tl_objective]
    if hasattr(spec.mapper, "timeout"):
        spec.mapper.timeout = max(timeout * 10, 10000)

    if output_root is None:
        output_dir = Path(tempfile.mkdtemp(prefix="cimloop_"))
        cleanup_output = True
    else:
        output_dir = (
            Path(output_root).expanduser().resolve()
            / "cimloop_generated"
            / "mapper_outputs"
            / f"{objective}_{cimloop_model}_{macro_name}_{layer_tag}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup_output = False
    try:
        mapper_result = tl.call_mapper(
            specification=spec,
            output_dir=str(output_dir),
            environment=cimloop_runtime_env(),
            log_to=str(output_dir / "timeloop-mapper.log"),
        )

        cycles = int(mapper_result.cycles)
        energy_pj = float(mapper_result.energy)  # Timeloop 报告 pJ
        mapping_str = getattr(mapper_result, "mapping", "") or ""
        computes = int(mapper_result.computes)

        return {
            "cycles": cycles,
            "energy_pj": energy_pj,
            "mapping": mapping_str,
            "computes": computes,
            "output_dir": str(output_dir),
            "workload_path": str(workload_path) if workload_path is not None else None,
            "rectangular_compatible": rectangular_compatible,
            "mapspace_template": mapspace_template,
        }
    finally:
        if cleanup_output:
            shutil.rmtree(output_dir, ignore_errors=True)


def _mem_index_by_name(acc, name: str) -> int:
    for mem in range(1, acc.Num_mem):
        if acc.mem2dict(mem) == name:
            return mem
    raise ValueError(f"CIMLoop mapping adapter cannot find memory '{name}' in accelerator")


def _allowed_mem_levels(acc, op: str) -> list[int]:
    op_idx = _OP_ORDER.index(op)
    return [mem for mem, flag in enumerate(acc.mappingArray[op_idx]) if flag == 1]


def _component_mem(component_names: list[str], op: str, acc, *, is_spatial: bool = False) -> int:
    names = {name.lower() for name in component_names}
    mem = {
        "dram": _mem_index_by_name(acc, "Dram"),
        "global": _mem_index_by_name(acc, "Global_buffer"),
        "input": _mem_index_by_name(acc, "Input_buffer"),
        "output": _mem_index_by_name(acc, "Output_buffer"),
        "ireg": _mem_index_by_name(acc, "IReg"),
        "oreg": _mem_index_by_name(acc, "OReg"),
        "macro": _mem_index_by_name(acc, "Macro"),
    }

    if "dummy_top" in names:
        return mem["dram"]
    if "inter_macro_in_system_spatial" in names or "inter_array_spatial" in names:
        return mem["global"]
    if "inter_column_spatial" in names or "inter_row_spatial" in names or "inter_1bit_x_1bit_mac_spatial" in names:
        if is_spatial:
            return {"I": mem["input"], "W": mem["global"], "O": mem["output"]}[op]
        return {"I": mem["ireg"], "W": mem["macro"], "O": mem["oreg"]}[op]
    if "input_buffer" in names:
        return {"I": mem["input"], "W": mem["global"], "O": mem["output"]}[op]
    if "output_center_offset_correct" in names:
        return {"I": mem["global"], "W": mem["global"], "O": mem["output"]}[op]
    if "output_register" in names or "timely_psubbuf" in names or "shift_add" in names or "adc" in names:
        return {"I": mem["ireg"], "W": mem["macro"], "O": mem["oreg"]}[op]
    if "row_drivers" in names:
        return {"I": mem["ireg"], "W": mem["macro"], "O": mem["oreg"]}[op]
    if "cim_unit" in names or "here_to_fix_a_bug" in names:
        return {"I": mem["ireg"], "W": mem["macro"], "O": mem["oreg"]}[op]
    return mem["global"]


_HEADER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+\[")
_LOOP_RE = re.compile(
    r"for\s+([A-Za-z]+)\s+in\s+\[0:(\d+)(?:,(\d+))?\)\s*(?:\((Spatial-[XY])\))?"
)


def _iter_mapping_blocks(mapping_text: str):
    headers: list[str] = []
    active_headers: list[str] = []
    for line in mapping_text.splitlines():
        header = _HEADER_RE.match(line)
        if header and not line.lstrip().startswith("|"):
            headers.append(header.group(1))
            continue
        if set(line.strip()) == {"-"} and headers:
            active_headers = headers
            headers = []
            continue
        loop = _LOOP_RE.search(line)
        if loop:
            yield active_headers, loop


def cimloop_mapping_residuals(raw: dict) -> list[dict]:
    """Return Timeloop residual factors from a CIMLoop mapper result."""
    mapping_text = raw.get("mapping") or ""
    residuals = []
    for component_names, loop in _iter_mapping_blocks(mapping_text):
        tl_dim, end_s, residual_s, spatial_tag = loop.groups()
        end = int(end_s)
        residual = int(residual_s) if residual_s is not None else end
        if residual == end:
            continue
        residuals.append({
            "dim": tl_dim,
            "miredo_dim": _TL_DIM_TO_MIREDO.get(tl_dim),
            "end": end,
            "residual": residual,
            "spatial": bool(spatial_tag),
            "components": list(component_names),
        })
    return residuals


def miredo_incompatible_residuals(raw: dict) -> list[dict]:
    """Residual factors that cannot be represented as a rectangular MIREDO LoopNest."""
    return [
        item
        for item in cimloop_mapping_residuals(raw)
        if item["dim"] in _MIREDO_RECTANGULAR_DIMS
    ]


def format_cimloop_residuals(residuals: list[dict]) -> str:
    if not residuals:
        return ""
    parts = []
    for item in residuals:
        location = "spatial" if item.get("spatial") else "temporal"
        components = "/".join(item.get("components") or ["unknown"])
        parts.append(
            f"{item['dim']} in [0:{item['end']},{item['residual']}) "
            f"at {location} {components}"
        )
    return "; ".join(parts)


def cimloop_result_to_baseline_layer(raw: dict, loopdim: dict, acc) -> BaselineLayer:
    mapping_text = raw.get("mapping") or ""
    if not mapping_text:
        raise ValueError("CIMLoop mapping adapter received an empty Timeloop mapping")

    temporal_by_op_mem = {op: {mem: [] for mem in _allowed_mem_levels(acc, op)} for op in _OP_ORDER}
    spatial_by_op_mem = {op: {mem: [] for mem in _allowed_mem_levels(acc, op)} for op in _OP_ORDER}

    for component_names, loop in _iter_mapping_blocks(mapping_text):
        tl_dim, end_s, residual_s, spatial_tag = loop.groups()
        if tl_dim not in _TL_DIM_TO_MIREDO:
            continue
        dim = _TL_DIM_TO_MIREDO[tl_dim]
        if dim == "B":
            continue
        end = int(end_s)
        residual = int(residual_s) if residual_s is not None else end
        if residual != end:
            if tl_dim in _MIREDO_RECTANGULAR_DIMS:
                raise ValueError(
                    "CIMLoop emitted an imperfect Timeloop factor that cannot be converted "
                    f"to a rectangular MIREDO LoopNest: {tl_dim} in [0:{end},{residual})"
                )
            continue
        if end <= 1:
            continue

        target = spatial_by_op_mem if spatial_tag else temporal_by_op_mem
        for op in _OP_ORDER:
            if dim not in _OPERAND_RELEVANCE[op]:
                continue
            mem = _component_mem(component_names, op, acc, is_spatial=bool(spatial_tag))
            if mem not in target[op]:
                allowed = _allowed_mem_levels(acc, op)
                mem = min(allowed, key=lambda candidate: abs(candidate - mem))
            target[op][mem].append((dim, end))

    temporal_mapping = {}
    spatial_mapping = {}
    for op in _OP_ORDER:
        temporal_mapping[op] = [
            temporal_by_op_mem[op][mem] for mem in reversed(_allowed_mem_levels(acc, op))
        ]
        spatial_mapping[op] = [
            spatial_by_op_mem[op][mem] for mem in reversed(_allowed_mem_levels(acc, op))
        ]

    double_buffer_flag = {
        op: [False] + [False] * len(temporal_mapping[op]) for op in _OP_ORDER
    }
    top_r_loop_size = {
        op: [1] * (len(temporal_mapping[op]) + 1) for op in _OP_ORDER
    }

    return BaselineLayer(
        source="cimloop",
        loop_dim={key: int(value) if value is not None else value for key, value in loopdim.items()},
        temporal_mapping=temporal_mapping,
        spatial_mapping=spatial_mapping,
        double_buffer_flag=double_buffer_flag,
        top_r_loop_size=top_r_loop_size,
        raw=raw,
    )

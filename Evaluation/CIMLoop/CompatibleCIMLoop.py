# CIMLoop 适配器: 工作负载匹配、mapper 调用、结果转换
# 模式参考: Evaluation/Zigzag_imc/CompatibleZigzag.py

import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path

from utils.CIMLoopUtils import (
    cimloop_cache_path,
    cimloop_models_root,
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
    keys = ["R", "S", "P", "Q", "C", "K", "G"]
    return "_".join(str(loopdim.get(k, 0)) for k in keys)


def run_cimloop_mapper(
    cimloop_model: str,
    layer_index: str,
    loopdim: dict,
    macro_name: str = "raella_isca_2023",
    objective: str = "latency",
    timeout: int = 600,
) -> dict:
    """运行 CIMLoop timeloop-mapper 求解单层, 返回原始结果字典。

    Returns: {"cycles": int, "energy_pj": float, "mapping": str, "computes": int}
    """
    ensure_cimloop_available()
    ensure_cimloop_submodule()

    # 检查缓存
    tag = _layer_tag(loopdim)
    cache_key = (cimloop_model, tag, macro_name, objective)
    if cache_key in _RESULT_CACHE:
        return _RESULT_CACHE[cache_key]

    pickle_path = cimloop_cache_path(macro_name, cimloop_model, tag, objective)
    if pickle_path.is_file():
        with open(pickle_path, "rb") as f:
            result = pickle.load(f)
        _RESULT_CACHE[cache_key] = result
        return result

    # 调用 CIMLoop API
    result = _invoke_mapper(cimloop_model, layer_index, macro_name, objective, timeout)

    # 持久化缓存
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(result, f)
    _RESULT_CACHE[cache_key] = result
    return result


def _invoke_mapper(
    cimloop_model: str,
    layer_index: str,
    macro_name: str,
    objective: str,
    timeout: int,
) -> dict:
    """通过 timeloopfe 构建 spec, 用 tl.call_mapper() 调用匹配版本的 timeloop-mapper。"""
    from utils.CIMLoopUtils import ensure_cimloop_scripts_on_path
    ensure_cimloop_scripts_on_path()

    import pytimeloop.timeloopfe.v4 as tl
    from processors import ArrayProcessor

    models_dir = str(cimloop_models_root())
    top_yaml = os.path.join(models_dir, "top.yaml.jinja2")

    spec = tl.Specification.from_yaml_files(
        top_yaml,
        processors=[ArrayProcessor],
        jinja_parse_data={
            "macro": macro_name,
            "iso": macro_name,
            "dnn": cimloop_model,
            "layer": layer_index,
            "system": "ws_dummy_buffer_many_macro",
        },
    )

    # 设置优化目标
    tl_objective = _OBJECTIVE_MAP.get(objective.lower(), "edp")
    spec.mapper.optimization_metrics = [tl_objective]
    if hasattr(spec.mapper, "timeout"):
        spec.mapper.timeout = max(timeout * 10, 10000)

    output_dir = tempfile.mkdtemp(prefix="cimloop_")
    try:
        mapper_result = tl.call_mapper(
            specification=spec,
            output_dir=output_dir,
            log_to=os.path.join(output_dir, "timeloop-mapper.log"),
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
        }
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ======================================================================
# 结果转换
# ======================================================================

def cimloop_result_to_miredo_units(raw: dict) -> tuple:
    """将 CIMLoop 结果转换为 MIREDO 单位。

    CIMLoop: cycles (整数), energy (pJ)
    MIREDO:  cycles (整数), energy (nJ)  — 见 ArchSpec.py pj_to_nj()
    """
    latency = float(raw["cycles"])
    energy = float(raw["energy_pj"]) * 1e-3  # pJ → nJ
    return latency, energy

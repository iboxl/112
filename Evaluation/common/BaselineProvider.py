import copy
import hashlib
import json
import pickle
from dataclasses import dataclass

from Simulator.Simulax import tranSimulator
from utils.GlobalUT import CONST
from utils.Workload import LoopNest
from utils.ZigzagUtils import (
    compare_ops_cme,
    convert_Zigzag_to_MIREDO,
    get_hardware_performance_zigzag,
    zigzag_cache_prefix,
)

from Evaluation.common.EvalCommon import objective_to_opt_flag, repo_root
from Evaluation.WeightStationaryGenerator import generate_weight_stationary_baseline


SUPPORTED_BASELINE_METHODS = ("zigzag", "ws")
BASELINE_METHOD_LABELS = {
    "zigzag": "ZigZag_IMC",
    "ws": "WS",
}


@dataclass
class BaselineRunResult:
    method: str
    objective: str
    latency: float
    energy: float
    profile: object
    dataflow: LoopNest
    metadata: dict

    @property
    def edp(self):
        return self.latency * self.energy * CONST.SCALINGFACTOR


_ZIGZAG_CME_CACHE = {}
_DEFAULT_FP_CACHE = {}


def _spec_fingerprint(spec) -> str:
    """12 位 sha256 前缀，作为 spec 的确定性 cache key。"""
    raw = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _default_spec_fingerprint(architecture: str):
    """查 architecture 对应的默认 Spec 指纹；None 表示没注册默认 Spec。"""
    if architecture in _DEFAULT_FP_CACHE:
        return _DEFAULT_FP_CACHE[architecture]
    from importlib import import_module
    from Evaluation.common.EvalCommon import _ARCHITECTURE_SPEC_BUILDERS
    module_path = _ARCHITECTURE_SPEC_BUILDERS.get(architecture)
    if module_path is None:
        _DEFAULT_FP_CACHE[architecture] = None
        return None
    try:
        default = import_module(module_path).default_spec()
        fp = _spec_fingerprint(default)
    except Exception:
        fp = None
    _DEFAULT_FP_CACHE[architecture] = fp
    return fp


def _resolve_default_spec(architecture):
    """从架构注册表拉默认 Spec；None = 未注册（退回 legacy importlib 路径）。"""
    from importlib import import_module
    from Evaluation.common.EvalCommon import _ARCHITECTURE_SPEC_BUILDERS
    module_path = _ARCHITECTURE_SPEC_BUILDERS.get(architecture)
    if module_path is None:
        return None
    try:
        return import_module(module_path).default_spec()
    except Exception:
        return None


def _load_zigzag_cmes(model_name, architecture, objective, spec=None):
    """加载 ZigZag CME 列表。
    - spec 非给定：取架构注册表的默认 Spec
    - 默认 Spec 对应非 fingerprinted pickle（复用旧缓存）；variant Spec 带 fingerprint 隔离
    - 架构未注册（极少）：回退到 legacy 字符串 importlib 路径
    """
    if spec is None:
        spec = _resolve_default_spec(architecture)

    fp = None
    use_default_path = True
    if spec is not None:
        fp = _spec_fingerprint(spec)
        default_fp = _default_spec_fingerprint(architecture)
        use_default_path = (default_fp is not None and fp == default_fp)

    cache_key = (model_name, architecture, objective, "default" if use_default_path else fp)
    if cache_key in _ZIGZAG_CME_CACHE:
        return _ZIGZAG_CME_CACHE[cache_key]

    opt_flag = objective_to_opt_flag(objective)
    model_path = repo_root() / "model" / f"{model_name}.onnx"
    compare_file_prefix = zigzag_cache_prefix(opt_flag, model_name, architecture)
    suffix = "" if use_default_path else f"_{fp}"
    compare_pickle = compare_file_prefix.with_name(compare_file_prefix.name + suffix).with_suffix(".pickle")
    compare_json = compare_file_prefix.with_name(compare_file_prefix.name + suffix).with_suffix(".json")

    if not compare_pickle.is_file():
        if spec is not None:
            from Evaluation.Zigzag_imc.zigzag_adapter import to_zigzag_accelerator
            acc_name = architecture if use_default_path else f"{architecture}_{fp}"
            accelerator_arg = to_zigzag_accelerator(spec, acc_name=acc_name)
        else:
            accelerator_arg = f"Architecture.{architecture}"  # 未注册架构 fallback
        get_hardware_performance_zigzag(
            workload=str(model_path),
            accelerator=accelerator_arg,
            mapping="Config.zigzag_mapping",
            opt=opt_flag,
            dump_filename_pattern=str(compare_json),
            pickle_filename=str(compare_pickle),
        )

    with open(compare_pickle, "rb") as fp_handle:
        cmes = pickle.load(fp_handle)
    _ZIGZAG_CME_CACHE[cache_key] = cmes
    return cmes


def run_zigzag_baseline(acc, ops, loopdim, model_name, architecture, objective):
    # 如果 acc 通过 from_spec 构造，拿到它的 source_spec；否则走默认路径
    spec = getattr(acc, "source_spec", None)
    cmes = _load_zigzag_cmes(
        model_name=model_name,
        architecture=architecture,
        objective=objective,
        spec=spec,
    )
    cme = next(c for c in cmes if compare_ops_cme(loopDim=loopdim, cme=c))
    loops = LoopNest(acc=acc, ops=ops)
    loops = convert_Zigzag_to_MIREDO(loops=loops, cme=cme)
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()
    return BaselineRunResult(
        method="zigzag",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": "zigzag_imc",
            "model": model_name,
            "architecture": architecture,
            "spec_fingerprint": _spec_fingerprint(spec) if spec is not None else "default",
        },
    )


def run_ws_baseline(acc, ops, objective):
    result = generate_weight_stationary_baseline(acc=copy.deepcopy(acc), ops=ops)
    return BaselineRunResult(
        method="ws",
        objective=objective,
        latency=result.latency,
        energy=result.energy,
        profile=result.profile,
        dataflow=result.dataflow,
        metadata={
            "policy": result.policy,
        },
    )


def run_baseline(method, acc, ops, loopdim, model_name, architecture, objective):
    method = method.lower()
    if method == "zigzag":
        return run_zigzag_baseline(
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=architecture,
            objective=objective,
        )
    if method == "ws":
        return run_ws_baseline(acc=acc, ops=ops, objective=objective)
    raise ValueError(f"Unsupported baseline method: {method}")

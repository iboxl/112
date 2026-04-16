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


SUPPORTED_BASELINE_METHODS = ("zigzag", "ws", "cimloop", "cosa", "cosa-constrained")
BASELINE_METHOD_LABELS = {
    "zigzag": "ZigZag_IMC",
    "ws": "WS",
    "cimloop": "CIMLoop",
    "cosa": "CoSA",
    "cosa-constrained": "CoSA-Constr",
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
_CIMLOOP_OUTPUTS_CACHE = {}
_COSA_OUTPUTS_CACHE = {}
_COSA_CONSTRAINED_OUTPUTS_CACHE = {}


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


def _cimloop_cache_root(architecture: str, objective: str, model_name: str, spec) -> 'Path':
    from pathlib import Path
    suffix = "" if spec is None else _cimloop_spec_suffix(architecture, spec)
    base = repo_root() / "Evaluation" / "CIMLoop" / "output"
    dirname = f"{objective.lower()}_{model_name}_{architecture}{suffix}"
    return base / dirname


def _cimloop_spec_suffix(architecture: str, spec) -> str:
    fp = _spec_fingerprint(spec)
    default_fp = _default_spec_fingerprint(architecture)
    if default_fp is not None and fp == default_fp:
        return ""
    return f"_{fp}"


def _load_cimloop_outputs(model_name, architecture, objective, spec=None):
    """Lazy per-layer cache. Returns a dict layer_fp → CIMLoopLayerOutput, loaded
    from index.pickle if present. Mapper runs on demand per missing layer.
    """
    import pickle
    if spec is None:
        spec = _resolve_default_spec(architecture)

    cache_root = _cimloop_cache_root(architecture, objective, model_name, spec)
    index_path = cache_root / "index.pickle"
    cache_key = (model_name, architecture, objective, _spec_fingerprint(spec) if spec is not None else "legacy")

    if cache_key in _CIMLOOP_OUTPUTS_CACHE:
        return _CIMLOOP_OUTPUTS_CACHE[cache_key], cache_root, index_path

    outputs: dict = {}
    if index_path.is_file():
        with open(index_path, "rb") as fp_handle:
            outputs = pickle.load(fp_handle)
    _CIMLOOP_OUTPUTS_CACHE[cache_key] = outputs
    return outputs, cache_root, index_path


def _save_cimloop_index(index_path, outputs):
    import pickle
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fp_handle:
        pickle.dump(outputs, fp_handle)


def _parse_art_summary_estimators(art_path) -> dict:
    """从 ART_summary.yaml 的 table_summary 构造 {component_name: [estimator, ...]}。
    这让评审能审查：每个 storage/compute 组件的能耗是由哪个 accelergy plug-in 估的，
    论文立论依赖的是 NeuroSim/CACTI 真实模型而非 dummy 回退。"""
    import yaml
    if not art_path or not art_path.is_file():
        return {"_error": f"ART_summary missing at {art_path}"}
    try:
        with open(art_path) as fh:
            art = yaml.safe_load(fh)
    except Exception as e:
        return {"_error": f"ART_summary parse failed: {type(e).__name__}: {e}"}
    table = (art or {}).get("ART_summary", {}).get("table_summary", [])
    sources = {}
    for entry in table:
        raw_name = str(entry.get("name", ""))
        # 剥 "[1..N]" 后缀（内含 "..."，必须先去后缀再按 "." 拆父路径），再取最后一个 "." 后的组件名
        name = raw_name
        if "[" in name:
            name = name.split("[", 1)[0]
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        pe = entry.get("primitive_estimations")
        if isinstance(pe, list) and pe:
            sources[name] = [p.get("estimator", "?") for p in pe if isinstance(p, dict)]
        elif isinstance(pe, str):
            sources[name] = [pe]
        # primitive_estimations: [] 的条目（inter_*_spatial / dummy_top 等 fanout 占位）
        # 不记录 — 它们不是能耗源。
    return sources


def _cosa_cache_root(architecture: str, objective: str, model_name: str, spec) -> 'Path':
    from pathlib import Path
    suffix = "" if spec is None else _cimloop_spec_suffix(architecture, spec)
    base = repo_root() / "Evaluation" / "CoSA" / "output"
    dirname = f"{objective.lower()}_{model_name}_{architecture}{suffix}"
    return base / dirname


def _load_cosa_outputs(model_name, architecture, objective, spec=None):
    import pickle
    if spec is None:
        spec = _resolve_default_spec(architecture)
    cache_root = _cosa_cache_root(architecture, objective, model_name, spec)
    index_path = cache_root / "index.pickle"
    cache_key = (model_name, architecture, objective, _spec_fingerprint(spec) if spec is not None else "legacy")
    if cache_key in _COSA_OUTPUTS_CACHE:
        return _COSA_OUTPUTS_CACHE[cache_key], cache_root, index_path
    outputs: dict = {}
    if index_path.is_file():
        with open(index_path, "rb") as fp_handle:
            outputs = pickle.load(fp_handle)
    _COSA_OUTPUTS_CACHE[cache_key] = outputs
    return outputs, cache_root, index_path


def _save_cosa_index(index_path, outputs):
    import pickle
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fp_handle:
        pickle.dump(outputs, fp_handle)


def run_cosa_baseline(acc, ops, loopdim, model_name, architecture, objective):
    g = int(max(1, loopdim.get("G", 1)))
    if g != 1:
        raise NotImplementedError(
            f"CoSA baseline: grouped convolution (G={g}) not supported; "
            "CoSA's cnn-layer problem shape has no G dimension. Layer excluded "
            "from CoSA comparison."
        )

    from Evaluation.CoSA.cosa_adapter import loopdim_fingerprint, run_cosa_mapper_for_layer
    from Evaluation.CoSA.CompatibleCoSA import convert_CoSA_to_MIREDO

    spec = getattr(acc, "source_spec", None)
    if spec is None:
        spec = _resolve_default_spec(architecture)

    outputs, cache_root, index_path = _load_cosa_outputs(
        model_name=model_name,
        architecture=architecture,
        objective=objective,
        spec=spec,
    )
    layer_fp = loopdim_fingerprint(loopdim)

    if layer_fp not in outputs:
        if spec is None:
            raise RuntimeError(
                f"run_cosa_baseline: no HardwareSpec resolved for architecture={architecture!r}; "
                "CoSA adapter requires a registered default_spec()"
            )
        work_dir = cache_root / "per_layer" / layer_fp
        outputs[layer_fp] = run_cosa_mapper_for_layer(
            spec=spec,
            loopdim=loopdim,
            objective=objective,
            work_dir=work_dir,
        )
        _save_cosa_index(index_path, outputs)

    out = outputs[layer_fp]
    loops = LoopNest(acc=acc, ops=ops)
    loops, legalization_meta = convert_CoSA_to_MIREDO(loops=loops, out=out, spec=spec)
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()

    return BaselineRunResult(
        method="cosa",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": "cosa_mip",
            "model": model_name,
            "architecture": architecture,
            "spec_fingerprint": _spec_fingerprint(spec) if spec is not None else "legacy",
            "mapper_runtime_s": out.runtime_s,
            "legalization_demoted_count": legalization_meta["demoted_count"],
            "legalization_demoted": legalization_meta["demoted"],
            "capacity_demoted_count": legalization_meta.get("capacity_demoted_count", 0),
            "capacity_demoted": legalization_meta.get("capacity_demoted", []),
        },
    )


def _cosa_constrained_cache_root(architecture: str, objective: str, model_name: str, spec) -> 'Path':
    from pathlib import Path
    suffix = "" if spec is None else _cimloop_spec_suffix(architecture, spec)
    base = repo_root() / "Evaluation" / "CoSA" / "output_constrained"
    dirname = f"{objective.lower()}_{model_name}_{architecture}{suffix}"
    return base / dirname


def _load_cosa_constrained_outputs(model_name, architecture, objective, spec=None):
    import pickle
    if spec is None:
        spec = _resolve_default_spec(architecture)
    cache_root = _cosa_constrained_cache_root(architecture, objective, model_name, spec)
    index_path = cache_root / "index.pickle"
    cache_key = (model_name, architecture, objective, _spec_fingerprint(spec) if spec is not None else "legacy")
    if cache_key in _COSA_CONSTRAINED_OUTPUTS_CACHE:
        return _COSA_CONSTRAINED_OUTPUTS_CACHE[cache_key], cache_root, index_path
    outputs: dict = {}
    if index_path.is_file():
        with open(index_path, "rb") as fp_handle:
            outputs = pickle.load(fp_handle)
    _COSA_CONSTRAINED_OUTPUTS_CACHE[cache_key] = outputs
    return outputs, cache_root, index_path


def _save_cosa_constrained_index(index_path, outputs):
    import pickle
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fp_handle:
        pickle.dump(outputs, fp_handle)


def run_cosa_constrained_baseline(acc, ops, loopdim, model_name, architecture, objective):
    """CoSA baseline with axis constraints injected into the Gurobi MIP.

    Uses a minimal fork of CoSA's mip_solver that adds x[(i,j,n,0)]==0
    constraints for dims not allowed at each physical spatial axis (dimX→RSC,
    dimY→K, cores→PQKG).  The resulting mapping is already axis-legal, so
    legalization demoted_count should be 0.

    G>1 (grouped conv) is not supported, same as the unconstrained path.

    OverSize risk: the constrained MIP allocates factors to different levels
    than unconstrained CoSA.  CoSA's simplified capacity model (single
    word-bits + part_ratios) may accept tile distributions that Simulax's
    per-operand precision check rejects.  Callers should catch
    ValueError('Dataflow Over MemSize Error') and record it as an anomaly.
    """
    g = int(max(1, loopdim.get("G", 1)))
    if g != 1:
        raise NotImplementedError(
            f"CoSA-constrained baseline: grouped convolution (G={g}) not supported; "
            "CoSA's cnn-layer problem shape has no G dimension. Layer excluded."
        )

    from Evaluation.CoSA.cosa_adapter import (
        loopdim_fingerprint,
        run_cosa_constrained_mapper_for_layer,
    )
    from Evaluation.CoSA.CompatibleCoSA import convert_CoSA_to_MIREDO

    spec = getattr(acc, "source_spec", None)
    if spec is None:
        spec = _resolve_default_spec(architecture)

    outputs, cache_root, index_path = _load_cosa_constrained_outputs(
        model_name=model_name,
        architecture=architecture,
        objective=objective,
        spec=spec,
    )
    layer_fp = loopdim_fingerprint(loopdim)

    if layer_fp not in outputs:
        if spec is None:
            raise RuntimeError(
                f"run_cosa_constrained_baseline: no HardwareSpec resolved for "
                f"architecture={architecture!r}; CoSA adapter requires a registered default_spec()"
            )
        work_dir = cache_root / "per_layer" / layer_fp
        outputs[layer_fp] = run_cosa_constrained_mapper_for_layer(
            spec=spec,
            loopdim=loopdim,
            objective=objective,
            work_dir=work_dir,
        )
        _save_cosa_constrained_index(index_path, outputs)

    out = outputs[layer_fp]
    loops = LoopNest(acc=acc, ops=ops)
    loops, legalization_meta = convert_CoSA_to_MIREDO(loops=loops, out=out, spec=spec)
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()

    return BaselineRunResult(
        method="cosa-constrained",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": "cosa_constrained_mip",
            "model": model_name,
            "architecture": architecture,
            "spec_fingerprint": _spec_fingerprint(spec) if spec is not None else "legacy",
            "mapper_runtime_s": out.runtime_s,
            "legalization_demoted_count": legalization_meta["demoted_count"],
            "legalization_demoted": legalization_meta["demoted"],
            "capacity_demoted_count": legalization_meta.get("capacity_demoted_count", 0),
            "capacity_demoted": legalization_meta.get("capacity_demoted", []),
        },
    )


def run_cimloop_baseline(acc, ops, loopdim, model_name, architecture, objective):
    from Evaluation.CIMLoop.cimloop_adapter import (
        loopdim_fingerprint,
        run_cimloop_mapper_for_layer,
    )
    from Evaluation.CIMLoop.CompatibleCIMLoop import convert_CIMLoop_to_MIREDO

    spec = getattr(acc, "source_spec", None)
    if spec is None:
        spec = _resolve_default_spec(architecture)

    outputs, cache_root, index_path = _load_cimloop_outputs(
        model_name=model_name,
        architecture=architecture,
        objective=objective,
        spec=spec,
    )
    layer_fp = loopdim_fingerprint(loopdim)

    if layer_fp not in outputs:
        if spec is None:
            raise RuntimeError(
                f"run_cimloop_baseline: no HardwareSpec resolved for architecture={architecture!r}; "
                "CIMLoop adapter requires a registered default_spec()"
            )
        work_dir = cache_root / "per_layer" / layer_fp
        outputs[layer_fp] = run_cimloop_mapper_for_layer(
            spec=spec,
            loopdim=loopdim,
            objective=objective,
            work_dir=work_dir,
        )
        _save_cimloop_index(index_path, outputs)

    out = outputs[layer_fp]
    loops = LoopNest(acc=acc, ops=ops)
    loops, legalization_meta = convert_CIMLoop_to_MIREDO(
        loops=loops, out=out, spec=spec, loopdim=loopdim,
    )
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()

    energy_sources = _parse_art_summary_estimators(getattr(out, "art_summary_path", None))

    return BaselineRunResult(
        method="cimloop",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": "cimloop_mapper",
            "model": model_name,
            "architecture": architecture,
            "spec_fingerprint": _spec_fingerprint(spec) if spec is not None else "legacy",
            "optimization_metric": list(out.optimization_metric),
            "mapper_runtime_s": out.runtime_s,
            "energy_sources": energy_sources,
            "capacity_demoted_count": legalization_meta.get("capacity_demoted_count", 0),
            "capacity_demoted": legalization_meta.get("capacity_demoted", []),
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
    if method == "cimloop":
        return run_cimloop_baseline(
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=architecture,
            objective=objective,
        )
    if method == "cosa":
        return run_cosa_baseline(
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=architecture,
            objective=objective,
        )
    if method == "cosa-constrained":
        return run_cosa_constrained_baseline(
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=architecture,
            objective=objective,
        )
    raise ValueError(f"Unsupported baseline method: {method}")

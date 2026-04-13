import copy
import pickle
from dataclasses import dataclass

from Simulator.Simulax import tranSimulator
from baseline.cosa_adapter import CoSABaselineAdapter
from Evaluation.Zigzag_imc.CompatibleZigzag import convert_baseline_to_MIREDO
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
from utils.UtilsFunction.OnnxParser import extract_loopdims


SUPPORTED_BASELINE_METHODS = ("zigzag", "ws", "cimloop", "cosa")
BASELINE_METHOD_LABELS = {
    "zigzag": "ZigZag_IMC",
    "ws": "WS",
    "cimloop": "CIMLoop",
    "cosa": "CoSA",
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
_COSA_ADAPTER_CACHE = {}
_COSA_MODEL_SUPPORT_CACHE = {}


def _loopdim_int(loopdim, key, default=1):
    value = loopdim.get(key, default)
    if value is None:
        return default
    return int(value)


def _format_grouped_loopdim(index, loopdim):
    fields = ["R", "S", "P", "Q", "C", "K", "G", "Stride"]
    dims = ", ".join(f"{key}={_loopdim_int(loopdim, key)}" for key in fields)
    return f"Layer-{index}({dims})"


def get_cosa_unsupported_reason(model_name, loopdims=None):
    """Return a human-readable reason when CoSA cannot run a whole model fairly."""
    cache_key = str(model_name)
    if loopdims is None and cache_key in _COSA_MODEL_SUPPORT_CACHE:
        return _COSA_MODEL_SUPPORT_CACHE[cache_key]

    if loopdims is None:
        model_path = repo_root() / "model" / f"{model_name}.onnx"
        if not model_path.is_file():
            reason = (
                f"CoSA baseline support cannot be checked for model '{model_name}' "
                f"because the ONNX model file is missing: {model_path}"
            )
            _COSA_MODEL_SUPPORT_CACHE[cache_key] = reason
            return reason
        _, loopdims = extract_loopdims(str(model_path))

    grouped = [
        (idx, loopdim)
        for idx, loopdim in enumerate(loopdims)
        if _loopdim_int(loopdim, "G", 1) != 1
    ]
    if not grouped:
        reason = None
    else:
        examples = "; ".join(
            _format_grouped_loopdim(idx, loopdim) for idx, loopdim in grouped[:3]
        )
        extra = "" if len(grouped) <= 3 else f"; ... total grouped/depthwise layers={len(grouped)}"
        reason = (
            f"CoSA whole-model baseline is disabled for model '{model_name}' because "
            f"it contains grouped/depthwise convolution layers with G>1: {examples}{extra}. "
            "The public CoSA formulation and repository interface use loop dimensions "
            "R,S,P,Q,C,K,N and do not expose a group dimension G; this MIREDO CoSA "
            "adapter therefore cannot export or convert these layers fairly. Running only "
            "the G=1 subset would not be a meaningful whole-model baseline."
        )

    _COSA_MODEL_SUPPORT_CACHE[cache_key] = reason
    return reason


def assert_cosa_model_supported(model_name, loopdims=None):
    reason = get_cosa_unsupported_reason(model_name=model_name, loopdims=loopdims)
    if reason is not None:
        raise ValueError(reason)


def _load_zigzag_cmes(model_name, architecture, objective):
    cache_key = (model_name, architecture, objective)
    if cache_key in _ZIGZAG_CME_CACHE:
        return _ZIGZAG_CME_CACHE[cache_key]

    opt_flag = objective_to_opt_flag(objective)
    model_path = repo_root() / "model" / f"{model_name}.onnx"
    compare_file_prefix = zigzag_cache_prefix(opt_flag, model_name, architecture)
    compare_pickle = compare_file_prefix.with_suffix(".pickle")
    compare_json = compare_file_prefix.with_suffix(".json")
    if not compare_pickle.is_file():
        get_hardware_performance_zigzag(
            workload=str(model_path),
            accelerator=f"Architecture.{architecture}",
            mapping="Config.zigzag_mapping",
            opt=opt_flag,
            dump_filename_pattern=str(compare_json),
            pickle_filename=str(compare_pickle),
        )

    with open(compare_pickle, "rb") as fp:
        cmes = pickle.load(fp)
    _ZIGZAG_CME_CACHE[cache_key] = cmes
    return cmes


def _get_cosa_adapter(model_name, architecture, map_path=None, output_root="output"):
    cache_key = (
        model_name,
        architecture,
        str(map_path) if map_path is not None else None,
        str(output_root),
    )
    adapter = _COSA_ADAPTER_CACHE.get(cache_key)
    if adapter is None:
        adapter = CoSABaselineAdapter(
            model=model_name,
            architecture=architecture,
            map_path=map_path,
            output_root=output_root,
        )
        _COSA_ADAPTER_CACHE[cache_key] = adapter
    return adapter


def _resolve_baseline_options(cimloop_macro, cosa_map, output_root, baseline_options):
    options = baseline_options or {}
    if cosa_map is None:
        cosa_map = options.get("cosa_map")
    if output_root == "output" and options.get("output_root") is not None:
        output_root = options["output_root"]
    if cimloop_macro == "raella_isca_2023" and options.get("cimloop_macro") is not None:
        cimloop_macro = options["cimloop_macro"]
    return cimloop_macro, cosa_map, output_root


def run_zigzag_baseline(acc, ops, loopdim, model_name, architecture, objective):
    cmes = _load_zigzag_cmes(model_name=model_name, architecture=architecture, objective=objective)
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


def run_cosa_baseline(acc, ops, loopdim, model_name, architecture, objective,
                      map_path=None, output_root="output"):
    assert_cosa_model_supported(model_name)
    adapter = _get_cosa_adapter(
        model_name=model_name,
        architecture=architecture,
        map_path=map_path,
        output_root=output_root,
    )
    baseline_layer = adapter.find_layer(loopdim)
    loops = LoopNest(acc=acc, ops=ops)
    loops = convert_baseline_to_MIREDO(loops=loops, baseline=baseline_layer)
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
            "policy": "cosa_baseline",
            "model": model_name,
            "architecture": architecture,
            "baseline_source": baseline_layer.source,
        },
    )


def run_cimloop_baseline(acc, ops, loopdim, model_name, architecture, objective,
                         macro_name="raella_isca_2023", output_root="output"):
    from Evaluation.CIMLoop.CompatibleCIMLoop import (
        cimloop_result_to_baseline_layer,
        cimloop_mapping_residuals,
        export_cimloop_workload,
        format_cimloop_residuals,
        miredo_incompatible_residuals,
        run_cimloop_mapper,
    )

    layer_name = (
        f"R{loopdim.get('R')}_S{loopdim.get('S')}_P{loopdim.get('P')}_Q{loopdim.get('Q')}_"
        f"C{loopdim.get('C')}_K{loopdim.get('K')}_G{loopdim.get('G', 1)}"
    )
    workload_path = export_cimloop_workload(
        loopdim=loopdim,
        model_name=model_name,
        layer_name=layer_name,
        output_root=output_root,
    )

    raw = run_cimloop_mapper(
        cimloop_model=model_name,
        layer_index=None,
        loopdim=loopdim,
        workload_path=workload_path,
        macro_name=macro_name,
        objective=objective,
        output_root=output_root,
        rectangular_compatible=False,
    )
    native_residuals = cimloop_mapping_residuals(raw)
    incompatible_residuals = miredo_incompatible_residuals(raw)
    rectangular_retry = False
    retry_reason = ""
    if incompatible_residuals:
        rectangular_retry = True
        retry_reason = format_cimloop_residuals(incompatible_residuals)
        raw = run_cimloop_mapper(
            cimloop_model=model_name,
            layer_index=None,
            loopdim=loopdim,
            workload_path=workload_path,
            macro_name=macro_name,
            objective=objective,
            output_root=output_root,
            rectangular_compatible=True,
        )
        retry_incompatible = miredo_incompatible_residuals(raw)
        if retry_incompatible:
            details = format_cimloop_residuals(retry_incompatible)
            raise ValueError(
                "CIMLoop rectangular-compatible retry still emitted MIREDO-incompatible "
                f"residual tiling: {details}"
            )
    baseline_layer = cimloop_result_to_baseline_layer(raw=raw, loopdim=loopdim, acc=acc)
    loops = LoopNest(acc=acc, ops=ops)
    loops = convert_baseline_to_MIREDO(loops=loops, baseline=baseline_layer)
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()

    return BaselineRunResult(
        method="cimloop",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": f"cimloop_{macro_name}",
            "model": model_name,
            "architecture": architecture,
            "macro": macro_name,
            "workload_path": str(workload_path),
            "mapping_output_dir": raw.get("output_dir"),
            "baseline_source": baseline_layer.source,
            "native_mapping_imperfect": bool(native_residuals),
            "native_incompatible_residuals": format_cimloop_residuals(incompatible_residuals),
            "rectangular_retry": rectangular_retry,
            "retry_reason": retry_reason,
            "mapspace_template": raw.get("mapspace_template"),
            "rectangular_compatible": raw.get("rectangular_compatible"),
        },
    )


def run_baseline(method, acc, ops, loopdim, model_name, architecture, objective,
                 cimloop_macro="raella_isca_2023", cosa_map=None, output_root="output",
                 baseline_options=None):
    method = method.lower()
    cimloop_macro, cosa_map, output_root = _resolve_baseline_options(
        cimloop_macro=cimloop_macro,
        cosa_map=cosa_map,
        output_root=output_root,
        baseline_options=baseline_options,
    )
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
    if method == "cosa":
        return run_cosa_baseline(
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=architecture,
            objective=objective,
            map_path=cosa_map,
            output_root=output_root,
        )
    if method == "cimloop":
        return run_cimloop_baseline(
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=architecture,
            objective=objective,
            macro_name=cimloop_macro,
            output_root=output_root,
        )
    raise ValueError(f"Unsupported baseline method: {method}")

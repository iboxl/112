import copy
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


def run_cimloop_baseline(acc, ops, loopdim, model_name, architecture, objective,
                         macro_name="raella_isca_2023"):
    from Evaluation.CIMLoop.CompatibleCIMLoop import (
        cimloop_result_to_miredo_units,
        find_matching_workload,
        run_cimloop_mapper,
    )

    match = find_matching_workload(model_name, loopdim)
    if match is None:
        raise ValueError(
            f"No CIMLoop workload match for {model_name} layer "
            f"R={loopdim.get('R')} S={loopdim.get('S')} "
            f"C={loopdim.get('C')} K={loopdim.get('K')} "
            f"P={loopdim.get('P')} Q={loopdim.get('Q')}"
        )

    raw = run_cimloop_mapper(
        cimloop_model=match["cimloop_model"],
        layer_index=match["layer_index"],
        loopdim=loopdim,
        macro_name=macro_name,
        objective=objective,
    )
    latency, energy = cimloop_result_to_miredo_units(raw)

    return BaselineRunResult(
        method="cimloop",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=None,
        dataflow=None,
        metadata={
            "policy": f"cimloop_{macro_name}",
            "model": model_name,
            "architecture": architecture,
            "macro": macro_name,
            "cimloop_model": match["cimloop_model"],
            "layer_index": match["layer_index"],
            "approximate_match": match["approximate"],
            "cimloop_cycles": raw["cycles"],
            "cimloop_energy_pj": raw["energy_pj"],
        },
    )


def run_baseline(method, acc, ops, loopdim, model_name, architecture, objective,
                 cimloop_macro="raella_isca_2023"):
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
            macro_name=cimloop_macro,
        )
    raise ValueError(f"Unsupported baseline method: {method}")

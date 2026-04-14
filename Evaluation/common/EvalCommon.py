import contextlib
import copy
import dataclasses
import datetime as dt
import json
import os
import pathlib
import platform
import pickle
import subprocess
import uuid

import gurobipy as gp
import psutil

from importlib import import_module

from Architecture.ArchSpec import CIM_Acc
from SolveMapping import SolveMapping
from utils.GlobalUT import CONST, FLAG, Logger
from utils.UtilsFunction.OnnxParser import extract_loopdims
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad


DEFAULT_MODELS = [
    "resnet18",
    "vgg19bn",
    "alexnet",
    "mobilenetV2",
    "EfficientNet-B0",
]


def repo_root():
    return pathlib.Path(__file__).resolve().parent.parent.parent


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def objective_to_opt_flag(objective):
    return {
        "Latency": "latency",
        "Energy": "energy",
        "EDP": "EDP",
    }[objective]


def objective_metric_value(objective, latency, energy):
    if objective == "Latency":
        return latency
    if objective == "Energy":
        return energy
    if objective == "EDP":
        return latency * energy * CONST.SCALINGFACTOR
    raise ValueError(f"Unsupported objective: {objective}")


def make_output_dir(prefix, output_dir=None):
    if output_dir is None:
        output_dir = f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_path = repo_root() / "output" / output_dir
    prepare_save_dir(str(output_path))
    return output_path


def setup_experiment_logger(output_dir, log_name="experiment.log"):
    Logger.setcfg(
        setcritical=False,
        setDebug=False,
        STD=False,
        file=str(pathlib.Path(output_dir) / log_name),
        nofile=False,
    )


def _run_git(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def collect_provenance(script_path):
    script_path = pathlib.Path(script_path).resolve()
    try:
        rel_script = str(script_path.relative_to(repo_root()))
    except ValueError:
        rel_script = str(script_path)

    version = gp.gurobi.version()
    solver_version = f"Gurobi {version[0]}.{version[1]}.{version[2]}"
    hardware_env = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.uname().processor or "unknown",
        "physical_cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        "logical_cores": psutil.cpu_count() or 1,
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    return {
        "repo": str(repo_root()),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_git(["rev-parse", "HEAD"]),
        "script": rel_script,
        "timestamp": now_iso(),
        "solver_version": solver_version,
        "hardware_env": hardware_env,
    }


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_experiment_json(output_dir, file_name, experiment_id, script_path, config, results, anomalies=None):
    payload = {
        "experiment_id": experiment_id,
        "provenance": collect_provenance(script_path),
        "config": config,
        "results": results,
        "anomalies": anomalies or [],
    }
    json_path = pathlib.Path(output_dir) / file_name
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
    return json_path


_ACC_CACHE_VERSION = 1
_acc_cache = None
_acc_cache_path = None


def _ensure_acc_cache_loaded():
    global _acc_cache, _acc_cache_path
    if _acc_cache is not None:
        return
    _acc_cache_path = repo_root() / "output" / ".acc_cache.pkl"
    if _acc_cache_path.is_file():
        try:
            with open(_acc_cache_path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("version") == _ACC_CACHE_VERSION:
                _acc_cache = blob["data"]
                return
        except Exception:
            pass
    _acc_cache = {}


def _save_acc_cache():
    if _acc_cache is None:
        return
    _acc_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_acc_cache_path, "wb") as fh:
        pickle.dump({"version": _ACC_CACHE_VERSION, "data": _acc_cache}, fh)


_ARCHITECTURE_SPEC_BUILDERS = {
    "CIM_ACC_TEMPLATE": "Architecture.templates.default",
}


def _try_build_from_spec(architecture):
    """优先 HardwareSpec 路径：返回 CIM_Acc 或 None（未注册/构建失败即 None）。"""
    module_path = _ARCHITECTURE_SPEC_BUILDERS.get(architecture)
    if module_path is None:
        return None
    try:
        spec_module = import_module(module_path)
        spec = spec_module.default_spec()
        return CIM_Acc.from_spec(spec)
    except Exception as exc:
        Logger.warning(f"HardwareSpec 路径失败({architecture}): {exc}；回退 legacy 构造")
        return None


def make_accelerator(architecture="CIM_ACC_TEMPLATE"):
    _ensure_acc_cache_loaded()
    if architecture not in _acc_cache:
        acc = _try_build_from_spec(architecture)
        if acc is None:
            acc_template = import_module(f"Architecture.{architecture}").accelerator
            acc = CIM_Acc(acc_template.cores[0])
        _acc_cache[architecture] = acc
        _save_acc_cache()
    return copy.deepcopy(_acc_cache[architecture])


def hardware_spec_from_acc(acc):
    memories = []
    for mem in range(1, acc.Num_mem):
        memories.append({
            "name": acc.mem2dict(mem),
            "size_bits": acc.memSize[mem],
            "bandwidth_bits_per_cycle": acc.bw[mem],
            "shared": acc.shareMemory[mem],
            "operands": [op for op, enabled in zip(["I", "W", "O"], [acc.mappingArray[t][mem] for t in range(3)]) if enabled],
        })
    return {
        "num_core": acc.Num_core,
        "dimX": acc.dimX,
        "dimY": acc.dimY,
        "spatial_unrolling": list(acc.SpUnrolling),
        "precision_final": acc.precision_final,
        "precision_psum": acc.precision_psum,
        "t_mac": acc.t_MAC,
        "cost_act_macro": acc.cost_ActMacro,
        "leakage_per_cycle": acc.leakage_per_cycle,
        "memories": memories,
    }


def classify_layer_type(loopdim):
    if loopdim["G"] > 1:
        if loopdim["C"] == 1 and loopdim["K"] == 1:
            return "depthwise"
        return "grouped_conv"
    if loopdim["R"] == 1 and loopdim["S"] == 1:
        return "pointwise"
    return f"conv{loopdim['R']}x{loopdim['S']}"


def classify_layer_family(loopdim):
    return "grouped_conv" if loopdim["G"] > 1 else "standard_conv"


def iter_model_layers(model_name):
    model_path = repo_root() / "model" / f"{model_name}.onnx"
    convs, loopdims = extract_loopdims(str(model_path))
    layers = []
    for layer_name, loopdim in zip(convs, loopdims):
        layers.append({
            "model": model_name,
            "layer": layer_name,
            "loopdim": copy.deepcopy(loopdim),
            "layer_type": classify_layer_type(loopdim),
            "layer_family": classify_layer_family(loopdim),
        })
    return layers


def normalize_loopdim_for_solver(loopdim):
    normalized = copy.deepcopy(loopdim)
    for dim_char in ["P", "Q", "H", "W"]:
        if normalized[dim_char] % 2 == 1 and normalized[dim_char] > 15:
            normalized[dim_char] += 1
    return normalized


@contextlib.contextmanager
def temporary_runtime_config(objective="Latency", time_limit=120, mip_focus=1,
                             enable_simu=True, debug_simu=False, presolve_search=True,
                             ablation_flags=None):
    old_state = {
        "CONST.FLAG_OPT": CONST.FLAG_OPT,
        "CONST.TIMELIMIT": CONST.TIMELIMIT,
        "CONST.MIPFOCUS": CONST.MIPFOCUS,
        "FLAG.SIMU": FLAG.SIMU,
        "FLAG.DEBUG_SIMU": FLAG.DEBUG_SIMU,
        "FLAG.PRESOLVE_SEARCH": FLAG.PRESOLVE_SEARCH,
        "FLAG.ABLATION_FIXED_DOUBLE_BUFFER": FLAG.ABLATION_FIXED_DOUBLE_BUFFER,
        "FLAG.ABLATION_SIMPLIFIED_PIPELINE": FLAG.ABLATION_SIMPLIFIED_PIPELINE,
    }
    try:
        CONST.FLAG_OPT = objective
        CONST.TIMELIMIT = time_limit
        CONST.MIPFOCUS = mip_focus
        FLAG.SIMU = enable_simu
        FLAG.DEBUG_SIMU = debug_simu
        FLAG.PRESOLVE_SEARCH = presolve_search
        if ablation_flags:
            for key, value in ablation_flags.items():
                setattr(FLAG, key, value)
        yield
    finally:
        CONST.FLAG_OPT = old_state["CONST.FLAG_OPT"]
        CONST.TIMELIMIT = old_state["CONST.TIMELIMIT"]
        CONST.MIPFOCUS = old_state["CONST.MIPFOCUS"]
        FLAG.SIMU = old_state["FLAG.SIMU"]
        FLAG.DEBUG_SIMU = old_state["FLAG.DEBUG_SIMU"]
        FLAG.PRESOLVE_SEARCH = old_state["FLAG.PRESOLVE_SEARCH"]
        FLAG.ABLATION_FIXED_DOUBLE_BUFFER = old_state["FLAG.ABLATION_FIXED_DOUBLE_BUFFER"]
        FLAG.ABLATION_SIMPLIFIED_PIPELINE = old_state["FLAG.ABLATION_SIMPLIFIED_PIPELINE"]


import hashlib

_CACHE_VERSION = 2
_mip_cache = None
_cache_path = None


def _default_cache_path():
    return repo_root() / "output" / ".mip_cache.pkl"


def _ensure_cache_loaded():
    global _mip_cache, _cache_path
    if _mip_cache is not None:
        return
    _cache_path = _default_cache_path()
    if _cache_path.is_file():
        try:
            with open(_cache_path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("version") == _CACHE_VERSION:
                _mip_cache = blob["data"]
                return
        except Exception:
            pass
    _mip_cache = {}


def _save_cache():
    if _mip_cache is None:
        return
    _cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_cache_path, "wb") as fh:
        pickle.dump({"version": _CACHE_VERSION, "data": _mip_cache}, fh)


def _hardware_fingerprint(acc):
    spec = hardware_spec_from_acc(acc)
    raw = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_cache_key(acc, solver_loopdim, objective, time_limit, mip_focus, ablation_flags):
    hw_key = _hardware_fingerprint(acc)
    dim_key = tuple(sorted(solver_loopdim.items()))
    abl_key = tuple(sorted(ablation_flags.items())) if ablation_flags else ()
    return (hw_key, dim_key, objective, time_limit, mip_focus, abl_key)


def clear_mip_cache():
    """Clear the persistent MIP result cache."""
    global _mip_cache
    _ensure_cache_loaded()
    _mip_cache.clear()
    _save_cache()


def mip_cache_get(acc, solver_loopdim, objective, time_limit, mip_focus, ablation_flags=None):
    """Look up a cached MIP result. Returns None on miss."""
    _ensure_cache_loaded()
    key = _make_cache_key(acc, solver_loopdim, objective, time_limit, mip_focus, ablation_flags)
    if key in _mip_cache:
        return copy.deepcopy(_mip_cache[key])
    return None


def mip_cache_put(acc, solver_loopdim, objective, time_limit, mip_focus, result, ablation_flags=None):
    """Store a MIP result in the persistent cache."""
    _ensure_cache_loaded()
    key = _make_cache_key(acc, solver_loopdim, objective, time_limit, mip_focus, ablation_flags)
    _mip_cache[key] = copy.deepcopy(result)
    _save_cache()


def run_miredo_layer(acc, loopdim, outputdir, objective="Latency", time_limit=120,
                     mip_focus=1, return_profile=True, ablation_flags=None):
    """Run MIREDO for one layer.

    Metric pruning starts from MIREDO's own incumbent instead of an external baseline.
    """
    solver_loopdim = normalize_loopdim_for_solver(loopdim)

    _ensure_cache_loaded()
    cache_key = _make_cache_key(acc, solver_loopdim, objective, time_limit, mip_focus, ablation_flags)
    if cache_key in _mip_cache:
        Logger.info(f"MIP cache hit: {solver_loopdim}")
        return copy.deepcopy(_mip_cache[cache_key])

    prepare_save_dir(str(outputdir))
    solver_ops = WorkLoad(loopDim=solver_loopdim)

    with temporary_runtime_config(
        objective=objective,
        time_limit=time_limit,
        mip_focus=mip_focus,
        enable_simu=True,
        debug_simu=False,
        presolve_search=True,
        ablation_flags=ablation_flags,
    ):
        solve_result = SolveMapping(
            acc=copy.deepcopy(acc),
            ops=solver_ops,
            bestMetric=CONST.MAX_POS,
            outputdir=str(outputdir),
            return_profile=return_profile,
        )

    if return_profile:
        result, mapping_profile = solve_result
    else:
        result, mapping_profile = solve_result, None

    dataflow = None
    dataflow_path = pathlib.Path(outputdir) / "Dataflow.pkl"
    if dataflow_path.is_file():
        with open(dataflow_path, "rb") as fh:
            dataflow = pickle.load(fh)

    layer_result = {
        "solver_latency": result[0],
        "solver_energy": result[1],
        "solver_edp": result[2],
        "simulator_latency": result[3],
        "simulator_energy": result[4],
        "simulator_profile": result[5],
        "mapping_profile": mapping_profile,
        "solver_loopdim": solver_loopdim,
        "dataflow": dataflow,
    }
    _mip_cache[cache_key] = copy.deepcopy(layer_result)
    _save_cache()
    return layer_result

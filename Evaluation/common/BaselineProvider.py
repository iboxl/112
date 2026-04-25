import hashlib
import json
from dataclasses import dataclass, field

from utils.GlobalUT import CONST

from Evaluation.common.EvalCommon import repo_root


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
    dataflow: object
    metadata: dict = field(default_factory=dict)

    @property
    def edp(self):
        return self.latency * self.energy * CONST.SCALINGFACTOR

    @classmethod
    def na(cls, method: str, objective: str, reason: str) -> "BaselineRunResult":
        return cls(
            method=method,
            objective=objective,
            latency=float("nan"),
            energy=float("nan"),
            profile=None,
            dataflow=None,
            metadata={"unsupported": True, "reason": reason},
        )

_DEFAULT_FP_CACHE = {}
_CIMLOOP_OUTPUTS_CACHE = {}
_COSA_OUTPUTS_CACHE = {}
_COSA_CONSTRAINED_OUTPUTS_CACHE = {}


def _spec_fingerprint(spec) -> str:
    raw = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _default_spec_fingerprint(architecture: str):
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
    from importlib import import_module
    from Evaluation.common.EvalCommon import _ARCHITECTURE_SPEC_BUILDERS
    module_path = _ARCHITECTURE_SPEC_BUILDERS.get(architecture)
    if module_path is None:
        return None
    try:
        return import_module(module_path).default_spec()
    except Exception:
        return None

def _cimloop_spec_suffix(architecture: str, spec) -> str:
    fp = _spec_fingerprint(spec)
    default_fp = _default_spec_fingerprint(architecture)
    if default_fp is not None and fp == default_fp:
        return ""
    return f"_{fp}"


def _cimloop_cache_root(architecture: str, objective: str, model_name: str, spec):
    suffix = "" if spec is None else _cimloop_spec_suffix(architecture, spec)
    base = repo_root() / "Evaluation" / "CIMLoop" / "output"
    dirname = f"{objective.lower()}_{model_name}_{architecture}{suffix}"
    return base / dirname


def _load_cimloop_outputs(model_name, architecture, objective, spec=None):
    import pickle
    if spec is None:
        spec = _resolve_default_spec(architecture)

    cache_root = _cimloop_cache_root(architecture, objective, model_name, spec)
    index_path = cache_root / "index.pickle"
    cache_key = (
        model_name, architecture, objective,
        _spec_fingerprint(spec) if spec is not None else "legacy",
    )

    if cache_key in _CIMLOOP_OUTPUTS_CACHE:
        return _CIMLOOP_OUTPUTS_CACHE[cache_key], cache_root, index_path

    outputs: dict = {}
    if index_path.is_file():
        with open(index_path, "rb") as fh:
            outputs = pickle.load(fh)
    _CIMLOOP_OUTPUTS_CACHE[cache_key] = outputs
    return outputs, cache_root, index_path


def _save_cimloop_index(index_path, outputs):
    import pickle
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fh:
        pickle.dump(outputs, fh)


def _parse_art_summary_estimators(art_path) -> dict:
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
    return sources


def _cosa_cache_root(architecture: str, objective: str, model_name: str, spec):
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
    cache_key = (
        model_name, architecture, objective,
        _spec_fingerprint(spec) if spec is not None else "legacy",
    )
    if cache_key in _COSA_OUTPUTS_CACHE:
        return _COSA_OUTPUTS_CACHE[cache_key], cache_root, index_path
    outputs: dict = {}
    if index_path.is_file():
        with open(index_path, "rb") as fh:
            outputs = pickle.load(fh)
    _COSA_OUTPUTS_CACHE[cache_key] = outputs
    return outputs, cache_root, index_path


def _save_cosa_index(index_path, outputs):
    import pickle
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fh:
        pickle.dump(outputs, fh)


def _cosa_constrained_cache_root(architecture: str, objective: str, model_name: str, spec):
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
    cache_key = (
        model_name, architecture, objective,
        _spec_fingerprint(spec) if spec is not None else "legacy",
    )
    if cache_key in _COSA_CONSTRAINED_OUTPUTS_CACHE:
        return _COSA_CONSTRAINED_OUTPUTS_CACHE[cache_key], cache_root, index_path
    outputs: dict = {}
    if index_path.is_file():
        with open(index_path, "rb") as fh:
            outputs = pickle.load(fh)
    _COSA_CONSTRAINED_OUTPUTS_CACHE[cache_key] = outputs
    return outputs, cache_root, index_path


def _save_cosa_constrained_index(index_path, outputs):
    import pickle
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fh:
        pickle.dump(outputs, fh)

_ADAPTER_FNS = {}


def _get_adapter(method: str):
    method = method.lower()
    if method in _ADAPTER_FNS:
        return _ADAPTER_FNS[method]

    if method == "ws":
        from Evaluation.WeightStationaryGenerator import (
            run_for_layer, supports_loopdim,
        )
    elif method == "zigzag":
        from Evaluation.Zigzag_imc.zigzag_adapter import (
            run_for_layer, supports_loopdim,
        )
    elif method == "cimloop":
        from Evaluation.CIMLoop.cimloop_adapter import (
            run_for_layer, supports_loopdim,
        )
    elif method == "cosa":
        from Evaluation.CoSA.cosa_adapter import (
            run_for_layer, supports_loopdim,
        )
    elif method == "cosa-constrained":
        from Evaluation.CoSA.cosa_adapter import (
            run_for_layer_constrained as run_for_layer,
            supports_loopdim_constrained as supports_loopdim,
        )
    else:
        raise ValueError(f"Unsupported baseline method: {method!r}")

    _ADAPTER_FNS[method] = (run_for_layer, supports_loopdim)
    return _ADAPTER_FNS[method]


def run_baseline(method, acc, ops, loopdim, model_name, architecture, objective,
                 raise_on_unsupported=True):
    method = method.lower()
    run_for_layer, supports_loopdim = _get_adapter(method)
    reason = supports_loopdim(loopdim)
    if reason is not None:
        if raise_on_unsupported:
            raise NotImplementedError(reason)
        return BaselineRunResult.na(method=method, objective=objective, reason=reason)
    return run_for_layer(
        acc=acc, ops=ops, loopdim=loopdim,
        model_name=model_name, architecture=architecture, objective=objective,
    )

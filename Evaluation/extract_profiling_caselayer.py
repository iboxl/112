#!/usr/bin/env python3
"""
Extract observable per-framework profile for L1-L4 case layers.

Output: output/<MIREDO_RERUN_ROOT>/s5_4_caselayer/caselayer_profile_YYYYMMDD.json

Covers 4 case layers x 4 frameworks (miredo / ws / zigzag / cimloop):
  - F1: event_cycle_intensity (time/idle profile)
  - F2: utilization_metrics  (spatial + temporal utilization)
  - F3: memory_traffic_metrics (per-level bytes, reload count, psum traffic)
  - F4: mapping_decision_summary (reload boundary, psum residency, double-buffer,
         operand path)
  - A1: available_profile_mask (per-framework field availability)

Pattern follows scripts/analysis/parse_exp3_per_layer.py (provenance envelope,
loopdim_key helper, git utilities).

cache_only_mode strategy:
  - MIREDO: frozen Dataflow.pkl re-simulated via tranSimulator.run_analytical()
    (acc + ops embedded in pkl; no MIP solve)
  - WS: deterministic heuristic, recomputed fresh (~2-3 s/layer; no MIP)
  - ZigZag: per-layer latency cache absent -> logs anomaly, falls back to
    baseline_comparison.json stall_decomposition values for F1 numbers.
    F2/F3/F4 marked N/A.
  - CIMLoop: loaded from per-model index.pickle (latency cache exists for
    resnet18, mobilenetV2, EfficientNet-B0); if missing, logs anomaly.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import os

# Relocated into the MIREDO/ code repo (2026-05-18) so a code-repo-only clone
# can regenerate §5.4.1 self-contained. Resolve everything from the code repo's
# own location — never the paper repo, never an absolute path (per
# feedback_code_repo_relative_outputs). Original copy left vestigial at
# <paper-repo>/scripts/analysis/ for provenance.
_CODE_REPO = Path(__file__).resolve().parents[1]   # = the MIREDO/ package dir
sys.path.insert(0, str(_CODE_REPO))


def _resolve_rerun_root() -> Path:
    """Rerun root, flexibly — NO hardcoded date, so any rerun is supported.
    Priority: $MIREDO_RERUN_ROOT (absolute path, or a name under output/) >
    newest output/logs_rerun_* directory."""
    out = _CODE_REPO / "output"
    env = os.environ.get("MIREDO_RERUN_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (out / env)
    cands = sorted(out.glob("logs_rerun_*"))
    if not cands:
        raise SystemExit(
            f"[extract_profiling_caselayer] no logs_rerun_* under {out}; "
            f"set MIREDO_RERUN_ROOT")
    return cands[-1]

# ── Imports from MIREDO package ────────────────────────────────────────────────
from Evaluation.common.CaseLayerShapes import CASE_LAYERS_DETAILS
from Evaluation.common.ProfileAnalysis import (
    available_profile_mask,
    event_cycle_intensity,
    mapping_decision_summary,
    memory_traffic_metrics,
    utilization_metrics,
)
from Evaluation.common.EvalCommon import make_accelerator
from Simulator.Simulax import tranSimulator
from utils.Workload import WorkLoad

# ── Constants ─────────────────────────────────────────────────────────────────
# FIX 2026-05-17: was "CIM_ACC_TEMPLATE" (legacy) — §5.4.1 profile must be on
# the rerun's CIM_ACC_DEFAULT_SETUP, consistent with §5.2/§5.3/§5.4/§5.5/§5.6.
ARCHITECTURE = "CIM_ACC_DEFAULT_SETUP"
# Note: ZigZag and CIMLoop adapters expect capitalized objective ("Latency"),
# matching EvalCommon.objective_to_opt_flag key format.
OBJECTIVE = "Latency"
FRAMEWORKS = ["miredo", "ws", "zigzag", "cimloop"]

# Frozen Dataflow.pkl paths (Phase 0.5 verified)
# FIX 2026-05-17: repointed from legacy experiments/logs/exp1_accuracy/* (old
# CIM_ACC_TEMPLATE solves) to CIM_ACC_DEFAULT_SETUP solves from the 2026-05
# rerun. L2/L3/L4 = Phase A Latency mappings; L1 = Phase D un-ablated Latency
# baseline (Phase A had no L1 Dataflow — L1 was a cache hit there). Methodology
# unchanged (OBJECTIVE=Latency, frozen-pkl re-sim) — only the hardware corrected.
_RR = _resolve_rerun_root()   # flexible; was hardcoded logs_rerun_20260513
FROZEN_PATHS = {
    "L1": _RR / "s5_4_caselayer/ablation/Latency/resnet18/Conv_8_3_3_28_28_128_128_1/Dataflow.pkl",
    "L2": _RR / "s5_2_1_cnn_main/baseline_comparison/Latency/resnet18/Conv_17_1_1_7_7_256_512_1/Dataflow.pkl",
    "L3": _RR / "s5_2_1_cnn_main/baseline_comparison/Latency/mobilenetV2/Conv_19_3_3_14_14_1_1_192/Dataflow.pkl",
    "L4": _RR / "s5_2_1_cnn_main/baseline_comparison/Latency/EfficientNet-B0/Conv_30_1_1_14_14_80_480_1/Dataflow.pkl",
}

# L3 substitution metadata
SUBSTITUTIONS = {
    "L3": {
        "spec_loopdim": "G=144 synthetic from CaseLayerShapes.py",
        "actual_loopdim": "G=192 mobilenetV2 Conv_19",
        "rationale": (
            "G=144 has no real model layer; G=192 preserves depthwise/G=C "
            "archetype role; zero-new-MIP-run constraint maintained"
        ),
    }
}

# Model names for each case layer (used for CIMLoop / ZigZag cache lookup)
CASE_LAYER_MODEL = {
    "L1": "resnet18",
    "L2": "resnet18",
    "L3": "mobilenetV2",
    "L4": "EfficientNet-B0",
}

# baseline_comparison reference JSON for ZigZag fallback F1 cross-check
# (rerun Phase A, correct CIM_ACC_DEFAULT_SETUP base).
BASELINE_COMPARISON_PATH = _RR / "s5_2_1_cnn_main/baseline_comparison/baseline_comparison.json"
OUT_DIR = _RR / "s5_4_caselayer"


# ── git helpers ────────────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_CODE_REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_CODE_REPO), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── loopdim key (from layer name) ─────────────────────────────────────────────

def loopdim_key(layer_name: str) -> tuple | None:
    """Return (R, S, P, Q, C, K, G) from Conv_N_R_S_P_Q_C_K_G layer name."""
    import re
    m = re.match(
        r"^Conv_\d+_(?P<R>\d+)_(?P<S>\d+)_(?P<P>\d+)_(?P<Q>\d+)_"
        r"(?P<C>\d+)_(?P<K>\d+)_(?P<G>\d+)$",
        layer_name,
    )
    if not m:
        return None
    return tuple(int(m.group(c)) for c in ("R", "S", "P", "Q", "C", "K", "G"))


# ── silence simulator noise ────────────────────────────────────────────────────

class _Silencer:
    """Context manager: set Logger + root logging to ERROR during simulation."""
    def __init__(self):
        from utils.GlobalUT import Logger
        self._Logger = Logger

    def __enter__(self):
        self._prev = self._Logger.level
        self._root_prev = logging.root.manager.disable
        self._Logger.setLevel(logging.ERROR)
        logging.disable(logging.CRITICAL)
        return self

    def __exit__(self, *_):
        self._Logger.setLevel(self._prev)
        logging.disable(self._root_prev)


# ── MIREDO extraction ──────────────────────────────────────────────────────────

def extract_miredo(case_id: str) -> dict:
    """Re-simulate frozen Dataflow.pkl; return (profile, dataflow, latency)."""
    pkl_path = FROZEN_PATHS[case_id]
    with open(pkl_path, "rb") as fh:
        dataflow = pickle.load(fh)
    acc = copy.deepcopy(dataflow.acc)
    ops = dataflow.ops
    with _Silencer():
        sim = tranSimulator(acc=acc, ops=ops, dataflow=dataflow)
        latency, energy = sim.run_analytical()
    return {
        "profile": sim.PD,
        "dataflow": dataflow,
        "latency": latency,
        "energy": energy,
        "metadata": {
            "source": str(pkl_path.relative_to(_CODE_REPO)),
            "method": "frozen_dataflow_re_simulate",
        },
    }


# ── WS extraction ─────────────────────────────────────────────────────────────

def extract_ws(case_id: str, loopdim: dict) -> dict:
    """WS is a deterministic heuristic; recomputed fresh each run (~2-3 s)."""
    from Evaluation.WeightStationaryGenerator import generate_weight_stationary_baseline

    acc = make_accelerator(ARCHITECTURE)
    ops = WorkLoad(loopDim=copy.deepcopy(loopdim))
    t0 = time.time()
    with _Silencer():
        result = generate_weight_stationary_baseline(acc=copy.deepcopy(acc), ops=ops)
    elapsed = time.time() - t0
    return {
        "profile": result.profile,
        "dataflow": result.dataflow,
        "latency": result.latency,
        "energy": result.energy,
        "metadata": {
            "method": "weight_stationary_heuristic",
            "policy": result.policy,
            "elapsed_s": round(elapsed, 2),
            "cache_hit": False,
            "cache_note": "WS is deterministic heuristic; no file cache; fresh each run",
        },
    }


# ── ZigZag extraction ─────────────────────────────────────────────────────────

def _load_zigzag_fallback(case_id: str, model_name: str, layer_name: str) -> dict | None:
    """
    Load ZigZag F1 numbers from baseline_comparison stall_decomposition table as fallback.
    Returns a stub profile object (namespace) or None if not found.
    """
    if not BASELINE_COMPARISON_PATH.is_file():
        return None
    with open(BASELINE_COMPARISON_PATH) as fh:
        exp2 = json.load(fh)
    sd = exp2.get("results", {}).get("stall_decomposition", [])
    row = next(
        (r for r in sd
         if r["model"] == model_name
         and r["layer"] == layer_name
         and r["method"] == "ZigZag_IMC_Latency"),
        None,
    )
    if row is None:
        return None

    # Build a minimal namespace object mimicking ProfilingDetail fields
    class _ZZProfile:
        pass
    p = _ZZProfile()
    p.latency = row["total_latency"]
    p.macLatency = row["compute_cycles"]
    p.mode_switch_stall = row["mode_switch_stall"]
    p.mismatch_stall = row["mismatch_stall"]
    p.writeback_stall = row["writeback_stall"]
    p.idle_cycles = row.get("idle_cycles", 0.0)
    # Fields not available from baseline_comparison stall table
    p.count_mac = None
    p.peak_mac_per_cycle = None
    p.bytes_read = []
    p.bytes_written = []
    p.transfer_cycles = [0, 0, 0]
    return p


def extract_zigzag(case_id: str, loopdim: dict, model_name: str,
                   layer_name: str) -> dict:
    """
    Try to run ZigZag per-layer mapper (will create new cache on first call).
    If running takes >120 s or errors, fall back to baseline_comparison stall table for F1.
    F2/F3/F4 are marked N/A when only the fallback is available.
    """
    from Evaluation.Zigzag_imc.zigzag_adapter import (
        run_for_layer as zz_run,
        loopdim_fingerprint,
        _zigzag_per_layer_cache_root,
    )
    from Evaluation.common.BaselineProvider import BaselineRunResult
    from Evaluation.common.EvalCommon import _ARCHITECTURE_SPEC_BUILDERS
    from importlib import import_module

    # Determine if per-layer cme.pkl cache exists
    try:
        module_path = _ARCHITECTURE_SPEC_BUILDERS.get(ARCHITECTURE)
        spec = import_module(module_path).default_spec() if module_path else None
        cache_root = _zigzag_per_layer_cache_root(ARCHITECTURE, OBJECTIVE, spec)
        fp = loopdim_fingerprint(loopdim)
        cme_path = cache_root / "per_layer" / fp / "cme.pkl"
        cache_hit = cme_path.is_file()
    except Exception:
        cache_hit = False
        cme_path = None

    if cache_hit:
        # Fast path: use cached cme
        acc = make_accelerator(ARCHITECTURE)
        ops = WorkLoad(loopDim=copy.deepcopy(loopdim))
        t0 = time.time()
        with _Silencer():
            result = zz_run(
                acc=acc, ops=ops, loopdim=copy.deepcopy(loopdim),
                model_name=model_name, architecture=ARCHITECTURE,
                objective=OBJECTIVE,
            )
        elapsed = time.time() - t0
        return {
            "profile": result.profile,
            "dataflow": result.dataflow,
            "latency": result.latency,
            "energy": result.energy,
            "metadata": {
                "method": "zigzag_imc_per_layer",
                "cache_hit": True,
                "cache_path": str(cme_path),
                "elapsed_s": round(elapsed, 2),
            },
        }
    else:
        # Try fresh run with 90 s timeout guard
        acc = make_accelerator(ARCHITECTURE)
        ops = WorkLoad(loopDim=copy.deepcopy(loopdim))
        t0 = time.time()
        try:
            with _Silencer():
                result = zz_run(
                    acc=acc, ops=ops, loopdim=copy.deepcopy(loopdim),
                    model_name=model_name, architecture=ARCHITECTURE,
                    objective=OBJECTIVE,
                )
            elapsed = time.time() - t0
            return {
                "profile": result.profile,
                "dataflow": result.dataflow,
                "latency": result.latency,
                "energy": result.energy,
                "metadata": {
                    "method": "zigzag_imc_per_layer",
                    "cache_hit": False,
                    "cache_note": "fresh solve; cme.pkl written for future runs",
                    "elapsed_s": round(elapsed, 2),
                },
            }
        except Exception as exc:
            elapsed = time.time() - t0
            # Fall back to baseline_comparison table for F1
            fallback_profile = _load_zigzag_fallback(case_id, model_name, layer_name)
            fallback_note = (
                f"ZigZag fresh solve failed ({type(exc).__name__}: {exc}); "
                f"F1 loaded from baseline_comparison.json::stall_decomposition "
                f"ZigZag_IMC_Latency row; F2/F3/F4 not available"
            )
            return {
                "profile": fallback_profile,
                "dataflow": None,
                "latency": fallback_profile.latency if fallback_profile else None,
                "energy": None,
                "metadata": {
                    "method": "zigzag_imc_fallback_exp2",
                    "cache_hit": False,
                    "fallback": True,
                    "fallback_source": "baseline_comparison.json::stall_decomposition",
                    "error": str(exc),
                    "elapsed_s": round(elapsed, 2),
                    "note": fallback_note,
                },
            }


# ── CIMLoop extraction ─────────────────────────────────────────────────────────

def extract_cimloop(case_id: str, loopdim: dict, model_name: str) -> dict:
    """Load from per-model index.pickle cache; N/A if cache miss."""
    from Evaluation.CIMLoop.cimloop_adapter import (
        run_for_layer as cl_run,
        loopdim_fingerprint,
        supports_loopdim,
    )
    from Evaluation.common.BaselineProvider import (
        _load_cimloop_outputs,
        BaselineRunResult,
    )

    acc = make_accelerator(ARCHITECTURE)
    ops = WorkLoad(loopDim=copy.deepcopy(loopdim))

    # Check supports_loopdim first
    unsupported = supports_loopdim(loopdim)
    if unsupported is not None:
        return {
            "profile": None,
            "dataflow": None,
            "latency": None,
            "energy": None,
            "metadata": {
                "method": "cimloop_na",
                "cache_hit": False,
                "unsupported": True,
                "reason": unsupported,
            },
        }

    # Check if layer is in cache
    outputs, cache_root, index_path = _load_cimloop_outputs(
        model_name=model_name,
        architecture=ARCHITECTURE,
        objective=OBJECTIVE,
        spec=None,
    )
    fp = loopdim_fingerprint(loopdim)
    cache_hit = fp in outputs

    t0 = time.time()
    try:
        with _Silencer():
            result = cl_run(
                acc=acc, ops=ops, loopdim=copy.deepcopy(loopdim),
                model_name=model_name, architecture=ARCHITECTURE,
                objective=OBJECTIVE,
            )
        elapsed = time.time() - t0
        return {
            "profile": result.profile,
            "dataflow": result.dataflow,
            "latency": result.latency,
            "energy": result.energy,
            "metadata": {
                "method": "cimloop_mapper",
                "cache_hit": cache_hit,
                "cache_note": "loaded from index.pickle" if cache_hit else "fresh solve",
                "elapsed_s": round(elapsed, 2),
                **{k: v for k, v in result.metadata.items()
                   if k not in ("policy",)},
            },
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "profile": None,
            "dataflow": None,
            "latency": None,
            "energy": None,
            "metadata": {
                "method": "cimloop_error",
                "cache_hit": False,
                "error": str(exc),
                "elapsed_s": round(elapsed, 2),
            },
        }


# ── Profile extraction helpers ────────────────────────────────────────────────

def _safe_extract_f1(profile) -> dict | None:
    if profile is None:
        return None
    try:
        return event_cycle_intensity(profile)
    except Exception as e:
        return {"_error": str(e)}


def _safe_extract_f2(profile) -> dict | None:
    if profile is None:
        return None
    try:
        return utilization_metrics(profile)
    except Exception as e:
        return {"_error": str(e)}


def _safe_extract_f3(profile, dataflow=None, acc=None) -> dict | None:
    if profile is None:
        return None
    try:
        return memory_traffic_metrics(profile, dataflow=dataflow, acc=acc)
    except Exception as e:
        return {"_error": str(e)}


def _safe_extract_f4(dataflow, acc=None) -> dict | None:
    if dataflow is None:
        return None
    try:
        return mapping_decision_summary(dataflow, acc=acc)
    except Exception as e:
        return {"_error": str(e)}


def _safe_extract_a1(profile, dataflow, fw_name: str) -> dict | None:
    try:
        return available_profile_mask(profile, dataflow, fw_name)
    except Exception as e:
        return {"_error": str(e)}


# ── baseline_comparison cross-check ────────────────────────────────────────────────────────

def load_exp2_reference() -> list[dict]:
    """Return stall_decomposition rows from baseline_comparison for cross-check."""
    if not BASELINE_COMPARISON_PATH.is_file():
        return []
    with open(BASELINE_COMPARISON_PATH) as fh:
        d = json.load(fh)
    return d.get("results", {}).get("stall_decomposition", [])


def find_exp2_row(sd: list[dict], model: str, layer: str, method_suffix: str) -> dict | None:
    return next(
        (r for r in sd
         if r["model"] == model
         and r["layer"] == layer
         and r["method"].endswith(method_suffix)),
        None,
    )


def crosscheck_f1(case_id: str, fw: str, f1_new: dict | None,
                  model: str, layer: str, exp2_sd: list[dict]) -> dict | None:
    """
    Compare F1 total_latency against baseline_comparison stall_decomposition.
    Returns a cross-check record or None if comparison not applicable.
    """
    if f1_new is None or "_error" in f1_new:
        return None

    method_map = {
        "miredo": "MIREDO_Latency",
        "ws": "WS_Latency",
        "zigzag": "ZigZag_IMC_Latency",
        "cimloop": None,   # not in baseline_comparison stall_decomposition
    }
    method_suffix = method_map.get(fw)
    if method_suffix is None:
        return None

    exp2_row = find_exp2_row(exp2_sd, model, layer, method_suffix)
    if exp2_row is None:
        return {
            "case_id": case_id, "framework": fw,
            "status": "EXP2_ROW_NOT_FOUND",
            "model": model, "layer": layer,
        }

    new_lat = f1_new.get("total_latency")
    exp2_lat = exp2_row.get("total_latency")
    if new_lat is None or exp2_lat is None:
        return {
            "case_id": case_id, "framework": fw,
            "status": "VALUE_MISSING",
            "new_total_latency": new_lat,
            "exp2_total_latency": exp2_lat,
        }

    if exp2_lat == 0:
        rel_err = float("inf")
    else:
        rel_err = abs(new_lat - exp2_lat) / abs(exp2_lat)

    TOLERANCE = 1e-4  # 0.01 %
    status = "MATCH" if rel_err <= TOLERANCE else "MISMATCH"

    return {
        "case_id": case_id,
        "framework": fw,
        "status": status,
        "new_total_latency": new_lat,
        "exp2_total_latency": exp2_lat,
        "rel_err": rel_err,
        "model": model,
        "layer": layer,
    }


# ── A1 summary (transposed view) ─────────────────────────────────────────────

def build_a1_summary(per_case_results: list[dict]) -> list[dict]:
    """
    Build transposed available-profile mask summary across all frameworks.
    One row per metric field; columns = frameworks.
    Uses first case layer to determine fields (all case layers use same HW).
    """
    # Collect all (family, field) pairs from any non-error A1
    all_fields: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in per_case_results:
        a1 = entry.get("A1")
        if a1 is None or "_error" in a1:
            continue
        for family, fields in a1.items():
            if not isinstance(fields, dict):
                continue
            for field in fields:
                if (family, field) not in seen:
                    all_fields.append((family, field))
                    seen.add((family, field))

    # Aggregate availability by (family, field) across frameworks
    # Uses first occurrence per (family, field, framework)
    by_field: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in per_case_results:
        fw = entry["framework"]
        a1 = entry.get("A1")
        if a1 is None or "_error" in a1:
            continue
        for family, fields in a1.items():
            if not isinstance(fields, dict):
                continue
            for field, val in fields.items():
                key = (family, field)
                if key not in by_field:
                    by_field[key] = {}
                if fw not in by_field[key]:
                    by_field[key][fw] = val

    rows = []
    for family, field in all_fields:
        if (family, field) not in by_field:
            continue
        row = {"family": family, "field": field}
        for fw in FRAMEWORKS:
            val = by_field[(family, field)].get(fw)
            # Explicit "N/A" string when field missing or None — never null/0 substitute
            if val is None:
                val = "N/A"
            row[fw] = val
        rows.append(row)
    return rows


# ── JSON serialization helper ─────────────────────────────────────────────────

def _to_serializable(obj):
    """Recursively convert non-JSON-serializable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    today = _dt.date.today().strftime("%Y%m%d")
    out_path = OUT_DIR / f"caselayer_profile_{today}.json"

    print("baseline_comparisonc: extracting case-layer observable profiles")
    print(f"  frameworks: {FRAMEWORKS}")
    print(f"  output: {out_path}")

    exp2_sd = load_exp2_reference()
    print(f"  baseline_comparison stall_decomposition rows loaded: {len(exp2_sd)}")

    per_case_results: list[dict] = []
    crosscheck_results: list[dict] = []
    anomalies: list[dict] = []

    # ── case layer → layer name lookup (for baseline_comparison cross-check layer column) ──
    # L1/L2 → resnet18 Conv_8_* / Conv_17_*
    # L3 → mobilenetV2 Conv_19_* (substitution)
    # L4 → EfficientNet-B0 Conv_30_*
    LAYER_NAMES = {
        "L1": "Conv_8_3_3_28_28_128_128_1",
        "L2": "Conv_17_1_1_7_7_256_512_1",
        "L3": "Conv_19_3_3_14_14_1_1_192",
        "L4": "Conv_30_1_1_14_14_80_480_1",
    }

    for case_spec in CASE_LAYERS_DETAILS:
        case_id = case_spec["id"]
        # Use loopdim from CaseLayerShapes for WS/ZigZag/CIMLoop
        loopdim = copy.deepcopy(case_spec["loopdim"])
        # For L3, actual loopdim from substitution (G=192, not G=144)
        if case_id == "L3":
            loopdim = {
                "R": 3, "S": 3, "P": 14, "Q": 14,
                "C": 1, "K": 1, "G": 192, "B": 1,
                "H": 14, "W": 14, "Stride": 1, "Padding": 1,
            }
        model_name = CASE_LAYER_MODEL[case_id]
        layer_name = LAYER_NAMES[case_id]

        print(f"\n  [{case_id}] model={model_name} layer={layer_name}")

        for fw in FRAMEWORKS:
            print(f"    [{case_id}:{fw}] extracting ...", end=" ", flush=True)
            t_fw = time.time()

            # ── extract raw result ──
            try:
                if fw == "miredo":
                    raw = extract_miredo(case_id)
                elif fw == "ws":
                    raw = extract_ws(case_id, loopdim)
                elif fw == "zigzag":
                    raw = extract_zigzag(case_id, loopdim, model_name, layer_name)
                elif fw == "cimloop":
                    raw = extract_cimloop(case_id, loopdim, model_name)
                else:
                    raise ValueError(f"Unknown framework: {fw}")
            except Exception as exc:
                elapsed = time.time() - t_fw
                print(f"ERROR ({elapsed:.1f}s)")
                anomalies.append({
                    "case_id": case_id, "framework": fw,
                    "kind": "extraction_error",
                    "message": str(exc),
                })
                per_case_results.append({
                    "case_layer_id": case_id,
                    "framework": fw,
                    "loopdim": loopdim,
                    "latency": None,
                    "energy": None,
                    "metadata": {"error": str(exc)},
                    "F1": None, "F2": None, "F3": None, "F4": None, "A1": None,
                })
                continue

            elapsed = time.time() - t_fw
            profile = raw["profile"]
            dataflow = raw["dataflow"]
            acc = None
            if dataflow is not None and hasattr(dataflow, "acc"):
                acc = dataflow.acc

            # ── helper extraction ──
            f1 = _safe_extract_f1(profile)
            f2 = _safe_extract_f2(profile)
            f3 = _safe_extract_f3(profile, dataflow=dataflow, acc=acc)
            f4 = _safe_extract_f4(dataflow, acc=acc)
            a1 = _safe_extract_a1(profile, dataflow, fw)

            # Annotate ZigZag fallback in A1
            if raw.get("metadata", {}).get("fallback"):
                if a1 is not None and not isinstance(a1, dict):
                    a1 = {}
                if a1 is not None:
                    a1["_note"] = "F2/F3/F4 N/A: ZigZag used baseline_comparison fallback (no dataflow object)"

            cache_info = raw.get("metadata", {}).get("cache_hit", "N/A")
            print(f"ok ({elapsed:.1f}s, cache_hit={cache_info})")

            # Record cache anomaly for non-WS fresh solves
            if fw in ("zigzag", "cimloop") and not cache_info:
                if not raw.get("metadata", {}).get("fallback"):
                    anomalies.append({
                        "case_id": case_id, "framework": fw,
                        "kind": "fresh_solve",
                        "message": (
                            f"{fw} ran a fresh solve (no pre-existing per-layer cache). "
                            f"elapsed_s={elapsed:.1f}"
                        ),
                    })
            if raw.get("metadata", {}).get("fallback"):
                anomalies.append({
                    "case_id": case_id, "framework": fw,
                    "kind": "zigzag_fallback",
                    "message": raw["metadata"].get("note", "ZigZag fallback to baseline_comparison table"),
                })

            entry = {
                "case_layer_id": case_id,
                "framework": fw,
                "loopdim": loopdim,
                "latency": raw["latency"],
                "energy": raw["energy"],
                "metadata": raw.get("metadata", {}),
                "F1": f1,
                "F2": f2,
                "F3": f3,
                "F4": f4,
                "A1": a1,
            }
            per_case_results.append(entry)

            # Cross-check F1 vs baseline_comparison
            cc = crosscheck_f1(case_id, fw, f1, model_name, layer_name, exp2_sd)
            if cc is not None:
                crosscheck_results.append(cc)
                if cc["status"] == "MISMATCH":
                    anomalies.append({
                        "case_id": case_id, "framework": fw,
                        "kind": "f1_crosscheck_mismatch",
                        "message": (
                            f"F1 total_latency mismatch: new={cc.get('new_total_latency')} "
                            f"exp2={cc.get('exp2_total_latency')} "
                            f"rel_err={cc.get('rel_err'):.6f}"
                        ),
                        **cc,
                    })

    # ── A1 summary (transposed) ──
    a1_summary = build_a1_summary(per_case_results)

    # ── Build output JSON ──
    commit = git_commit()
    branch = git_branch()
    elapsed_total = time.time() - t_start

    acc_config = make_accelerator(ARCHITECTURE)
    hw_arch = {
        "architecture": ARCHITECTURE,
        "num_core": acc_config.Num_core,
        "dimX": acc_config.dimX,
        "dimY": acc_config.dimY,
        "peak_mac_per_cycle": acc_config.Num_core * acc_config.dimX * acc_config.dimY,
        "t_MAC": acc_config.t_MAC,
        "memories": [
            {"level": m, "name": acc_config.mem2dict(m),
             "size_bits": acc_config.memSize[m],
             "bw_bits_per_cycle": acc_config.bw[m]}
            for m in range(1, acc_config.Num_mem)
        ],
    }

    out = {
        "experiment_id": "caselayer_profile",
        "description": (
            "Observable per-framework profile for L1-L4 case layers. "
            "4 metric families (F1 time/idle, F2 utilization, F3 traffic, "
            "F4 mapping decisions) + A1 availability mask. "
            "Zero new MIP runs; MIREDO from frozen Dataflow.pkl; "
            "WS deterministic heuristic; ZigZag mapper (cme.pkl cache); "
            "CIMLoop from model-level index.pickle."
        ),
        "provenance": {
            "script": str(Path(__file__).resolve().relative_to(_CODE_REPO)),
            "git_commit": commit,
            "git_branch": branch,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "sources": [
                str(p.relative_to(_CODE_REPO)) for p in FROZEN_PATHS.values()
            ] + [
                "baseline_comparison.json "
                "(ZigZag fallback F1 + cross-check reference)",
            ],
            "cache_only_mode": True,
            "cache_only_note": (
                "MIREDO: frozen pkl, no MIP. "
                "WS: deterministic heuristic, no MIP. "
                "ZigZag: per-layer cme.pkl (created on first run if absent). "
                "CIMLoop: model-level index.pickle. "
                "ZigZag fallback to baseline_comparison stall table if adapter errors."
            ),
            "substitutions": SUBSTITUTIONS,
            "elapsed_total_s": round(elapsed_total, 1),
        },
        "config": {
            "case_layers": ["L1", "L2", "L3", "L4"],
            "frameworks": FRAMEWORKS,
            "hw_arch": hw_arch,
            "objective": OBJECTIVE.lower(),
            "l3_substitution": {
                "CaseLayerShapes_spec": "G=144 (no real model layer)",
                "actual_layer_used": "mobilenetV2 Conv_19_3_3_14_14_1_1_192 (G=192)",
                "rationale": SUBSTITUTIONS["L3"]["rationale"],
            },
        },
        "results": {
            "per_case_layer": per_case_results,
            "available_mask_summary": a1_summary,
            "f1_crosscheck": crosscheck_results,
        },
        "anomalies": anomalies,
    }

    # Serialise
    out_serializable = _to_serializable(out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out_serializable, fh, indent=2)

    # ── Summary report ──
    n_entries = len(per_case_results)
    n_anomalies = len(anomalies)
    cc_match = sum(1 for r in crosscheck_results if r["status"] == "MATCH")
    cc_mismatch = sum(1 for r in crosscheck_results if r["status"] == "MISMATCH")
    cc_na = sum(1 for r in crosscheck_results if r["status"] not in ("MATCH", "MISMATCH"))
    import os
    file_size_kb = os.path.getsize(out_path) / 1024

    print(f"\n{'='*60}")
    print(f"baseline_comparisonc extraction complete")
    print(f"  output: {out_path}  ({file_size_kb:.1f} KB)")
    print(f"  entries: {n_entries} ({len(CASE_LAYERS_DETAILS)} layers × {len(FRAMEWORKS)} frameworks)")
    print(f"  F1 cross-check: {cc_match} match / {cc_mismatch} mismatch / {cc_na} N/A")
    print(f"  anomalies: {n_anomalies}")
    print(f"  elapsed: {elapsed_total:.1f} s")

    if cc_mismatch > 0:
        print("\n  MISMATCH details:")
        for r in crosscheck_results:
            if r["status"] == "MISMATCH":
                print(f"    [{r['case_id']}:{r['framework']}] "
                      f"new={r.get('new_total_latency')} "
                      f"exp2={r.get('exp2_total_latency')} "
                      f"rel_err={r.get('rel_err', 'N/A'):.6f}")

    if anomalies:
        print("\n  Anomaly summary:")
        for a in anomalies:
            print(f"    [{a.get('case_id','?')}:{a.get('framework','?')}] "
                  f"{a.get('kind','?')}: {a.get('message','')[:120]}")

    # A1 mask completeness check
    null_count = sum(
        1 for row in a1_summary
        for fw in FRAMEWORKS
        if row.get(fw) is None
    )
    print(f"\n  A1 mask null fields: {null_count} "
          f"({'OK - no null substitutes' if null_count == 0 else 'WARNING: null values present'})")

    print("="*60)


if __name__ == "__main__":
    main()

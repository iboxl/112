# Evaluation/RunBaselineWallTime.py
# baseline_walltime: Per-layer solver wall-time across all baselines on MIREDO case layers L1-L4
import argparse
import copy
import datetime
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# CoSA's check_timeloop_version() raises UnboundLocalError when TIMELOOP_DIR is
# unset (it falls into the except branch but then unconditionally references
# `output`). The version check is purely advisory — set a placeholder so the
# module import succeeds.
os.environ.setdefault("TIMELOOP_DIR", "/tmp/_unused_timeloop_dir")

from Evaluation.common.BaselineProvider import (
    run_baseline,
    SUPPORTED_BASELINE_METHODS,
    _cimloop_cache_root,
    _cosa_cache_root,
    _cosa_constrained_cache_root,
    _resolve_default_spec,
)
from Evaluation.common.EvalCommon import make_accelerator, run_miredo_layer, make_output_dir
from Evaluation.common.CaseLayerShapes import CASE_LAYERS_DETAILS
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad

# Architecture for this run. main() overrides from --architecture.
# Rerun default CIM_ACC_DEFAULT_SETUP matches Phase A-E; legacy was CIM_ACC_TEMPLATE.
# Single-process script, so a module global is safe for threading the arch into
# the cold-cache cleanup helpers and the per-method timers.
ARCH = "CIM_ACC_DEFAULT_SETUP"


def get_provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "repo": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # code repo, portable (was hardcoded stray)
        "commit": commit,
        "script": "Evaluation/RunBaselineWallTime.py",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _invalidate_cimloop_cache(layer_spec):
    """Remove the CIMLoop on-disk cache for CIM_ACC_TEMPLATE / Latency so timing is cold."""
    try:
        spec = _resolve_default_spec(ARCH)
        model_name = layer_spec["source"].replace(" ", "_")
        cache_root = _cimloop_cache_root(ARCH, "Latency", model_name, spec)
        if cache_root.exists():
            shutil.rmtree(str(cache_root), ignore_errors=True)
            print(f"    [cache] removed CIMLoop cache: {cache_root}", flush=True)
        else:
            print(f"    [cache] CIMLoop cache not present (already cold): {cache_root}", flush=True)
    except Exception as exc:
        print(f"    [cache] WARNING: could not invalidate CIMLoop cache: {exc}", flush=True)


def _invalidate_cosa_cache(layer_spec):
    """Remove the CoSA on-disk caches (both regular and constrained) so timing is cold."""
    try:
        spec = _resolve_default_spec(ARCH)
        model_name = layer_spec["source"].replace(" ", "_")
        for root_fn, label in [
            (_cosa_cache_root, "CoSA"),
            (_cosa_constrained_cache_root, "CoSA-legal"),
        ]:
            cache_root = root_fn(ARCH, "Latency", model_name, spec)
            if cache_root.exists():
                shutil.rmtree(str(cache_root), ignore_errors=True)
                print(f"    [cache] removed {label} cache: {cache_root}", flush=True)
            else:
                print(f"    [cache] {label} cache not present (already cold): {cache_root}", flush=True)
    except Exception as exc:
        print(f"    [cache] WARNING: could not invalidate CoSA cache: {exc}", flush=True)


def _is_na_result(result):
    """Detect whether a BaselineRunResult is an 'unsupported' (n/a) record."""
    if result is None:
        return True
    # BaselineRunResult.na() sets metadata["unsupported"] = True
    meta = getattr(result, "metadata", {}) or {}
    if meta.get("unsupported", False):
        return True
    # Fallback: both latency and energy are NaN
    lat = getattr(result, "latency", None)
    if lat is not None:
        try:
            if math.isnan(lat):
                return True
        except (TypeError, ValueError):
            pass
    return False


def time_baseline(method, layer_spec, output_dir):
    """Time a single baseline method on a single layer (cold cache)."""
    acc = make_accelerator(ARCH)
    loopdim = copy.deepcopy(layer_spec["loopdim"])
    ops = WorkLoad(loopDim=loopdim)
    model_name = layer_spec["source"].replace(" ", "_")

    layer_dir = output_dir / layer_spec["id"] / method
    prepare_save_dir(str(layer_dir))

    # Invalidate on-disk caches so timing reflects a real cold run
    if method == "cimloop":
        _invalidate_cimloop_cache(layer_spec)
    elif method in ("cosa", "cosa_legal"):
        _invalidate_cosa_cache(layer_spec)

    t0 = time.time()
    err = None
    result = None
    try:
        result = run_baseline(
            method=method,
            acc=acc,
            ops=ops,
            loopdim=loopdim,
            model_name=model_name,
            architecture=ARCH,
            objective="Latency",
            raise_on_unsupported=False,
            use_cache=False,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0

    is_na = _is_na_result(result)
    latency = None
    mapper_wall_sec = None
    if result is not None and not is_na:
        latency = getattr(result, "latency", None)
        meta = getattr(result, "metadata", {}) or {}
        mapper_wall_sec = meta.get("mapper_wall_sec")

    return {
        "method": method,
        "wall_sec": wall,
        "mapper_wall_sec": mapper_wall_sec,
        "latency": latency,
        "is_na": is_na,
        "error": err,
        "cache_state": "cold",
    }


def _invalidate_miredo_mip_cache_key(layer_spec, time_limit, mip_focus):
    """Drop the specific MIP cache entry for this layer's default config.

    Preserves all other entries in `output/.mip_cache.pkl` (other layers,
    ablation flags, alternate time-limits remain cached). This makes our
    timing a real fresh solve while honoring the user's instruction to
    preserve EXP-7c data more broadly.
    """
    try:
        from Evaluation.common.EvalCommon import (
            _ensure_cache_loaded, _make_cache_key, _save_cache,
            normalize_loopdim_for_solver,
        )
        # Load the persistent cache into module memory
        _ensure_cache_loaded()
        from Evaluation.common import EvalCommon as EC

        acc = make_accelerator(ARCH)
        solver_loopdim = normalize_loopdim_for_solver(copy.deepcopy(layer_spec["loopdim"]))
        key = _make_cache_key(
            acc=acc, solver_loopdim=solver_loopdim, objective="Latency",
            time_limit=time_limit, mip_focus=mip_focus, ablation_flags=None,
        )
        if EC._mip_cache is not None and key in EC._mip_cache:
            del EC._mip_cache[key]
            _save_cache()
            print(f"    [cache] dropped MIP cache entry for {layer_spec['id']} (default config)", flush=True)
        else:
            print(f"    [cache] no MIP cache entry for {layer_spec['id']} (already cold)", flush=True)
    except Exception as exc:
        print(f"    [cache] WARNING: could not invalidate MIREDO cache: {exc}", flush=True)


def time_miredo(layer_spec, output_dir, time_limit, mip_focus):
    """Time MIREDO on a single layer. The specific cache entry for this
    (layer, default config) is dropped before the call so timing reflects a
    real solve, consistent with EXP-7c's per-layer total_sec_wall."""
    layer_dir = output_dir / layer_spec["id"] / "miredo"
    prepare_save_dir(str(layer_dir))

    _invalidate_miredo_mip_cache_key(layer_spec, time_limit, mip_focus)

    t0 = time.time()
    err = None
    miredo = None
    try:
        miredo = run_miredo_layer(
            acc=make_accelerator(ARCH),
            loopdim=copy.deepcopy(layer_spec["loopdim"]),
            outputdir=layer_dir,
            objective="Latency",
            time_limit=time_limit,
            mip_focus=mip_focus,
            return_profile=True,
            ablation_flags=None,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0

    # run_miredo_layer returns a dict with keys:
    #   solver_latency, solver_energy, solver_edp,
    #   simulator_latency, simulator_energy, simulator_profile,
    #   mapping_profile, solver_loopdim, dataflow
    # Mirror exactly the extraction pattern from RunAccelerationControl.py.
    latency = None
    mapper_wall_sec = None
    if miredo is not None and isinstance(miredo, dict):
        sim_lat = miredo.get("simulator_latency")
        sol_lat = miredo.get("solver_latency")
        # Prefer simulator_latency; fall back to solver_latency
        if sim_lat is not None and sim_lat < 1e17:
            latency = sim_lat
        else:
            latency = sol_lat
        # MIREDO has no post-hoc tranSimulator wrap added by the runner; the
        # per-scheme simulator inside SolveMapping is part of MIREDO's own
        # cost-model evaluation (analogous to ZigZag's CostModelStage).
        # mapper_wall_sec therefore equals the outer wall (SolveMapping +
        # negligible cache/IO). Prefer mapping_profile.timing_total_sec when
        # available for tighter accounting.
        prof = miredo.get("mapping_profile")
        if prof is not None and hasattr(prof, "timing_total_sec"):
            mapper_wall_sec = float(prof.timing_total_sec)
        else:
            mapper_wall_sec = wall

    return {
        "method": "miredo",
        "wall_sec": wall,
        "mapper_wall_sec": mapper_wall_sec,
        "latency": latency,
        "is_na": False,
        "error": err,
        "cache_state": "cold",
    }


def main():
    parser = argparse.ArgumentParser(
        description="baseline_walltime: Per-layer solver wall-time across all baselines on L1-L4"
    )
    parser.add_argument("--layer-ids", nargs="+", default=["L1", "L2", "L3", "L4"])
    parser.add_argument(
        "--methods", nargs="+",
        default=["ws", "zigzag", "cimloop", "cosa", "cosa_legal", "miredo"],
    )
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument(
        "--architecture", default="CIM_ACC_DEFAULT_SETUP",
        help="Architecture registry key (rerun default: CIM_ACC_DEFAULT_SETUP, "
             "matching Phase A-E; legacy was CIM_ACC_TEMPLATE)"
    )
    parser.add_argument(
        "--output-json",
        # FIX 2026-05-17: code-repo-relative (portable; MIREDO/output/).
        default=os.path.join(os.path.dirname(__file__), "..", "output",
                             "baseline_walltime.json"),
    )
    args = parser.parse_args()

    global ARCH
    ARCH = args.architecture
    print(f"Architecture: {ARCH}", flush=True)

    output_dir = make_output_dir("baseline_walltime", None)
    print(f"Output directory: {output_dir}", flush=True)

    prov = get_provenance()
    rows = []

    spec_by_id = {s["id"]: s for s in CASE_LAYERS_DETAILS}

    for layer_id in args.layer_ids:
        if layer_id not in spec_by_id:
            print(f"WARNING: unknown layer id '{layer_id}', skipping.", flush=True)
            continue
        spec = spec_by_id[layer_id]

        for method in args.methods:
            print(f"\n=== {spec['id']} / {method} ===", flush=True)

            if method == "miredo":
                r = time_miredo(spec, output_dir, args.time_limit, args.mip_focus)
            else:
                r = time_baseline(method, spec, output_dir)

            r.update({
                "layer_id": spec["id"],
                "model_source": spec["source"],
                "loopdim": spec["loopdim"],
            })
            rows.append(r)

            mw = r.get("mapper_wall_sec")
            mw_str = f"{mw:.3f}s" if isinstance(mw, (int, float)) else "n/a"
            print(
                f"  -> {r['method']}: wall={r['wall_sec']:.3f}s  "
                f"mapper={mw_str}  "
                f"latency={r['latency']}  na={r['is_na']}  "
                f"cache={r.get('cache_state')}  err={r['error']}",
                flush=True,
            )

            # Incremental write — no previous EXP file is touched
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            out = {
                "experiment_id": "baseline_walltime",
                "provenance": prov,
                "config": {
                    "time_limit": args.time_limit,
                    "mip_focus": args.mip_focus,
                    "objective": "Latency",
                    "architecture": ARCH,
                    "architecture_key": ARCH,
                    "cache_policy": "all_methods_cold_per_row; per-key MIP cache invalidation for MIREDO; zigzag now passes use_cache=False through adapter interface (no .pkl files deleted); preserves other entries in output/.mip_cache.pkl",
                    "wall_sec_includes": "end-to-end (mapper + MIREDO simulator wrap for ZigZag/CoSA/CIMLoop; MIREDO has no external wrap, so its wall_sec equals SolveMapping wall)",
                    "mapper_wall_sec_includes": "mapper-only wall time (ZigZag MainStage / WS rule application / CoSA MIP / CIMLoop timeloop-mapper / MIREDO SolveMapping); excludes the post-hoc tranSimulator wrap added by MIREDO's adapter pipeline for ZigZag/CoSA/CIMLoop",
                },
                "results": rows,
            }
            with open(args.output_json, "w") as fh:
                json.dump(out, fh, indent=2, default=str)
            print(f"  -> JSON updated ({len(rows)} records): {args.output_json}", flush=True)

    # ── Summary tables ────────────────────────────────────────────────────
    method_order = ["ws", "zigzag", "cimloop", "cosa", "cosa_legal", "miredo"]
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer_id"], {})[r["method"]] = r

    def _print_table(field, title):
        print(f"\n\n=== {title} ===", flush=True)
        header = f"{'Layer':6s}  {'WS':>10s}  {'ZigZag':>10s}  {'CIMLoop':>10s}  {'CoSA':>10s}  {'CoSA-c':>10s}  {'MIREDO':>10s}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for lid in ["L1", "L2", "L3", "L4"]:
            if lid not in by_layer:
                continue
            layer_rows = by_layer[lid]
            cells = []
            for m in method_order:
                if m not in layer_rows:
                    cells.append("     -")
                else:
                    r = layer_rows[m]
                    if r["is_na"]:
                        cells.append("    n/a")
                    elif r["error"]:
                        cells.append("    ERR")
                    else:
                        v = r.get(field)
                        if isinstance(v, (int, float)):
                            cells.append(f"{v:>9.3f}s")
                        else:
                            cells.append("    n/a")
            print(f"{lid:6s}  {'  '.join(cells)}", flush=True)

    _print_table("wall_sec", "SUMMARY: end-to-end wall_sec (mapper + simulator wrap)")
    _print_table("mapper_wall_sec", "SUMMARY: mapper-only wall_sec (excludes MIREDO simulator wrap)")

    # Anomaly detection
    print("\n=== ANOMALY CHECK ===", flush=True)
    anomalies = []
    for r in rows:
        if r["error"] and not r["is_na"]:
            anomalies.append(
                f"  UNEXPECTED ERROR  {r['layer_id']}/{r['method']}: {r['error']}"
            )
    if not anomalies:
        print("  No unexpected errors.", flush=True)
    else:
        for a in anomalies:
            print(a, flush=True)

    print(f"\nFinal JSON: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

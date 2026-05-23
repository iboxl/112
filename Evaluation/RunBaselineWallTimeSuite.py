"""walltime_suite: Full-suite per-layer wall-time across baselines.

Diagnostic experiment for case-layer selection. NOT a paper artifact.

Scope:
  - Methods: ws, zigzag, cosa, cosa_legal
    (cimloop and miredo intentionally excluded; cimloop already matches the
     paper, miredo is not a baseline.)
  - Networks: ResNet-18, VGG19BN, AlexNet, MobileNet-v2, EfficientNet-B0
    (CoSA family skips MobileNet-v2 and EfficientNet-B0 entirely because the
     cnn-layer schema does not express grouped/depthwise conv.)
  - Cold cache per layer.
  - Strict sequential execution (one method × one layer at a time) to avoid
    CPU/L3/memory-bandwidth contention that would inflate wall measurements.

Output: experiments/parsed_metrics/walltime_suite_full_suite_walltime_baselines_<date>.json
Each row carries: method, wall_sec, mapper_wall_sec, latency, is_na, error,
layer_id, model, layer_family, MAC count, loopdim. The driver writes the JSON
incrementally so a crash mid-run still leaves usable partial data.
"""
import argparse
import copy
import datetime
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# CoSA's check_timeloop_version() raises UnboundLocalError when TIMELOOP_DIR is
# unset. Set a placeholder so the module import succeeds (version check is
# advisory only).
os.environ.setdefault("TIMELOOP_DIR", "/tmp/_unused_timeloop_dir")

from Evaluation.common.BaselineProvider import (
    run_baseline,
    _cosa_cache_root,
    _cosa_constrained_cache_root,
    _resolve_default_spec,
)
from Evaluation.common.EvalCommon import (
    DEFAULT_MODELS,
    iter_model_layers,
    make_accelerator,
    make_output_dir,
    classify_layer_family,
)
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad


# Networks where the CoSA family is not run (cnn-layer schema can't express
# grouped/depthwise conv that dominates these models).
COSA_SKIP_NETWORKS = {"mobilenetV2", "EfficientNet-B0"}

BASELINE_METHODS = ["ws", "zigzag", "cosa", "cosa_legal"]


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
        "script": "Evaluation/RunBaselineWallTimeSuite.py",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _invalidate_cosa_cache(model_name):
    """Drop on-disk CoSA caches for this model (both regular and constrained)."""
    try:
        spec = _resolve_default_spec("CIM_ACC_TEMPLATE")
        for root_fn, label in [
            (_cosa_cache_root, "CoSA"),
            (_cosa_constrained_cache_root, "CoSA-legal"),
        ]:
            cache_root = root_fn("CIM_ACC_TEMPLATE", "Latency", model_name, spec)
            if cache_root.exists():
                shutil.rmtree(str(cache_root), ignore_errors=True)
    except Exception as exc:
        print(f"    [cache] WARNING: could not invalidate CoSA cache for {model_name}: {exc}", flush=True)


def _is_na_result(result):
    if result is None:
        return True
    meta = getattr(result, "metadata", {}) or {}
    if meta.get("unsupported", False):
        return True
    lat = getattr(result, "latency", None)
    if lat is not None:
        try:
            import math
            if math.isnan(lat):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _mac_count(loopdim):
    def _v(k):
        x = loopdim.get(k, 1)
        return 1 if x is None else int(x)
    return _v("R") * _v("S") * _v("P") * _v("Q") * _v("C") * _v("K") * _v("G") * _v("B")


def time_baseline(method, model_name, layer_dict, output_dir):
    acc = make_accelerator("CIM_ACC_TEMPLATE")
    loopdim = copy.deepcopy(layer_dict["loopdim"])
    ops = WorkLoad(loopDim=loopdim)

    layer_dir = output_dir / model_name / layer_dict["layer"] / method
    prepare_save_dir(str(layer_dir))

    # CoSA family caches are invalidated per-(model,layer) at outer driver level
    # before the first call against this layer.

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
            architecture="CIM_ACC_TEMPLATE",
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


def _write_json(out_path, prov, results, models, methods, started_at, finished_at=None):
    out = {
        "experiment_id": "walltime_suite",
        "purpose": (
            "Diagnostic full-suite per-layer wall-time across baselines for "
            "case-layer selection. NOT a paper artifact."
        ),
        "provenance": prov,
        "config": {
            "models": models,
            "methods": methods,
            "cosa_skip_networks": sorted(COSA_SKIP_NETWORKS),
            "architecture": "CIM_ACC_TEMPLATE",
            "objective": "Latency",
            "execution_mode": (
                "strict_sequential: one method x one layer at a time; no "
                "intra-process or cross-method concurrency"
            ),
            "cache_policy": (
                "all baselines cold per layer (use_cache=False); "
                "CoSA on-disk caches dropped per-(model,layer) before each first call"
            ),
            "wall_sec_includes": (
                "end-to-end (mapper + post-hoc tranSimulator wrap added by "
                "MIREDO's adapter pipeline for ZigZag/CoSA/CoSA-c; WS has no "
                "MIREDO-side simulator wrap)"
            ),
            "mapper_wall_sec_includes": (
                "mapper-only wall (ZigZag MainStage / WS rule application / "
                "CoSA MIP / CoSA-c MIP); excludes the post-hoc tranSimulator wrap"
            ),
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "results": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    os.replace(tmp, out_path)


def main():
    parser = argparse.ArgumentParser(description="walltime_suite full-suite baseline wall-time")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--methods", nargs="+", default=BASELINE_METHODS)
    parser.add_argument(
        "--output-json",
        # FIX 2026-05-17: code-repo-relative (portable; MIREDO/output/).
        default=os.path.join(os.path.dirname(__file__), "..", "output",
                             "walltime_suite_baselines.json"),
    )
    args = parser.parse_args()

    output_dir = make_output_dir("walltime_suite", None)
    print(f"Output directory: {output_dir}", flush=True)

    prov = get_provenance()
    started_at = datetime.datetime.now().astimezone().isoformat()

    rows = []

    # Plan: enumerate (model, layer) up front so we can show progress & skip
    # CoSA family for grouped-conv-only networks at the model granularity
    # (faster + clean record).
    plan = []  # list of (model_name, layer_dict)
    for model_name in args.models:
        try:
            layers = iter_model_layers(model_name)
        except Exception as exc:
            print(f"!! cannot enumerate layers for {model_name}: {exc}", flush=True)
            continue
        for ld in layers:
            plan.append((model_name, ld))

    total_units = sum(
        len(args.methods) - (
            len([m for m in args.methods if m.startswith("cosa")])
            if model_name in COSA_SKIP_NETWORKS else 0
        )
        for model_name, _ in plan
    )
    print(f"Total work units (method x layer): {total_units}", flush=True)
    unit_idx = 0
    t_start = time.time()

    # Track which (model) have had their CoSA caches dropped already
    cosa_invalidated = set()

    for model_name, layer_dict in plan:
        layer_id = layer_dict["layer"]
        loopdim = layer_dict["loopdim"]
        layer_family = classify_layer_family(loopdim)
        macs = _mac_count(loopdim)

        for method in args.methods:
            unit_idx += 1
            # CoSA family skips entire grouped-conv-dominant networks.
            if method.startswith("cosa") and model_name in COSA_SKIP_NETWORKS:
                rows.append({
                    "method": method,
                    "model": model_name,
                    "layer_id": layer_id,
                    "layer_family": layer_family,
                    "macs": macs,
                    "loopdim": loopdim,
                    "wall_sec": None,
                    "mapper_wall_sec": None,
                    "latency": None,
                    "is_na": True,
                    "error": None,
                    "cache_state": "skipped",
                    "skip_reason": "cosa_grouped_conv_network",
                })
                _write_json(args.output_json, prov, rows, args.models, args.methods, started_at)
                continue

            # Drop CoSA caches once per model per family
            if method.startswith("cosa") and (model_name, "cosa") not in cosa_invalidated:
                _invalidate_cosa_cache(model_name)
                cosa_invalidated.add((model_name, "cosa"))

            elapsed = time.time() - t_start
            avg = elapsed / max(1, unit_idx - 1)
            eta_str = f"ETA {(total_units - unit_idx + 1) * avg:.0f}s" if unit_idx > 1 else "ETA -"
            print(
                f"\n[{unit_idx}/{total_units}] {model_name} / {layer_id} / "
                f"{method}  (family={layer_family}, MACs={macs}, elapsed={elapsed:.0f}s, {eta_str})",
                flush=True,
            )

            r = time_baseline(method, model_name, layer_dict, output_dir)
            r.update({
                "model": model_name,
                "layer_id": layer_id,
                "layer_family": layer_family,
                "macs": macs,
                "loopdim": loopdim,
            })
            rows.append(r)

            mw = r.get("mapper_wall_sec")
            mw_str = f"{mw:.3f}s" if isinstance(mw, (int, float)) else "n/a"
            print(
                f"  -> wall={r['wall_sec']:.3f}s mapper={mw_str} "
                f"latency={r['latency']} na={r['is_na']} err={r['error']}",
                flush=True,
            )

            _write_json(args.output_json, prov, rows, args.models, args.methods, started_at)

    finished_at = datetime.datetime.now().astimezone().isoformat()
    _write_json(args.output_json, prov, rows, args.models, args.methods, started_at, finished_at)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n=== ANOMALY CHECK ===", flush=True)
    anomalies = [r for r in rows if r.get("error") and not r.get("is_na")]
    if not anomalies:
        print("  No unexpected errors.", flush=True)
    else:
        for a in anomalies:
            print(f"  ERR  {a['model']}/{a['layer_id']}/{a['method']}: {a['error']}", flush=True)

    print(f"\nFinal JSON: {args.output_json}", flush=True)
    print(f"Total wall: {(time.time() - t_start):.1f}s for {len(rows)} records", flush=True)


if __name__ == "__main__":
    main()

# Evaluation/RunAccelerationControl.py
# dynlb_losslessness: Empirical losslessness verification for LB pruning on L1-L5
import argparse
import copy
import json
import os
import sys
import time
import datetime
import subprocess
import socket
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Evaluation.common.EvalCommon import (
    make_accelerator, run_miredo_layer, hardware_spec_from_acc, make_output_dir
)
from Evaluation.common.CaseLayerShapes import CASE_LAYERS_DETAILS
from utils.UtilsFunction.ToolFunction import prepare_save_dir


def get_provenance():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "repo": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # code repo, portable (was hardcoded stray)
        "commit": commit,
        "script": "Evaluation/RunAccelerationControl.py",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="dynlb_losslessness: Empirical losslessness verification for LB pruning"
    )
    parser.add_argument("--time-limit", type=int, default=60,
                        help="MIP time limit per scheme (seconds)")
    parser.add_argument("--mip-focus", type=int, default=1,
                        help="Gurobi MIPFocus parameter")
    parser.add_argument(
        "--output-json",
        # FIX 2026-05-17: code-repo-relative (portable, lands in this checkout's
        # MIREDO/output/) — was an absolute path into a stray non-_v1 dir.
        default=os.path.join(os.path.dirname(__file__), "..", "output",
                             "dynlb_losslessness.json"),
        help="Path for the combined results JSON"
    )
    parser.add_argument(
        "--layer-ids", nargs="+", default=["L3", "L2", "L4", "L1"],
        help="Layer IDs to run (order determines execution sequence)"
    )
    parser.add_argument(
        "--architecture", default="CIM_ACC_DEFAULT_SETUP",
        help="Architecture registry key (rerun default: CIM_ACC_DEFAULT_SETUP, "
             "matching Phase A-E; legacy was CIM_ACC_TEMPLATE)"
    )
    args = parser.parse_args()

    # Create base output directory (timestamped so it never clashes with acceleration_profile)
    output_dir = make_output_dir("dynlb_losslessness", None)
    print(f"Output directory: {output_dir}", flush=True)

    acc = make_accelerator(args.architecture)

    results = []
    prov = get_provenance()

    # Layer id -> spec map for fast lookup
    spec_by_id = {s["id"]: s for s in CASE_LAYERS_DETAILS}

    for layer_id in args.layer_ids:
        if layer_id not in spec_by_id:
            print(f"WARNING: unknown layer id '{layer_id}', skipping.", flush=True)
            continue
        spec = spec_by_id[layer_id]

        for mode in ["lb_on", "lb_off"]:
            print(f"\n=== {spec['id']} / {mode} ===", flush=True)

            ablation_flags = {"ABLATION_DISABLE_LB_PRUNING": True} if mode == "lb_off" else None

            layer_dir = output_dir / spec["id"] / mode
            prepare_save_dir(str(layer_dir))

            t0 = time.time()
            layer_result = run_miredo_layer(
                acc=copy.deepcopy(acc),
                loopdim=copy.deepcopy(spec["loopdim"]),
                outputdir=layer_dir,
                objective="Latency",
                time_limit=args.time_limit,
                mip_focus=args.mip_focus,
                return_profile=True,
                ablation_flags=ablation_flags,
            )
            wall = time.time() - t0

            # run_miredo_layer returns a dict with keys:
            #   solver_latency, solver_energy, solver_edp,
            #   simulator_latency, simulator_energy, simulator_profile,
            #   mapping_profile, solver_loopdim, dataflow
            # Use simulator_latency as the best_metric (primary validated result).
            # Fall back to solver_latency if simulator is not available.
            sim_lat = layer_result.get("simulator_latency")
            sol_lat = layer_result.get("solver_latency")
            best_metric = sim_lat if (sim_lat is not None and sim_lat < 1e17) else sol_lat

            mp = layer_result.get("mapping_profile")

            row = {
                "layer_id": spec["id"],
                "model_source": spec["source"],
                "loopdim": spec["loopdim"],
                "mode": mode,
                "best_metric": best_metric,
                "simulator_latency": sim_lat,
                "solver_latency": sol_lat,
                "num_schemes_initial": getattr(mp, "num_schemes_initial", None),
                "num_schemes_after_dominance": getattr(mp, "num_schemes_after_dominance", None),
                "num_schemes_after_static_lb": getattr(mp, "num_schemes_after_static_lb", None),
                "num_schemes_dynamic_lb_pruned": getattr(mp, "num_schemes_dynamic_lb_pruned", None),
                "num_schemes_after_dynamic_lb": getattr(mp, "num_schemes_after_dynamic_lb", None),
                "num_schemes_with_solution": getattr(mp, "num_schemes_with_solution", None),
                "total_sec_wall": wall,
                "mip_cumulative_sec": getattr(mp, "timing_mip_cumulative_sec", None),
                "mip_wall_sec": getattr(mp, "timing_mip_wall_sec", None),
            }
            results.append(row)

            print(
                f"  -> best_metric={best_metric:.4g}  wall={wall:.1f}s  "
                f"schemes_initial={getattr(mp, 'num_schemes_initial', '?')}  "
                f"after_static_lb={getattr(mp, 'num_schemes_after_static_lb', '?')}  "
                f"after_dynamic_lb={getattr(mp, 'num_schemes_after_dynamic_lb', '?')}",
                flush=True
            )

            # Save incrementally so a crash does not lose earlier results
            out = {
                "experiment_id": "dynlb_losslessness",
                "provenance": prov,
                "config": {
                    "time_limit": args.time_limit,
                    "mip_focus": args.mip_focus,
                    "objective": "Latency",
                    "architecture": args.architecture,
                    "architecture_key": args.architecture,
                },
                "results": results,
            }
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, "w") as fh:
                json.dump(out, fh, indent=2, default=str)
            print(f"  -> JSON updated ({len(results)} records): {args.output_json}", flush=True)

    # ── Summary: losslessness verification ────────────────────────────────
    print("\n\n=== dynlb_losslessness Losslessness Summary ===", flush=True)
    by_layer = {}
    for row in results:
        by_layer.setdefault(row["layer_id"], {})[row["mode"]] = row

    all_equal = True
    for lid in args.layer_ids:
        if lid not in by_layer:
            print(f"  {lid}: incomplete (missing)", flush=True)
            continue
        r = by_layer[lid]
        if "lb_on" not in r or "lb_off" not in r:
            print(f"  {lid}: incomplete (only {list(r.keys())})", flush=True)
            continue
        on_val = r["lb_on"]["best_metric"]
        off_val = r["lb_off"]["best_metric"]
        if on_val is None or off_val is None:
            print(f"  {lid}: WARN best_metric is None", flush=True)
            all_equal = False
            continue
        if on_val == 0 or off_val == 0:
            match = (on_val == off_val)
        else:
            rel_diff = abs(on_val - off_val) / max(abs(on_val), abs(off_val))
            match = rel_diff < 1e-4
            if not match:
                all_equal = False
        speedup = r["lb_off"]["total_sec_wall"] / max(r["lb_on"]["total_sec_wall"], 1e-6)
        status = "EQUAL" if match else "MISMATCH"
        print(
            f"  {lid}: lb_on={on_val:.6g}  lb_off={off_val:.6g}  "
            f"rel_diff={abs(on_val - off_val) / max(abs(on_val), 1e-9):.2e}  "
            f"slowdown_without_lb={speedup:.1f}x  [{status}]",
            flush=True
        )

    print(
        f"\nVerdict: {'ALL EQUAL — LB pruning is lossless' if all_equal else 'DISCREPANCY DETECTED — investigate'}",
        flush=True
    )
    print(f"Final JSON: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

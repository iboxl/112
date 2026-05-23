# Evaluation/RunFlexFactControl.py
# flexfact_losslessness: Empirical losslessness verification for FlexFact (flexible factorization) on L1-L4
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
from utils.factorization import flexible_factorization, prime_factors
from utils.Workload import WorkLoad
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
        "script": "Evaluation/RunFlexFactControl.py",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _flexfact_compression_profile(loopdim, ops, flexfact_disabled):
    """Return per-dimension factor counts for flex and prime factorizations.

    When flexfact_disabled=True both flex_factors and prime_factors should
    match, confirming FlexFact is fully bypassed.
    """
    compression = {}
    for dim_char in ops.dim2Dict[1:]:   # skip '-' sentinel at index 0
        bound = loopdim[dim_char]
        if bound <= 1:
            pf_count = 0
            ff_count = 1 if bound == 1 else 0
        else:
            pf = prime_factors(bound)
            ff = flexible_factorization(bound)
            pf_count = len(pf)
            ff_count = len(ff)
        compression[f"dimension_{dim_char}"] = {
            "prime_factors": pf_count,
            "flex_factors": ff_count,
            "flex_equals_prime": (ff_count == pf_count),
        }
    return compression


def main():
    parser = argparse.ArgumentParser(
        description="flexfact_losslessness: Empirical losslessness verification for FlexFact"
    )
    parser.add_argument("--time-limit", type=int, default=60,
                        help="MIP time limit per scheme (seconds)")
    parser.add_argument("--mip-focus", type=int, default=1,
                        help="Gurobi MIPFocus parameter")
    parser.add_argument(
        "--output-json",
        # FIX 2026-05-17: code-repo-relative (portable; MIREDO/output/).
        default=os.path.join(os.path.dirname(__file__), "..", "output",
                             "flexfact_losslessness.json"),
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

    # Create base output directory (timestamped so it never clashes with dynlb_losslessness)
    output_dir = make_output_dir("flexfact_losslessness", None)
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

        for mode in ["flexfact_on", "flexfact_off"]:
            print(f"\n=== {spec['id']} / {mode} ===", flush=True)

            ablation_flags = {"ABLATION_DISABLE_FLEXFACT": True} if mode == "flexfact_off" else None
            flexfact_disabled = (mode == "flexfact_off")

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

            # Extract MIP size from the best solver profile
            solver_profile = getattr(mp, "best_solver_profile", None) if mp is not None else None
            mip_variables = getattr(solver_profile, "num_vars", None)
            mip_constraints = getattr(solver_profile, "num_constrs", None)

            # Compute FlexFact compression profile using the real loopdim
            ops_for_profile = WorkLoad(loopDim=copy.deepcopy(spec["loopdim"]))
            flexfact_compression = _flexfact_compression_profile(
                spec["loopdim"], ops_for_profile, flexfact_disabled
            )

            row = {
                "layer_id": spec["id"],
                "model_source": spec["source"],
                "loopdim": spec["loopdim"],
                "mode": mode,
                "best_metric": best_metric,
                "simulator_latency": sim_lat,
                "solver_latency": sol_lat,
                "mip_variables": mip_variables,
                "mip_constraints": mip_constraints,
                "num_schemes_initial": getattr(mp, "num_schemes_initial", None),
                "num_schemes_after_static_lb": getattr(mp, "num_schemes_after_static_lb", None),
                "num_schemes_after_dynamic_lb": getattr(mp, "num_schemes_after_dynamic_lb", None),
                "num_schemes_with_solution": getattr(mp, "num_schemes_with_solution", None),
                "total_sec_wall": wall,
                "mip_cumulative_sec": getattr(mp, "timing_mip_cumulative_sec", None),
                "mip_wall_sec": getattr(mp, "timing_mip_wall_sec", None),
                "flexfact_compression": flexfact_compression,
            }
            results.append(row)

            print(
                f"  -> best_metric={best_metric:.4g}  wall={wall:.1f}s  "
                f"mip_vars={mip_variables}  mip_cons={mip_constraints}  "
                f"schemes_initial={getattr(mp, 'num_schemes_initial', '?')}",
                flush=True
            )

            # Save incrementally so a crash does not lose earlier results
            out = {
                "experiment_id": "flexfact_losslessness",
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
    print("\n\n=== flexfact_losslessness FlexFact Losslessness Summary ===", flush=True)
    by_layer = {}
    for row in results:
        by_layer.setdefault(row["layer_id"], {})[row["mode"]] = row

    all_equal = True
    print(
        f"{'Layer':<6}  {'flexfact_on':>14}  {'mip_vars_on':>11}  {'mip_cons_on':>11}  "
        f"{'wall_on':>8}  |  {'flexfact_off':>14}  {'mip_vars_off':>12}  {'mip_cons_off':>12}  "
        f"{'wall_off':>9}  {'verdict':<12}  {'vars_growth%':>12}  {'cons_growth%':>12}",
        flush=True
    )
    for lid in ["L1", "L2", "L3", "L4"]:
        if lid not in by_layer:
            print(f"  {lid}: incomplete (missing)", flush=True)
            continue
        r = by_layer[lid]
        if "flexfact_on" not in r or "flexfact_off" not in r:
            print(f"  {lid}: incomplete (only {list(r.keys())})", flush=True)
            continue
        on_row = r["flexfact_on"]
        off_row = r["flexfact_off"]

        on_val = on_row["best_metric"]
        off_val = off_row["best_metric"]
        if on_val is None or off_val is None:
            print(f"  {lid}: WARN best_metric is None", flush=True)
            all_equal = False
            continue

        if on_val == 0 or off_val == 0:
            match = (on_val == off_val)
            rel_diff = 0.0
        else:
            rel_diff = abs(on_val - off_val) / max(abs(on_val), abs(off_val))
            match = rel_diff < 1e-4

        if not match:
            all_equal = False

        on_vars = on_row.get("mip_variables")
        off_vars = off_row.get("mip_variables")
        on_cons = on_row.get("mip_constraints")
        off_cons = off_row.get("mip_constraints")

        vars_growth = (
            (off_vars - on_vars) / max(on_vars, 1) * 100.0
            if (on_vars is not None and off_vars is not None) else float("nan")
        )
        cons_growth = (
            (off_cons - on_cons) / max(on_cons, 1) * 100.0
            if (on_cons is not None and off_cons is not None) else float("nan")
        )

        verdict = "EQUAL" if match else "MISMATCH"
        on_wall = on_row["total_sec_wall"]
        off_wall = off_row["total_sec_wall"]
        print(
            f"{lid:<6}  {on_val:>14.6g}  {str(on_vars):>11}  {str(on_cons):>11}  "
            f"{on_wall:>7.1f}s  |  {off_val:>14.6g}  {str(off_vars):>12}  {str(off_cons):>12}  "
            f"{off_wall:>8.1f}s  {verdict:<12}  {vars_growth:>11.1f}%  {cons_growth:>11.1f}%",
            flush=True
        )
        print(
            f"       rel_diff={rel_diff:.2e}",
            flush=True
        )

    print(
        f"\nVerdict: {'ALL EQUAL — FlexFact is lossless' if all_equal else 'DISCREPANCY DETECTED — investigate'}",
        flush=True
    )
    print(f"Final JSON: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

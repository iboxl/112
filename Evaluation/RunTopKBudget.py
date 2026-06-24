# Evaluation/RunTopKBudget.py
# topk_budget: Top-K x per-candidate budget trade-off grid on case layers L1-L5
#
# For each (layer, budget, K) cell: run MIREDO with objective="Latency",
# time_limit=budget, ablation_flags={"ACCEL_TOP_K": K} (None if K="all").
# MIP cache is cleared at script start so every cell is a cold solve.
# Results are saved incrementally to experiments/parsed_metrics/.
import argparse
import copy
import datetime
import json
import os
import pathlib
import platform
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Evaluation.common.EvalCommon import (
    make_accelerator,
    run_miredo_layer,
    hardware_spec_from_acc,
    make_output_dir,
    clear_mip_cache,
    _default_cache_path,
)
from Evaluation.common.CaseLayerShapes import CASE_LAYERS_DETAILS
from utils.UtilsFunction.ToolFunction import prepare_save_dir


def get_provenance(script_path):
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
        "script": script_path,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def _reset_mip_cache():
    """Move (not delete) the existing MIP cache so every cell is a cold solve."""
    cache_path = _default_cache_path()
    if cache_path.is_file():
        backup = cache_path.with_suffix(".pkl.bak_topk_budget")
        shutil.move(str(cache_path), str(backup))
        print(f"[cache] moved existing MIP cache -> {backup}", flush=True)
    else:
        print("[cache] no existing MIP cache found (already cold)", flush=True)
    # Also clear in-memory cache
    clear_mip_cache()


def _relative_loss(val, baseline):
    """Relative quality loss vs baseline: positive = worse than baseline."""
    if baseline is None or baseline == 0 or val is None:
        return None
    return (val - baseline) / baseline


def main():
    parser = argparse.ArgumentParser(
        description="topk_budget: top-K x budget trade-off grid on case layers L1-L5"
    )
    parser.add_argument(
        "--layer-ids", nargs="+", default=["L1", "L2", "L3", "L4", "L5"],
        help="Layer IDs to run",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[60, 30, 15, 5],
        help="MIP time limits (seconds) to sweep",
    )
    parser.add_argument(
        "--top-ks", nargs="+", default=["all", "10", "5", "3"],
        help="Top-K values to sweep; 'all' means no truncation",
    )
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument(
        "--output-json",
        # FIX 2026-05-17: code-repo-relative (portable; MIREDO/output/).
        default=os.path.join(os.path.dirname(__file__), "..", "output",
                             "topk_budget_caselayer.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--skip-cache-reset", action="store_true",
        help="Skip moving the MIP cache (for debugging/resume only)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing rows from --output-json and skip any already-completed "
             "(layer_id, budget_sec, top_k) cell. Implies --skip-cache-reset.",
    )
    parser.add_argument(
        "--architecture", default="CIM_ACC_DEFAULT_SETUP",
        help="Architecture registry key (rerun default: CIM_ACC_DEFAULT_SETUP, "
             "matching Phase A-E; legacy was CIM_ACC_TEMPLATE)"
    )
    args = parser.parse_args()

    # ── Normalise top-k list ───────────────────────────────────────────────
    top_ks = []
    for v in args.top_ks:
        if str(v).lower() == "all":
            top_ks.append("all")
        else:
            top_ks.append(int(v))

    output_dir = make_output_dir("topk_budget", None)
    print(f"Output directory: {output_dir}", flush=True)

    # ── Resume: preload existing rows and build skip-set ─────────────────
    results = []
    completed = set()
    if args.resume and os.path.isfile(args.output_json):
        try:
            with open(args.output_json) as fh:
                prior = json.load(fh)
            results = list(prior.get("results", []))
            for r in results:
                key = (r.get("layer_id"), int(r.get("budget_sec")),
                       "all" if r.get("top_k") in ("all", None) else int(r.get("top_k")))
                completed.add(key)
            print(
                f"[resume] preloaded {len(results)} existing rows from {args.output_json} "
                f"({len(completed)} unique cells will be skipped)",
                flush=True,
            )
        except Exception as exc:
            print(f"[resume] failed to load existing JSON: {exc}; starting fresh", flush=True)
            results = []
            completed = set()

    # ── Cold-cache policy ─────────────────────────────────────────────────
    # Each cell uses a unique (acc, loopdim, objective, time_limit, ablation_flags)
    # key so cells never cross-contaminate.  We additionally move the on-disk
    # cache before the run to guarantee no residual state from earlier experiments.
    if args.resume:
        # Resume implies skip-cache-reset: prior cells already poisoned the cache
        # with completed (key, time_limit, ablation_flags) entries, but those keys
        # are unique per cell so they cannot contaminate remaining cells.
        print("[resume] not resetting MIP cache (cell keys are unique).", flush=True)
    elif not args.skip_cache_reset:
        _reset_mip_cache()

    prov = get_provenance("Evaluation/RunTopKBudget.py")
    prov["cache_policy"] = (
        "mip_cache cleared/moved at script start; each cell has unique "
        "(time_limit, ablation_flags) key so no inter-cell reuse is possible"
    )
    if args.resume:
        prov["resume_run"] = {
            "preloaded_rows": len(results),
            "preloaded_at": datetime.datetime.now().astimezone().isoformat(),
        }

    acc_template = make_accelerator(args.architecture)
    spec_by_id = {s["id"]: s for s in CASE_LAYERS_DETAILS}

    total_cells = len(args.layer_ids) * len(args.budgets) * len(top_ks)
    cell_idx = 0

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    for layer_id in args.layer_ids:
        if layer_id not in spec_by_id:
            print(f"WARNING: unknown layer id '{layer_id}', skipping.", flush=True)
            continue
        spec = spec_by_id[layer_id]

        for budget in args.budgets:
            for top_k in top_ks:
                cell_idx += 1
                k_label = "all" if top_k == "all" else str(top_k)
                key = (layer_id, int(budget), top_k if top_k == "all" else int(top_k))
                if key in completed:
                    print(
                        f"\n=== [{cell_idx}/{total_cells}] {layer_id} / budget={budget}s / K={k_label} -- SKIP (already in JSON) ===",
                        flush=True,
                    )
                    continue
                print(
                    f"\n=== [{cell_idx}/{total_cells}] {layer_id} / budget={budget}s / K={k_label} ===",
                    flush=True,
                )

                ablation_flags = {"ACCEL_TOP_K": None if top_k == "all" else int(top_k)}
                layer_dir = output_dir / spec["id"] / f"budget{budget}" / f"K{k_label}"
                prepare_save_dir(str(layer_dir))

                t0 = time.time()
                layer_result = run_miredo_layer(
                    acc=copy.deepcopy(acc_template),
                    loopdim=copy.deepcopy(spec["loopdim"]),
                    outputdir=layer_dir,
                    objective="Latency",
                    time_limit=budget,
                    mip_focus=args.mip_focus,
                    return_profile=True,
                    ablation_flags=ablation_flags,
                )
                wall = time.time() - t0

                sim_lat = layer_result.get("simulator_latency")
                sol_lat = layer_result.get("solver_latency")
                best_metric = sim_lat if (sim_lat is not None and sim_lat < 1e17) else sol_lat

                mp = layer_result.get("mapping_profile")
                solver_profile = getattr(mp, "best_solver_profile", None) if mp is not None else None
                mip_variables = getattr(solver_profile, "num_vars", None)
                mip_constraints = getattr(solver_profile, "num_constrs", None)

                num_after_top_k = getattr(mp, "num_schemes_after_top_k", None)

                row = {
                    "layer_id": spec["id"],
                    "model_source": spec["source"],
                    "loopdim": spec["loopdim"],
                    "budget_sec": budget,
                    "top_k": top_k,
                    "best_metric": best_metric,
                    "simulator_latency": sim_lat,
                    "solver_latency": sol_lat,
                    "total_sec_wall": wall,
                    "mip_cumulative_sec": getattr(mp, "timing_mip_cumulative_sec", None),
                    "mip_wall_sec": getattr(mp, "timing_mip_wall_sec", None),
                    "num_schemes_initial": getattr(mp, "num_schemes_initial", None),
                    "num_schemes_after_static_lb": getattr(mp, "num_schemes_after_static_lb", None),
                    "num_schemes_after_top_k": num_after_top_k,
                    "num_schemes_after_dynamic_lb": getattr(mp, "num_schemes_after_dynamic_lb", None),
                    "num_schemes_with_solution": getattr(mp, "num_schemes_with_solution", None),
                    "mip_variables": mip_variables,
                    "mip_constraints": mip_constraints,
                }
                results.append(row)

                print(
                    f"  -> best={best_metric:.6g}  wall={wall:.1f}s  "
                    f"schemes: initial={getattr(mp, 'num_schemes_initial', '?')} "
                    f"-> static_lb={getattr(mp, 'num_schemes_after_static_lb', '?')} "
                    f"-> top_k={num_after_top_k} "
                    f"-> dynamic_lb={getattr(mp, 'num_schemes_after_dynamic_lb', '?')}  "
                    f"solutions={getattr(mp, 'num_schemes_with_solution', '?')}",
                    flush=True,
                )

                # Incremental save
                out = {
                    "experiment_id": "topk_budget",
                    "provenance": prov,
                    "config": {
                        "time_limits": args.budgets,
                        "top_ks": [str(k) for k in top_ks],
                        "mip_focus": args.mip_focus,
                        "objective": "Latency",
                        "architecture": args.architecture,
                        "architecture_key": args.architecture,
                        "cache_policy": prov["cache_policy"],
                    },
                    "results": results,
                }
                with open(args.output_json, "w") as fh:
                    json.dump(out, fh, indent=2, default=str)
                print(f"  -> JSON updated ({len(results)}/{total_cells} cells): {args.output_json}", flush=True)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print("\n\n=== topk_budget Sanity Checks ===", flush=True)
    anomalies = []

    # 1. Default-path invariance: (L1, budget=60, K="all") vs EXP-7d L1 reference
    EXP7D_L1_LATENCY_REF = 358624
    gate_row = next(
        (r for r in results
         if r["layer_id"] == "L1" and r["budget_sec"] == 60 and r["top_k"] == "all"),
        None,
    )
    if gate_row is not None:
        bm = gate_row["best_metric"]
        static_lb = gate_row["num_schemes_after_static_lb"]
        dyn_lb = gate_row["num_schemes_after_dynamic_lb"]
        rel = abs(bm - EXP7D_L1_LATENCY_REF) / EXP7D_L1_LATENCY_REF if bm else float("nan")
        improved = bm is not None and bm <= EXP7D_L1_LATENCY_REF
        print(
            f"Default-path invariance (L1, 60s, K=all):\n"
            f"  best_metric={bm:.6g}  ref={EXP7D_L1_LATENCY_REF}  rel_diff={rel:.2e}  "
            f"improved_or_equal={'YES' if improved else 'NO'}\n"
            f"  num_schemes_after_static_lb={static_lb} (ref=1600)\n"
            f"  num_schemes_after_dynamic_lb={dyn_lb} (ref=30)",
            flush=True,
        )
        if not (improved or rel < 1e-4):
            msg = f"GATE FAIL: best_metric={bm} deviates from ref={EXP7D_L1_LATENCY_REF} by {rel:.2e} (>1e-4) and is worse"
            print(f"  *** {msg}", flush=True)
            anomalies.append(msg)
        else:
            print("  GATE PASS", flush=True)
        if static_lb is not None and static_lb != 1600:
            msg = f"GATE WARN: L1/60s/all num_schemes_after_static_lb={static_lb}, expected 1600"
            print(f"  *** {msg}", flush=True)
            anomalies.append(msg)
        if dyn_lb is not None and dyn_lb != 30:
            msg = f"GATE WARN: L1/60s/all num_schemes_after_dynamic_lb={dyn_lb}, expected 30"
            print(f"  *** {msg}", flush=True)
            anomalies.append(msg)
    else:
        print("Default-path invariance gate: L1/60s/all cell not found.", flush=True)

    # 2. Top-K monotone: num_schemes_after_top_k == min(K, num_schemes_after_static_lb)
    print("\nTop-K monotone check:", flush=True)
    for r in results:
        k = r["top_k"]
        if k == "all":
            continue
        nat = r["num_schemes_after_top_k"]
        nas = r["num_schemes_after_static_lb"]
        expected = min(int(k), nas) if nas is not None else None
        if nat != expected:
            msg = (
                f"TOP-K MONOTONE FAIL: {r['layer_id']}/budget={r['budget_sec']}/K={k}: "
                f"after_top_k={nat}, expected min({k},{nas})={expected}"
            )
            print(f"  *** {msg}", flush=True)
            anomalies.append(msg)

    # 3. Quality monotone check (soft): K=10 beating K=all by >5% is an anomaly
    print("\nQuality monotone anomalies (K=10 beats K=all by >5%):", flush=True)
    found_any = False
    by_layer_budget = {}
    for r in results:
        by_layer_budget.setdefault((r["layer_id"], r["budget_sec"]), {})[r["top_k"]] = r

    for (lid, budget), kmap in sorted(by_layer_budget.items()):
        base = kmap.get("all", {}).get("best_metric")
        k10 = kmap.get(10, {}).get("best_metric")
        if base is None or k10 is None or base == 0:
            continue
        rel = (k10 - base) / base  # negative = K=10 is better than all
        if rel < -0.05:
            msg = f"ANOMALY: {lid}/budget={budget}: K=10 latency={k10:.6g} better than K=all latency={base:.6g} by {abs(rel)*100:.1f}%"
            print(f"  {msg}", flush=True)
            anomalies.append(msg)
            found_any = True
    if not found_any:
        print("  None found.", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n\n=== topk_budget Summary Table ===", flush=True)
    k_order = ["all"] + sorted([k for k in top_ks if k != "all"])
    budget_order = sorted(args.budgets, reverse=True)

    for layer_id in args.layer_ids:
        if layer_id not in spec_by_id:
            continue
        print(f"\nLayer: {layer_id}", flush=True)
        # Get baseline: budget=60, K=all
        baseline_cell = by_layer_budget.get((layer_id, 60), {}).get("all")
        baseline_metric = baseline_cell["best_metric"] if baseline_cell else None

        # Header
        header_cells = ["budget/K"]
        for k in k_order:
            header_cells.append(f"K={k}_latency")
            header_cells.append(f"K={k}_wall")
            header_cells.append(f"K={k}_loss%")
        print("  " + "  ".join(f"{h:>14}" for h in header_cells), flush=True)

        for budget in budget_order:
            row_cells = [f"{budget}s"]
            for k in k_order:
                cell = by_layer_budget.get((layer_id, budget), {}).get(k)
                if cell is None:
                    row_cells += ["-", "-", "-"]
                    continue
                bm = cell["best_metric"]
                wall = cell["total_sec_wall"]
                loss = _relative_loss(bm, baseline_metric)
                row_cells.append(f"{bm:.4g}" if bm is not None else "-")
                row_cells.append(f"{wall:.1f}s")
                row_cells.append(f"{loss*100:+.1f}%" if loss is not None else "-")
            print("  " + "  ".join(f"{c:>14}" for c in row_cells), flush=True)

    # ── Append anomalies and re-save JSON ─────────────────────────────────
    out = {
        "experiment_id": "EXP-7f",
        "provenance": prov,
        "config": {
            "time_limits": args.budgets,
            "top_ks": [str(k) for k in top_ks],
            "mip_focus": args.mip_focus,
            "objective": "Latency",
            "architecture": args.architecture,
            "architecture_key": args.architecture,
            "cache_policy": prov["cache_policy"],
        },
        "results": results,
        "anomalies": anomalies,
    }
    with open(args.output_json, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"\nFinal JSON: {args.output_json}", flush=True)
    print(f"Total cells: {len(results)}", flush=True)
    if anomalies:
        print(f"\n*** {len(anomalies)} anomaly(ies) detected ***", flush=True)
        for a in anomalies:
            print(f"  {a}", flush=True)
    else:
        print("No anomalies detected.", flush=True)


if __name__ == "__main__":
    main()

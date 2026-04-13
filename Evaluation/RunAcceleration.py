import argparse
import copy
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Evaluation.common.BaselineProvider import (
    SUPPORTED_BASELINE_METHODS,
    run_baseline,
)
from Evaluation.common.EvalCommon import (
    DEFAULT_MODELS,
    hardware_spec_from_acc,
    iter_model_layers,
    make_accelerator,
    make_output_dir,
    objective_metric_value,
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad
from utils.factorization import flexible_factorization, prime_factors


def _flexfact_compression(loopdim, ops):
    compression = {}
    start_time = time.time()
    for dim_char in ops.dim2Dict[1:]:
        bound = loopdim[dim_char]
        if bound <= 1:
            prime = []
            flexible = [1]
        else:
            prime = prime_factors(bound)
            flexible = flexible_factorization(bound)
        compression[f"dimension_{dim_char}"] = {
            "prime_factors": len(prime),
            "flex_factors": len(flexible),
        }
    return compression, time.time() - start_time


def main():
    parser = argparse.ArgumentParser(description="EXP-7 acceleration effect profiling")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--architecture", default="ZigzagAcc")
    parser.add_argument("--objective", default="Latency", choices=["Latency", "Energy", "EDP"])
    parser.add_argument("--baselines", nargs="+", choices=SUPPORTED_BASELINE_METHODS, default=["ws", "zigzag"])
    parser.add_argument("--cosa-map", default=None,
                        help="Path to a CoSA map_16.yaml file or directory; omit to generate locally.")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp7_acceleration", args.output_dir)
    setup_experiment_logger(output_dir, "exp7.log")

    acceleration_effects = []
    anomalies = []

    for model_name in args.models:
        model_layers = iter_model_layers(model_name)
        if args.maxLayers is not None:
            model_layers = model_layers[:args.maxLayers]

        for layer in model_layers:
            loopdim = copy.deepcopy(layer["loopdim"])
            ops = WorkLoad(loopDim=loopdim)
            layer_dir = output_dir / model_name / layer["layer"]
            prepare_save_dir(str(layer_dir))

            baseline_metrics = []
            for baseline_method in args.baselines:
                try:
                    baseline_result = run_baseline(
                        method=baseline_method,
                        acc=make_accelerator(args.architecture),
                        ops=ops,
                        loopdim=loopdim,
                        model_name=model_name,
                        architecture=args.architecture,
                        objective=args.objective,
                        cosa_map=args.cosa_map,
                        output_root=output_dir,
                    )
                    baseline_metrics.append(
                        objective_metric_value(args.objective, baseline_result.latency, baseline_result.energy)
                    )
                except Exception as exc:
                    anomalies.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "method": baseline_method,
                        "kind": "baseline_error",
                        "message": str(exc),
                    })

            best_metric = min(baseline_metrics) * 2 if baseline_metrics else None

            try:
                miredo = run_miredo_layer(
                    acc=make_accelerator(args.architecture),
                    loopdim=loopdim,
                    outputdir=layer_dir,
                    objective=args.objective,
                    time_limit=args.timeLimit,
                    mip_focus=args.mipFocus,
                    best_metric=best_metric,
                    return_profile=True,
                )
                mapping_profile = miredo["mapping_profile"]
                solver_profile = mapping_profile.best_solver_profile if mapping_profile is not None else None
                compression, flexfact_sec = _flexfact_compression(loopdim, ops)

                acceleration_effects.append({
                    "model": model_name,
                    "layer": layer["layer"],
                    "num_schemes_initial": mapping_profile.num_schemes_initial if mapping_profile else None,
                    "num_schemes_after_dominance": mapping_profile.num_schemes_after_dominance if mapping_profile else None,
                    "num_schemes_after_static_lb": mapping_profile.num_schemes_after_static_lb if mapping_profile else None,
                    "num_schemes_dynamic_lb_pruned": mapping_profile.num_schemes_dynamic_lb_pruned if mapping_profile else None,
                    "num_schemes_after_dynamic_lb": mapping_profile.num_schemes_after_dynamic_lb if mapping_profile else None,
                    "num_schemes_submitted": mapping_profile.num_schemes_submitted if mapping_profile else None,
                    "num_schemes_with_solution": mapping_profile.num_schemes_with_solution if mapping_profile else None,
                    "num_schemes_no_solution": mapping_profile.num_schemes_no_solution if mapping_profile else None,
                    "survival_rate_pct": (
                        mapping_profile.num_schemes_after_dynamic_lb / max(1, mapping_profile.num_schemes_initial) * 100.0
                        if mapping_profile else None
                    ),
                    "flexfact_compression": compression,
                    "mip_variables": solver_profile.num_vars if solver_profile else None,
                    "mip_constraints": solver_profile.num_constrs if solver_profile else None,
                    "best_solver_profile": {
                        "status": solver_profile.status,
                        "sol_count": solver_profile.sol_count,
                        "mip_gap": solver_profile.mip_gap,
                        "best_bound": solver_profile.best_bound,
                        "num_bin_vars": solver_profile.num_bin_vars,
                        "node_count": solver_profile.node_count,
                        "callback_mipsol_updates": solver_profile.callback_mipsol_updates,
                        "callback_dynamic_terminations": solver_profile.callback_dynamic_terminations,
                    } if solver_profile else None,
                    "solve_time_breakdown": {
                        "enumeration_sec": mapping_profile.timing_enumeration_sec if mapping_profile else None,
                        "flexfact_sec": flexfact_sec,
                        "scoring_pruning_sec": mapping_profile.timing_scoring_pruning_sec if mapping_profile else None,
                        "mip_cumulative_sec": mapping_profile.timing_mip_cumulative_sec if mapping_profile else None,
                        "mip_solving_sec": mapping_profile.timing_mip_wall_sec if mapping_profile else None,
                        "total_sec": mapping_profile.timing_total_sec if mapping_profile else None,
                    },
                })
            except Exception as exc:
                anomalies.append({
                    "model": model_name,
                    "layer": layer["layer"],
                    "kind": "runtime_error",
                    "message": str(exc),
                })

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-7.json",
        experiment_id="EXP-7",
        script_path=__file__,
        config={
            "models": args.models,
            "architecture": hardware_spec_from_acc(acc),
            "time_limit": args.timeLimit,
            "objective": args.objective,
            "baselines_for_upper_bound": args.baselines,
        },
        results={
            "acceleration_effects": acceleration_effects,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

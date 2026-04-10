import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Evaluation.common.BaselineProvider import run_baseline
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
from Evaluation.common.ProfileAnalysis import stall_decomposition
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad


METHOD_LABELS = {
    "ws": "WS",
    "zigzag": "ZigZag_IMC",
}


def _empty_total():
    return {
        "total_latency": 0.0,
        "total_energy": 0.0,
        "total_edp": 0.0,
    }


def _accumulate(total, latency, energy):
    total["total_latency"] += latency
    total["total_energy"] += energy
    total["total_edp"] += latency * energy


def main():
    parser = argparse.ArgumentParser(description="EXP-2 baseline comparison")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--objectives", nargs="+", default=["Latency", "Energy", "EDP"])
    parser.add_argument("--baselines", nargs="+", default=["ws", "zigzag"])
    parser.add_argument("--architecture", default="ZigzagAcc")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp2_compare", args.output_dir)
    setup_experiment_logger(output_dir, "exp2.log")

    per_model = []
    per_layer = []
    stall_rows = []
    anomalies = [{
        "method": "CoSA_adapted",
        "kind": "unsupported",
        "message": "Evaluation/CoSA integration is not available in the current repository.",
    }]

    for model_name in args.models:
        model_layers = iter_model_layers(model_name)
        if args.maxLayers is not None:
            model_layers = model_layers[:args.maxLayers]

        totals = {}
        for objective in args.objectives:
            totals[f"MIREDO_{objective}"] = _empty_total()
            for baseline_method in args.baselines:
                totals[f"{METHOD_LABELS[baseline_method]}_{objective}"] = _empty_total()

        latency_layer_records = {}

        for objective in args.objectives:
            for layer in model_layers:
                loopdim = copy.deepcopy(layer["loopdim"])
                ops = WorkLoad(loopDim=loopdim)
                layer_dir = output_dir / objective / model_name / layer["layer"]
                prepare_save_dir(str(layer_dir))

                baseline_results = {}
                for baseline_method in args.baselines:
                    try:
                        baseline_result = run_baseline(
                            method=baseline_method,
                            acc=make_accelerator(args.architecture),
                            ops=ops,
                            loopdim=loopdim,
                            model_name=model_name,
                            architecture=args.architecture,
                            objective=objective,
                        )
                        baseline_results[baseline_method] = baseline_result
                        _accumulate(
                            totals[f"{METHOD_LABELS[baseline_method]}_{objective}"],
                            baseline_result.latency,
                            baseline_result.energy,
                        )
                        decomposition = stall_decomposition(baseline_result.profile)
                        stall_rows.append({
                            "model": model_name,
                            "layer": layer["layer"],
                            "objective": objective,
                            "method": f"{METHOD_LABELS[baseline_method]}_{objective}",
                            **decomposition,
                        })
                    except Exception as exc:
                        anomalies.append({
                            "model": model_name,
                            "layer": layer["layer"],
                            "method": baseline_method,
                            "objective": objective,
                            "kind": "baseline_error",
                            "message": str(exc),
                        })

                best_metric = None
                if baseline_results:
                    baseline_metrics = [
                        objective_metric_value(objective, result.latency, result.energy)
                        for result in baseline_results.values()
                    ]
                    best_metric = min(baseline_metrics) * 2

                try:
                    miredo = run_miredo_layer(
                        acc=make_accelerator(args.architecture),
                        loopdim=loopdim,
                        outputdir=layer_dir,
                        objective=objective,
                        time_limit=args.timeLimit,
                        mip_focus=args.mipFocus,
                        best_metric=best_metric,
                        return_profile=True,
                    )
                    _accumulate(
                        totals[f"MIREDO_{objective}"],
                        miredo["simulator_latency"],
                        miredo["simulator_energy"],
                    )
                    decomposition = stall_decomposition(miredo["simulator_profile"])
                    stall_rows.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "objective": objective,
                        "method": f"MIREDO_{objective}",
                        **decomposition,
                    })

                    if objective == "Latency":
                        latency_layer_records[layer["layer"]] = {
                            "model": model_name,
                            "layer": layer["layer"],
                            "miredo_latency": miredo["simulator_latency"],
                        }
                        for baseline_method, baseline_result in baseline_results.items():
                            key = f"{baseline_method}_latency"
                            t_baseline = baseline_result.latency
                            t_miredo = miredo["simulator_latency"]
                            latency_layer_records[layer["layer"]][key] = t_baseline
                            latency_layer_records[layer["layer"]][f"speedup_vs_{baseline_method}"] = (
                                t_baseline / max(1e-9, t_miredo)
                            )
                            latency_layer_records[layer["layer"]][f"improvement_pct_vs_{baseline_method}"] = (
                                (t_baseline - t_miredo) / max(1e-9, t_baseline) * 100.0
                            )
                except Exception as exc:
                    anomalies.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "method": "MIREDO",
                        "objective": objective,
                        "kind": "runtime_error",
                        "message": str(exc),
                    })

        per_model.append({
            "model": model_name,
            "results_by_method": totals,
        })
        per_layer.extend(latency_layer_records.values())

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-2.json",
        experiment_id="EXP-2",
        script_path=__file__,
        config={
            "models": args.models,
            "architecture": hardware_spec_from_acc(acc),
            "time_limit": args.timeLimit,
            "objective": args.objectives,
            "baselines": args.baselines,
            "edp_note": "total_edp is raw latency*energy without CONST.SCALINGFACTOR",
        },
        results={
            "per_model": per_model,
            "per_layer": per_layer,
            "stall_decomposition": stall_rows,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

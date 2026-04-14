import argparse
import copy
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
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from Evaluation.common.ProfileAnalysis import summarize_dataflow_decisions
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad


def main():
    parser = argparse.ArgumentParser(description="EXP-5 latency-energy tradeoff")
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenetV2"])
    parser.add_argument("--objectives", nargs="+", default=["Latency", "Energy", "EDP"])
    parser.add_argument("--baselines", nargs="+", choices=SUPPORTED_BASELINE_METHODS, default=["ws", "zigzag"])
    parser.add_argument("--architecture", default="ZigzagAcc")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--representativeLayers", type=int, default=3)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp5_tradeoff", args.output_dir)
    setup_experiment_logger(output_dir, "exp5.log")

    tradeoff_points = []
    decision_comparison = []
    anomalies = []

    for model_name in args.models:
        model_layers = iter_model_layers(model_name)
        if args.maxLayers is not None:
            model_layers = model_layers[:args.maxLayers]

        decision_by_layer = {}
        latency_ranking = []

        for objective in args.objectives:
            total_latency = 0.0
            total_energy = 0.0

            for layer in model_layers:
                loopdim = copy.deepcopy(layer["loopdim"])
                ops = WorkLoad(loopDim=loopdim)
                layer_dir = output_dir / objective / model_name / layer["layer"]
                prepare_save_dir(str(layer_dir))

                for baseline_method in args.baselines:
                    try:
                        run_baseline(
                            method=baseline_method,
                            acc=make_accelerator(args.architecture),
                            ops=ops,
                            loopdim=loopdim,
                            model_name=model_name,
                            architecture=args.architecture,
                            objective=objective,
                        )
                    except Exception as exc:
                        anomalies.append({
                            "model": model_name,
                            "layer": layer["layer"],
                            "objective": objective,
                            "method": baseline_method,
                            "kind": "baseline_error",
                            "message": str(exc),
                        })

                try:
                    miredo = run_miredo_layer(
                        acc=make_accelerator(args.architecture),
                        loopdim=loopdim,
                        outputdir=layer_dir,
                        objective=objective,
                        time_limit=args.timeLimit,
                        mip_focus=args.mipFocus,
                        return_profile=True,
                    )
                    total_latency += miredo["simulator_latency"]
                    total_energy += miredo["simulator_energy"]

                    decision_by_layer.setdefault(layer["layer"], {})
                    if miredo["dataflow"] is not None:
                        decision_by_layer[layer["layer"]][objective.lower()] = summarize_dataflow_decisions(miredo["dataflow"])

                    if objective == "Latency":
                        latency_ranking.append((miredo["simulator_latency"], layer["layer"]))
                except Exception as exc:
                    anomalies.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "objective": objective,
                        "method": "MIREDO",
                        "kind": "runtime_error",
                        "message": str(exc),
                    })

            tradeoff_points.append({
                "model": model_name,
                "mode": objective,
                "total_latency": total_latency,
                "total_energy": total_energy,
            })

        representative_layers = [
            layer_name for _, layer_name in sorted(latency_ranking, reverse=True)[:args.representativeLayers]
        ]
        for layer_name in representative_layers:
            row = {
                "model": model_name,
                "layer": layer_name,
            }
            row["latency_mode"] = decision_by_layer.get(layer_name, {}).get("latency")
            row["energy_mode"] = decision_by_layer.get(layer_name, {}).get("energy")
            row["edp_mode"] = decision_by_layer.get(layer_name, {}).get("edp")
            decision_comparison.append(row)

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-5.json",
        experiment_id="EXP-5",
        script_path=__file__,
        config={
            "models": args.models,
            "architecture": hardware_spec_from_acc(acc),
            "time_limit": args.timeLimit,
            "objective": args.objectives,
            "baselines_evaluated": args.baselines,
        },
        results={
            "tradeoff_points": tradeoff_points,
            "decision_comparison": decision_comparison,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

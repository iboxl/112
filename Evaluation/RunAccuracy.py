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
    classify_layer_family,
    hardware_spec_from_acc,
    iter_model_layers,
    make_accelerator,
    make_output_dir,
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from Evaluation.common.ProfileAnalysis import dominant_stall_type
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.Workload import WorkLoad


def main():
    parser = argparse.ArgumentParser(description="EXP-1 latency accuracy validation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--architecture", default="CIM_ACC_TEMPLATE")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--baseline", choices=SUPPORTED_BASELINE_METHODS, default="zigzag")
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp1_accuracy", args.output_dir)
    setup_experiment_logger(output_dir, "exp1.log")

    per_layer = []
    anomalies = []

    for model_name in args.models:
        layers = iter_model_layers(model_name)
        if args.maxLayers is not None:
            layers = layers[:args.maxLayers]

        for layer in layers:
            layer_dir = output_dir / model_name / layer["layer"]
            prepare_save_dir(str(layer_dir))
            try:
                loopdim = copy.deepcopy(layer["loopdim"])
                ops = WorkLoad(loopDim=loopdim)
                baseline = run_baseline(
                    method=args.baseline,
                    acc=make_accelerator(args.architecture),
                    ops=ops,
                    loopdim=loopdim,
                    model_name=model_name,
                    architecture=args.architecture,
                    objective="Latency",
                )
                miredo = run_miredo_layer(
                    acc=make_accelerator(args.architecture),
                    loopdim=loopdim,
                    outputdir=layer_dir,
                    objective="Latency",
                    time_limit=args.timeLimit,
                    mip_focus=args.mipFocus,
                    return_profile=True,
                )
                analytical_latency = miredo["solver_latency"]
                simulator_latency = miredo["simulator_latency"]
                relative_error_pct = abs(analytical_latency - simulator_latency) / max(1.0, simulator_latency) * 100.0
                per_layer.append({
                    "model": model_name,
                    "layer": layer["layer"],
                    "layer_type": layer["layer_type"],
                    "layer_family": classify_layer_family(loopdim),
                    "analytical_latency": analytical_latency,
                    "simulator_latency": simulator_latency,
                    "relative_error_pct": relative_error_pct,
                    "dominant_stall_type": dominant_stall_type(miredo["simulator_profile"]),
                    "baseline_method": baseline.method,
                })
                if relative_error_pct > 10.0:
                    anomalies.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "kind": "high_relative_error",
                        "relative_error_pct": relative_error_pct,
                    })
            except Exception as exc:
                anomalies.append({
                    "model": model_name,
                    "layer": layer["layer"],
                    "kind": "runtime_error",
                    "message": str(exc),
                })

    error_values = [entry["relative_error_pct"] for entry in per_layer]
    summary = {
        "mean_relative_error_pct": sum(error_values) / max(1, len(error_values)),
        "max_relative_error_pct": max(error_values) if error_values else None,
        "pct_within_5pct": (
            sum(1 for entry in per_layer if entry["relative_error_pct"] <= 5.0) / max(1, len(per_layer))
        ),
        "error_by_layer_type": {},
    }

    for family in ["standard_conv", "grouped_conv"]:
        family_entries = [entry for entry in per_layer if entry["layer_family"] == family]
        if not family_entries:
            continue
        family_err = [entry["relative_error_pct"] for entry in family_entries]
        summary["error_by_layer_type"][family] = {
            "mean_relative_error_pct": sum(family_err) / len(family_err),
            "max_relative_error_pct": max(family_err),
        }

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-1.json",
        experiment_id="EXP-1",
        script_path=__file__,
        config={
            "models": args.models,
            "architecture": hardware_spec_from_acc(acc),
            "time_limit": args.timeLimit,
            "objective": "Latency",
            "baseline": args.baseline,
        },
        results={
            "per_layer": per_layer,
            "summary": summary,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

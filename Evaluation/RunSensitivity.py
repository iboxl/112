import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from Evaluation.common.BaselineProvider import run_ws_baseline
from Evaluation.common.EvalCommon import (
    hardware_spec_from_acc,
    iter_model_layers,
    make_accelerator,
    make_output_dir,
    objective_metric_value,
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from Evaluation.common.HardwareVariants import build_hardware_variant
from utils.Workload import WorkLoad


DEFAULT_SWEEPS = {
    "core_count": [0.5, 1.0, 2.0, 4.0],
    "buffer_capacity": [0.25, 0.5, 1.0, 2.0],
    "gbuf_core_bw": [0.5, 1.0, 2.0, 4.0],
    "macro_input_precision": [4, 8, 16],
}


def _format_value(value):
    if isinstance(value, float):
        return f"{value}x"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="EXP-4 hardware sensitivity study")
    parser.add_argument("--models", nargs="+", default=["resnet18"])
    parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_SWEEPS.keys()))
    parser.add_argument("--architecture", default="CIM_ACC_TEMPLATE")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp4_sensitivity", args.output_dir)
    setup_experiment_logger(output_dir, "exp4.log")

    sensitivity = []
    anomalies = []

    base_acc = make_accelerator(args.architecture)
    # sensitivity 直接在 HardwareSpec 上变异，之后 from_spec 构造 CIM_Acc，保证
    # MIREDO / baseline 两侧同源。目前仅支持 CIM_ACC_TEMPLATE 架构的默认 Spec。
    base_spec = default_spec()

    for parameter in args.parameters:
        for value in DEFAULT_SWEEPS[parameter]:
            variant_spec = build_hardware_variant(base_spec, parameter, value)
            variant_acc = CIM_Acc.from_spec(variant_spec)

            for model_name in args.models:
                model_layers = iter_model_layers(model_name)
                if args.maxLayers is not None:
                    model_layers = model_layers[:args.maxLayers]

                miredo_total = 0.0
                baseline_total = 0.0

                for layer in model_layers:
                    loopdim = copy.deepcopy(layer["loopdim"])
                    ops = WorkLoad(loopDim=loopdim)
                    layer_dir = output_dir / parameter / str(value) / model_name / layer["layer"]
                    try:
                        baseline = run_ws_baseline(acc=variant_acc, ops=ops, objective="EDP")
                        miredo = run_miredo_layer(
                            acc=variant_acc,
                            loopdim=loopdim,
                            outputdir=layer_dir,
                            objective="EDP",
                            time_limit=args.timeLimit,
                            mip_focus=args.mipFocus,
                            return_profile=False,
                        )
                        baseline_total += objective_metric_value("EDP", baseline.latency, baseline.energy)
                        miredo_total += objective_metric_value("EDP", miredo["simulator_latency"], miredo["simulator_energy"])
                    except Exception as exc:
                        anomalies.append({
                            "model": model_name,
                            "layer": layer["layer"],
                            "parameter": parameter,
                            "value": value,
                            "kind": "runtime_error",
                            "message": str(exc),
                        })

                sensitivity.append({
                    "parameter": parameter,
                    "value": _format_value(value),
                    "model": model_name,
                    "miredo_edp": miredo_total,
                    "baseline_edp": baseline_total,
                    "speedup": baseline_total / max(1e-9, miredo_total) if miredo_total else None,
                    "improvement_pct": (baseline_total - miredo_total) / max(1e-9, baseline_total) * 100.0 if baseline_total else None,
                })

    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-4.json",
        experiment_id="EXP-4",
        script_path=__file__,
        config={
            "models": args.models,
            "architecture": hardware_spec_from_acc(base_acc),
            "time_limit": args.timeLimit,
            "objective": "EDP",
            "parameters": args.parameters,
            "sweeps": {parameter: DEFAULT_SWEEPS[parameter] for parameter in args.parameters},
            "baseline": "ws",
        },
        results={
            "sensitivity": sensitivity,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from Evaluation.common.BaselineProvider import run_baseline
from Evaluation.common.EvalCommon import (
    hardware_spec_from_acc,
    make_accelerator,
    make_output_dir,
    objective_metric_value,
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from Evaluation.common.HardwareVariants import build_hardware_variant, DEFAULT_SWEEPS
from Evaluation.common.CaseLayerShapes import (
    all_layer_ids,
    layer_selection_config,
    layers_by_ids,
    select_model_layers,
)
from utils.Workload import WorkLoad


def _is_valid_result(result):
    return (
        result.latency is not None and result.energy is not None and
        math.isfinite(float(result.latency)) and math.isfinite(float(result.energy))
    )


def _accumulate(total, miredo_edp, baseline_edp):
    total["miredo_edp"] += miredo_edp
    total["baseline_edp"] += baseline_edp
    total["n_layers"] += 1


def main():
    parser = argparse.ArgumentParser(description="EXP-4 hardware sensitivity sweep")
    parser.add_argument("--models", nargs="+", default=["resnet18"],
                        help="Models to sweep when --layerSource=model.")
    parser.add_argument("--layerSource", choices=("model", "representative"), default="model",
                        help="model runs full/selected model layers; representative runs the L1-L4 registry.")
    parser.add_argument(
        "--layers", nargs="+", default=None,
        help="Layer subset. For model source: exact layer names/aliases, 1-based positions, "
             "idx:N, or model:<selector>. For representative source: IDs "
             f"from {all_layer_ids()}. Default: all layers.",
    )
    parser.add_argument(
        "--baselines", nargs="+", choices=("ws", "zigzag"),
        default=["ws", "zigzag"],
        help="Baselines to compare against.",
    )
    parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_SWEEPS.keys()))
    parser.add_argument("--architecture", default="CIM_ACC_TEMPLATE")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    if args.layerSource == "representative":
        representative_layers = layers_by_ids(args.layers)
        if args.maxLayers is not None:
            representative_layers = representative_layers[:args.maxLayers]
        layer_groups = [("representative", representative_layers)]
        layer_selection = layer_selection_config(
            layer_source="representative",
            layer_selectors=args.layers,
            max_layers=args.maxLayers,
        )
    else:
        layer_groups = [
            (model_name, select_model_layers(
                model_name, layer_selectors=args.layers, max_layers=args.maxLayers,
            ))
            for model_name in args.models
        ]
        layer_selection = layer_selection_config(
            layer_source="model",
            layer_selectors=args.layers,
            max_layers=args.maxLayers,
        )

    output_dir = make_output_dir("exp4_sensitivity", args.output_dir)
    setup_experiment_logger(output_dir, "exp4.log")

    sensitivity_totals = {}
    sensitivity_order = []
    per_layer = []
    anomalies = []

    base_acc = make_accelerator(args.architecture)
    base_spec = default_spec()

    for parameter in args.parameters:
        for value in DEFAULT_SWEEPS[parameter]:
            variant_spec = build_hardware_variant(base_spec, parameter, value)
            variant_acc = CIM_Acc.from_spec(variant_spec)

            for group_name, selected_layers in layer_groups:
                for layer in selected_layers:
                    layer_id = layer.get("layer_id", layer["layer"])
                    loopdim = copy.deepcopy(layer["loopdim"])
                    ops = WorkLoad(loopDim=loopdim)
                    layer_dir = output_dir / parameter / str(value) / group_name / layer["layer"]

                    try:
                        miredo = run_miredo_layer(
                            acc=variant_acc, loopdim=loopdim, outputdir=layer_dir,
                            objective="EDP", time_limit=args.timeLimit,
                            mip_focus=args.mipFocus, return_profile=False,
                        )
                        miredo_edp = objective_metric_value(
                            "EDP", miredo["simulator_latency"], miredo["simulator_energy"]
                        )
                    except Exception as exc:
                        anomalies.append({
                            "model": group_name, "layer": layer["layer"],
                            "layer_id": layer_id, "parameter": parameter, "value": value,
                            "kind": "miredo_error", "message": str(exc),
                        })
                        continue

                    for bname in args.baselines:
                        try:
                            res = run_baseline(
                                method=bname, acc=variant_acc, ops=ops,
                                loopdim=loopdim, model_name=group_name,
                                architecture=args.architecture, objective="EDP",
                                raise_on_unsupported=False,
                            )
                            if not _is_valid_result(res):
                                reason = res.metadata.get("reason", "unsupported")
                                anomalies.append({
                                    "model": group_name, "layer": layer["layer"],
                                    "layer_id": layer_id, "parameter": parameter,
                                    "value": value, "baseline": bname,
                                    "kind": "baseline_unsupported",
                                    "message": reason,
                                })
                                continue
                            baseline_edp = objective_metric_value("EDP", res.latency, res.energy)
                        except Exception as exc:
                            anomalies.append({
                                "model": group_name, "layer": layer["layer"],
                                "layer_id": layer_id, "parameter": parameter,
                                "value": value, "baseline": bname,
                                "kind": "baseline_error", "message": str(exc),
                            })
                            continue

                        key = (parameter, str(value), group_name, bname)
                        if key not in sensitivity_totals:
                            sensitivity_totals[key] = {
                                "parameter": parameter,
                                "value": str(value),
                                "model": group_name,
                                "baseline": bname,
                                "miredo_edp": 0.0,
                                "baseline_edp": 0.0,
                                "n_layers": 0,
                            }
                            sensitivity_order.append(key)
                        _accumulate(sensitivity_totals[key], miredo_edp, baseline_edp)

                        per_layer.append({
                            "parameter": parameter,
                            "value": str(value),
                            "model": group_name,
                            "layer": layer["layer"],
                            "layer_id": layer_id,
                            "layer_index": layer.get("layer_index"),
                            "layer_source": layer.get("layer_source"),
                            "representative_label": layer.get("label"),
                            "representative_source": layer.get("source"),
                            "baseline": bname,
                            "miredo_edp": miredo_edp,
                            "baseline_edp": baseline_edp,
                            "speedup": baseline_edp / max(1e-9, miredo_edp) if miredo_edp else None,
                            "improvement_pct": (
                                (baseline_edp - miredo_edp) / max(1e-9, baseline_edp) * 100.0
                                if baseline_edp else None
                            ),
                        })

    sensitivity = []
    for key in sensitivity_order:
        row = sensitivity_totals[key]
        baseline_edp = row["baseline_edp"]
        miredo_edp = row["miredo_edp"]
        sensitivity.append({
            **row,
            "speedup": baseline_edp / max(1e-9, miredo_edp) if miredo_edp else None,
            "improvement_pct": (
                (baseline_edp - miredo_edp) / max(1e-9, baseline_edp) * 100.0
                if baseline_edp else None
            ),
        })

    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-4.json",
        experiment_id="EXP-4",
        script_path=__file__,
        config={
            "models": args.models,
            "layer_selection": layer_selection,
            "architecture": hardware_spec_from_acc(base_acc),
            "time_limit": args.timeLimit,
            "objective": "EDP",
            "parameters": args.parameters,
            "sweeps": {parameter: DEFAULT_SWEEPS[parameter] for parameter in args.parameters},
            "baselines": args.baselines,
        },
        results={
            "sensitivity": sensitivity,
            "per_layer": per_layer,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

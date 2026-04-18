import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def _pct_tile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main():
    parser = argparse.ArgumentParser(description="EXP-1b energy accuracy validation (MIP solver vs simulator)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--architecture", default="CIM_ACC_TEMPLATE")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp1b_energy_accuracy", args.output_dir)
    setup_experiment_logger(output_dir, "exp1b.log")

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
                miredo = run_miredo_layer(
                    acc=make_accelerator(args.architecture),
                    loopdim=loopdim,
                    outputdir=layer_dir,
                    objective="Latency",
                    time_limit=args.timeLimit,
                    mip_focus=args.mipFocus,
                    return_profile=True,
                )
                solver_energy = miredo["solver_energy"]
                simulator_energy = miredo["simulator_energy"]
                if simulator_energy <= 0:
                    anomalies.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "kind": "nonpositive_simulator_energy",
                        "simulator_energy": simulator_energy,
                    })
                    continue

                relative_error_pct = abs(solver_energy - simulator_energy) / simulator_energy * 100.0
                signed_error_pct = (solver_energy - simulator_energy) / simulator_energy * 100.0
                per_layer.append({
                    "model": model_name,
                    "layer": layer["layer"],
                    "layer_type": layer["layer_type"],
                    "layer_family": classify_layer_family(loopdim),
                    "analytical_energy": solver_energy,
                    "simulator_energy": simulator_energy,
                    "relative_error_pct": relative_error_pct,
                    "signed_error_pct": signed_error_pct,
                    "dominant_stall_type": dominant_stall_type(miredo["simulator_profile"]),
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

    errors = [entry["relative_error_pct"] for entry in per_layer]
    errors_sorted = sorted(errors)
    n = len(errors_sorted)

    summary = {
        "n_layers": n,
        "mean_relative_error_pct": (sum(errors) / n) if n else None,
        "median_relative_error_pct": _pct_tile(errors_sorted, 0.5),
        "p95_relative_error_pct": _pct_tile(errors_sorted, 0.95),
        "max_relative_error_pct": max(errors) if errors else None,
        "pct_within_1pct": (sum(1 for e in errors if e <= 1.0) / n) if n else None,
        "pct_within_5pct": (sum(1 for e in errors if e <= 5.0) / n) if n else None,
        "pct_within_10pct": (sum(1 for e in errors if e <= 10.0) / n) if n else None,
        "mean_signed_error_pct": (sum(entry["signed_error_pct"] for entry in per_layer) / n) if n else None,
        "frac_overestimate": (sum(1 for entry in per_layer if entry["signed_error_pct"] > 0) / n) if n else None,
        "error_by_layer_family": {},
        "error_by_dominant_stall": {},
    }

    for family in ["standard_conv", "grouped_conv"]:
        ents = [e for e in per_layer if e["layer_family"] == family]
        if not ents:
            continue
        errs = [e["relative_error_pct"] for e in ents]
        summary["error_by_layer_family"][family] = {
            "n": len(ents),
            "mean_relative_error_pct": sum(errs) / len(errs),
            "max_relative_error_pct": max(errs),
        }

    stall_buckets = {}
    for e in per_layer:
        stall_buckets.setdefault(e["dominant_stall_type"], []).append(e["relative_error_pct"])
    for stall, errs in stall_buckets.items():
        summary["error_by_dominant_stall"][str(stall)] = {
            "n": len(errs),
            "mean_relative_error_pct": sum(errs) / len(errs),
            "max_relative_error_pct": max(errs),
        }

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-1b.json",
        experiment_id="EXP-1b",
        script_path=__file__,
        config={
            "models": args.models,
            "architecture": hardware_spec_from_acc(acc),
            "objective": "Latency",
            "time_limit": args.timeLimit,
            "mip_focus": args.mipFocus,
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

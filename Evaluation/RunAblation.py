import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Evaluation.common.EvalCommon import (
    hardware_spec_from_acc,
    iter_model_layers,
    make_accelerator,
    make_output_dir,
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from utils.UtilsFunction.ToolFunction import prepare_save_dir


VARIANT_LABELS = {
    "Latency": "latency-only",
    "Energy": "energy-only",
    "EDP": "edp",
}

STRUCTURAL_VARIANTS = {
    "fixed-double-buffer": {"ABLATION_FIXED_DOUBLE_BUFFER": True},
    "simplified-pipeline": {"ABLATION_SIMPLIFIED_PIPELINE": True},
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


def _degradation_vs_full(result_total, full_total):
    return {
        "degradation_latency_pct": (
            (result_total["total_latency"] - full_total["total_latency"]) / max(1e-9, full_total["total_latency"]) * 100.0
            if full_total["total_latency"] else None
        ),
        "degradation_edp_pct": (
            (result_total["total_edp"] - full_total["total_edp"]) / max(1e-9, full_total["total_edp"]) * 100.0
            if full_total["total_edp"] else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-3 objective-function ablation")
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenetV2"])
    parser.add_argument("--objectives", nargs="+", default=["Latency", "Energy", "EDP"])
    parser.add_argument("--structural", nargs="*", default=None,
                        choices=list(STRUCTURAL_VARIANTS.keys()),
                        help="Structural ablation variants to run (e.g. fixed-double-buffer)")
    parser.add_argument("--architecture", default="CIM_ACC_TEMPLATE")
    parser.add_argument("--timeLimit", type=int, default=120)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("exp3_ablation", args.output_dir)
    setup_experiment_logger(output_dir, "exp3.log")

    ablation_results = []
    anomalies = []
    full_totals_by_model = {}

    for model_name in args.models:
        model_layers = iter_model_layers(model_name)
        if args.maxLayers is not None:
            model_layers = model_layers[:args.maxLayers]

        totals_by_objective = {}

        for objective in args.objectives:
            totals = _empty_total()

            for layer in model_layers:
                loopdim = copy.deepcopy(layer["loopdim"])
                layer_dir = output_dir / objective / model_name / layer["layer"]
                prepare_save_dir(str(layer_dir))

                try:
                    miredo = run_miredo_layer(
                        acc=make_accelerator(args.architecture),
                        loopdim=loopdim,
                        outputdir=layer_dir,
                        objective=objective,
                        time_limit=args.timeLimit,
                        mip_focus=args.mipFocus,
                        return_profile=False,
                    )
                    _accumulate(
                        totals,
                        miredo["simulator_latency"],
                        miredo["simulator_energy"],
                    )
                except Exception as exc:
                    anomalies.append({
                        "model": model_name,
                        "layer": layer["layer"],
                        "objective": objective,
                        "kind": "runtime_error",
                        "message": str(exc),
                    })

            totals_by_objective[objective] = totals

        full_total = totals_by_objective["Latency"]
        full_totals_by_model[model_name] = full_total
        for objective in args.objectives:
            totals = totals_by_objective[objective]
            ablation_results.append({
                "variant": VARIANT_LABELS[objective],
                "model": model_name,
                "total_latency": totals["total_latency"],
                "total_energy": totals["total_energy"],
                "total_edp": totals["total_edp"],
                "degradation_vs_full": _degradation_vs_full(totals, full_total),
            })

    # ── Structural ablation variants ──────────────────────────────────
    if args.structural:
        for variant_name in args.structural:
            ablation_flags = STRUCTURAL_VARIANTS[variant_name]
            for model_name in args.models:
                model_layers = iter_model_layers(model_name)
                if args.maxLayers is not None:
                    model_layers = model_layers[:args.maxLayers]

                totals = _empty_total()
                for layer in model_layers:
                    loopdim = copy.deepcopy(layer["loopdim"])
                    layer_dir = output_dir / variant_name / model_name / layer["layer"]
                    prepare_save_dir(str(layer_dir))
                    try:
                        miredo = run_miredo_layer(
                            acc=make_accelerator(args.architecture),
                            loopdim=loopdim,
                            outputdir=layer_dir,
                            objective="Latency",
                            time_limit=args.timeLimit,
                            mip_focus=args.mipFocus,
                            return_profile=False,
                            ablation_flags=ablation_flags,
                        )
                        _accumulate(totals, miredo["simulator_latency"], miredo["simulator_energy"])
                    except Exception as exc:
                        anomalies.append({
                            "model": model_name,
                            "layer": layer["layer"],
                            "variant": variant_name,
                            "kind": "runtime_error",
                            "message": str(exc),
                        })

                full_total = full_totals_by_model.get(model_name, _empty_total())
                ablation_results.append({
                    "variant": variant_name,
                    "model": model_name,
                    "total_latency": totals["total_latency"],
                    "total_energy": totals["total_energy"],
                    "total_edp": totals["total_edp"],
                    "degradation_vs_full": _degradation_vs_full(totals, full_total),
                })

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="EXP-3.json",
        experiment_id="EXP-3",
        script_path=__file__,
        config={
            "models": args.models,
            "objectives": args.objectives,
            "architecture": hardware_spec_from_acc(acc),
            "time_limit": args.timeLimit,
            "mip_focus": args.mipFocus,
            "structural_variants": args.structural or [],
        },
        results={
            "ablation_results": ablation_results,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

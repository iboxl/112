import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Evaluation.common.EvalCommon import (
    hardware_spec_from_acc,
    make_accelerator,
    make_output_dir,
    run_miredo_layer,
    save_experiment_json,
    setup_experiment_logger,
)
from Evaluation.common.CaseLayerShapes import layer_selection_config, select_model_layers
from utils.UtilsFunction.ToolFunction import prepare_save_dir


VARIANT_LABELS = {
    "Latency": "latency-only",
    "Energy": "energy-only",
    "EDP": "edp",
}

STRUCTURAL_VARIANTS = {
    "fixed-double-buffer": {"ABLATION_FIXED_DOUBLE_BUFFER": True},
    "simplified-pipeline": {"ABLATION_SIMPLIFIED_PIPELINE": True},
    "psum-capacity-only": {"ABLATION_PSUM_CAPACITY_ONLY": True},
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
    # Whole-model EDP = (total latency) x (total energy), not the per-layer sum;
    # recomputed from the running totals so it is the product of the finals
    # after the last accumulate call. MIREDO still optimizes per-layer EDP.
    total["total_edp"] = total["total_latency"] * total["total_energy"]


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
    parser = argparse.ArgumentParser(description="objective_ablation")
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenetV2"])
    parser.add_argument("--objectives", nargs="+", default=["Latency", "Energy", "EDP"])
    parser.add_argument("--structural", nargs="*", default=None,
                        choices=list(STRUCTURAL_VARIANTS.keys()),
                        help="Structural ablation variants to run (e.g. fixed-double-buffer)")
    parser.add_argument("--architecture", default="CIM_ACC_DEFAULT_SETUP")
    parser.add_argument("--timeLimit", type=int, default=60)
    parser.add_argument("--mipFocus", type=int, default=1)
    parser.add_argument("--maxLayers", type=int, default=None)
    parser.add_argument("--layers", nargs="+", default=None,
                        help="Optional layer subset: exact names/aliases, 1-based positions, idx:N, or model:<selector>.")
    parser.add_argument("-o", "--outputdir", dest="output_dir", default=None)
    args = parser.parse_args()

    output_dir = make_output_dir("objective_ablation", args.output_dir)
    setup_experiment_logger(output_dir, "objective_ablation.log")

    ablation_results = []
    anomalies = []
    # Per-(model, layer) full-MIREDO baseline, keyed PER LAYER and never summed
    # to a model row. A --layers selection may pick several non-contiguous case
    # layers from one model (e.g. two ResNet case layers); accumulating them into
    # one "model" total reports a synthetic network that does not exist and hides
    # each case layer's individual degradation. Always emit one row per layer.
    full_by_layer = {}

    for model_name in args.models:
        model_layers = select_model_layers(
            model_name, layer_selectors=args.layers, max_layers=args.maxLayers,
        )

        for layer in model_layers:
            layer_key = (model_name, layer["layer"])
            totals_by_objective = {}

            for objective in args.objectives:
                loopdim = copy.deepcopy(layer["loopdim"])
                layer_dir = output_dir / objective / model_name / layer["layer"]
                prepare_save_dir(str(layer_dir))

                totals = _empty_total()
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
            full_by_layer[layer_key] = full_total
            for objective in args.objectives:
                totals = totals_by_objective[objective]
                ablation_results.append({
                    "variant": VARIANT_LABELS[objective],
                    "model": model_name,
                    "layer": layer["layer"],
                    "latency": totals["total_latency"],
                    "energy": totals["total_energy"],
                    "edp": totals["total_edp"],
                    "degradation_vs_full": _degradation_vs_full(totals, full_total),
                })

    # ── Structural ablation variants ──────────────────────────────────
    if args.structural:
        for variant_name in args.structural:
            ablation_flags = STRUCTURAL_VARIANTS[variant_name]
            for model_name in args.models:
                model_layers = select_model_layers(
                    model_name, layer_selectors=args.layers, max_layers=args.maxLayers,
                )

                for layer in model_layers:
                    loopdim = copy.deepcopy(layer["loopdim"])
                    layer_dir = output_dir / variant_name / model_name / layer["layer"]
                    prepare_save_dir(str(layer_dir))
                    totals = _empty_total()
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

                    full_total = full_by_layer.get((model_name, layer["layer"]), _empty_total())
                    ablation_results.append({
                        "variant": variant_name,
                        "model": model_name,
                        "layer": layer["layer"],
                        "latency": totals["total_latency"],
                        "energy": totals["total_energy"],
                        "edp": totals["total_edp"],
                        "degradation_vs_full": _degradation_vs_full(totals, full_total),
                    })

    acc = make_accelerator(args.architecture)
    json_path = save_experiment_json(
        output_dir=output_dir,
        file_name="objective_ablation.json",
        experiment_id="objective_ablation",
        script_path=__file__,
        config={
            "models": args.models,
            "objectives": args.objectives,
            "architecture": hardware_spec_from_acc(acc),
            "architecture_key": args.architecture,
            "time_limit": args.timeLimit,
            "mip_focus": args.mipFocus,
            "structural_variants": args.structural or [],
            "layer_selection": layer_selection_config(
                layer_selectors=args.layers, max_layers=args.maxLayers,
            ),
        },
        results={
            "ablation_results": ablation_results,
        },
        anomalies=anomalies,
    )
    print(json_path)


if __name__ == "__main__":
    main()

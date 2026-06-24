#!/usr/bin/env python3
"""Phase G — §5.3.1 fidelity + §5.3.2 ranking + §5.3.3 gap closure, analysis-only
over Phase A's EDP-mode logs (no new MIP solves).

Each Phase A layer dir holds a Scheme-Summary.log recording every spatial
candidate whose inner MIP produced a feasible solution, with both the
analytical (Solver-) and simulator (Simu-) latency/energy for that mapping,
and a SolPool/ of per-candidate Gurobi Solver.log files carrying the MIP
optimality gap.

§5.3.1 fidelity : for each layer's analytical-EDP-winner mapping, the
                  latency / energy relative error |solver - sim| / sim.
§5.3.2 ranking  : per layer, does the analytical-EDP-best candidate also have
                  the minimum simulator EDP.
§5.3.3 closure  : per layer, the winning candidate's MIP gap. A genuine zero-gap
                  closure (obj!=0, gap==0 at the 60s main solve) is "proven
                  optimal"; obj==0 with gap==0 is the degenerate 20s-prescreen
                  artifact; gap>0 is an open (McCormick-slack) gap.

EXTENDED 2026-06-08 from CNN-only to all evaluated workloads (CNN + Transformer).
Each workload group has logged unique-shape layers; the rest are shape-twins
(identical signature) that MIP-cache-hit their twin and inherit the outcome.

Output: <rerun-root>/_analysis/ : 5_3_1_fidelity.json, 5_3_2_ranking.json,
            5_3_3_closure.json (each with all-workload `results` + per-workload
            `by_group` + `per_layer`).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_CODE_REPO = Path(__file__).resolve().parents[1]


def _resolve_rerun_root() -> Path:
    """Rerun root, flexibly — NO hardcoded date, so any rerun is supported.
    Priority: $MIREDO_RERUN_ROOT (absolute path, or a name under output/) >
    newest output/logs_rerun_* directory."""
    out = _CODE_REPO / "output"
    env = os.environ.get("MIREDO_RERUN_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (out / env)
    cands = sorted(out.glob("logs_rerun_*"))
    if not cands:
        raise SystemExit(
            f"[RunPhaseGAnalysis] no logs_rerun_* under {out}; "
            f"set MIREDO_RERUN_ROOT")
    return cands[-1]


_RR = _resolve_rerun_root()
OUT_DIR = _RR / "_analysis"

# Workload groups: each maps to a Phase-A EDP source dir + its network/block
# names. CNN draws from the cnn_main comparison; Transformer from the HW-
# Transformer template comparison. Both log per-candidate analytical+simulator
# in Scheme-Summary.log and per-candidate MIP gap in SolPool/Solver.log.
GROUPS = {
    "cnn": {
        "base": _RR / "s5_2_1_cnn_main" / "baseline_comparison" / "EDP",
        # AlexNet excluded: not part of the paper's 4-CNN suite (169 CNN layers).
        "nets": ["resnet18", "vgg19bn", "mobilenetV2", "EfficientNet-B0"],
    },
    "transformer": {
        "base": _RR / "s5_2_2_transformer" / "main" / "EDP",
        "nets": ["bert_base", "gpt2_medium_block", "tinyllama_block"],
    },
}

BLOCK_RE = re.compile(
    r"^Scheme\s+(\d+)\s+End:\s*Latency-(\d+(?:\.\d+)?)\s*,\s*"
    r"Energy-(\d+(?:\.\d+)?)\s*,\s*EDP-(\d+(?:\.\d+)?)\s*\n"
    r"\s*\|---\s*Latency Relative Error:[^\n]*Solver-(\d+(?:\.\d+)?)\s*and\s*Simu-(\d+(?:\.\d+)?)\s*\n"
    r"\s*\|---\s*Energy\s*Relative Error:[^\n]*Solver-(\d+(?:\.\d+)?)\s*and\s*Simu-(\d+(?:\.\d+)?)",
    re.MULTILINE,
)

# Layer dir name is Conv_<idx>_... (CNN) or MatMul_<idx>_... / Gemm_<idx>_...
# (Transformer); the shape signature (everything after the idx) is the MIP-cache
# key for shape-equivalence (twins share it).
SIG_RE = re.compile(r"(?:Conv|MatMul|Gemm)_\d+_(.+)$")

# Gurobi per-candidate final line: "Best objective X, best bound Y, gap Z%".
GAP_RE = re.compile(
    r"Best objective\s+([0-9.eE+-]+),\s+best bound\s+([0-9.eE+-]+),\s+gap\s+([0-9.eE+-]+)%")


def shape_sig(layer_name: str) -> str:
    m = SIG_RE.match(layer_name)
    return m.group(1) if m else layer_name


def parse_layer_log(log_path: Path):
    text = log_path.read_text()
    entries = []
    for m in BLOCK_RE.finditer(text):
        solver_lat = float(m.group(5))
        simu_lat = float(m.group(6))
        solver_eng = float(m.group(7))
        simu_eng = float(m.group(8))
        entries.append({
            "scheme_id": int(m.group(1)),
            "analytical_latency": solver_lat,
            "analytical_energy": solver_eng,
            "analytical_edp": solver_lat * solver_eng,
            "simulator_latency": simu_lat,
            "simulator_energy": simu_eng,
            "simulator_edp": simu_lat * simu_eng,
        })
    return entries


def analyse_layer(entries):
    if not entries:
        return None
    winner = min(entries, key=lambda e: e["analytical_edp"])
    simu_best = min(entries, key=lambda e: e["simulator_edp"])
    simu_gap = ((winner["simulator_edp"] - simu_best["simulator_edp"])
                / max(1e-12, simu_best["simulator_edp"]) * 100.0)
    lat_err = (abs(winner["analytical_latency"] - winner["simulator_latency"])
               / max(1.0, winner["simulator_latency"]) * 100.0)
    eng_err = (abs(winner["analytical_energy"] - winner["simulator_energy"])
               / max(1.0, winner["simulator_energy"]) * 100.0)
    return {
        "num_solved": len(entries),
        "rank_agrees": winner["scheme_id"] == simu_best["scheme_id"],
        "simulator_gap_pct": simu_gap,
        "analytical_winner_scheme": winner["scheme_id"],
        "simulator_best_scheme": simu_best["scheme_id"],
        "fid_lat_err_pct": lat_err,
        "fid_eng_err_pct": eng_err,
        "fid_lat_overest": winner["analytical_latency"] >= winner["simulator_latency"],
        "fid_eng_overest": winner["analytical_energy"] >= winner["simulator_energy"],
    }


def winner_gap(layer_dir: Path):
    """Lowest final MIP objective across the layer's SolPool candidates, and that
    winner's relaxation gap (%). Returns (obj, gap) or (None, None) if no log."""
    sp = layer_dir / "SolPool"
    if not sp.is_dir():
        return (None, None)
    best_o = best_g = None
    for solver_log in sp.glob("**/Solver.log"):
        o = g = None
        try:
            txt = solver_log.read_text(errors="ignore")
        except Exception:
            continue
        for m in GAP_RE.finditer(txt):
            o = float(m.group(1))
            g = float(m.group(3))
        if o is None:
            continue
        if best_o is None or o < best_o:
            best_o, best_g = o, g
    return (best_o, best_g)


def closure_label(obj, gap):
    """Match closure_report.py: genuine zero-gap = obj!=0 & gap==0 (proven
    optimal at the 60s main solve); prescreen artifact = obj==0 (degenerate
    20s prescreen); open = gap>0; no_gap_data if missing."""
    if obj is None or gap is None:
        return "no_gap_data"
    if obj == 0.0 and gap == 0.0:
        return "prescreen_artifact"
    if gap == 0.0:
        return "genuine_zero"
    return "open"


def _mean(x):
    return sum(x) / len(x) if x else 0.0


def _worst(x):
    return max(x) if x else 0.0


def _med(x):
    return x[len(x) // 2] if x else 0.0


def aggregate(per_layer):
    """Fidelity + ranking + closure aggregates over a per-layer record list."""
    valid = [r for r in per_layer if r.get("rank_agrees") is not None]
    multi = [r for r in valid if r.get("num_solved", 0) >= 2]
    agree = [r for r in valid if r["rank_agrees"]]
    dis = [r for r in valid if not r["rank_agrees"]]
    dgap = sorted(r["simulator_gap_pct"] for r in dis)
    allgap = sorted(r["simulator_gap_pct"] for r in valid)
    lat_errs = sorted(r["fid_lat_err_pct"] for r in valid)
    eng_errs = sorted(r["fid_eng_err_pct"] for r in valid)

    fidelity = {
        "total_layers": len(per_layer),
        "valid_layers": len(valid),
        "logged_layers": sum(1 for r in per_layer if r.get("source") == "logged"),
        "cache_inherited_layers": sum(1 for r in per_layer if r.get("source") == "cache_inherited"),
        "latency_error_pct": {
            "mean": _mean(lat_errs), "worst": _worst(lat_errs), "median": _med(lat_errs),
            "overestimate_share_pct": 100.0 * sum(
                1 for r in valid if r["fid_lat_overest"]) / max(1, len(valid)),
        },
        "energy_error_pct": {
            "mean": _mean(eng_errs), "worst": _worst(eng_errs), "median": _med(eng_errs),
            "overestimate_share_pct": 100.0 * sum(
                1 for r in valid if r["fid_eng_overest"]) / max(1, len(valid)),
        },
    }
    ranking = {
        "total_layers": len(per_layer),
        "valid_layers": len(valid),
        "multi_solved_layers": len(multi),
        "rank_agrees_count": len(agree),
        "rank_agrees_pct": 100.0 * len(agree) / max(1, len(valid)),
        "disagreement_layers": len(dis),
        "disagreement_simu_gap_pct": {
            "mean": _mean(dgap), "worst": _worst(dgap), "median": _med(dgap)},
        "all_layer_mean_simu_gap_pct": _mean(allgap),
        "all_layer_worst_simu_gap_pct": _worst(allgap),
    }
    # closure (MIP optimality gap on the winning candidate)
    genuine = sum(1 for r in per_layer if r.get("closure") == "genuine_zero")
    artifact = sum(1 for r in per_layer if r.get("closure") == "prescreen_artifact")
    nodata = sum(1 for r in per_layer if r.get("closure") in (None, "no_gap_data"))
    openg = [r["winner_gap_pct"] for r in per_layer
             if r.get("closure") == "open" and r.get("winner_gap_pct") is not None]
    opens = sorted(x for x in openg if x > 0.05)

    def c(lo, hi):
        return sum(1 for x in openg if lo < x <= hi)

    closure = {
        "total_layers": len(per_layer),
        "with_gap_data": len(per_layer) - nodata,
        "proven_zero_gap": genuine,
        "prescreen_artifact": artifact,
        "open_gap_layers": len(openg),
        "no_gap_data": nodata,
        "open_gap_pct": {
            "n_gt_0p05": len(opens),
            "min": opens[0] if opens else 0.0,
            "median": _med(opens),
            "max": opens[-1] if opens else 0.0,
        },
        "hist": {"==0": genuine, "(0,1]": c(0, 1), "(1,5]": c(1, 5),
                 "(5,15]": c(5, 15), "(15,35]": c(15, 35),
                 "(35,60]": c(35, 60), ">60": sum(1 for x in openg if x > 60)},
    }
    return fidelity, ranking, closure


def scan_group(base: Path, nets):
    """Build the per-layer record list for one workload group (logged + twins)."""
    logged = {}
    per_layer = []
    pending = []
    if not base.is_dir():
        print(f"[warn] group base not found: {base}", file=sys.stderr)
        return per_layer
    for net in nets:
        net_dir = base / net
        if not net_dir.is_dir():
            continue
        for layer_dir in sorted(net_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            sig = shape_sig(layer_dir.name)
            log = layer_dir / "Scheme-Summary.log"
            if log.is_file():
                summ = analyse_layer(parse_layer_log(log))
                if summ is None:
                    pending.append((net, layer_dir.name, sig))
                    continue
                obj, gap = winner_gap(layer_dir)
                summ.update({"network": net, "layer": layer_dir.name, "sig": sig,
                             "source": "logged", "winner_obj": obj,
                             "winner_gap_pct": gap, "closure": closure_label(obj, gap)})
                logged.setdefault(sig, summ)
                per_layer.append(summ)
            else:
                pending.append((net, layer_dir.name, sig))

    for net, lname, sig in pending:
        twin = logged.get(sig)
        if twin is None:
            per_layer.append({"network": net, "layer": lname, "sig": sig,
                              "source": "no_twin", "num_solved": 0,
                              "rank_agrees": None, "winner_gap_pct": None,
                              "closure": "no_gap_data"})
            continue
        rec = dict(twin)
        rec.update({"network": net, "layer": lname, "sig": sig,
                    "source": "cache_inherited"})
        per_layer.append(rec)
    return per_layer


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups_pl = {g: scan_group(GROUPS[g]["base"], GROUPS[g]["nets"]) for g in GROUPS}
    by_group = {}
    for g, pl in groups_pl.items():
        f, r, c = aggregate(pl)
        by_group[g] = {"fidelity": f, "ranking": r, "closure": c}
    combined = [r for pl in groups_pl.values() for r in pl]
    cf, cr, cc = aggregate(combined)

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(_CODE_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "script": "MIREDO/Evaluation/RunPhaseGAnalysis.py",
        "commit": commit,
        "timestamp": datetime.now().astimezone().isoformat(),
        "groups": {g: str(GROUPS[g]["base"]) for g in GROUPS},
        "objective": "EDP",
        "note": "Analysis-only over Phase A EDP-mode Scheme-Summary.log (fidelity"
                "+ranking) and SolPool/Solver.log (MIP gap closure); no new MIP "
                "solves. Extended 2026-06-08 to all workloads (CNN + Transformer).",
    }

    def per_layer_view(keys):
        return [{k: r.get(k) for k in keys} for r in combined]

    fid_out = {"experiment_id": "5_3_1_fidelity", "provenance": prov,
               "results": cf, "by_group": {g: by_group[g]["fidelity"] for g in by_group},
               "per_layer": combined}
    rank_out = {"experiment_id": "5_3_2_ranking", "provenance": prov,
                "results": cr, "by_group": {g: by_group[g]["ranking"] for g in by_group},
                "per_layer": per_layer_view(("network", "layer", "num_solved",
                    "rank_agrees", "simulator_gap_pct", "winner_gap_pct",
                    "closure", "source"))}
    clo_out = {"experiment_id": "5_3_3_closure", "provenance": prov,
               "results": cc, "by_group": {g: by_group[g]["closure"] for g in by_group},
               "per_layer": per_layer_view(("network", "layer", "winner_obj",
                    "winner_gap_pct", "closure", "source"))}

    (OUT_DIR / "5_3_1_fidelity.json").write_text(json.dumps(fid_out, indent=2))
    (OUT_DIR / "5_3_2_ranking.json").write_text(json.dumps(rank_out, indent=2))
    (OUT_DIR / "5_3_3_closure.json").write_text(json.dumps(clo_out, indent=2))

    def report(tag, f, r, c):
        print(f"\n===== {tag} =====")
        print(f"  layers: {f['total_layers']} total "
              f"({f['logged_layers']} logged + {f['cache_inherited_layers']} cache-inherited), "
              f"valid {f['valid_layers']}")
        print(f"  §5.3.1 fidelity:  lat mean {f['latency_error_pct']['mean']:.2f}% / "
              f"worst {f['latency_error_pct']['worst']:.2f}% / med {f['latency_error_pct']['median']:.2f}%"
              f"  | eng mean {f['energy_error_pct']['mean']:.2f}% / "
              f"worst {f['energy_error_pct']['worst']:.2f}% / med {f['energy_error_pct']['median']:.2f}%")
        print(f"  §5.3.2 ranking:   agree {r['rank_agrees_count']}/{r['valid_layers']} "
              f"({r['rank_agrees_pct']:.1f}%), disagree {r['disagreement_layers']} "
              f"(simu-gap mean {r['disagreement_simu_gap_pct']['mean']:.2f}% / "
              f"worst {r['disagreement_simu_gap_pct']['worst']:.2f}%), "
              f"all-layer mean gap {r['all_layer_mean_simu_gap_pct']:.2f}%")
        print(f"  §5.3.3 closure:   proven zero-gap {c['proven_zero_gap']}/{c['total_layers']} "
              f"(prescreen-artifact {c['prescreen_artifact']}, open {c['open_gap_layers']}, "
              f"no-data {c['no_gap_data']}); open-gap hist {c['hist']}")

    for g in GROUPS:
        report(g, by_group[g]["fidelity"], by_group[g]["ranking"], by_group[g]["closure"])
    report("ALL WORKLOADS (combined)", cf, cr, cc)
    print(f"\nOutputs: {OUT_DIR}/5_3_1_fidelity.json, 5_3_2_ranking.json, 5_3_3_closure.json")


if __name__ == "__main__":
    main()

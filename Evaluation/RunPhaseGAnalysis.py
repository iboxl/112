#!/usr/bin/env python3
"""Phase G — §5.3.1 fidelity + §5.3.2 ranking, analysis-only over Phase A's
EDP-mode Scheme-Summary.log files (no new MIP solves).

Each Phase A layer dir holds a Scheme-Summary.log recording every spatial
candidate whose inner MIP produced a feasible solution, with both the
analytical (Solver-) and simulator (Simu-) latency/energy for that mapping.

§5.3.1 fidelity : for each layer's analytical-EDP-winner mapping, the
                  latency / energy relative error |solver - sim| / sim.
                  Aggregated mean / worst / upward-bias share over 174 layers.
§5.3.2 ranking  : per layer, does the analytical-EDP-best candidate also have
                  the minimum simulator EDP. Aggregated agreement count and the
                  simulator-EDP gap on disagreement layers.

101 layers carry a logged Scheme-Summary; the remaining 73 are shape-twins
(identical R_S_P_Q_C_K_G signature) that MIP-cache-hit their twin and inherit
its fidelity + ranking outcome — matching the legacy EXP-8 cache-inherited
methodology (total = 174).

Output: <rerun-root>/_analysis/ (rerun root resolved flexibly — see
            _resolve_rerun_root): 5_3_1_fidelity.json, 5_3_2_ranking.json,
            diff_report.md
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# FIX 2026-05-17: portable + relocated + renamed. Code repo = this file's
# MIREDO/ ; rerun tree lives in the code repo's output/ (was experiments/);
# Phase A dir renamed exp2_cnn_comparison → baseline_comparison.
_CODE_REPO = Path(__file__).resolve().parents[1]


def _resolve_rerun_root() -> Path:
    """Rerun root, flexibly — NO hardcoded date, so any rerun is supported.
    Priority: $MIREDO_RERUN_ROOT (absolute path, or a name under output/) >
    newest output/logs_rerun_* directory. (Same contract as
    Evaluation/extract_profiling_caselayer.py:_resolve_rerun_root.)"""
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
PHASE_A_EDP = _RR / "s5_2_1_cnn_main" / "baseline_comparison" / "EDP"
OUT_DIR = _RR / "_analysis"
LEGACY = _CODE_REPO.parent / "experiments" / "parsed_metrics"

NETWORKS = ["resnet18", "vgg19bn", "alexnet", "mobilenetV2", "EfficientNet-B0"]

BLOCK_RE = re.compile(
    r"^Scheme\s+(\d+)\s+End:\s*Latency-(\d+(?:\.\d+)?)\s*,\s*"
    r"Energy-(\d+(?:\.\d+)?)\s*,\s*EDP-(\d+(?:\.\d+)?)\s*\n"
    r"\s*\|---\s*Latency Relative Error:[^\n]*Solver-(\d+(?:\.\d+)?)\s*and\s*Simu-(\d+(?:\.\d+)?)\s*\n"
    r"\s*\|---\s*Energy\s*Relative Error:[^\n]*Solver-(\d+(?:\.\d+)?)\s*and\s*Simu-(\d+(?:\.\d+)?)",
    re.MULTILINE,
)

# Layer dir name is Conv_<idx>_<R>_<S>_<P>_<Q>_<C>_<K>_<G>; the shape signature
# (everything after the idx) is the MIP-cache key for shape-equivalence.
SIG_RE = re.compile(r"Conv_\d+_(.+)$")


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
    # §5.3.1 fidelity from the analytical winner mapping
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PHASE_A_EDP.is_dir():
        sys.exit(f"Phase A EDP dir not found: {PHASE_A_EDP}")

    logged = {}          # sig -> summary (first logged occurrence)
    per_layer = []       # all 174 records (logged + inherited)
    pending_inherit = [] # (network, layer, sig) with no log yet

    for network in NETWORKS:
        net_dir = PHASE_A_EDP / network
        if not net_dir.is_dir():
            continue
        for layer_dir in sorted(net_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            sig = shape_sig(layer_dir.name)
            log = layer_dir / "Scheme-Summary.log"
            if log.is_file():
                entries = parse_layer_log(log)
                summ = analyse_layer(entries)
                if summ is None:
                    pending_inherit.append((network, layer_dir.name, sig))
                    continue
                summ.update({"network": network, "layer": layer_dir.name,
                             "sig": sig, "source": "logged"})
                logged.setdefault(sig, summ)
                per_layer.append(summ)
            else:
                pending_inherit.append((network, layer_dir.name, sig))

    inherited = 0
    for network, layer, sig in pending_inherit:
        twin = logged.get(sig)
        if twin is None:
            per_layer.append({"network": network, "layer": layer, "sig": sig,
                              "source": "no_twin", "num_solved": 0,
                              "rank_agrees": None})
            continue
        rec = dict(twin)
        rec.update({"network": network, "layer": layer, "sig": sig,
                    "source": "cache_inherited"})
        per_layer.append(rec)
        inherited += 1

    valid = [r for r in per_layer if r.get("rank_agrees") is not None]
    multi = [r for r in valid if r.get("num_solved", 0) >= 2]
    agree = [r for r in valid if r["rank_agrees"]]
    dis = [r for r in valid if not r["rank_agrees"]]
    dgap = sorted(r["simulator_gap_pct"] for r in dis)
    allgap = sorted(r["simulator_gap_pct"] for r in valid)
    lat_errs = sorted(r["fid_lat_err_pct"] for r in valid)
    eng_errs = sorted(r["fid_eng_err_pct"] for r in valid)

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    def worst(x):
        return max(x) if x else 0.0

    fidelity = {
        "total_layers": len(per_layer),
        "valid_layers": len(valid),
        "logged_layers": sum(1 for r in per_layer if r.get("source") == "logged"),
        "cache_inherited_layers": inherited,
        "latency_error_pct": {
            "mean": mean(lat_errs), "worst": worst(lat_errs),
            "median": lat_errs[len(lat_errs)//2] if lat_errs else 0.0,
            "overestimate_share_pct": 100.0 * sum(
                1 for r in valid if r["fid_lat_overest"]) / max(1, len(valid)),
        },
        "energy_error_pct": {
            "mean": mean(eng_errs), "worst": worst(eng_errs),
            "median": eng_errs[len(eng_errs)//2] if eng_errs else 0.0,
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
            "mean": mean(dgap), "worst": worst(dgap),
            "median": dgap[len(dgap)//2] if dgap else 0.0,
        },
        "all_layer_mean_simu_gap_pct": mean(allgap),
        "all_layer_worst_simu_gap_pct": worst(allgap),
    }

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    prov = {
        "script": "MIREDO/Evaluation/RunPhaseGAnalysis.py",
        "commit": commit,
        "timestamp": datetime.now().astimezone().isoformat(),
        "source": str(PHASE_A_EDP),
        "objective": "EDP",
        "note": "Analysis-only over Phase A EDP-mode Scheme-Summary.log; "
                "no new MIP solves. Fidelity + ranking from the same EDP "
                "mapping set (paper §5.3.1/§5.3.2 single-set framing).",
    }

    fid_out = {"experiment_id": "5_3_1_fidelity", "provenance": prov,
               "results": fidelity, "per_layer": per_layer}
    rank_out = {"experiment_id": "5_3_2_ranking", "provenance": prov,
                "results": ranking, "per_layer": [
                    {k: r.get(k) for k in ("network", "layer", "num_solved",
                     "rank_agrees", "simulator_gap_pct", "source")}
                    for r in per_layer]}

    (OUT_DIR / "5_3_1_fidelity.json").write_text(json.dumps(fid_out, indent=2))
    (OUT_DIR / "5_3_2_ranking.json").write_text(json.dumps(rank_out, indent=2))

    # ── Diff report vs legacy paper numbers ───────────────────────────────
    def load_legacy(pat):
        for p in sorted(LEGACY.glob(pat), reverse=True):
            try:
                return json.load(open(p)), p.name
            except Exception:
                continue
        return None, None

    lines = []
    lines.append("# Phase G diff report — rerun (CIM_ACC_DEFAULT_SETUP, new HW) vs legacy paper data")
    lines.append(f"\nGenerated {datetime.now():%Y-%m-%d %H:%M}, source {PHASE_A_EDP}\n")
    lines.append("## §5.3.1 Analytical-model fidelity (EDP-mode mappings)\n")
    lines.append(f"- layers: {fidelity['total_layers']} total "
                 f"({fidelity['logged_layers']} logged + {fidelity['cache_inherited_layers']} cache-inherited)")
    lines.append(f"- **latency error**: mean {fidelity['latency_error_pct']['mean']:.2f}%, "
                 f"worst {fidelity['latency_error_pct']['worst']:.2f}%, "
                 f"overestimate share {fidelity['latency_error_pct']['overestimate_share_pct']:.0f}%")
    lines.append(f"- **energy error**: mean {fidelity['energy_error_pct']['mean']:.2f}%, "
                 f"worst {fidelity['energy_error_pct']['worst']:.2f}%, "
                 f"overestimate share {fidelity['energy_error_pct']['overestimate_share_pct']:.0f}%")
    lines.append("- paper §5.3.1 (legacy, old HW + Latency-obj): latency mean 3.84% / worst 9.47%, "
                 "energy mean 0.71% / worst 3.54%, 69% lat & 68% eng overestimate")
    lines.append("")
    lines.append("## §5.3.2 Rank consistency (EDP-mode)\n")
    lines.append(f"- rank agrees {ranking['rank_agrees_count']}/{ranking['valid_layers']} "
                 f"({ranking['rank_agrees_pct']:.1f}%)")
    lines.append(f"- disagreement layers: {ranking['disagreement_layers']}, "
                 f"simu-EDP gap mean {ranking['disagreement_simu_gap_pct']['mean']:.2f}% "
                 f"/ worst {ranking['disagreement_simu_gap_pct']['worst']:.2f}%")
    lines.append(f"- all-layer mean simu gap {ranking['all_layer_mean_simu_gap_pct']:.2f}%")
    lines.append("- paper §5.3.2 (legacy): 147/174 (84.5%), disagreement median 0.83% / "
                 "mean 1.73% / worst 8.23%, all-layer 0.27%")
    lines.append("")
    # §5.6.1 dynamic-LB — compute from the actual (BYPASS) dynlb json, NEVER hardcode.
    # The legacy hardcoded "6.4e4x/6.0e4x" was a non-BYPASS MIP-cache-contamination
    # artifact (lb_on served from Phase A's shared key in ~2ms vs lb_off cold solve).
    dynlb_summary = ("source 5_6_1_dynlb_control.json absent at report time — "
                     "rerun Phase F §5.6.1a, then re-read")
    dynlb_json = _RR / "s5_6_perlayer_cost" / "5_6_1_dynlb_control.json"
    if dynlb_json.is_file():
        try:
            dd = json.load(open(dynlb_json))["results"]
            by = {}
            for r in dd:
                by.setdefault(r["layer_id"], {})[r["mode"]] = r
            ratios, prunes, lossless = [], [], True
            for m in by.values():
                on, off = m.get("lb_on"), m.get("lb_off")
                if not on or not off:
                    continue
                if on.get("mip_wall_sec"):
                    ratios.append(off["mip_wall_sec"] / on["mip_wall_sec"])
                init = on.get("num_schemes_initial") or 1
                prunes.append(on.get("num_schemes_dynamic_lb_pruned", 0) / init)
                lossless = lossless and (on.get("best_metric") == off.get("best_metric"))
            if ratios:
                dynlb_summary = (
                    f"wall ratio lb_off/lb_on {min(ratios):.2f}-{max(ratios):.2f}x, "
                    f"scheme prune {min(prunes)*100:.0f}-{max(prunes)*100:.0f}%, "
                    f"lossless={'yes (rel_diff=0)' if lossless else 'NO — INVESTIGATE'}")
        except Exception as exc:
            dynlb_summary = f"present but unreadable ({exc}) — re-run §5.6.1a"

    lines.append("## §5.3.3 / §5.2.2 optimality cert (source-of-truth pointers, NOT hardcoded)\n")
    lines.append("- CNN anchors: read `_analysis/optimality_chain_cnn` from THIS rerun "
                 "(do NOT reuse stale hardcoded cert cycles).")
    lines.append("- attention QK^T tile: read the `VerifyBruteforce --case attention_tiny` "
                 "(CIM_ACC_DEFAULT_SETUP_TRANSFORMER, --objective both) result from THIS rerun "
                 "under `output/Eval_Result/` (VerifyOptimalityChain anchors are CNN-only, so the "
                 "attention cert comes from VerifyBruteforce, not the chain; do NOT reuse stale values).")
    lines.append("")
    lines.append("## Action items for paper editor\n")
    lines.append("- §5.3.1: replace mean/worst lat & energy error with the rerun numbers above (new HW, EDP-mode)")
    lines.append("- §5.3.2: replace 147/174 + gap stats with the rerun numbers above")
    lines.append("- §5.3.3: read cert cycles from `_analysis/optimality_chain_cnn` (do NOT reuse stale 70/71/151 or any hardcoded value)")
    lines.append("- §5.4.1: force-Q 23.5% claim unverified on new HW (OBuf 16KB removes the writeback bottleneck) — re-verify or drop")
    lines.append(f"- §5.6.1: dynamic-LB — {dynlb_summary} (computed from 5_6_1_dynlb_control.json; "
                 "the legacy '6.4e4x/6.0e4x' was a non-BYPASS cache-contamination artifact — do NOT use)")
    lines.append("- §5.6 walltime: regenerate Table VII from 5_6_3_walltime.json (new HW)")

    (OUT_DIR / "diff_report.md").write_text("\n".join(lines) + "\n")

    print("§5.3.1 fidelity:")
    print(f"  lat err mean={fidelity['latency_error_pct']['mean']:.2f}% "
          f"worst={fidelity['latency_error_pct']['worst']:.2f}%")
    print(f"  eng err mean={fidelity['energy_error_pct']['mean']:.2f}% "
          f"worst={fidelity['energy_error_pct']['worst']:.2f}%")
    print("§5.3.2 ranking:")
    print(f"  agree {ranking['rank_agrees_count']}/{ranking['valid_layers']} "
          f"({ranking['rank_agrees_pct']:.1f}%), "
          f"all-layer mean gap {ranking['all_layer_mean_simu_gap_pct']:.2f}%")
    print(f"\nOutputs: {OUT_DIR}/5_3_1_fidelity.json, 5_3_2_ranking.json, diff_report.md")


if __name__ == "__main__":
    main()

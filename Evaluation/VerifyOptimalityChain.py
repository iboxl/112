#!/usr/bin/env python3
"""
Brute-force optimality validation chain on real EfficientNet-B0 layers.

Target layers (3 sub_mip_open SE-squeeze + 3 smallest proven SE-squeeze + optional depthwise):
For each layer:
  1. Construct a legal spatial scheme (deterministic, layer-shape-driven)
  2. Run brute-force enumeration over all temporal mappings under that scheme
  3. Run native MIP under the same scheme
  4. Compare best simulator latency

Output: parsed_metrics-compatible JSON with per-layer brute-force vs MIP gap.
"""

import os
import sys
import time
import copy
import math
import json

# Ensure we run from MIREDO/ root
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from utils.Workload import WorkLoad
from utils.GlobalUT import CONST, FLAG, Logger
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from Evaluation.VerifyBruteforce import run_verify
from Evaluation.common.EvalCommon import save_experiment_json

Logger.setcfg(setcritical=False, setDebug=False, STD=False, file="", nofile=True)
import logging
logging.disable(logging.CRITICAL)

CONST.FLAG_OPT = "Latency"
CONST.MIPFOCUS = 1
FLAG.GUROBI_OUTPUT = False
FLAG.SIMU = False


# dim2Dict = ['-','R','S','P','Q','C','K','G']  (8 dims, index 0-7)
# scheme = [[cores axes], [dimX axes], [dimY axes]] — 3 rows × 8 cols, all 1 except spatial unrolling
def make_scheme_for_se(C, K, num_cores=8, dimX=32, dimY=16):
    """Spatial scheme for an SE-squeeze 1x1 GEMV. Intentionally leaves at least
    one dimension partially in temporal to avoid the all-spatial trivial case
    that returns no active_dims (which crashes run_verify on a bare stats dict).

    Strategy: pack C onto dimX up to dimX/2 (leaving factor 2+ in temporal
    when possible), pack K onto dimY similarly. If C/K are very small, fall
    back to a normal pack but ensure at least *some* temporal axis exists.
    """
    sc = [[1]*8, [1]*8, [1]*8]  # cores, dimX, dimY

    # ---- Strategy: try to leave at least one temporal axis ----
    # If C >= 4: put min(C // 2, dimX) on dimX (C // 2 keeps factor 2 temporal)
    # Else: put C fully on dimX
    if C >= 4:
        cx = min(C // 2, dimX)
    else:
        cx = min(C, dimX)
    sc[1][5] = max(1, cx)

    if K >= 4:
        ky = min(K // 2, dimY)
    else:
        ky = min(K, dimY)
    sc[2][6] = max(1, ky)

    # If somehow still no temporal (everything == 1 spatial), bump back to full pack.
    # The run_verify path handles non-trivial active_dims correctly.
    rem_C = max(1, C // sc[1][5])
    rem_K = max(1, K // sc[2][6])
    if rem_C == 1 and rem_K == 1:
        # Trivial case unavoidable: pack fully and accept the path.
        sc[1][5] = min(C, dimX)
        sc[2][6] = min(K, dimY)
    return sc


def make_scheme_for_depthwise_3x3_p7(G, num_cores=8, dimX=32, dimY=16):
    """For depthwise 3x3 P=Q=7 G=large layer: cores split G, dimX gets R*S spread on G,
    dimY gets G further. Conservative legal scheme.
    """
    sc = [[1]*8, [1]*8, [1]*8]
    # cores: G
    sc[0][7] = num_cores
    # dimX: G as well (until dimX filled)
    rem_G = max(1, G // num_cores)
    g_on_x = min(rem_G, dimX)
    sc[1][7] = g_on_x
    # dimY: G further
    rem_G_after_x = max(1, rem_G // g_on_x)
    g_on_y = min(rem_G_after_x, dimY)
    sc[2][7] = g_on_y
    return sc


# ── Layer targets ───────────────────────────────────────────────
TARGETS = [
    # 3 sub_mip_open SE-squeeze layers
    {
        "name": "EfficientNet-B0/Conv_2_1_1_1_1_32_8_1",
        "category": "sub_mip_open",
        "loopdim": {"R": 1, "S": 1, "C": 32, "K": 8, "P": 1, "Q": 1, "G": 1, "B": 1,
                    "H": 1, "W": 1, "Stride": 1, "Padding": 0},
        "scheme_fn": lambda d: make_scheme_for_se(d["C"], d["K"]),
    },
    {
        "name": "EfficientNet-B0/Conv_3_1_1_1_1_8_32_1",
        "category": "sub_mip_open",
        "loopdim": {"R": 1, "S": 1, "C": 8, "K": 32, "P": 1, "Q": 1, "G": 1, "B": 1,
                    "H": 1, "W": 1, "Stride": 1, "Padding": 0},
        "scheme_fn": lambda d: make_scheme_for_se(d["C"], d["K"]),
    },
    {
        "name": "EfficientNet-B0/Conv_8_1_1_1_1_4_96_1",
        "category": "sub_mip_open",
        "loopdim": {"R": 1, "S": 1, "C": 4, "K": 96, "P": 1, "Q": 1, "G": 1, "B": 1,
                    "H": 1, "W": 1, "Stride": 1, "Padding": 0},
        "scheme_fn": lambda d: make_scheme_for_se(d["C"], d["K"]),
    },
    # 3 smallest proven SE-squeeze layers (real, in proven set, validate MIP=brute-force)
    {
        "name": "EfficientNet-B0/Conv_7_1_1_1_1_96_4_1",
        "category": "proven_real_se",
        "loopdim": {"R": 1, "S": 1, "C": 96, "K": 4, "P": 1, "Q": 1, "G": 1, "B": 1,
                    "H": 1, "W": 1, "Stride": 1, "Padding": 0},
        "scheme_fn": lambda d: make_scheme_for_se(d["C"], d["K"]),
    },
    {
        "name": "EfficientNet-B0/Conv_12_1_1_1_1_144_6_1",
        "category": "proven_real_se",
        "loopdim": {"R": 1, "S": 1, "C": 144, "K": 6, "P": 1, "Q": 1, "G": 1, "B": 1,
                    "H": 1, "W": 1, "Stride": 1, "Padding": 0},
        "scheme_fn": lambda d: make_scheme_for_se(d["C"], d["K"]),
    },
    {
        "name": "EfficientNet-B0/Conv_13_1_1_1_1_6_144_1",
        "category": "proven_real_se",
        "loopdim": {"R": 1, "S": 1, "C": 6, "K": 144, "P": 1, "Q": 1, "G": 1, "B": 1,
                    "H": 1, "W": 1, "Stride": 1, "Padding": 0},
        "scheme_fn": lambda d: make_scheme_for_se(d["C"], d["K"]),
    },
    # 1 mid-size real depthwise layer (proven set) for non-trivial brute-force
    # mobilenetV2/Conv_40_3_3_7_7_1_1_576: 3x3, P=Q=7, G=576, log10(cost)=5.40
    # Slightly smaller than the toy 3x3 C=K=32 (log10=5.65, 20h), expected 5-15h.
    {
        "name": "mobilenetV2/Conv_40_3_3_7_7_1_1_576",
        "category": "proven_real_depthwise",
        "loopdim": {"R": 3, "S": 3, "C": 1, "K": 1, "P": 7, "Q": 7, "G": 576, "B": 1,
                    "H": 7, "W": 7, "Stride": 1, "Padding": 1},
        "scheme_fn": lambda d: make_scheme_for_depthwise_3x3_p7(d["G"]),
    },
]


def silent_log_fn(msg):
    """Quiet log for batch driver (still goes to per-layer file)."""
    pass


def run_one(target):
    name = target["name"]
    print(f"\n{'='*70}\n[+] {target['category']} | {name}\n{'='*70}", flush=True)
    spec = default_spec()
    acc = CIM_Acc.from_spec(spec)
    ops = WorkLoad(loopDim=target["loopdim"])
    scheme = target["scheme_fn"](target["loopdim"])
    print(f"  ops: {target['loopdim']}", flush=True)
    print(f"  scheme: {scheme}", flush=True)

    spatial = [math.prod(col) for col in zip(*scheme)]
    print(f"  spatial unrolling: {spatial}", flush=True)
    tu = [math.ceil(x / y) if y > 0 else x for x, y in zip(ops.dim2bound, spatial)]
    print(f"  temporal unrolling: {tu}", flush=True)

    # Per-layer log file
    log_dir = "/home/xiaolin/pro/overleaf/MIREDO/experiments/logs/optimality_chain"
    os.makedirs(log_dir, exist_ok=True)
    layer_log = os.path.join(log_dir, f"{name.replace('/','_')}.log")
    def lf(msg):
        with open(layer_log, "a") as f:
            f.write(f"{msg}\n")

    t0 = time.time()
    try:
        # run_verify writes its own EXP-6 file; we extract numbers from return value
        bf_lat, mip_simu_lat = run_verify(
            acc, ops, scheme,
            mip_timelimit=120,
            log_fn=lf,
            arch_spec=spec,
        )
    except KeyError as e:
        # Trivial-spatial case where active_dims=[] returns a sparse stats dict.
        # Fallback: explicitly handle as "single mapping" — bf_lat == mip_lat trivially.
        print(f"  trivial spatial-only mapping (no temporal axes); KeyError {e}", flush=True)
        # Compute the single feasible mapping's latency via direct simulator call.
        from utils.Workload import LoopNest, Mapping
        from Simulator.Simulax import tranSimulator
        sm_list = []
        for u in range(acc.Num_SpUr):
            for d in range(1, ops.Num_dim):
                if scheme[u][d] > 1:
                    sm_list.append(Mapping(dim=d, dimSize=scheme[u][d],
                                           mem=[acc.SpUr2Mem[u, op] for op in range(3)]))
        loops = LoopNest(acc=acc, ops=ops)
        loops.tm = []
        loops.sm = sm_list
        loops.usr_defined_double_flag = [[0]*3 for _ in range(acc.Num_mem + 1)]
        loops.psum_flag = None
        try:
            simu = tranSimulator(acc=acc, ops=ops, dataflow=loops)
            bf_lat, _bf_e = simu.run()
        except Exception as ee:
            print(f"  fallback simulator failed: {ee}", flush=True)
            return {"name": name, "category": target["category"], "error": f"trivial+sim_fail: {ee}"}
        # MIP under same scheme: build standalone Solver
        from utils.SolverTSS import Solver
        from utils.UtilsFunction.ToolFunction import prepare_save_dir
        CONST.FLAG_OPT = "Latency"
        CONST.TIMELIMIT = 120
        FLAG.SIMU = False
        spatial_local = [math.prod(col) for col in zip(*scheme)]
        tu_local = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_local)]
        mip_dir = "/tmp/optimality_chain_mip_tmp"
        prepare_save_dir(mip_dir)
        solver = Solver(acc=acc, ops=ops, tu=tu_local, su=scheme,
                        metric_ub=CONST.MAX_POS, outputdir=mip_dir)
        try:
            solver.run()
            if solver.model is not None and solver.model.SolCount > 0:
                simu_mip = tranSimulator(acc=acc, ops=ops, dataflow=solver.dataflow)
                mip_simu_lat, _ = simu_mip.run()
            else:
                mip_simu_lat = None
        finally:
            solver.close()
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        return {"name": name, "category": target["category"], "error": str(e)}
    else:
        elapsed = time.time() - t0

    gap_pct = None
    if bf_lat is not None and mip_simu_lat is not None and bf_lat > 0 and bf_lat < float("inf"):
        gap_pct = (mip_simu_lat - bf_lat) / bf_lat * 100

    res = {
        "name": name,
        "category": target["category"],
        "loopdim": target["loopdim"],
        "scheme": scheme,
        "spatial_unrolling": spatial,
        "temporal_unrolling": tu,
        "bruteforce_optimal_latency": bf_lat,
        "mip_simulator_latency": mip_simu_lat,
        "optimality_gap_pct": gap_pct,
        "is_optimal": (gap_pct is not None and abs(gap_pct) < 0.01),
        "wall_seconds": round(elapsed, 2),
    }
    print(f"  bf_optimal_lat={bf_lat}", flush=True)
    print(f"  mip_simu_lat={mip_simu_lat}", flush=True)
    print(f"  gap%={gap_pct}", flush=True)
    print(f"  wall={elapsed:.1f}s", flush=True)
    return res


def main():
    out_dir = "/home/xiaolin/pro/overleaf/MIREDO/experiments/parsed_metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"EXP-6c_optimality_chain_caselayers_{time.strftime('%Y%m%d_%H%M%S')}.json")

    all_results = []
    for target in TARGETS:
        r = run_one(target)
        all_results.append(r)
        # Save partial after each
        save_experiment_json(
            output_dir=out_dir,
            file_name=os.path.basename(out_file),
            experiment_id="EXP-6c",
            script_path=__file__,
            config={
                "verification_method": "bruteforce_real_layers",
                "targets": [{"name": t["name"], "category": t["category"]} for t in TARGETS],
                "completed_so_far": len(all_results),
            },
            results={
                "verification": all_results,
                "summary": {
                    "total_targets": len(TARGETS),
                    "completed": len(all_results),
                    "n_optimal": sum(1 for r in all_results if r.get("is_optimal")),
                },
            },
            anomalies=[],
        )
        print(f"\n  >> partial result saved to {out_file}", flush=True)

    # Final summary
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for r in all_results:
        print(f"  {r['category']:20} {r['name']:50}  gap={r.get('optimality_gap_pct')}%  bf_lat={r.get('bruteforce_optimal_latency')}  mip_lat={r.get('mip_simulator_latency')}")


if __name__ == "__main__":
    main()

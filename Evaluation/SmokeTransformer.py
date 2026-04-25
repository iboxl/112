"""Phase A7 smoke test: run MIREDO + WS on one ViT-Base encoder layer
(HW-Transformer). Verifies: ONNX parse → MatMul extraction → WorkLoad
→ MIREDO solver (parallel dispatch) → WS baseline → sanity-compare.

Invocation:
    cd MIREDO/
    python -m Evaluation.SmokeTransformer
"""

from __future__ import annotations

import copy
import os
import sys
import time


def main():
    # Repo root = MIREDO/ so imports resolve
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from Architecture.templates.transformer import transformer_spec
    from Architecture.ArchSpec import CIM_Acc
    from utils.Workload import WorkLoad
    from utils.GlobalUT import CONST, FLAG
    from utils.UtilsFunction.OnnxParser import extract_loopdims
    from Evaluation.WeightStationaryGenerator import generate_weight_stationary_baseline
    from SolveMapping import SolveMapping

    out_dir = os.path.join(repo, "output", "smoke_transformer")
    os.makedirs(out_dir, exist_ok=True)
    # SolveMapping's worker appends to this log; create empty
    with open(os.path.join(out_dir, "Scheme-Summary.log"), "w"):
        pass

    onnx_path = os.path.join(repo, "model", "vit_b_16.onnx")
    print(f"[Smoke] loading {onnx_path}")
    names, dims = extract_loopdims(onnx_path, allow_matmul=True)
    print(f"[Smoke] extracted {len(names)} layers")

    # Target: encoder-0 FFN1 (representative matmul, big enough to exercise solver)
    target_name = "MatMul_3_197_768_3072_1"
    assert target_name in names, f"Target layer {target_name} not in extraction"
    idx = names.index(target_name)
    ld = dims[idx]
    print(f"[Smoke] target layer: {target_name} -> {ld}")

    spec = transformer_spec()
    acc = CIM_Acc.from_spec(spec)
    wl = WorkLoad(ld)
    print(f"[Smoke] Num_MAC = {wl.Num_MAC:,}")
    print(f"[Smoke] HW: {spec.cores} cores x {spec.macro.dimX}x{spec.macro.dimY}, "
          f"peak = {spec.cores * spec.macro.dimX * spec.macro.dimY / 8:,.0f} MACs/cycle")

    # --- WS baseline ---
    t0 = time.perf_counter()
    ws = generate_weight_stationary_baseline(
        acc=copy.deepcopy(acc), ops=wl, quiet=True, enable_double_buffer=True
    )
    print(f"[WS] t={time.perf_counter()-t0:.1f}s  "
          f"latency={ws.latency:.3e}  energy={ws.energy:.3e}  policy={ws.policy}")

    # --- MIREDO MIP ---
    CONST.TIMELIMIT = 30
    CONST.MIPFOCUS = 1
    CONST.FLAG_OPT = "EDP"
    FLAG.GUROBI_OUTPUT = False
    FLAG.SIMU = False

    t0 = time.perf_counter()
    result = SolveMapping(
        acc=copy.deepcopy(acc),
        ops=wl,
        bestMetric=[float("inf"), float("inf"), float("inf")],
        outputdir=out_dir,
    )
    print(f"[MIREDO] t={time.perf_counter()-t0:.1f}s  "
          f"latency={result[0]:.3e}  energy={result[1]:.3e}  edp={result[2]:.3e}")

    if ws.latency > 0 and result[0] > 0:
        print(f"[speedup MIREDO/WS]  L={ws.latency/result[0]:.2f}x  "
              f"E={ws.energy/result[1]:.2f}x  "
              f"EDP={ws.latency*ws.energy/(result[0]*result[1]):.2f}x")


if __name__ == "__main__":
    main()

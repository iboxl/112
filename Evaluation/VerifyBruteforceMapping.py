"""gap_attribution: four-quadrant attribution of the 3x3 model gap.

For each case (A: 3x3_C16K16_P7, B: 3x3_C32K32_P7), we compute:

                          simu              MIP analytical
    M_bf (bruteforce)     known / verified  TO COMPUTE
    M_mip (MIP-selected)  known / verified  known (= MIP obj)

Interpretation of analytical(M_bf) vs analytical(M_mip):
  - analytical(M_bf) < analytical(M_mip):
      MIP's analytical objective prefers M_bf; MIP failed to find it within
      the 120s budget. Root cause = solver-time insufficient.
      Prediction: a longer MIP run should close the gap.
  - analytical(M_bf) > analytical(M_mip):
      MIP's analytical objective prefers M_mip. The model's ranking disagrees
      with the simulator's ranking. Root cause = analytical model approximation.
      Cannot be closed by more solver time.
  - analytical(M_bf) ~= analytical(M_mip):
      Model cannot distinguish them. Subtle tightness issue.

The bruteforce-optimal mappings are reconstructed from the printed structures in
the 2026-04-21 Case A / Case B log files (Evaluation/../output/
brute_force_result_case{A,B}_final.log). The double-buffer flag is not printed
in the log; we recover it by sweeping valid dbl configs for the
reconstructed (sm, tm) and keeping the one whose simulator latency equals the
known bruteforce optimum.

Runs one MIP-pin-and-solve per case. Writes a single JSON to
output/Eval_Result/gap_attribution_<timestamp>.json.
"""

import os
import sys
import time
import copy
import itertools
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from utils.Workload import WorkLoad, LoopNest, Mapping
from utils.SolverTSS import Solver
from utils.GlobalUT import *
from Simulator.Simulax import tranSimulator


# ────────────────────────────────────────────────────────────────────────────
# Hardcoded M_bf structures parsed from the 2026-04-21 bruteforce logs.
# Memory indices: 1=Dram, 2=Global_buffer, 3=Output_buffer, 4=Input_buffer,
#                 5=OReg, 6=IReg, 7=Macro
# Schemes match what VerifyBruteforce.py used for the completed runs.
# ────────────────────────────────────────────────────────────────────────────

CASES = {
    'A': {
        'tag': '3x3_C16K16_P7',
        'ops': {'R': 3, 'S': 3, 'C': 16, 'K': 16, 'P': 7, 'Q': 7,
                'G': 1, 'B': 1, 'H': 7, 'W': 7, 'Stride': 1, 'Padding': 1},
        'scheme': [[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 16, 1, 1],
                   [1, 1, 1, 1, 1, 1, 16, 1]],
        # M_bf temporal mapping from the log:
        #   0 for R in 3    [Dram, Dram, Dram]
        #   1 for P in 7    [Dram, Dram, Dram]
        #   2 for Q in 7    [Dram, Dram, Global_buffer]
        #   3 for S in 3    [Global_buffer, Macro, Global_buffer]
        # dim2Dict index: -=0, R=1, S=2, P=3, Q=4, C=5, K=6, G=7
        'mbf_tm': [
            {'dim': 1, 'size': 3, 'mem': [1, 1, 1]},  # R: Dram/Dram/Dram
            {'dim': 3, 'size': 7, 'mem': [1, 1, 1]},  # P: Dram/Dram/Dram
            {'dim': 4, 'size': 7, 'mem': [1, 1, 2]},  # Q: Dram/Dram/Global_buffer
            {'dim': 2, 'size': 3, 'mem': [2, 7, 2]},  # S: Global_buffer/Macro/Global_buffer
        ],
        'bruteforce_lat': 5299,
        'bruteforce_energy': 1758.9199694529966,
        'mip_obj_known': 5386,        # from 2026-04-21 log
        'mip_simu_known': 5374,       # from 2026-04-21 log
        'model_gap_known_pct': 1.42,
    },
    'B': {
        'tag': '3x3_C32K32_P7',
        'ops': {'R': 3, 'S': 3, 'C': 32, 'K': 32, 'P': 7, 'Q': 7,
                'G': 1, 'B': 1, 'H': 7, 'W': 7, 'Stride': 1, 'Padding': 1},
        'scheme': [[1, 1, 1, 1, 1, 1, 2, 1],
                   [1, 1, 1, 1, 1, 32, 1, 1],
                   [1, 1, 1, 1, 1, 1, 16, 1]],
        # M_bf temporal mapping from the log:
        #   0 for R in 3    [Dram, Dram, Dram]
        #   1 for P in 7    [Dram, Dram, Dram]
        #   2 for Q in 7    [Dram, Dram, Output_buffer]
        #   3 for S in 3    [Global_buffer, Macro, Output_buffer]
        'mbf_tm': [
            {'dim': 1, 'size': 3, 'mem': [1, 1, 1]},
            {'dim': 3, 'size': 7, 'mem': [1, 1, 1]},
            {'dim': 4, 'size': 7, 'mem': [1, 1, 3]},  # O at Output_buffer
            {'dim': 2, 'size': 3, 'mem': [2, 7, 3]},
        ],
        'bruteforce_lat': 6485,
        'bruteforce_energy': 3364.1483950280285,
        'mip_obj_known': 6763,
        'mip_simu_known': 6625,
        'model_gap_known_pct': 2.16,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# LoopNest reconstruction + dbl sweep
# ────────────────────────────────────────────────────────────────────────────

def build_spatial_mapping(scheme, acc, ops):
    """Replicate VerifyBruteforce's sm_list construction."""
    sm_list = []
    for u in range(acc.Num_SpUr):
        for d in range(1, ops.Num_dim):
            if scheme[u][d] > 1:
                sm_list.append(Mapping(dim=d, dimSize=scheme[u][d],
                                        mem=[acc.SpUr2Mem[u, op] for op in range(3)]))
    return sm_list


def build_loopnest_for_Mbf(case, acc, ops, dbl_cfg):
    """Construct a LoopNest matching M_bf (structural + provided dbl_cfg).
    Returns a LoopNest or raises on infeasible preprogress."""
    loops = LoopNest(acc=acc, ops=ops)
    loops.tm = [Mapping(dim=t['dim'], dimSize=t['size'], mem=list(t['mem']))
                for t in case['mbf_tm']]
    loops.sm = build_spatial_mapping(case['scheme'], acc, ops)
    loops.usr_defined_double_flag = [row[:] for row in dbl_cfg]
    loops.psum_flag = None
    return loops


def eligible_dbl_pairs(loops, acc):
    """Return sorted list of (m, op) pairs eligible for double-buffer toggling
    (matches VerifyBruteforce.get_double_configs)."""
    used = set()
    for tm in loops.tm:
        for op in range(3):
            m = tm.mem[op]
            if 1 <= m < acc.Num_mem and acc.double_config[m][op]:
                used.add((m, op))
    return sorted(used)


def sweep_dbl_for_target(case, acc, ops, target_lat):
    """Enumerate all dbl configs on top of M_bf's (sm, tm); return the first
    config whose simulator latency matches target_lat (cycle). Returns (dbl_cfg,
    simu_lat, simu_energy, num_tried, num_feasible, num_matching)."""
    eligible = None
    matched_cfg = None
    matched_energy = None
    num_tried = 0
    num_feasible = 0
    num_matching = 0

    no_dbl = [[0] * 3 for _ in range(acc.Num_mem + 1)]
    try:
        probe = build_loopnest_for_Mbf(case, acc, ops, no_dbl)
        eligible = eligible_dbl_pairs(probe, acc)
    except Exception as exc:
        raise RuntimeError(f"Failed to build M_bf probe: {exc}")

    for mask in range(1 << len(eligible)):
        cfg = [row[:] for row in no_dbl]
        for bit, (m, op) in enumerate(eligible):
            if mask & (1 << bit):
                cfg[m][op] = 1
        num_tried += 1

        try:
            loops = build_loopnest_for_Mbf(case, acc, ops, cfg)
            simu = tranSimulator(acc=acc, ops=ops, dataflow=loops)
            lat, energy = simu.run()
            num_feasible += 1
        except (ValueError, KeyError, IndexError, ZeroDivisionError,
                TypeError, AttributeError):
            continue

        if abs(lat - target_lat) < 0.5:
            num_matching += 1
            if matched_cfg is None:
                matched_cfg = cfg
                matched_energy = energy

    return matched_cfg, matched_energy, num_tried, num_feasible, num_matching


# ────────────────────────────────────────────────────────────────────────────
# MIP variable pinning — monkey-patch model.optimize on the first call
# ────────────────────────────────────────────────────────────────────────────

def build_var_pins_from_loopnest(loops, acc, ops, factors, tu):
    """Translate a LoopNest into {gurobi_var_name -> 0/1 value} pins.

    The MIP's decision binaries:
      Indic_factor2Loop_({dim_name},{f_idx},{loop_idx})
      indic_factor2Mem_({dim_name},{f_idx},{op_name},{mem_name})
      Indic_doubleMem_({mem_name},{op_name})

    For each case here, every nontrivial dim has exactly one factor (prime),
    so f_idx=0 for all pinned factors.
    """
    op_names = ['I', 'W', 'O']
    dim_names = ops.dim2Dict
    Num_Loops = sum(len(f) for f in factors[1:ops.Num_dim] if f != [1])

    pins = {}

    # Step 1: Determine, for each (d, f) with factors[d] != [1],
    # which loop level i it occupies in the provided LoopNest.
    # In M_bf each dim appears in exactly one tm entry with factor matching
    # factors[d][0]. Assert consistency.
    for d in range(1, ops.Num_dim):
        if factors[d] == [1]:
            continue
        for f in range(len(factors[d])):
            # Find the loop level i whose (dim, size) matches (d, factors[d][f]).
            # When a dim has multiple factors, the match must be by (dim, size)
            # taking each occurrence in tm in order. Here, all dims have a
            # single factor, so match is unique by dim.
            target_size = factors[d][f]
            matches = [i for i, m in enumerate(loops.tm)
                       if m.dim == d and m.dimSize == target_size]
            if len(matches) != 1:
                # For degenerate matching (multiple identical factors), fall
                # back to ordered assignment: the f-th occurrence of dim d in
                # tm order. But all current cases have 1 factor per dim.
                raise ValueError(
                    f"Ambiguous tm match for dim={dim_names[d]} "
                    f"size={target_size}: {matches} candidates")
            loop_idx = matches[0]

            # Pin indic_factor2Loop
            for i in range(Num_Loops):
                name = f"Indic_factor2Loop_({dim_names[d]},{f},{i})"
                pins[name] = 1 if i == loop_idx else 0

            # Pin indic_factor2Mem per operand
            for op in range(3):
                for m in range(1, acc.Num_mem):
                    if acc.mappingArray[op][m] != 1:
                        continue
                    # the factor (d,f) is "at memory m" for op iff tm[loop_idx].mem[op] >= m
                    # Actually the MIP semantics: indic_factor2Mem[d,f,op,m] = 1
                    # means this factor is assigned to memory m for op, i.e.,
                    # the loop's memory index for op is exactly m.
                    is_at_m = (loops.tm[loop_idx].mem[op] == m)
                    name = f"indic_factor2Mem_({dim_names[d]},{f},{op_names[op]},{acc.mem2dict(m)})"
                    pins[name] = 1 if is_at_m else 0

    # Step 2: Pin indic_doubleMem from usr_defined_double_flag.
    for m in range(1, acc.Num_mem):
        for op in range(3):
            if acc.mappingArray[op][m] != 1:
                continue
            if acc.double_config[m][op] == 0:
                continue  # var not created by MIP
            name = f"Indic_doubleMem_({acc.mem2dict(m)},{op_names[op]})"
            pins[name] = int(loops.usr_defined_double_flag[m][op])

    return pins


def apply_pins_and_resolve(solver, pins, timelimit=60):
    """After solver.run() has solved once, pin all decision vars via LB/UB
    and re-optimize. KEEP the existing multi-objective setup — resetting it
    caused the objective to go to 0 and the solver to park latency vars at
    their UB instead of minimizing.

    With all primary binaries pinned, the feasible region collapses to a
    single point (assuming pin consistency); both Latency and Energy
    objectives therefore reduce to computing the values at that point.

    Returns (analytical_latency, analytical_energy, gurobi_status, pin_stats).
    """
    model = solver.model
    missing = []
    applied_count = 0
    for vname, val in pins.items():
        v = model.getVarByName(vname)
        if v is None:
            missing.append(vname)
        else:
            v.LB = float(val)
            v.UB = float(val)
            applied_count += 1
    Logger.critical(
        f"[DiagnosePin] applied {applied_count} pins; {len(missing)} missing")
    if missing[:5]:
        Logger.info(f"[DiagnosePin] first missing: {missing[:5]}")

    res_latency = model.getVarByName("res_latency")
    res_energy = model.getVarByName("res_energy")
    if res_latency is None:
        raise RuntimeError("res_latency variable not found after solver.run()")

    model.setParam('TimeLimit', timelimit)
    model.update()
    model.optimize()

    status = int(model.Status)
    if model.SolCount == 0:
        return None, None, status, (applied_count, missing)

    latency_cycles = res_latency.X * CONST.SCALE_LATENCY
    energy_value = res_energy.X if res_energy is not None else None
    return latency_cycles, energy_value, status, (applied_count, missing)


# ────────────────────────────────────────────────────────────────────────────
# Per-case diagnostic
# ────────────────────────────────────────────────────────────────────────────

def diagnose_case(case_id, mip_timelimit_mbf=60, mip_timelimit_mmip=1800):
    case = CASES[case_id]
    Logger.critical("=" * 72)
    Logger.critical(f"DIAGNOSE {case_id}: {case['tag']}")
    Logger.critical("=" * 72)

    acc = CIM_Acc.from_spec(default_spec())
    ops = WorkLoad(loopDim=case['ops'])
    scheme = case['scheme']
    import math
    spatial = [math.prod(col) for col in zip(*scheme)]
    tu = [math.ceil(x / y) if y > 0 else x for x, y in zip(ops.dim2bound, spatial)]

    # Step 1: recover M_bf's full LoopNest (dbl_cfg via sweep)
    Logger.critical(f"[step 1] sweep dbl configs for M_bf; target latency={case['bruteforce_lat']}")
    t0 = time.time()
    dbl_cfg, matched_energy, n_tried, n_feasible, n_matching = sweep_dbl_for_target(
        case, acc, ops, case['bruteforce_lat'])
    sweep_elapsed = time.time() - t0
    Logger.critical(f"  swept {n_tried} configs / {n_feasible} feasible / {n_matching} matched")
    if dbl_cfg is None:
        Logger.critical("  FAILED: no dbl config reproduces bruteforce latency. "
                        "tm/sm parse likely wrong.")
        return {
            'case_id': case_id, 'status': 'mbf_unreconstructable',
            'sweep_tried': n_tried, 'sweep_feasible': n_feasible,
            'sweep_matching': n_matching, 'sweep_elapsed_s': sweep_elapsed,
        }

    mbf_loops = build_loopnest_for_Mbf(case, acc, ops, dbl_cfg)
    mbf_simu_lat, mbf_simu_energy = tranSimulator(acc=acc, ops=ops, dataflow=mbf_loops).run()
    Logger.critical(f"  M_bf simu = {mbf_simu_lat:.0f} cycles / {mbf_simu_energy:.2f} nJ "
                    f"(expected {case['bruteforce_lat']})")

    # Step 2: long MIP for M_mip — this also gives us the Gurobi model to reuse for M_bf pinning.
    from utils.UtilsFunction.ToolFunction import prepare_save_dir
    import logging
    logging.disable(logging.CRITICAL)

    outdir = os.path.join(os.path.dirname(__file__), '..', 'output',
                           f'#Diag_{case_id}')
    prepare_save_dir(outdir)

    CONST.FLAG_OPT = "Latency"
    CONST.MIPFOCUS = 1
    CONST.TIMELIMIT = mip_timelimit_mmip
    FLAG.GUROBI_OUTPUT = False
    FLAG.SIMU = False

    Logger.critical(f"[step 2] long MIP for M_mip (TL={CONST.TIMELIMIT}s)")
    solver = Solver(acc=CIM_Acc.from_spec(default_spec()), ops=ops, tu=tu, su=scheme,
                    metric_ub=CONST.MAX_POS, outputdir=outdir)
    t2 = time.time()
    solver.run()
    mip_solve_elapsed = time.time() - t2

    if solver.model.SolCount == 0:
        analytical_mmip = None
        simu_mmip_lat = None
        simu_mmip_energy = None
        mip_gap_mmip = None
        Logger.critical("  MIP returned no solution in long run — diagnostic cannot continue")
        return {
            'case_id': case_id, 'case_tag': case['tag'],
            'status': 'mmip_no_solution',
            'mip_solve_elapsed_s_mmip': round(mip_solve_elapsed, 3),
        }
    else:
        analytical_mmip = float(solver.result[0])
        try:
            mip_gap_mmip = float(solver.model.MIPGap)
        except Exception:
            mip_gap_mmip = None
        try:
            simu_mmip = tranSimulator(acc=CIM_Acc.from_spec(default_spec()),
                                       ops=ops, dataflow=solver.dataflow)
            simu_mmip_lat, simu_mmip_energy = simu_mmip.run()
        except Exception as exc:
            simu_mmip_lat, simu_mmip_energy = None, None
            Logger.error(f"  simu(M_mip) failed: {exc}")
        Logger.critical(f"  analytical(M_mip) = {analytical_mmip:.4f}  "
                        f"simu(M_mip) = {simu_mmip_lat}  "
                        f"MIPGap = {mip_gap_mmip}  "
                        f"elapsed = {mip_solve_elapsed:.1f}s")

    # Step 3: pin vars on the SAME solved model and re-optimize for analytical(M_bf)
    Logger.critical(f"[step 3] pin M_bf on solved model, re-optimize (TL={mip_timelimit_mbf}s)")
    pins = build_var_pins_from_loopnest(mbf_loops, acc, ops, solver.FACTORS, tu)
    Logger.critical(f"  prepared {len(pins)} var pins "
                    f"({sum(1 for v in pins.values() if v == 1)} set to 1)")

    t1 = time.time()
    analytical_mbf, energy_mbf, mip_status_bf, pin_stats_tuple = apply_pins_and_resolve(
        solver, pins, timelimit=mip_timelimit_mbf)
    bf_solve_elapsed = time.time() - t1

    if analytical_mbf is None:
        Logger.critical(f"  MIP infeasible/no-solution under M_bf pins (status={mip_status_bf})")
        Logger.critical("  → M_bf may NOT be in MIP feasible space (structural mismatch)")
    else:
        Logger.critical(f"  analytical(M_bf) = {analytical_mbf:.4f} cycles  "
                        f"(status={mip_status_bf}, elapsed={bf_solve_elapsed:.1f}s)")

    try:
        solver.close()
    except Exception:
        pass

    # Interpretation
    interpretation = None
    if analytical_mbf is not None and analytical_mmip is not None:
        diff = analytical_mbf - analytical_mmip
        rel = diff / analytical_mmip * 100 if analytical_mmip > 0 else 0.0
        if abs(rel) < 0.5:
            interpretation = (
                f"analytical(M_bf) ~= analytical(M_mip) (Δ={rel:+.2f}%). "
                "The MIP model cannot distinguish the two mappings; "
                "simu diverges because of tightness/precision, not ranking.")
        elif diff < 0:
            interpretation = (
                f"analytical(M_bf) < analytical(M_mip) (Δ={rel:+.2f}%). "
                "MIP model prefers M_bf. Root cause = SOLVER TIME: Gurobi did "
                "not find M_bf within the 120s budget. A longer MIP run should "
                "close the gap. Not a model-approximation defect.")
        else:
            interpretation = (
                f"analytical(M_bf) > analytical(M_mip) (Δ={rel:+.2f}%). "
                "MIP model prefers M_mip. Root cause = MODEL APPROXIMATION: "
                "the analytical latency ranks M_mip above M_bf, so no amount "
                "of additional solver time would reach M_bf. This quantifies "
                "the 3x3 model-approximation gap.")

    return {
        'case_id': case_id,
        'case_tag': case['tag'],
        'status': 'ok' if analytical_mbf is not None else 'mip_infeasible_under_Mbf_pin',
        'mbf': {
            'structure_source': 'log parse, 2026-04-21 bruteforce run',
            'dbl_cfg_matching_target': dbl_cfg,
            'sweep_tried': n_tried,
            'sweep_feasible': n_feasible,
            'sweep_matching': n_matching,
            'sweep_elapsed_s': round(sweep_elapsed, 3),
            'simu_lat_verified': float(mbf_simu_lat),
            'simu_energy_verified': float(mbf_simu_energy),
            'simu_lat_known': case['bruteforce_lat'],
        },
        'analytical_mbf': analytical_mbf,
        'analytical_mbf_energy': energy_mbf,
        'mip_status_under_pin': mip_status_bf,
        'mip_solve_elapsed_s_bf': round(bf_solve_elapsed, 3),
        'pin_applied_count': pin_stats_tuple[0],
        'pin_missing_count': len(pin_stats_tuple[1]),
        'pin_missing_sample': pin_stats_tuple[1][:8],
        'analytical_mmip_long': analytical_mmip,
        'simu_mmip_long': simu_mmip_lat,
        'simu_mmip_long_energy': simu_mmip_energy,
        'mipgap_mmip_long': mip_gap_mmip,
        'mip_solve_elapsed_s_mmip': round(mip_solve_elapsed, 3),
        'mip_timelimit_mmip': mip_timelimit_mmip,
        'known_reference': {
            'mip_obj_120s': case['mip_obj_known'],
            'mip_simu_120s': case['mip_simu_known'],
            'model_gap_120s_pct': case['model_gap_known_pct'],
        },
        'interpretation': interpretation,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="gap_attribution four-quadrant diagnostic")
    parser.add_argument('--cases', default='A,B',
                        help="comma-separated case IDs (subset of A,B)")
    parser.add_argument('--tl-mbf', type=int, default=60,
                        help="MIP time limit for M_bf pinned solve (should converge "
                             "instantly since everything is pinned)")
    parser.add_argument('--tl-mmip', type=int, default=1800,
                        help="MIP time limit for long M_mip solve to separate "
                             "solver-time from model-gap")
    parser.add_argument('--output', default=None,
                        help="JSON output path; default = "
                             "output/Eval_Result/gap_attribution_<ts>.json")
    args = parser.parse_args()

    Logger.setcfg(setcritical=False, setDebug=False, STD=True, file="", nofile=True)

    results = []
    for case_id in args.cases.split(','):
        case_id = case_id.strip()
        if case_id not in CASES:
            Logger.error(f"Unknown case: {case_id}")
            continue
        results.append(diagnose_case(case_id,
                                     mip_timelimit_mbf=args.tl_mbf,
                                     mip_timelimit_mmip=args.tl_mmip))

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'Eval_Result')
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = args.output or os.path.join(out_dir, f'gap_attribution_{ts}.json')

    payload = {
        'experiment_id': 'gap_attribution',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'script': 'Evaluation/VerifyBruteforceMapping.py',
        'mip_timelimit_mbf_s': args.tl_mbf,
        'mip_timelimit_mmip_s': args.tl_mmip,
        'cases': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    Logger.critical("=" * 72)
    Logger.critical(f"Diagnostic JSON: {out_path}")
    Logger.critical("=" * 72)
    for r in results:
        Logger.critical(f"  [{r['case_id']}] {r.get('interpretation', '(no interp)')}")

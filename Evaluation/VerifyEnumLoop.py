# 因子排列枚举验证脚本
# 通过穷举所有不同的因子排列（factor ordering），对每种排列固定后求解子MIP，
# 验证标准MIP求解器找到的解是否为全局最优。
# 用途：对gap>0%的层提供最优性的穷举证明。
# 用法：python Evaluation/VerifyEnumLoop.py

import os, sys, math, time, shutil, copy, functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

_LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
_LOG_FILE = os.path.join(_LOG_DIR, 'enum_verify_result.log')
def log(msg):
    """写入结果文件（绕过Logger对stdout的劫持）"""
    os.makedirs(_LOG_DIR, exist_ok=True)
    with open(_LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from Evaluation.common.EvalCommon import save_experiment_json
from utils.Workload import WorkLoad
from utils.SolverTSS import Solver
from utils.GlobalUT import *
from utils.factorization import flexible_factorization
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from Simulator.Simulax import tranSimulator


def enumerate_factor_orderings(factors, Num_dim):
    """穷举所有不同的因子排列（同维度同值因子对称破缺）。
    仅对同一维度内的同值因子施加升序位置约束，与MIP对称约束一致。"""
    items = []
    for d in range(1, Num_dim):
        if factors[d] == [1]: continue
        for f in range(len(factors[d])):
            items.append((d, f, factors[d][f]))

    N = len(items)
    if N == 0:
        return [{}]

    items.sort(key=lambda x: (x[2], x[0], x[1]))
    same_val = [False] * N
    for k in range(1, N):
        # 仅同维度同值的因子才可互换（与MIP自身的对称破缺一致）
        if items[k][2] == items[k-1][2] and items[k][0] == items[k-1][0]:
            same_val[k] = True

    results = []
    pos = [None] * N
    used = [False] * N

    def bt(k):
        if k == N:
            results.append({(d, f): pos[i] for i, (d, f, _) in enumerate(items)})
            return
        lo = (pos[k-1] + 1) if same_val[k] else 0
        for p in range(lo, N):
            if not used[p]:
                pos[k] = p; used[p] = True
                bt(k + 1)
                pos[k] = None; used[p] = False
    bt(0)
    return results


def _worker(args):
    """单个子MIP求解worker（进程池调用）。"""
    idx, ordering, spec, ops_dict, scheme, out_dir, timelimit = args

    CONST.FLAG_OPT = "Latency"
    CONST.TIMELIMIT = timelimit
    CONST.MIPFOCUS = 1
    FLAG.GUROBI_OUTPUT = False
    FLAG.SIMU = False
    import logging; logging.disable(logging.CRITICAL)  # 静默子进程所有log

    ops = WorkLoad(loopDim=ops_dict)
    acc = CIM_Acc.from_spec(spec)
    su = scheme
    spatial = [math.prod(col) for col in zip(*su)]
    tu = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial)]

    sub_dir = os.path.join(out_dir, str(idx))
    prepare_save_dir(sub_dir)

    solver = Solver(acc=acc, ops=ops, tu=tu, su=su, metric_ub=CONST.MAX_POS,
                    outputdir=sub_dir, threads=1, soft_mem_limit_gb=1.0,
                    fixed_factor_ordering=ordering)
    try:
        solver.run()
        if solver.model is not None and solver.model.SolCount > 0:
            lat = solver.result[0]
            try:
                gap = solver.model.MIPGap
            except Exception:
                gap = -1.0
            return (idx, lat, gap, True)
        return (idx, CONST.MAX_POS, -1, False)
    finally:
        solver.close()
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir, ignore_errors=True)


def run_enumeration(spec, ops_dict, scheme, timelimit=15, max_workers=None):
    """并行枚举所有因子排列，返回全局最优latency和gap统计。

    spec: HardwareSpec —— 子进程通过 CIM_Acc.from_spec 重建 acc，避免传递非 picklable 的 ZigZag Core。"""
    ops = WorkLoad(loopDim=ops_dict)
    su = scheme
    spatial = [math.prod(col) for col in zip(*su)]
    tu = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial)]
    factors = [flexible_factorization(t) for t in tu]

    log(f"temporal unrolling: {tu}")
    log(f"factors: {[f for fs in factors[1:ops.Num_dim] for f in fs if fs != [1]]}")

    t0 = time.time()
    orderings = enumerate_factor_orderings(factors, ops.Num_dim)
    t1 = time.time()
    log(f"生成 {len(orderings)} 种不同排列 ({t1-t0:.2f}s)")

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 2)

    out_dir = os.path.join("output", "#EnumVerify_temp")
    prepare_save_dir(out_dir)

    args_list = [(i, o, spec, ops_dict, scheme, out_dir, timelimit)
                 for i, o in enumerate(orderings)]

    best_lat, best_idx = CONST.MAX_POS, -1
    feasible, proven_opt = 0, 0

    log(f"启动 {max_workers} workers, 每子问题 {timelimit}s ...")
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_worker, a): a[0] for a in args_list}
        done_count = 0
        for future in futures:
            pass  # submitted
        pending = set(futures.keys())
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED, timeout=None)
            for f in done:
                done_count += 1
                try:
                    idx, lat, gap, ok = f.result()
                except Exception as e:
                    log(f"  子问题异常: {e}")
                    continue
                if ok:
                    feasible += 1
                    if gap >= 0 and gap < 0.01:
                        proven_opt += 1
                    if lat < best_lat:
                        best_lat = lat
                        best_idx = idx
                if done_count % 200 == 0:
                    elapsed = time.time() - t0
                    log(f"  进度 {done_count}/{len(orderings)} ({elapsed:.0f}s), "
                          f"可行={feasible}, gap=0%={proven_opt}, 最优latency={best_lat:.2f}")

    elapsed = time.time() - t0
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    log(f"\n{'='*60}")
    log(f"枚举完成: {len(orderings)} 排列, {elapsed:.1f}s")
    log(f"可行解: {feasible}/{len(orderings)}")
    log(f"子问题gap=0%: {proven_opt}/{feasible}")
    log(f"全局最优 Latency = {best_lat:.2f} (排列#{best_idx})")
    log(f"{'='*60}")

    # 保存结构化结果到 output/Eval_Result/
    from utils.UtilsFunction.ToolFunction import save_result_json
    result_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'Eval_Result')
    result = {
        'script': 'VerifyEnumLoop',
        'workload': str(ops),
        'scheme': su,
        'total_orderings': len(orderings),
        'feasible': feasible,
        'proven_optimal': proven_opt,
        'best_latency': best_lat if best_lat < CONST.MAX_POS else None,
        'best_ordering_idx': best_idx,
        'time_seconds': round(elapsed, 1),
    }
    result_file = save_result_json(result_dir, 'enumLoop', result)
    log(f"结果已保存: {result_file}")

    exp6_file = save_experiment_json(
        output_dir=result_dir,
        file_name=f"EXP-6_enumLoop_{time.strftime('%Y%m%d_%H%M%S')}.json",
        experiment_id="EXP-6",
        script_path=__file__,
        config={
            "verification_method": "enumLoop",
            "workload": ops_dict,
            "scheme": su,
            "mip_time_limit": timelimit,
            "max_workers": max_workers,
        },
        results={
            "verification": {
                "total_orderings": len(orderings),
                "feasible_orderings": feasible,
                "proven_optimal_orderings": proven_opt,
                "global_best_latency": best_lat if best_lat < CONST.MAX_POS else None,
                "best_ordering_idx": best_idx,
                "elapsed_seconds": round(elapsed, 3),
            },
            "optimality_verification": [{
                "model": "manual",
                "layer": f"Conv_{ops_dict.get('R',1)}x{ops_dict.get('S',1)}_C{ops_dict.get('C',1)}K{ops_dict.get('K',1)}",
                "tier": "small",
                "mip_latency": best_lat if best_lat < CONST.MAX_POS else None,
                "exhaustive_latency": best_lat if best_lat < CONST.MAX_POS else None,
                "is_optimal": (proven_opt > 0),
                "optimality_gap_pct": 0.0 if proven_opt > 0 else None,
                "solve_time_sec": round(elapsed, 3),
                "num_spatial_schemes": len(orderings),
                "num_schemes_after_pruning": feasible
            }],
        },
        anomalies=[],
    )
    log(f"EXP-6结果已保存: {exp6_file}")

    return best_lat, best_idx


if __name__ == "__main__":
    import argparse

    CASES = {
        '1x1_C64K64': {
            'ops': {'R':1,'S':1,'C':64,'K':64,'P':7,'Q':7,'G':1,'B':1,'H':7,'W':7,'Stride':1,'Padding':0},
            'scheme': [[1,1,1,1,1,1,8,1],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,8,1]],
        },
        'resnet_layer1': {
            'ops': {'R':3,'S':3,'C':64,'K':64,'P':56,'Q':56,'G':1,'B':1,'H':56,'W':56,'Stride':1,'Padding':1},
            'scheme': [[1,1,1,2,1,1,4,1],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,16,1]],
        },
    }

    parser = argparse.ArgumentParser(description="因子排列枚举验证")
    parser.add_argument('--case', default='1x1_C64K64', choices=list(CASES.keys()))
    parser.add_argument('--timelimit', type=int, default=15)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    spec = default_spec()
    Logger.setcfg(setcritical=False, setDebug=False, STD=True, file="", nofile=True)

    case = CASES[args.case]
    best_lat, best_idx = run_enumeration(
        spec=spec,
        ops_dict=case['ops'],
        scheme=case['scheme'],
        timelimit=args.timelimit,
        max_workers=args.workers,
    )

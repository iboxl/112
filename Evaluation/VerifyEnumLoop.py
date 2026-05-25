# 因子排列枚举验证脚本
# 通过穷举所有不同的因子排列（factor ordering），对每种排列固定后求解子MIP，
# 验证标准MIP求解器找到的解是否为全局最优。
# 用途：对gap>0%的层提供最优性的穷举证明。
# 用法：python Evaluation/VerifyEnumLoop.py

import atexit
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


_WORKER_CACHE = {}


def _worker_init(spec, ops_dict, scheme, timelimit, metric, shared_ub_name=None):
    """ProcessPoolExecutor initializer — build immutable per-worker state ONCE.

    Caches acc / ops / tu / su / scheme settings so each task reuses them
    instead of rebuilding (CIM_Acc.from_spec is ~0.5-1s; over 36k tasks this
    is the dominant overhead). _worker reads from _WORKER_CACHE.

    Also opens cross-worker shared upper bound (SharedUB) so each ordering's
    sub-MIP can prune subtrees with LP bound > current global best objective.
    """
    CONST.FLAG_OPT = metric
    CONST.TIMELIMIT = timelimit
    CONST.MIPFOCUS = 1
    FLAG.GUROBI_OUTPUT = False
    FLAG.SIMU = False
    import logging; logging.disable(logging.CRITICAL)
    ops = WorkLoad(loopDim=ops_dict)
    acc = CIM_Acc.from_spec(spec)
    spatial = [math.prod(col) for col in zip(*scheme)]
    tu = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial)]

    shared_ub = None
    shm_handle = None
    if shared_ub_name is not None:
        from utils.Tools import SharedUB
        from multiprocessing.shared_memory import SharedMemory
        shm_handle = SharedMemory(name=shared_ub_name)
        shared_ub = SharedUB(shm_handle)
        import atexit
        atexit.register(lambda h=shm_handle: (h.close() if h is not None else None))

    _WORKER_CACHE.update(dict(ops=ops, acc=acc, tu=tu, su=scheme,
                              metric=metric, timelimit=timelimit,
                              shared_ub=shared_ub, shm_handle=shm_handle))


def _worker(args):
    """单个子MIP求解worker（进程池调用）。"""
    if len(args) == 8:
        idx, ordering, spec, ops_dict, scheme, out_dir, timelimit, metric = args
    else:
        idx, ordering, spec, ops_dict, scheme, out_dir, timelimit = args
        metric = "Latency"

    shared_ub = None
    # Fast path: reuse cached acc/ops/tu/su built once by initializer
    if _WORKER_CACHE:
        ops = _WORKER_CACHE["ops"]
        acc = _WORKER_CACHE["acc"]
        tu = _WORKER_CACHE["tu"]
        su = _WORKER_CACHE["su"]
        shared_ub = _WORKER_CACHE.get("shared_ub")
        CONST.FLAG_OPT = _WORKER_CACHE["metric"]
        CONST.TIMELIMIT = _WORKER_CACHE["timelimit"]
        CONST.MIPFOCUS = 1
        FLAG.GUROBI_OUTPUT = False
        FLAG.SIMU = False
    else:
        # Slow path (legacy / no-initializer call)
        CONST.FLAG_OPT = metric
        CONST.TIMELIMIT = timelimit
        CONST.MIPFOCUS = 1
        FLAG.GUROBI_OUTPUT = False
        FLAG.SIMU = False
        import logging; logging.disable(logging.CRITICAL)
        ops = WorkLoad(loopDim=ops_dict)
        acc = CIM_Acc.from_spec(spec)
        su = scheme
        spatial = [math.prod(col) for col in zip(*su)]
        tu = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial)]

    sub_dir = os.path.join(out_dir, str(idx))
    prepare_save_dir(sub_dir)

    # Get current global upper bound from shared memory (pruning cutoff)
    metric_ub = shared_ub.value if shared_ub is not None else CONST.MAX_POS

    solver = Solver(acc=acc, ops=ops, tu=tu, su=su, metric_ub=metric_ub,
                    outputdir=sub_dir, threads=1, soft_mem_limit_gb=1.0,
                    fixed_factor_ordering=ordering, shared_ub=shared_ub)
    try:
        solver.run()
        if solver.model is not None and solver.model.SolCount > 0:
            lat = solver.result[0]
            try:
                gap = solver.model.MIPGap
            except Exception:
                gap = -1.0
            try:
                status = int(solver.model.Status)
            except Exception:
                status = -1
            try:
                # Raw physical metric (model.ObjVal is in scaled model units for
                # Latency/EDP). Keeps selection/report consistent with the raw
                # value propagated to SharedUB below.
                _mi = {"Latency": 0, "Energy": 1, "EDP": 2}.get(CONST.FLAG_OPT, 0)
                obj_val = float(solver.result[_mi])
            except Exception:
                obj_val = float('nan')
            try:
                # Convert the dual bound from scaled model units to the same raw
                # physical metric so the per-ordering gap log stays coherent.
                if CONST.FLAG_OPT == "EDP":
                    _scaled_to_raw = CONST.SCALE_LATENCY / solver.edp_scaling_factor
                elif CONST.FLAG_OPT == "Latency":
                    _scaled_to_raw = CONST.SCALE_LATENCY
                else:
                    _scaled_to_raw = 1.0
                obj_bound = float(solver.model.ObjBound) * _scaled_to_raw
            except Exception:
                obj_bound = float('nan')
            # Propagate this ordering's metric to global SharedUB so subsequent
            # orderings can prune subtrees with LP bound > current best
            if shared_ub is not None:
                metric_index = {"Latency": 0, "Energy": 1, "EDP": 2}.get(CONST.FLAG_OPT)
                if metric_index is not None:
                    shared_ub.update_min(solver.result[metric_index])
            return (idx, lat, gap, True, status, obj_val, obj_bound)
        try:
            status = int(solver.model.Status) if solver.model is not None else -1
        except Exception:
            status = -1
        return (idx, CONST.MAX_POS, -1, False, status, float('nan'), float('nan'))
    finally:
        solver.close()
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir, ignore_errors=True)


def _safe_unlink(shm_name):
    """Defensively unlink a SharedMemory segment; swallows FileNotFoundError."""
    try:
        from multiprocessing.shared_memory import SharedMemory as _SHM
        _h = _SHM(name=shm_name, create=False, size=8)
        _h.close()
        _h.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def run_enumeration(spec, ops_dict, scheme, timelimit=15, max_workers=None, metric="Latency"):
    """并行枚举所有因子排列，返回全局最优latency和gap统计。

    spec: HardwareSpec —— 子进程通过 CIM_Acc.from_spec 重建 acc，避免传递非 picklable 的 ZigZag Core。"""
    ops = WorkLoad(loopDim=ops_dict)
    su = scheme
    spatial = [math.prod(col) for col in zip(*su)]
    tu = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial)]
    acc = CIM_Acc.from_spec(spec)
    factors = [flexible_factorization(t, acc.placement_depth) for t in tu]

    log(f"temporal unrolling: {tu}")
    log(f"factors: {[f for fs in factors[1:ops.Num_dim] for f in fs if fs != [1]]}")

    t0 = time.time()
    orderings = enumerate_factor_orderings(factors, ops.Num_dim)
    t1 = time.time()
    log(f"生成 {len(orderings)} 种不同排列 ({t1-t0:.2f}s)")

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 2)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output', "#EnumVerify_temp")
    prepare_save_dir(out_dir)

    args_list = [(i, o, spec, ops_dict, scheme, out_dir, timelimit, metric)
                 for i, o in enumerate(orderings)]

    best_lat, best_idx = CONST.MAX_POS, -1
    feasible, proven_opt = 0, 0
    per_ordering = []  # list of dicts: {idx, status, gap, obj_val, obj_bound, lat, ok}

    log(f"启动 {max_workers} workers, 每子问题 {timelimit}s ...")
    t0 = time.time()

    # Cross-worker shared upper bound for cutoff pruning (mirrors production
    # multi-scheme search). Initialize to MAX_POS so first ordering imposes no
    # cutoff; once any ordering finds an objective, subsequent orderings prune
    # subtrees with LP bound exceeding the running global best.
    from multiprocessing.shared_memory import SharedMemory
    import struct as _struct
    _shm = SharedMemory(create=True, size=8)
    _struct.pack_into('d', _shm.buf, 0, CONST.MAX_POS)
    shm_name = _shm.name
    log(f"SharedUB created: {shm_name}, init={CONST.MAX_POS}")
    atexit.register(_safe_unlink, shm_name)

    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx,
            initializer=_worker_init,
            initargs=(spec, ops_dict, scheme, timelimit, metric, shm_name),
        ) as executor:
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
                        res = f.result()
                    except Exception as e:
                        log(f"  子问题异常: {e}")
                        continue
                    # backward-compat: old 4-tuple, new 7-tuple
                    if len(res) == 7:
                        idx, lat, gap, ok, status, obj_val, obj_bound = res
                    else:
                        idx, lat, gap, ok = res
                        status, obj_val, obj_bound = -1, float('nan'), float('nan')
                    per_ordering.append({
                        'idx': idx, 'ok': bool(ok), 'status': status,
                        'gap': float(gap) if gap is not None else None,
                        'obj_val': obj_val, 'obj_bound': obj_bound,
                        'mip_analytical_obj': float(obj_val) if (obj_val is not None and obj_val == obj_val) else None,
                    })
                    if ok:
                        feasible += 1
                        # OPTIMAL = 2 in Gurobi; treat that as proven optimal regardless of MIPGap floor
                        is_opt = (status == 2) or (gap is not None and 0 <= gap < 1e-4)
                        if is_opt:
                            proven_opt += 1
                        # Use solver objective value (not solver.result[0] which is always latency)
                        cmp_val = obj_val if metric != "Latency" and obj_val == obj_val else lat
                        if cmp_val < best_lat:
                            best_lat = cmp_val
                            best_idx = idx
                    if done_count % 200 == 0:
                        elapsed = time.time() - t0
                        log(f"  进度 {done_count}/{len(orderings)} ({elapsed:.0f}s), "
                              f"可行={feasible}, gap=0%={proven_opt}, 最优latency={best_lat:.2f}")
        per_ordering.sort(key=lambda r: r['idx'])
        for r in per_ordering:
            log(f"  ord#{r['idx']:3d} ok={r['ok']} status={r['status']} gap={r['gap']!r} obj_val={r['obj_val']!r} obj_bound={r['obj_bound']!r} analytical_obj={r['mip_analytical_obj']!r}")
    finally:
        elapsed = time.time() - t0
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

        # Clean up SharedMemory — runs even on KeyboardInterrupt or pool exception
        try:
            final_ub = _struct.unpack_from('d', _shm.buf, 0)[0]
            log(f"Final SharedUB value: {final_ub}")
            _shm.close()
            _shm.unlink()
        except Exception as e:
            log(f"  SharedUB cleanup warning: {e}")

    log(f"\n{'='*60}")
    log(f"枚举完成: {len(orderings)} 排列, {elapsed:.1f}s")
    log(f"可行解: {feasible}/{len(orderings)}")
    log(f"子问题gap=0%: {proven_opt}/{feasible}")
    log(f"全局最优 MIP analytical objective = {best_lat:.2f} (排列#{best_idx})  [FLAG.SIMU=False, so no simulator run]")
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
        'per_ordering': per_ordering,
    }
    result_file = save_result_json(result_dir, 'enumLoop', result)
    log(f"结果已保存: {result_file}")

    exp6_file = save_experiment_json(
        output_dir=result_dir,
        file_name=f"enumloop_verify_{time.strftime('%Y%m%d_%H%M%S')}.json",
        experiment_id="enumloop_verify",
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
                "global_best_objective": best_lat if best_lat < CONST.MAX_POS else None,
                "global_best_latency_note": "FLAG.SIMU=False; reported value is the MIP analytical objective at the best ordering, not a simulator-evaluated latency",
                "best_ordering_idx": best_idx,
                "elapsed_seconds": round(elapsed, 3),
                "per_ordering": per_ordering,
            },
            "optimality_verification": [{
                "model": "manual",
                "layer": f"Conv_{ops_dict.get('R',1)}x{ops_dict.get('S',1)}_C{ops_dict.get('C',1)}K{ops_dict.get('K',1)}",
                "tier": "small",
                "mip_analytical_objective_global_best": best_lat if best_lat < CONST.MAX_POS else None,
                "is_search_complete": (proven_opt == feasible == len(orderings)),
                "proven_optimal_subMIPs": proven_opt,
                "total_subMIPs": len(orderings),
                "solve_time_sec": round(elapsed, 3),
                "num_factor_orderings": len(orderings),
                "num_feasible_orderings": feasible
            }],
        },
        anomalies=[],
    )
    log(f"enumloop_verify结果已保存: {exp6_file}")

    return best_lat, best_idx, per_ordering


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
        '3x3_C16K16': {
            'ops': {'R':3,'S':3,'C':16,'K':16,'P':7,'Q':7,'G':1,'B':1,'H':7,'W':7,'Stride':1,'Padding':1},
            'scheme': [[1,1,1,1,1,1,1,1],[1,1,1,1,1,16,1,1],[1,1,1,1,1,1,16,1]],
        },
        '3x3_C32K32': {
            'ops': {'R':3,'S':3,'C':32,'K':32,'P':7,'Q':7,'G':1,'B':1,'H':7,'W':7,'Stride':1,'Padding':1},
            'scheme': [[1,1,1,1,1,1,2,1],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,16,1]],
        },
    }

    parser = argparse.ArgumentParser(description="因子排列枚举验证")
    parser.add_argument('--case', default='1x1_C64K64', choices=list(CASES.keys()))
    parser.add_argument('--timelimit', type=int, default=15)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--objective', default='latency', choices=('latency', 'edp', 'both'),
                        help="MIP objective: latency (default), edp, or both. "
                             "'both' runs the enumeration twice (once per objective) "
                             "with independent SharedUB cutoffs and reports both certs.")
    args = parser.parse_args()
    objectives = {'latency': ('Latency',), 'edp': ('EDP',),
                  'both': ('Latency', 'EDP')}[args.objective]

    spec = default_spec()
    Logger.setcfg(setcritical=False, setDebug=False, STD=True, file="", nofile=True)

    case = CASES[args.case]
    results_per_obj = {}
    for obj in objectives:
        log("\n" + "#" * 60)
        log(f"# 枚举 objective = {obj}")
        log("#" * 60)
        best, best_idx, _ = run_enumeration(
            spec=spec,
            ops_dict=case['ops'],
            scheme=case['scheme'],
            timelimit=args.timelimit,
            max_workers=args.workers,
            metric=obj,
        )
        results_per_obj[obj] = {'best_objective_value': best, 'best_ordering_idx': best_idx}
    log("\n" + "=" * 60)
    log("Per-objective enumeration summary:")
    for obj, r in results_per_obj.items():
        log(f"  {obj:<8}: global-best obj_value={r['best_objective_value']:.3e} (ordering #{r['best_ordering_idx']})")
    log("=" * 60)

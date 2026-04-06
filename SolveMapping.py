# this file is prepared for project 511
# Created by iboxl

from utils.Workload import WorkLoad
from utils.Tools import get_ConfigFile, append_scheme_summary, detect_parallel_config, auto_parallel_config, SharedUB
from utils.SolverTSS import Solver
import gurobipy as gp
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import *
from Architecture.ArchSpec import CIM_Acc
import pickle
from utils.UtilsFunction.ToolFunction import prepare_save_dir, get_Spatial_Unrolling
from utils.UtilsFunction.SchemeFunction import score_scheme, scheme_objective_lb, dominance_filter
import shutil
import time, math, os, struct
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

_worker_env = None  # cached Gurobi Env, set by _init_worker

def _init_worker(threads_per_worker, runtime_config):
    """ProcessPoolExecutor initializer: create one Gurobi Env per worker process."""
    global _worker_env
    if runtime_config is not None:
        for key, value in runtime_config["CONST"].items():
            setattr(CONST, key, value)
        for key, value in runtime_config["FLAG"].items():
            setattr(FLAG, key, value)
    _worker_env = gp.Env(empty=True)
    _worker_env.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
    _worker_env.setParam('ThreadLimit', max(1, int(threads_per_worker)))
    _worker_env.start()


def solve_scheme_worker(count:int, origin_index:int, scheme, acc:CIM_Acc, ops:WorkLoad, metric_ub:float, outputdir_root:str, outputdir_scheme:str, threads_per_worker:int, soft_mem_limit_gb:float, runtime_config=None, shared_ub_name=None):
    if runtime_config is None:
        # Sequential mode: apply config directly (no initializer)
        pass
    # In parallel mode, config was already applied by _init_worker

    # Open cross-worker shared memory for metric upper bound
    shared_ub = None
    _shm_handle = None
    if shared_ub_name is not None:
        _shm_handle = SharedMemory(name=shared_ub_name)
        shared_ub = SharedUB(_shm_handle)

    prepare_save_dir(outputdir_scheme)

    spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
    temporal_unrolling = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]

    # Provable lower-bound pruning: skip model building entirely if the
    # analytical bound already exceeds the current best objective.
    effective_ub = metric_ub
    if shared_ub is not None:
        effective_ub = min(effective_ub, shared_ub.value)
    lat_lb, eng_lb, edp_lb = scheme_objective_lb(acc, ops, scheme, temporal_unrolling)
    obj_lb = {"Latency": lat_lb, "Energy": eng_lb, "EDP": edp_lb}.get(CONST.FLAG_OPT)
    if obj_lb is not None and obj_lb > effective_ub:
        if _shm_handle is not None:
            _shm_handle.close()
        return {
            "count": count, "origin_index": origin_index,
            "solver_result": [CONST.MAX_POS]*3, "sim_l": CONST.MAX_POS,
            "sim_e": CONST.MAX_POS, "profile": None, "dataflow": None,
            "has_solution": False, "outputdir_root": outputdir_root,
        }

    solver = Solver(
        acc=acc,
        ops=ops,
        tu=temporal_unrolling,
        su=scheme,
        metric_ub=metric_ub,
        outputdir=outputdir_scheme,
        threads=threads_per_worker,
        soft_mem_limit_gb=soft_mem_limit_gb,
        shared_ub=shared_ub,
        env=_worker_env,
    )

    sim_l, sim_e, profile = CONST.MAX_POS, CONST.MAX_POS, None
    try:
        solver.run()

        has_solution = solver.model is not None and solver.model.SolCount > 0
        if has_solution:
            # Immediately propagate solution to shared state for cross-worker pruning
            if shared_ub is not None:
                metric_index = {"Latency": 0, "Energy": 1, "EDP": 2}.get(CONST.FLAG_OPT)
                if metric_index is not None:
                    shared_ub.update_min(solver.result[metric_index])

            if FLAG.SIMU:
                simu = tranSimulator(acc=acc, ops=ops, dataflow=solver.dataflow, DEBUG_SIMU=FLAG.DEBUG_SIMU)
                sim_l, sim_e = simu.run()
                profile = simu.PD
            else:
                sim_l, sim_e = solver.result[0], solver.result[1]
        elif os.path.exists(outputdir_scheme):
            shutil.rmtree(outputdir_scheme)

        return {
            "count": count,
            "origin_index": origin_index,
            "solver_result": solver.result,
            "sim_l": sim_l,
            "sim_e": sim_e,
            "profile": profile,
            "dataflow": solver.dataflow if has_solution else None,
            "has_solution": has_solution,
            "outputdir_root": outputdir_root,
        }
    finally:
        solver.close()
        if _shm_handle is not None:
            _shm_handle.close()


def update_best(result_pack:dict, best_metric:float, result, best_count:int, best_dataflow, solCount:int):
    count = result_pack["count"]
    solver_result = result_pack["solver_result"]
    sim_l = result_pack["sim_l"]
    sim_e = result_pack["sim_e"]
    profile = result_pack["profile"]
    has_solution = result_pack["has_solution"]

    if has_solution:
        solCount += 1
        result_msg = (
            f"Scheme {count:<3} End: Latency-{round(sim_l,3):<15}, Energy-{round(sim_e,3):<15}, "
            f"EDP-{round(sim_l * sim_e,3):<15}"
        )
        lat_msg = (
            f"      |--- Latency Accuracy: {round(100-abs(solver_result[0]-sim_l)/sim_l*100,2)}%, "
            f"Solver-{round(solver_result[0]):<10} and Simu-{round(sim_l,3):<10}"
        )
        eng_msg = (
            f"      |--- Energy  Accuracy: {round(100-abs(solver_result[1]-sim_e)/sim_e*100,2)}%, "
            f"Solver-{round(solver_result[1]):<10} and Simu-{round(sim_e,3):<10}"
        )

        append_scheme_summary(result_pack["outputdir_root"], result_msg)
        append_scheme_summary(result_pack["outputdir_root"], lat_msg)
        append_scheme_summary(result_pack["outputdir_root"], eng_msg)

        Logger.info(result_msg)
        Logger.info(lat_msg)
        Logger.info(eng_msg)
    else:
        result_msg = f"Scheme {count:<3} End: NO BETTER SOLUTION"
        append_scheme_summary(result_pack["outputdir_root"], result_msg)
        Logger.info(result_msg)

    assert CONST.FLAG_OPT in ["Latency", "Energy", "EDP"], "No Such Metric for Optimization, Please Check CONST.FLAG_OPT"
    metric_index = {"Latency": 0, "Energy": 1, "EDP": 2}[CONST.FLAG_OPT]
    if solver_result[metric_index] <= best_metric:
        result = solver_result + [sim_l, sim_e] + [profile]
        best_metric = solver_result[metric_index]
        best_count = count
        best_dataflow = result_pack["dataflow"]

    return best_metric, result, best_count, best_dataflow, solCount

def SolveMapping(acc:CIM_Acc, ops:WorkLoad, bestMetric:int, outputdir:str, singleIter=False, **kwargs):
    time_begin = time.time()

    if FLAG.INPUT_STATIONARY and (acc.core.size_input_buffer * acc.num_core >= ops.dim_M * ops.dim_K * ops.input.bitwidth):
        Logger.debug("Sufficient Buffer Resources for Input")

    count, solCount = 0, 0
    best_metric = bestMetric
    best_count = 0
    best_dataflow = None
    result = [CONST.MAX_POS] * 5 + [None]

    # ── Step 1: enumerate & prune spatial schemes ──────────────────────────
    if singleIter:
        scheme = kwargs["Spatial_unrolling"]
        assert scheme is not None, "Single Iteration Mode Requires Spatial Unrolling Input as Scheme."
        scheme_records = [{
            "origin_index": 1,
            "scheme": scheme,
            "meta": score_scheme(acc=acc, ops=ops, scheme=scheme),
        }]
    else:
        scheme_records = []
        for origin_index, scheme in enumerate(
            get_Spatial_Unrolling(ops.dim2bound, acc.mappingRule, acc.SpUnrolling),
            start=1,
        ):
            meta = score_scheme(acc=acc, ops=ops, scheme=scheme)
            scheme_records.append({
                "origin_index": origin_index,
                "scheme": scheme,
                "meta": meta,
            })

        total_candidates = len(scheme_records)

        # ── Dominance pruning (optional, off by default) ─────────────────
        # NOT provably safe: a dominator with prime temporal factors may have
        # fewer loop levels than the dominated scheme, limiting memory hierarchy
        # tiling flexibility.  ~44% of dominance pairs exhibit this factor-count
        # asymmetry.  Experimentally validated (never loses optimum on tested
        # workloads), but cannot guarantee zero optimality loss in general.
        # Enable via SolveMapping(..., dominance_pruning=True) for ablation.
        if kwargs.get("dominance_pruning", False):
            dominance_safe = True
            for m in range(1, acc.Num_mem):
                if acc.mem2dict(m) in ('OReg', 'IReg', 'Macro'):
                    continue
                if acc.shareMemory[m]:
                    combined_volume = 0
                    for op in range(3):
                        if not acc.mappingArray[op][m]: continue
                        max_spur = math.prod(acc.SpUnrolling[u] for u in range(acc.Num_SpUr) if m <= acc.SpUr2Mem[u, op])
                        prec = acc.precision_psum if op == 2 else acc.precision[m, op]
                        combined_volume += max_spur * prec
                    if combined_volume > acc.memSize[m]:
                        dominance_safe = False; break
                else:
                    for op in range(3):
                        if not acc.mappingArray[op][m]: continue
                        max_spur = math.prod(acc.SpUnrolling[u] for u in range(acc.Num_SpUr) if m <= acc.SpUr2Mem[u, op])
                        prec = acc.precision_psum if op == 2 else acc.precision[m, op]
                        if max_spur * prec > acc.memSize[m]:
                            dominance_safe = False; break
                    if not dominance_safe: break

            if dominance_safe:
                scheme_records = dominance_filter(scheme_records)
                if len(scheme_records) < total_candidates:
                    Logger.info(f"Dominance pruning: {total_candidates} -> {len(scheme_records)} schemes.")
            else:
                Logger.info(f"Dominance pruning skipped: spatial tile exceeds buffer capacity.")

        # ── Sort by utilization score ──────────────────────────────────────
        # High-utilization first → finds good bounds early → tightens metric_ub
        # → accelerates downstream analytical-LB pruning.
        scheme_records.sort(key=lambda r: (r["meta"]["sort_key"], -r["origin_index"]), reverse=True)

        # ── Analytical lower-bound screening (Propositions 1-3) ────────────
        # Discard schemes whose provable LB exceeds the initial metric UB.
        # The dynamic version in solve_scheme_worker re-checks against the
        # live shared_ub during parallel execution.
        if bestMetric < CONST.MAX_POS:
            before = len(scheme_records)
            survived = []
            for rec in scheme_records:
                lat_lb, eng_lb, edp_lb = scheme_objective_lb(acc, ops, rec["scheme"], rec["meta"]["temporal_unrolling"])
                obj_lb = {"Latency": lat_lb, "Energy": eng_lb, "EDP": edp_lb}.get(CONST.FLAG_OPT)
                if obj_lb is None or obj_lb <= bestMetric:
                    survived.append(rec)
            if len(survived) < before:
                Logger.info(f"Analytical LB screening: {before} -> {len(survived)} schemes (LB <= bestMetric).")
                scheme_records = survived

    num_schemes = len(scheme_records)

    for count, scheme_record in enumerate(scheme_records, start=1):
        scheme_record["count"] = count
        scheme_record["outputdir_scheme"] = outputdir if singleIter else os.path.join(outputdir, "SolPool", str(count))

    # ── Step 2: two-phase adaptive parallel configuration ─────────────────
    parallel_config = detect_parallel_config()
    usable = parallel_config["usable_cores"]
    avail_mem = parallel_config["available_mem_gb"]
    runtime_config = {"CONST": dict(vars(CONST)), "FLAG": dict(vars(FLAG))}

    # Phase 1 (Scout): top schemes — likely feasible, give more threads for
    # faster MIP solving.  Phase 2 (Sweep): remaining schemes — mostly
    # infeasible, maximize worker throughput for fast presolve screening.
    sweep_threads, sweep_workers = auto_parallel_config(usable, avail_mem, num_schemes)
    scout_threads = min(usable, max(sweep_threads * 2, 4))
    scout_workers = max(1, usable // scout_threads)

    # Memory cap for scout (sweep already capped inside auto_parallel_config)
    mem_per_worker = 2.0
    max_by_mem = max(1, int(avail_mem * 0.8 / mem_per_worker))
    scout_workers = min(scout_workers, max_by_mem)

    # Boundary: scout processes enough batches to cover the feasible range.
    scout_size = min(num_schemes, scout_workers * 3)

    use_parallel = (not singleIter) and usable >= 2
    if not use_parallel:
        scout_threads = usable
        scout_workers = 1
        sweep_threads = usable
        sweep_workers = 1
        scout_size = num_schemes

    soft_mem_scout = max(1.0, avail_mem * 0.8 / scout_workers)
    soft_mem_sweep = max(1.0, avail_mem * 0.8 / sweep_workers)

    Logger.critical(
        f"Auto Parallel Config: physical={parallel_config['physical_cores']}, logical={parallel_config['logical_cores']}, "
        f"usable={usable}, scout={scout_workers}w×{scout_threads}t(top-{scout_size}), "
        f"sweep={sweep_workers}w×{sweep_threads}t(rest-{max(0,num_schemes-scout_size)}), "
        f"schemes={num_schemes}"
    )

    scheme_iter = iter(scheme_records)

    if not use_parallel:
        for scheme_record in scheme_iter:
            count = scheme_record["count"]
            meta = scheme_record["meta"]
            append_scheme_summary(
                outputdir,
                f"Scheme {count:<3} : "
                f"util_prod={meta['util_product']:.4f}, min_util={meta['util_min']:.3f}, "
                f"avg_util={meta['util_avg']:.3f}] "
                f"Beginning: SpUr-{meta['spatial_unrolling']}, TpUr-{meta['temporal_unrolling']}"
            )
            worker_args = (
                count,
                scheme_record["origin_index"],
                scheme_record["scheme"],
                acc,
                ops,
                best_metric,
                outputdir,
                scheme_record["outputdir_scheme"],
                usable,
                soft_mem_scout,
            )
            result_pack = solve_scheme_worker(*worker_args)
            best_metric, result, best_count, best_dataflow, solCount = update_best(
                result_pack, best_metric, result, best_count, best_dataflow, solCount
            )
    else:
        _shm = SharedMemory(create=True, size=8)
        struct.pack_into('d', _shm.buf, 0, best_metric)
        _shm_name = _shm.name

        def _run_phase(executor, records, threads, soft_mem, max_workers):
            nonlocal count, best_metric, result, best_count, best_dataflow, solCount
            pending = {}
            rec_iter = iter(records)
            inflight_limit = max_workers * 2  # bound pending futures to limit memory

            def submit(rec):
                nonlocal count, best_metric
                scheme_count = rec["count"]
                meta = rec["meta"]
                append_scheme_summary(
                    outputdir,
                    f"Scheme {scheme_count:<3} : "
                    f"util_prod={meta['util_product']:.4f}, min_util={meta['util_min']:.3f}, "
                    f"avg_util={meta['util_avg']:.3f}] "
                    f"Beginning: SpUr-{meta['spatial_unrolling']}, TpUr-{meta['temporal_unrolling']}"
                )
                args = (
                    scheme_count, rec["origin_index"], rec["scheme"],
                    acc, ops, best_metric, outputdir, rec["outputdir_scheme"],
                    threads, soft_mem, runtime_config, _shm_name,
                )
                future = executor.submit(solve_scheme_worker, *args)
                pending[future] = scheme_count
                count = scheme_count

            # Fill pipeline
            for rec in rec_iter:
                submit(rec)
                if len(pending) >= inflight_limit:
                    break

            # Drain + refill
            while pending:
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result_pack = future.result()
                    best_metric, result, best_count, best_dataflow, solCount = update_best(
                        result_pack, best_metric, result, best_count, best_dataflow, solCount
                    )
                    SharedUB(_shm).update_min(best_metric)
                for rec in rec_iter:
                    submit(rec)
                    if len(pending) >= inflight_limit:
                        break

        mp_context = mp.get_context("spawn")
        scout_records = scheme_records[:scout_size]
        sweep_records = scheme_records[scout_size:]

        try:
            # Phase 1: Scout — more threads, top-ranked schemes
            with ProcessPoolExecutor(max_workers=scout_workers, mp_context=mp_context,
                                     initializer=_init_worker, initargs=(scout_threads, runtime_config)) as executor:
                _run_phase(executor, scout_records, scout_threads, soft_mem_scout, scout_workers)

            # Phase 2: Sweep — more workers, remaining schemes with tight shared_ub
            if sweep_records:
                with ProcessPoolExecutor(max_workers=sweep_workers, mp_context=mp_context,
                                         initializer=_init_worker, initargs=(sweep_threads, runtime_config)) as executor:
                    _run_phase(executor, sweep_records, sweep_threads, soft_mem_sweep, sweep_workers)
        finally:
            _shm.close()
            _shm.unlink()
    
    if count == 0:
        raise ValueError("No feasible spatial scheme found after pre-screening")
    if solCount == 0:
        raise ValueError("SOLVER IIS")

    time_end = time.time()
    if not singleIter:
        Logger.info(f"Total valid loop nest found: {count}, best_count: {best_count}")
        Logger.info(f"Solving Time within whole layer: {round(time_end - time_begin,1)}s")

    file_name = os.path.join(outputdir, "Dataflow.pkl")
    with open(file_name, 'wb') as file:
        pickle.dump(best_dataflow, file)

    return result

if __name__ == "__main__":
    
    import uuid
    import time
    outFolder = os.path.join("output",f"#SolveMappingTest_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid1().hex[:8]}")
    prepare_save_dir(outFolder)

    start_time = time.time()

    Logger.setcfg(setcritical=False, setDebug=True, STD=False, file=os.path.join(outFolder,'112.log'), nofile=False)
    cfg = get_ConfigFile('cim_template.cfg')

    from Architecture.ZigzagAcc import accelerator as acc_zz
    accelerator = CIM_Acc(acc_zz.cores[0])

    Logger.debug("Running SolveMapping for debugging and testing Solver (MIP model), only one iteration with given scheme")

    CONST.FLAG_OPT="Latency"
    # CONST.FLAG_OPT="Energy"
    # CONST.FLAG_OPT="EDP"
    
    CONST.MIPFOCUS = 1
    # CONST.MIPFOCUS = 2
    # CONST.MIPFOCUS = 3
    
    CONST.TIMELIMIT = 77
    # CONST.TIMELIMIT = 600

    # # # # # RestNet-layer-0
    # ops = WorkLoad(loopDim={'R': 7, 'S': 7, 'C': 3, 'K':64, 'P': 112, 'Q': 112, 'G': 1, 'B': 1, 'H': 224, 'W': 224, 'Stride': 2, 'Padding': 3})
    # Spatial_unrolling = [[1,1,1,2,1,1,4,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,7,1,1,3,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-1
    ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 64, 'K':64, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1})
    Spatial_unrolling =    [[1,1,1,2,1,1,4,1],
                        #   [-,R,S,P,Q,C,K,G],
                            [1,1,1,1,1,32,1,1],
                            [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-12
    # ops = WorkLoad(loopDim={'R': 1, 'S': 1, 'C': 128, 'K':256, 'P': 14, 'Q': 14, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 2, 'Padding': 0})
    # Spatial_unrolling = [[1,1,1,2,2,1,2,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-15
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 256, 'K':512, 'P': 7, 'Q': 7, 'G': 1, 'B': 1, 'H': 14, 'W': 14, 'Stride': 2, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,8,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-16
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 512, 'K':512, 'P': 7, 'Q': 7, 'G': 1, 'B': 1, 'H': 7, 'W': 7, 'Stride': 1, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,8,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-17
    # ops = WorkLoad(loopDim={'R': 1, 'S': 1, 'C': 256, 'K':512, 'P': 7, 'Q': 7, 'G': 1, 'B': 1, 'H': 14, 'W': 14, 'Stride': 2, 'Padding': 0})
    # Spatial_unrolling = [[1,1,1,1,1,1,8,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # MobileNetV2-depthwise Conv_1  (G=32,  P=Q=112, stride=1, SpUr from full-model Scheme4)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 112, 'Q': 112, 'G': 32, 'B': 1, 'H': 112, 'W': 112, 'Stride': 1, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,8,1,1,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    # # # # # MobileNetV2-depthwise Conv_10 (G=144, P=Q=28,  stride=2, SpUr from full-model Scheme2)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 28, 'Q': 28, 'G': 144, 'B': 1, 'H': 56, 'W': 56, 'Stride': 2, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,2,1,1,4],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    # # # # # MobileNetV2-depthwise Conv_40 (G=576, P=Q=7,   stride=2, SpUr from full-model Scheme1)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 7, 'Q': 7, 'G': 576, 'B': 1, 'H': 14, 'W': 14, 'Stride': 2, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,1,8],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    # # # # # MobileNetV2-depthwise Conv_43 (G=960, P=Q=7,   stride=1, SpUr from full-model Scheme1)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 7, 'Q': 7, 'G': 960, 'B': 1, 'H': 7, 'W': 7, 'Stride': 1, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,1,8],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    lat, eng, edp, c_lat, c_eng, ds = SolveMapping(acc=accelerator, ops=ops, bestMetric=1e10, outputdir=outFolder, singleIter=True, Spatial_unrolling=Spatial_unrolling)

    end_time = time.time()
    Logger.critical(f"SingleIter-Running SolveMapping Cost: {round(end_time - start_time,1)}s")

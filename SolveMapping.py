# this file is prepared for project 511
# Created by iboxl

from utils.Workload import WorkLoad
from utils.Tools import get_ConfigFile, append_scheme_summary, detect_parallel_config
from utils.SolverTSS import Solver
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import *
from Architecture.ArchSpec import CIM_Acc
import pickle
from utils.UtilsFunction.ToolFunction import prepare_save_dir, get_Spatial_Unrolling
import shutil
import time, math, os
import heapq
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

def score_scheme(acc:CIM_Acc, ops:WorkLoad, scheme):
    axis_products = [math.prod(axis_scheme) for axis_scheme in scheme]
    axis_utils = [x / y if y > 0 else 1.0 for x, y in zip(axis_products, acc.SpUnrolling)]

    spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
    temporal_unrolling = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]

    util_product = math.prod(axis_utils) if axis_utils else 1.0
    util_min = min(axis_utils) if axis_utils else 1.0
    util_avg = sum(axis_utils) / len(axis_utils) if axis_utils else 1.0
    temporal_logsum = sum(math.log(max(1, tu)) for tu in temporal_unrolling)

    return {
        "spatial_unrolling": spatial_unrolling,
        "temporal_unrolling": temporal_unrolling,
        "util_product": util_product,
        "util_min": util_min,
        "util_avg": util_avg,
        "sort_key": (util_product, util_min, util_avg, -temporal_logsum),
    }


def solve_scheme_worker(count:int, origin_index:int, scheme, acc:CIM_Acc, ops:WorkLoad, metric_ub:float, outputdir_root:str, outputdir_scheme:str, threads_per_worker:int, soft_mem_limit_gb:float, runtime_config=None):
    if runtime_config is not None:
        for key, value in runtime_config["CONST"].items():
            setattr(CONST, key, value)
        for key, value in runtime_config["FLAG"].items():
            setattr(FLAG, key, value)

    prepare_save_dir(outputdir_scheme)

    spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
    temporal_unrolling = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]

    solver = Solver(
        acc=acc,
        ops=ops,
        tu=temporal_unrolling,
        su=scheme,
        metric_ub=metric_ub,
        outputdir=outputdir_scheme,
        threads=threads_per_worker,
        soft_mem_limit_gb=soft_mem_limit_gb
    )

    sim_l, sim_e, profile = CONST.MAX_POS, CONST.MAX_POS, None
    try:
        solver.run()

        has_solution = solver.model is not None and solver.model.SolCount > 0
        if has_solution:
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

    parallel_config = detect_parallel_config()
    requested_threads = kwargs.get("threads_per_worker")
    requested_workers = kwargs.get("max_workers")

    if requested_threads is None and requested_workers is None:
        threads_per_worker = parallel_config["threads_per_worker"]
        max_workers = parallel_config["max_workers"]
    elif requested_threads is None:
        max_workers = max(1, int(requested_workers))
        threads_per_worker = max(1, parallel_config["usable_cores"] // max_workers)
    elif requested_workers is None:
        threads_per_worker = max(1, int(requested_threads))
        max_workers = max(1, parallel_config["usable_cores"] // threads_per_worker)
    else:
        threads_per_worker = max(1, int(requested_threads))
        max_workers = max(1, int(requested_workers))

    if singleIter:
        scheme = kwargs["Spatial_unrolling"]
        assert scheme is not None, "Single Iteration Mode Requires Spatial Unrolling Input as Scheme."
        scheme_records = [{
            "origin_index": 1,
            "scheme": scheme,
            "meta": score_scheme(acc=acc, ops=ops, scheme=scheme),
        }]
    else:
        scheme_topk = CONST.SPATIAL_SCHEME_TOPK
        topk_heap = []
        total_candidates = 0

        for origin_index, scheme in enumerate(
            get_Spatial_Unrolling(ops.dim2bound, acc.mappingRule, acc.SpUnrolling),
            start=1,
        ):
            total_candidates += 1
            meta = score_scheme(acc=acc, ops=ops, scheme=scheme)
            record = {
                "origin_index": origin_index,
                "scheme": scheme,
                "meta": meta,
            }

            rank = (meta["sort_key"], -origin_index)
            if len(topk_heap) < scheme_topk:
                heapq.heappush(topk_heap, (rank, record))
            elif rank > topk_heap[0][0]:
                heapq.heapreplace(topk_heap, (rank, record))

        scheme_records = [item[1] for item in sorted(topk_heap, key=lambda item: item[0], reverse=True)]
        if total_candidates > scheme_topk:
            Logger.info(f"Spatial scheme top-k pruning: keep {scheme_topk}/{total_candidates} candidates by score_scheme.")

    for count, scheme_record in enumerate(scheme_records, start=1):
        scheme_record["count"] = count
        scheme_record["outputdir_scheme"] = outputdir if singleIter else os.path.join(outputdir, "SolPool", str(count))

    scheme_iter = iter(scheme_records)
    use_parallel = (not singleIter) and max_workers > 1
    runtime_config = None
    if use_parallel:
        runtime_config = {"CONST": dict(vars(CONST)), "FLAG": dict(vars(FLAG))}
    if not use_parallel and requested_threads is None:
        max_workers = 1
        threads_per_worker = parallel_config["usable_cores"]

    max_inflight = kwargs.get("max_inflight", max_workers * 2)
    soft_mem_limit_gb = kwargs.get(
        "soft_mem_limit_gb",
        max(1.0, parallel_config["available_mem_gb"] * 0.8 / max_workers),
    )

    Logger.critical(
        f"Auto Parallel Config: physical={parallel_config['physical_cores']}, logical={parallel_config['logical_cores']}, "
        f"usable={parallel_config['usable_cores']}, workers={max_workers}, threads/worker={threads_per_worker}, "
        f"soft_mem_limit={round(soft_mem_limit_gb,2)} GB"
    )

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
                threads_per_worker,
                soft_mem_limit_gb,
            )
            result_pack = solve_scheme_worker(*worker_args)
            best_metric, result, best_count, best_dataflow, solCount = update_best(
                result_pack, best_metric, result, best_count, best_dataflow, solCount
            )
    else:
        def submit_next(executor, pending):
            nonlocal count, best_metric
            try:
                scheme_record = next(scheme_iter)
            except StopIteration:
                return False

            scheme_count = scheme_record["count"]
            meta = scheme_record["meta"]
            append_scheme_summary(
                outputdir,
                f"Scheme {scheme_count:<3} : "
                f"util_prod={meta['util_product']:.4f}, min_util={meta['util_min']:.3f}, "
                f"avg_util={meta['util_avg']:.3f}] "
                f"Beginning: SpUr-{meta['spatial_unrolling']}, TpUr-{meta['temporal_unrolling']}"
            )
            worker_args = (
                scheme_count,
                scheme_record["origin_index"],
                scheme_record["scheme"],
                acc,
                ops,
                best_metric,
                outputdir,
                scheme_record["outputdir_scheme"],
                threads_per_worker,
                soft_mem_limit_gb,
                runtime_config,
            )
            future = executor.submit(solve_scheme_worker, *worker_args)
            pending[future] = scheme_count
            count = scheme_count
            return True

        pending = {}
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            while len(pending) < max_inflight and submit_next(executor, pending):
                pass

            while pending:
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result_pack = future.result()
                    best_metric, result, best_count, best_dataflow, solCount = update_best(
                        result_pack, best_metric, result, best_count, best_dataflow, solCount
                    )
                while len(pending) < max_inflight and submit_next(executor, pending):
                    pass
    
    if count == 0:
        raise ValueError("No feasible spatial scheme survived top-k screening")
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

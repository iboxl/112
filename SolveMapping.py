# this file is prepared for project 511
# Created by iboxl

from utils.Workload import WorkLoad
from utils.Tools import *
import gurobipy as gp
from gurobipy import GRB, quicksum
from utils.SolverTSS import Solver
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import *
from Architecture.ArchSpec import CIM_Acc
import pickle
from utils.UtilsFunction.ToolFunction import get_Spatial_Unrolling, prepare_save_dir
import shutil
import time

def SolveMapping(acc:CIM_Acc, ops:WorkLoad, bestMetric:int, outputdir:str, singleIter=False, **kwargs):
    time_begin = time.time()

    if FLAG.INPUT_STATIONARY and (acc.core.size_input_buffer * acc.num_core >= ops.dim_M * ops.dim_K * ops.input.bitwidth):
        Logger.debug("Sufficient Buffer Resources for Input")
        Latency = 0
        Energy = 0
        EDP = 0

    '''
    Generator for spliting temporal/spatial mapping
    '''
    generator = get_Spatial_Unrolling(ops.dim2bound, acc.mappingRule, acc.SpUnrolling, 0.5)

    count, solCount = 0, 0
    best_metric = bestMetric
    best_count = 0
    best_dataflow = None
    result = [CONST.MAX_POS] * 5 + [None]

    for scheme in generator:
        if singleIter == True:
            Spatial_unrolling = kwargs['Spatial_unrolling']
            assert Spatial_unrolling is not None, "Single Iteration Mode Requires Spatial Unrolling Input as Scheme."
            scheme = Spatial_unrolling

        count += 1

        spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
        temporal_unrolling= [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]
        
        outputdir_scheme = os.path.join(os.path.join(outputdir, 'SolPool'), str(count))
        prepare_save_dir(outputdir_scheme)

        Logger.info('\n' + f"Scheme {count:<3} Beginning: SpUr-{spatial_unrolling}, TpUr-{temporal_unrolling}")
        
        solver = Solver(acc=acc, ops=ops, tu=temporal_unrolling, su=scheme, metric_ub=best_metric, outputdir=outputdir_scheme)

        solver.run()

        if (solver.model.status == GRB.OPTIMAL or solver.model.status == GRB.SUBOPTIMAL or (solver.model.status == GRB.Status.TIME_LIMIT and solver.model.SolCount)) and FLAG.SIMU:

            simu = tranSimulator(acc=acc, ops=ops, dataflow=solver.dataflow)
            # Logger.debug(simu.debugLog())
            sim_l, sim_e = simu.run()
            
            solCount += 1
            Logger.critical(f"Solver Result: Latency-{solver.result[0]}, Energy-{solver.result[1]}")
            Logger.info(f"Scheme {count:<3} End: Latency-{round(sim_l,3):<15}, Energy-{round(sim_e,3):<15}, EDP-{round(sim_l * sim_e,3):<15}")
            Logger.info(f"* * *Latency Accuracy: {round(100-abs(solver.result[0]-sim_l)/sim_l*100,2)}%, Solver-{round(solver.result[0]):<10} and Simu-{round(sim_l,3):<10}")
            Logger.info(f"* * * Energy Accuracy: {round(100-abs(solver.result[1]-sim_e)/sim_e*100,2)}%, Solver-{round(solver.result[1]):<10} and Simu-{round(sim_e,3):<10}")

        else:
            sim_l, sim_e = CONST.MAX_POS, CONST.MAX_POS
            Logger.info(f"Scheme {count:<3} End: NO BETTER SOLUTION")


            # 删除目录及其所有内容
            if os.path.exists(outputdir_scheme):
                shutil.rmtree(outputdir_scheme)
        
        assert CONST.FLAG_OPT in ["Latency", "Energy", "EDP"], "No Such Metric for Optimization, Please Check CONST.FLAG_OPT"
        for f_i, f in enumerate(["Latency", "Energy", "EDP"]):
            if f == CONST.FLAG_OPT:
                if solver.result[f_i] <= best_metric:
                    result = solver.result + [sim_l, sim_e] + [simu.PD]
                    best_metric = solver.result[f_i]
                    best_count = count
                    best_dataflow = solver.dataflow

        solver.model.dispose()

        if singleIter == True:
            exit()
    
    if count == 0:
        raise ValueError("No Feasible Sol Found, Need to reset MIN_UTIL_COEFFICIENT")
    if solCount == 0:
        raise ValueError("SOLVER IIS")
            
    Logger.info(f"\nTotal valid loop nest found: {count}, best_count: {best_count}")

    time_end = time.time()
    Logger.info(f"Solving Time within layer: {round(time_end - time_begin,1)}s")

    file_name = os.path.join(outputdir, "Dataflow.pkl")
    with open(file_name, 'wb') as file:
        pickle.dump(best_dataflow, file)

    return result

if __name__ == "__main__":
    
    import uuid
    import time
    outFolder = os.path.join("output",f"#SolveMappingTest_{uuid.uuid1()}")
    prepare_save_dir(outFolder)

    start_time = time.time()

    Logger.setcfg(setcritical=False, setDebug=True, STD=False, file=os.path.join(outFolder,'112.log'), nofile=False)
    cfg = get_ConfigFile('cim_template.cfg')

    from Architecture.ZigzagAcc import accelerator as acc_zz
    accelerator = CIM_Acc(acc_zz.cores[0])

    Logger.debug("Running SolveMapping for debugging and testing Solver (MIP model), only one iteration with given scheme")

    CONST.FLAG_OPT="Latency"
    # CONST.FLAG_OPT="EDP"
    
    CONST.MIPFOCUS = 1

    ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 64, 'K':64, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1})
    Spatial_unrolling =    [[1,1,1,2,1,1,4,1],
                        #   [-,R,S,P,Q,C,K,G],
                            [1,1,1,1,1,32,1,1],
                            [1,1,1,1,1,1,16,1]]

    # ops = WorkLoad(loopDim={'R': 7, 'S': 7, 'C': 3, 'K':64, 'P': 112, 'Q': 112, 'G': 1, 'B': 1, 'H': 224, 'W': 224, 'Stride': 2, 'Padding': 3})
    # Spatial_unrolling = [[1,1,1,2,1,1,4,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,7,1,1,3,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    lat, eng, edp, c_lat, c_eng, ds = SolveMapping(acc=accelerator, ops=ops, bestMetric=1e10, outputdir=outFolder, singleIter=True, Spatial_unrolling=Spatial_unrolling)
    Logger.critical(f"* * MIREDO-Running  * *   Latency:{lat}")
    # Logger.critical(f"S_latency={round(lat)}, S_Energy={round(eng,2)}, S_EDP={edp}")
    # Logger.critical(f"C_latency={round(c_lat)}, C_Energy={round(c_eng,2)}")

    end_time = time.time()
    Logger.critical(f"Running SolveMapping Cost: {round(end_time - start_time,1)}s")

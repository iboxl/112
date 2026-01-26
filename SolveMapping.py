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

def SolveMapping(acc:CIM_Acc, ops:WorkLoad, cycBound:int, outputdir:str):
    Logger.critical(ops)

    if FLAG.INPUT_STATIONARY and (acc.core.size_input_buffer * acc.num_core >= ops.dim_M * ops.dim_K * ops.input.bitwidth):
        Logger.debug("Sufficient Buffer Resources for Input")
        Latency = 0
        Energy = 0
        EDP = 0

    '''
    Generator for spliting temporal/spatial mapping
    '''
    generator = get_Spatial_Unrolling(ops.dim2bound, acc.mappingRule, acc.SpUnrolling, 0.5)

    count = 0
    best_l = 1e10
    best_count = 0
    best_dataflow = None
    result = None

    for scheme in generator:
        count += 1

        spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
        temporal_unrolling= [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]
        
        outputdir_scheme = os.path.join(outputdir, str(count))
        prepare_save_dir(outputdir_scheme)

        solver = Solver(acc=acc, ops=ops, tu=temporal_unrolling, su=scheme, cycBound=cycBound, outputdir=outputdir_scheme)

        solver.run()

        if (solver.model.status == GRB.OPTIMAL or solver.model.status == GRB.SUBOPTIMAL or (solver.model.status == GRB.Status.TIME_LIMIT and solver.model.SolCount)) and FLAG.SIMU:

            simu = tranSimulator(acc=acc, ops=ops, dataflow=solver.dataflow)
            # Logger.debug(simu.debugLog())
            sim_l, sim_e = simu.run()
            simu.LModeling()
        else:
            sim_l, sim_e = 0, 0
            raise ValueError("SOLVER IIS")
        
        if sim_l <= best_l:
            result = solver.result + [sim_l, sim_e] + [simu.PD]
            best_l = sim_l
            best_count = count
            best_dataflow = solver.dataflow

        solver.model.dispose()

            
    Logger.info(f"\nTotal valid loop nest found: {count}, best_count: {best_count}")

    file_name = os.path.join(outputdir, "Dataflow.pkl")
    with open(file_name, 'wb') as file:
        pickle.dump(best_dataflow, file)

    return result

if __name__ == "__main__":
    Logger.setcfg(setcritical=False, setDebug=True, STD=False, file='112.log', nofile=False)
    cfg = get_ConfigFile('cim_template.cfg')

    from Architecture.ZigzagAcc import accelerator as acc_zz
    accelerator = CIM_Acc(acc_zz.cores[0])

    # CONST.FLAG_OPT="Latency"
    # CONST.FLAG_OPT="EDP"
    ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 64, 'K':64, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1},
                            )
    lat, eng, edp, c_lat, c_eng, ds = SolveMapping(acc=accelerator, ops=ops, cycBound=2e7, outputdir="./")
    Logger.critical(f"* * MIREDO-Running  * *   Latency:{lat}")
    # Logger.critical(f"S_latency={round(lat)}, S_Energy={round(eng,2)}, S_EDP={edp}")
    # Logger.critical(f"C_latency={round(c_lat)}, C_Energy={round(c_eng,2)}")

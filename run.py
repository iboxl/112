# this file is prepared for project 419
# Created by iboxl

from utils.Tools import *
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import Operands, WorkLoad, LoopNest
from SolveMapping import SolveMapping
import argparse
from utils.GlobalUT import *
import pickle
import uuid
from utils.UtilsFunction.OnnxParser import extract_loopdims
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from utils.CompatibleZigzag import convert_zz_to_miredo, compare_ops_cme
from Simulator.Simulax import tranSimulator
import time, copy
from importlib import import_module


def get_Args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", nargs="?", const=True, default=False, help="Enable debug mode.")
    parser.add_argument("--logger", nargs="?", const=True, default=False, help="Just print Logger")        # W.T.D. exchange ture and false
    parser.add_argument("--srun", nargs="?", const=True, default=False, help="batch srun with critical message")
    parser.add_argument("--noLogFile", nargs="?", const=True, default=False, help="No log file")
    # parser.add_argument("--EX", nargs="?", const=True, default=False, help="exchange input & weight")
    # parser.add_argument("--BM", nargs="?", const=True, default=False, help="using blocking Dim_M")
    # parser.add_argument("--RS", nargs="?", const=True, default=False, help="FLAG: Row traverse")
    parser.add_argument("--WS", nargs="?", const=True, default=False, help="FLAG: Weight stationary")
    parser.add_argument("--IS", nargs="?", const=True, default=False, help="FLAG: Buffer/Input stationary")
    # parser.add_argument("--OS", nargs="?", const=True, default=False, help="FLAG: Rejust dim/block to avoid overSize")
    # parser.add_argument("--GM", nargs="?", const=True, default=False, help="using blocking Dim_K with GAMMA")
    parser.add_argument("--NoPreSolve", nargs="?", const=True, default=False, help="dont search presolve by alpha&beta")
    parser.add_argument("--SIMU", nargs="?", const=True, default=False, help="using simulator calc")

    parser.add_argument('-c', '--cfg', dest='cfg', required=False, 
                        type=str, default='cim_template.cfg', help = 'config File Name')
    parser.add_argument('-m', '--model', dest='model', required=False, 
                        type=str, default='resnet18', help = 'NN model Name')
    parser.add_argument('-log', '--log_file', dest='log', required=False, 
                        type=str, default='112.log', help = 'Log file Name')
    parser.add_argument('-opt', '--flag_opt', dest='opt', choices=["Latency", "Energy", "EDP"], required=False, 
                        type=str, default="Feasible", help = 'kind of model optimization: 0=Feasible solution, 1=MIN_latency  2=MIN_energy  3=MIN_EDP')
    parser.add_argument('-class', '--num_classes', dest='classes', choices=[10, 1000], required=False, 
                        type=int, default=1000, help = '10=CIFAR 1000=ImageNet')
    parser.add_argument('-t', '--time', dest='time_limit', required=False, 
                        type=int, default=CONST.TIMELIMIT, help = 'time limitation for solving gurobi model')
    # parser.add_argument('-n', '--name', dest='dataflow_name', required=False, 
    #                     type=str, default='ds', help = 'save dataflow in FigName')
    parser.add_argument('-o', '--outputdir', dest='output_dir', required=False, 
                        type=str, default=f'test_{uuid.uuid1()}', help = 'save output files in folder')
    parser.add_argument('-arch', '--architecture', dest='architecture', required=False, 
                        type=str, default=f'ZigzagAcc', help = 'save output files in folder')
    args = parser.parse_args()

    return args
 
def __main__(**kwargs):

    args = get_Args()
    start_time = time.time()
    outFolder = os.path.join("output",args.output_dir)
    prepare_save_dir(outFolder)

    Logger.setcfg(setcritical=args.srun, setDebug=args.debug, STD=args.logger, file=os.path.join(outFolder,args.log), nofile=args.noLogFile)

    CONST.FLAG_OPT              = args.opt
    CONST.TIMELIMIT             = args.time_limit
    FLAG.WEIGHT_STATIONARY      = args.WS
    FLAG.INPUT_STATIONARY       = args.IS and (not args.RS)
    FLAG.DEBUG_SIMU             = args.SIMU
    FLAG.PRESOLVE_SEARCH        = not args.NoPreSolve
    FLAG.DEBUG_PER_LAYER_DETAIL = False               # illegal Tmp setting

    Logger.info("* " * 50)
    Logger.info(f"model={args.model}, Architecture={args.architecture}, Weight_stationary={FLAG.WEIGHT_STATIONARY}, " + 
                f"Optimization_Flag={CONST.FLAG_OPT}, MIPFOCUS={CONST.MIPFOCUS}" )
    Logger.info("* " * 50)

    model = f"model/{args.model}.onnx"

    convs, loopdims = extract_loopdims(model)

    match CONST.FLAG_OPT:
        case "Latency":
            opt_flag = "latency" 
        case "Energy":
            opt_flag = "energy" 
        case "EDP":
            opt_flag = "EDP" 
        case _:
            opt_flag = "latency"            # flagOPT = infeasible
    
    compare_filePrefix = f"zzz_outputs/{opt_flag}_{args.model}_{args.architecture}"
    if os.path.isfile(f"{compare_filePrefix}.pickle") == False:     # Zigzag-CME is not exist, running ZZ-opt
        Logger.info("Running Zigzag to generate CME-compared")
        start_time_zz = time.time()
        from zigzag.api import get_hardware_performance_zigzag
        energy, latency, cmeAll = get_hardware_performance_zigzag(
            workload = model,
            accelerator = f"Architecture.{args.architecture}",
            mapping = "Config.zigzag_mapping",
            opt=opt_flag,
            dump_filename_pattern=f"{compare_filePrefix}.json",
            pickle_filename=f"{compare_filePrefix}.pickle"
        )
        end_time_zz = time.time()
        Logger.critical(f"Zigzag solve cost time: {end_time_zz - start_time_zz}")     
    with open(f"{compare_filePrefix}.pickle", 'rb') as fp:
        cmes= pickle.load(fp)

    assert len(convs) == len(loopdims)

    cache = {}

    latency_zz, energy_zz, latency_mi, energy_mi = 0, 0, 0, 0
    
    acc_template = import_module(f"Architecture.{args.architecture}").accelerator
    accelerator_eval  = CIM_Acc(acc_template.cores[0])

    for i, (Conv, loopdim) in enumerate(zip(convs, loopdims)):
        # if i == 9 :
        #     pass
        # else:
        #     continue

        Logger.info('\n\n'+'* '*20+f"Layer {i}"+' *'*20)
        ops = WorkLoad(loopDim=loopdim)
        Logger.info(ops)

        key = tuple(sorted(loopdim.items()))
        pstr, cache_flag = "", False

        try:                                   
            (l_solver, e_solver, l_simu, e_simu, l_zz, e_zz) = cache[key]                  
            Logger.info("Get Result From Cache")
            cache_flag = True
        except KeyError:              
            outputdir_layer = os.path.join(outFolder,Conv)
            prepare_save_dir(outputdir_layer)
            Logger.changeFile(new_file = os.path.join(outputdir_layer,"Evaluation-Layer.log"))
            Logger.info(ops)
            Logger.info('\n' + '* '*30 + '\n')

            cme_compare = None
            cme_compare = next(c for c in cmes if compare_ops_cme(loopDim=loopdim, cme=c))
            assert cme_compare is not None     
            cache_flag = False

            accelerator = copy.deepcopy(accelerator_eval)

            loops = LoopNest(acc=accelerator,ops=ops)
            loops = convert_zz_to_miredo(loops=loops, cme=cme_compare)
            loops.usr_defined_double_flag[accelerator.Macro2mem][1] = accelerator.double_Macro

            try:
                Logger.info("Running: Zigzag-in-MIREDO-Simulator - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                simu = tranSimulator(acc=accelerator, ops=ops, dataflow=loops)
                l_zz, e_zz = simu.run()
            except ValueError as e:  
                Logger.error('Wrong Match') 
                Logger.changeFile(new_file = os.path.join(outFolder,args.log))
                Logger.error(e) 
                continue
            PD_Z = simu.PD

            l_zz_modeling = simu.LModeling()
            # simu.idealExec()
            
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#   
            accelerator = copy.deepcopy(accelerator_eval)
            newdim = copy.deepcopy(loopdim)
            for dChar in ['P','Q','H','W']: #newdim[dChar] += (loopdim[dChar] % 2)
                if loopdim[dChar] % 2==1 and loopdim[dChar]>15:
                    newdim[dChar] += 1

            match CONST.FLAG_OPT:
                case "Latency":
                    bestMetric = l_zz
                case "Energy":
                    bestMetric = e_zz
                case "EDP":
                    bestMetric = l_zz * e_zz * CONST.SCALINGFACTOR
                case _:
                    bestMetric = 1e9            # flagOPT = infeasible

            l_solver, e_solver, edp_solver, l_simu, e_simu, PD_M = SolveMapping(acc=accelerator, ops=WorkLoad(loopDim=newdim), bestMetric=bestMetric * 1.5, outputdir=outputdir_layer)
            cache[key] = (l_solver, e_solver, l_simu, e_simu, l_zz, e_zz)
            
            pstr += "\n-------- MemHierarchy ----- DB_M -- DB_Z --- PowerRate --- Power_M - Power_E -----\n"
            for m in range(1,accelerator.Num_mem):
                dm, dz = PD_M.doubleFlag, PD_Z.doubleFlag
                pstr += '-'*8 + f" {accelerator.mem2dict(m):<15}: [ {dm[m][0]} {dm[m][1]} {dm[m][2]} | {dz[m][0]} {dz[m][1]} {dz[m][2]} ]  "
                memCost_rate = '-'*8 if PD_Z.memCost[m]==0 else round(PD_M.memCost[m]/PD_Z.memCost[m]*100,2)
                pstr += f"{memCost_rate:>8}%   ({PD_M.memCost[m]:.2e} / {PD_Z.memCost[m]:.2e}) pJ" + '\n'
            pstr += '\n-------- Multiply & ADD : -----------------  '
            pstr += f"{round(PD_M.macEnergy/PD_Z.macEnergy*100,2):>8}%   ({PD_M.macEnergy:.2e} / {PD_Z.macEnergy:.2e}) pJ" + '\n\n'

            pstr += f"Dynamic Power: {round(PD_M.dynamic_power/PD_Z.dynamic_power*100,2):>8}%   ({PD_M.dynamic_power:.2e} / {PD_Z.dynamic_power:.2e}) pJ" + '\n'
            pstr += f"Leakage Power: {round(PD_M.leakage_power/PD_Z.leakage_power*100,2):>8}%   ({PD_M.leakage_power:.2e} / {PD_Z.leakage_power:.2e}) pJ" + '\n'
            pstr += f"On-chip Power: {round(PD_M.dynamic_power_onChip/PD_Z.dynamic_power_onChip*100,2):>8}%   ({PD_M.dynamic_power_onChip:.2e} / {PD_Z.dynamic_power_onChip:.2e}) pJ" + '\n'

        pstr += '\n'
        pstr += f"* * * Zigzag-Running  * * *  Latency:{round(l_zz,3):<15}, Energy:{round(e_zz,3):<20}, EDP:{round(l_zz *e_zz,3):.5e}, LModeling:{round(l_zz_modeling,3):<15}" + '\n'
        pstr += f"* * * MIREDO-Running  * * *  Latency:{round(l_simu,3):<15}, Energy:{round(e_simu,3):<20}, EDP:{round(l_simu*e_simu,3):.5e}" + '\n'
        pstr += f"MIP Solver Accuracy of Layer: {round(l_simu/l_solver*100,2)}%    (S){round(l_simu,3):<15} (M){round(l_solver,3):<15} " + '\n'

        rstr = f"Optimization Of Layer-{i}: Latency-({round(l_simu/l_zz*100,2)}%), Energy-({round(e_simu/e_zz*100,2)}%), EDP-({round((l_simu*e_simu)/(l_zz * e_zz)*100,2)}%)"
        
        if cache_flag == False:
            Logger.info(pstr)
            Logger.critical(rstr)
            Logger.changeFile(new_file = os.path.join(outFolder,args.log))
        
        latency_mi += l_simu
        energy_mi  += e_simu
        latency_zz += l_zz
        energy_zz  += e_zz

        Logger.info(pstr)
        Logger.critical(rstr)

    # exit()
    Logger.info("* " * 50)
    Logger.info('\n\n'+'* '*20+f"The WHOLE Model"+' *'*20)
    Logger.info(f"* * * Zigzag-Running  * * *  Latency:{round(latency_zz,3):<15}, Energy:{round(energy_zz,3):<15}, EDP:{round(latency_zz * energy_zz,3):.5e}")
    Logger.info(f"* * * MIREDO-Running  * * *  Latency:{round(latency_mi,3):<15}, Energy:{round(energy_mi,3):<15}, EDP:{round(latency_mi * energy_mi,3):.5e}")
    Logger.info(f"* * *  Optimization  * * *   Latency:{round(latency_mi/latency_zz*100,2)}%, Energy:{round(energy_mi/energy_zz*100,2)}%, EDP:{round((latency_mi * energy_mi)/(latency_zz * energy_zz)*100,2)}%")
    # exit()

        # res_energy Unit=nj    = 1e-9j  
        # res_latency Unit      = cycle
        # Power Unit            = mw
    # power = (res_e * CONST.SCALINGFACTOR * 1e-12 * 500 * 1e6 / res_l) * 1e3
    # Logger.debug(f"scaling factor: {CONST.SCALINGFACTOR}")
    end_time = time.time()
    Logger.critical(f"Solving The Whole Model Cost: {round(end_time - start_time,1)}s")

    Logger.recover_stdout()
    if os.path.isfile("pcacti_report.txt"):
        os.remove("pcacti_report.txt")                                  # remove cacti output file

__main__()
# this file is prepared for project 419
# Created by iboxl

from utils.Tools import *
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import Operands, WorkLoad, LoopNest
from SolveMapping import SolveMapping
import argparse
from utils.GlobalUT import *
import uuid
from utils.UtilsFunction.OnnxParser import extract_loopdims
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from Simulator.Simulax import tranSimulator
from Evaluation.Zigzag_imc.CompatibleZigzag import convert_baseline_to_MIREDO
from baseline.zigzag_adapter import ZigzagBaselineAdapter
from baseline.cosa_adapter import CoSABaselineAdapter
from baseline.cimloop_adapter import CimloopBaselineAdapter
from utils.ZigzagUtils import zigzag_cache_prefix, get_hardware_performance_zigzag, convert_Zigzag_to_MIREDO, compare_ops_cme
from Evaluation.WeightStationaryGenerator import generate_weight_stationary_baseline
from Evaluation.common.EvalCommon import make_accelerator, normalize_loopdim_for_solver, mip_cache_get, mip_cache_put
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
    parser.add_argument('-f', '--mipFocus', dest='mipFocus', choices=[0,1,2,3], required=False, 
                        type=int, default=1, help = '0=balanced, 1=feasibility, 2=optimality, 3=best bound')
    parser.add_argument('-class', '--num_classes', dest='classes', choices=[10, 1000], required=False, 
                        type=int, default=1000, help = '10=CIFAR 1000=ImageNet')
    parser.add_argument('-t', '--time', dest='time_limit', required=False, 
                        type=int, default=CONST.TIMELIMIT, help = 'time limitation for solving gurobi model')
    # parser.add_argument('-n', '--name', dest='dataflow_name', required=False, 
    #                     type=str, default='ds', help = 'save dataflow in FigName')
    parser.add_argument('-o', '--outputdir', dest='output_dir', required=False,
                        type=str, default=f'test_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid1().hex[:8]}',
                        help = 'save output files in folder')
    parser.add_argument('-arch', '--architecture', dest='architecture', required=False,
                        type=str, default=f'ZigzagAcc', help = 'save output files in folder')
    parser.add_argument('--baseline', dest='baseline', required=False,
                        choices=['zigzag', 'cosa', 'cimloop'], type=str, default='zigzag',
                        help='baseline mapping provider')
    parser.add_argument('--cosa_map', dest='cosa_map', required=False, type=str, default=None,
                        help='optional path to CoSA map_16.yaml (or directory). If omitted, CoSA uses --model/--architecture like Zigzag.')
    parser.add_argument('--cimloop_map', dest='cimloop_map', required=False, type=str, default=None,
                        help='path to CIMLoop baseline yaml file or directory containing layer baseline yaml files')
    parser.add_argument('--cimloop_macro', dest='cimloop_macro', required=False, type=str,
                        default=None, help='optional CIMLoop macro override in generate mode')
    parser.add_argument('--cimloop_system', dest='cimloop_system', required=False, type=str,
                        default=None, help='optional CIMLoop system override in generate mode')
    parser.add_argument('--cimloop_tile', dest='cimloop_tile', required=False, type=str, default=None,
                        help='optional CIMLoop tile name used in generate mode')
    parser.add_argument('--cimloop_chip', dest='cimloop_chip', required=False, type=str, default=None,
                        help='optional CIMLoop chip name used in generate mode')
    parser.add_argument('--cimloop_iso', dest='cimloop_iso', required=False, type=str, default=None,
                        help='optional CIMLoop iso macro name used in generate mode')
    parser.add_argument('--cimloop_hw_from_arch', dest='cimloop_hw_from_arch', action='store_true', default=True,
                        help='derive CIMLoop hardware profile from MIREDO Architecture (default on)')
    parser.add_argument('--no_cimloop_hw_from_arch', dest='cimloop_hw_from_arch', action='store_false',
                        help='disable architecture-driven CIMLoop hardware profile and use manual/default options')
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
    CONST.MIPFOCUS              = args.mipFocus
    FLAG.INPUT_STATIONARY       = args.IS
    FLAG.DEBUG_SIMU             = args.SIMU
    FLAG.PRESOLVE_SEARCH        = not args.NoPreSolve
    FLAG.DEBUG_PER_LAYER_DETAIL = False               # illegal Tmp setting

    Logger.info("* " * 50)
    Logger.info(f"model={args.model}, Architecture={args.architecture}, " +
                f"Optimization_Flag={CONST.FLAG_OPT}, MIPFOCUS={CONST.MIPFOCUS}, Baseline={args.baseline}" )
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
    
    if args.baseline == "zigzag":
        baseline_adapter = ZigzagBaselineAdapter(
            model=args.model,
            architecture=args.architecture,
            opt_flag=opt_flag,
        )
    elif args.baseline == "cosa":
        baseline_adapter = CoSABaselineAdapter(
            model=args.model,
            architecture=args.architecture,
            map_path=args.cosa_map,
            output_root=outFolder,
        )
    elif args.baseline == "cimloop":
        baseline_adapter = CimloopBaselineAdapter(
            model=args.model,
            architecture=args.architecture,
            map_path=args.cimloop_map,
            output_root=outFolder,
            macro=args.cimloop_macro,
            system=args.cimloop_system,
            tile=args.cimloop_tile,
            chip=args.cimloop_chip,
            iso=args.cimloop_iso,
            hardware_from_arch=args.cimloop_hw_from_arch,
        )
    else:
        raise ValueError(f"Unsupported baseline provider: {args.baseline}")

    assert len(convs) == len(loopdims)

    latency_base, energy_base, latency_mi, energy_mi = 0, 0, 0, 0

    accelerator_eval = make_accelerator(args.architecture)

    for i, (Conv, loopdim) in enumerate(zip(convs, loopdims)):
        # if i == 9 :
        #     pass
        # else:
        #     continue

        Logger.info('\n\n'+'* '*20+f"Layer {i}"+' *'*20)
        ops = WorkLoad(loopDim=loopdim)
        Logger.info(ops)

        newdim = normalize_loopdim_for_solver(loopdim)
        pstr, cache_flag = "", False

        cached = mip_cache_get(accelerator_eval, newdim, CONST.FLAG_OPT, CONST.TIMELIMIT, CONST.MIPFOCUS)
        if cached is not None:
            l_solver = cached["solver_latency"]
            e_solver = cached["solver_energy"]
            l_simu   = cached["simulator_latency"]
            e_simu   = cached["simulator_energy"]
            Logger.info("Get MIP Result From Persistent Cache")
            cache_flag = True

        if not cache_flag:
            outputdir_layer = os.path.join(outFolder,Conv)
            prepare_save_dir(outputdir_layer)
            Logger.changeFile(new_file = os.path.join(outputdir_layer,"Evaluation-Layer.log"), mode="w")
            Logger.info(ops)
            Logger.info('\n' + '* '*30 + '\n')

            try:
                baseline_layer = baseline_adapter.find_layer(loopdim)
            except ValueError as e:
                Logger.error(f"Failed to get baseline layer from {args.baseline}")
                Logger.changeFile(new_file = os.path.join(outFolder,args.log))
                Logger.error(e)
                continue
            cache_flag = False

            accelerator = copy.deepcopy(accelerator_eval)

            loops = LoopNest(acc=accelerator,ops=ops)
            try:
                loops = convert_baseline_to_MIREDO(loops=loops, baseline=baseline_layer)
            except ValueError as e:
                Logger.error("Baseline-to-MIREDO consistency check failed")
                Logger.changeFile(new_file = os.path.join(outFolder,args.log))
                Logger.error(e)
                continue
            loops.usr_defined_double_flag[accelerator.Macro2mem][1] = accelerator.double_Macro

            try:
                Logger.info(f"Running: {args.baseline}-in-MIREDO-Simulator - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                simu = tranSimulator(acc=accelerator, ops=ops, dataflow=loops)
                l_base, e_base = simu.run()
                PD_B = simu.PD
            else:
                Logger.info("Running: WS-Baseline - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                ws_result = generate_weight_stationary_baseline(
                    acc=accelerator,
                    ops=WorkLoad(loopDim=newdim),
                )
                loops = ws_result.dataflow
                l_base, e_base = ws_result.latency, ws_result.energy
                PD_B = ws_result.profile
                Logger.info(f"WS baseline policy: {ws_result.policy}")
        except ValueError as e:
            Logger.error('Wrong Match')
            Logger.changeFile(new_file = os.path.join(outFolder,args.log))
            Logger.error(e)
            continue

        if not cache_flag:
            accelerator = copy.deepcopy(accelerator_eval)

            match CONST.FLAG_OPT:
                case "Latency":
                    bestMetric = l_base
                case "Energy":
                    bestMetric = e_base
                case "EDP":
                    bestMetric = l_base * e_base * CONST.SCALINGFACTOR
                case _:
                    bestMetric = 1e9            # flagOPT = infeasible

            l_solver, e_solver, edp_solver, l_simu, e_simu, PD_M = SolveMapping(acc=accelerator, ops=WorkLoad(loopDim=newdim), bestMetric=bestMetric*2, outputdir=outputdir_layer)
            mip_cache_put(accelerator_eval, newdim, CONST.FLAG_OPT, CONST.TIMELIMIT, CONST.MIPFOCUS, {
                "solver_latency": l_solver, "solver_energy": e_solver,
                "solver_edp": l_solver * e_solver * CONST.SCALINGFACTOR,
                "simulator_latency": l_simu, "simulator_energy": e_simu,
                "simulator_profile": PD_M, "mapping_profile": None,
                "solver_loopdim": newdim, "dataflow": None,
            })
            
            pstr += "\n-------- MemHierarchy ----- DB_M -- DB_Z --- MIREDO/Baseline --- MIREDO - Baseline -----\n"
            for m in range(1,accelerator.Num_mem):
                dm, dz = PD_M.doubleFlag, PD_B.doubleFlag
                pstr += '-'*8 + f" {accelerator.mem2dict(m):<15}: [ {dm[m][0]} {dm[m][1]} {dm[m][2]} | {dz[m][0]} {dz[m][1]} {dz[m][2]} ]  "
                energy_ratio = '-'*8 if PD_B.memCost[m]==0 else round(PD_M.memCost[m]/PD_B.memCost[m],3)
                pstr += f"{energy_ratio:>8}x   ({PD_M.memCost[m]:.2e} / {PD_B.memCost[m]:.2e}) pJ" + '\n'
            pstr += '\n-------- Multiply & ADD : -----------------  '
            pstr += f"{round(PD_M.macEnergy/PD_B.macEnergy,3):>8}x   ({PD_M.macEnergy:.2e} / {PD_B.macEnergy:.2e}) pJ" + '\n\n'

            pstr += f"Dynamic Power: {round(PD_M.dynamic_power/PD_B.dynamic_power,3):>8}x   ({PD_M.dynamic_power:.2e} / {PD_B.dynamic_power:.2e}) pJ" + '\n'
            pstr += f"Leakage Power: {round(PD_M.leakage_power/PD_B.leakage_power,3):>8}x   ({PD_M.leakage_power:.2e} / {PD_B.leakage_power:.2e}) pJ" + '\n'
            pstr += f"On-chip Power: {round(PD_M.dynamic_power_onChip/PD_B.dynamic_power_onChip,3):>8}x   ({PD_M.dynamic_power_onChip:.2e} / {PD_B.dynamic_power_onChip:.2e}) pJ" + '\n'

        pstr += '\n'
        pstr += f"* * * Baseline-Running * * *  Latency:{round(l_zz,3):<15}, Energy:{round(e_zz,3):<20}, EDP:{round(l_zz *e_zz,3):.5e}" + '\n'
        pstr += f"* * * MIREDO-Running  * * *  Latency:{round(l_simu,3):<15}, Energy:{round(e_simu,3):<20}, EDP:{round(l_simu*e_simu,3):.5e}" + '\n'
        pstr += f"MIP Solver Latency Relative Error: {round(abs(l_solver-l_simu)/l_simu*100,2)}%    (Simu){round(l_simu,3):<15} (Solver){round(l_solver,3):<15} " + '\n'
        pstr += f"MIP Solver Energy  Relative Error: {round(abs(e_solver-e_simu)/e_simu*100,2)}%    (Simu){round(e_simu,3):<15} (Solver){round(e_solver,3):<15} " + '\n'

        rstr = f"Speedup Of Layer-{i}: Latency-({round(l_base/l_simu,3)}x), Energy-({round(e_base/e_simu,3)}x), EDP-({round((l_base*e_base)/(l_simu*e_simu),3)}x)"
        
        if cache_flag == False:
            Logger.info(pstr)
            Logger.critical(rstr)
            Logger.changeFile(new_file = os.path.join(outFolder,args.log))
        
        latency_mi += l_simu
        energy_mi  += e_simu
        latency_base += l_base
        energy_base  += e_base

        Logger.info(pstr)
        Logger.critical(rstr)

    # exit()
    Logger.info("* " * 50)
    Logger.info('\n\n'+'* '*20+f"The WHOLE Model"+' *'*20)
    Logger.info(f"* * * Baseline-Running * * *  Latency:{round(latency_zz,3):<15}, Energy:{round(energy_zz,3):<15}, EDP:{round(latency_zz * energy_zz,3):.5e}")
    Logger.info(f"* * * MIREDO-Running  * * *  Latency:{round(latency_mi,3):<15}, Energy:{round(energy_mi,3):<15}, EDP:{round(latency_mi * energy_mi,3):.5e}")

    if latency_zz > 0 and energy_zz > 0:
        opt_latency = round(latency_mi / latency_zz * 100, 2)
        opt_energy = round(energy_mi / energy_zz * 100, 2)
        opt_edp = round((latency_mi * energy_mi) / (latency_zz * energy_zz) * 100, 2)
        Logger.info(
            f"* * *  Optimization  * * *   "
            f"Latency:{opt_latency}%, Energy:{opt_energy}%, EDP:{opt_edp}%"
        )
    else:
        Logger.warning(
            "Optimization summary unavailable because baseline aggregate metric is zero. "
            "This usually means no valid baseline layer finished replay."
        )
    # exit()

        # res_energy Unit=nj    = 1e-9j  
        # res_latency Unit      = cycle
        # Power Unit            = mw
    # power = (res_e * CONST.SCALINGFACTOR * 1e-12 * 500 * 1e6 / res_l) * 1e3
    # Logger.debug(f"scaling factor: {CONST.SCALINGFACTOR}")
    end_time = time.time()
    Logger.critical(f"Solving The Whole Model Cost: {round(end_time - start_time,1)}s")

    Logger.recover_stdout()

if __name__ == "__main__":
    __main__()

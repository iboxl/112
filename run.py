# this file is prepared for project 511
# Created by iboxl

import torch
import torch.nn as nn
from utils.Tools import *
from Architecture.Accelerator import CIM_acc
from Architecture.ArchSpec import CIM_Acc
from Architecture.ZigzagAcc import accelerator as acc_zz
from utils.Workload import Operands, WorkLoad
from SolveMapping import SolveMapping
import argparse
from utils.GlobalUT import *
import pickle
import uuid

def get_Args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", nargs="?", const=True, default=False, help="Enable debug mode.")
    parser.add_argument("--logger", nargs="?", const=True, default=False, help="Just print Logger")        # W.T.D. exchange ture and false
    parser.add_argument("--srun", nargs="?", const=True, default=False, help="batch srun with critical message")
    parser.add_argument("--noLogFile", nargs="?", const=True, default=False, help="No log file")
    parser.add_argument("--EX", nargs="?", const=True, default=False, help="exchange input & weight")
    parser.add_argument("--BM", nargs="?", const=True, default=False, help="using blocking Dim_M")
    parser.add_argument("--RS", nargs="?", const=True, default=False, help="FLAG: Row traverse")
    parser.add_argument("--IS", nargs="?", const=True, default=False, help="FLAG: Buffer/Input stationary")
    parser.add_argument("--OS", nargs="?", const=True, default=False, help="FLAG: Rejust dim/block to avoid overSize")
    parser.add_argument("--GM", nargs="?", const=True, default=False, help="using blocking Dim_K with GAMMA")
    parser.add_argument("--NoPreSolve", nargs="?", const=True, default=False, help="dont search presolve by alpha&beta")
    parser.add_argument("--SIMU", nargs="?", const=True, default=False, help="using simulator calc")

    parser.add_argument('-c', '--cfg', dest='cfg', required=False, 
                        type=str, default='cim_template.cfg', help = 'config File Name')
    parser.add_argument('-m', '--model', dest='model', required=False, 
                        type=str, default='resnet18', help = 'NN model Name')
    parser.add_argument('-log', '--log_file', dest='log', required=False, 
                        type=str, default='419.log', help = 'Log file Name')
    parser.add_argument('-opt', '--flag_opt', dest='opt', choices=["Latency", "Energy", "EDP"], required=False, 
                        type=str, default="Feasible", help = 'kind of model optimization: 0=Feasible solution, 1=MIN_latency  2=MIN_energy  3=MIN_EDP')
    parser.add_argument('-class', '--num_classes', dest='classes', choices=[10, 1000], required=False, 
                        type=int, default=1000, help = '10=CIFAR 100=ImageNet')
    parser.add_argument('-t', '--time', dest='time_limit', required=False, 
                        type=int, default=CONST.TIMELIMIT, help = 'time limitation for solving gurobi model')
    parser.add_argument('-n', '--name', dest='dataflow_name', required=False, 
                        type=str, default='ds', help = 'save dataflow in FigName')
    args = parser.parse_args()

    return args
 
def __main__(**kwargs):

    args = get_Args()

    Logger.setcfg(setcritical=args.srun, setDebug=args.debug, STD=args.logger, file=args.log, nofile=args.noLogFile)
    model = get_Model(args.model, num_classes=args.classes)
    cfg = get_ConfigFile(args.cfg)
    # accelerator = CIM_acc(cfg)
    accelerator = CIM_Acc(acc_zz.cores[0])
    ds_name = args.dataflow_name
    CONST.FLAG_OPT              = args.opt
    CONST.TIMELIMIT             = args.time_limit
    FLAG.ROW_STATIONARY         = args.RS
    FLAG.OPS_EXCHANGE           = args.EX and (not args.RS)
    FLAG.BLOCK_M                = args.BM and (not args.RS)
    FLAG.INPUT_STATIONARY       = args.IS and (not args.RS)
    FLAG.OUTPUT_STATIONARY      = args.OS and (not args.RS)
    FLAG.GAMMA                  = args.GM and (not args.RS)
    FLAG.DEBUG_SIMU             = args.SIMU
    FLAG.PRESOLVE_SEARCH        = not args.NoPreSolve
    FLAG.DEBUG_PER_LAYER_DETAIL = False               # illegal Tmp setting

    Logger.info("* " * 50)
    Logger.info(f"config={args.cfg}, model={args.model}")
    Logger.info(f"flag_opt={args.opt}, Block_m={FLAG.BLOCK_M}, gamma={FLAG.GAMMA}, row_stationary={FLAG.ROW_STATIONARY}, "
                #   + f"output_stationary={FLAG.OUTPUT_STATIONARY}, "
                #   + f"ops_exchange={FLAG.OPS_EXCHANGE}, , output_stationary={FLAG.OUTPUT_STATIONARY}"
                  )
    Logger.info("* " * 50)

    if args.classes == 10:
        input_tensor = torch.randn(1, 3, 28, 28)
    else:
        input_tensor = torch.randn(1, 3, 224, 224)

    set_hook(model)
    with torch.no_grad():
        output = model(input_tensor)

    # debug_get_im2col_info(FLAG_DEBUG=True)
    # exit()

    res_l_eachLayer, res_e_eachLayer, res_p_eachLayer = [[0 for _ in range(len(conv_im2col_info))]for __ in range(3)]
    cal_l, cal_e = 0, 0
    cache = {}
    dataflow = []

    for idx, (layer_name, info) in enumerate(conv_im2col_info.items()):
        ops = WorkLoad(cfg, loopDim={'R': info['R'], 'S': info['S'], 'C': info['C'], 'K': info['K'], 
                                      'P': info['P'], 'Q': info['Q'], 'G': info['G'], 'B': info['B'],
                                      'H': info['H'], 'W': info['W'], 'Stride': info['Stride'], 'Padding': info['Padding']}, 
                        min_factor=2, max_factor=7)
        # Logger.critical(ops)
        # continue

        r,s,c,k,p,q,g = [info['R'], info['S'], info['C'], info['K'], info['P'], info['Q'], info['G']]
        Logger.info('\n'+'* '*20+f"Layer {idx}"+' *'*20)
        if (r,s,c,k,p,q,g) in cache:
            Logger.debug(f"Get Cost Result From Cache")
            [res_latency, res_energy, res_edp, res_cal_l, res_cal_e, res_ds] = cache[(r,s,c,k,p,q,g)]
        else:
            lat,eng,edp,c_lat,c_eng,ds = SolveMapping(acc=accelerator, ops=ops)
            res_latency, res_energy, res_edp, res_cal_l, res_cal_e, res_ds = [lat, eng, edp, c_lat, c_eng, ds]
            cache[(r,s,c,k,p,q,g)] = [res_latency, res_energy, res_edp, res_cal_l, res_cal_e, res_ds]
        dataflow.append(res_ds)
            
        Logger.info(f"Layer {idx}: latency = {int(res_latency)}, energy = {round(res_energy, 1)}, EDP = {round(res_edp, 1)}")
        if FLAG.DEBUG_SIMU:
            Logger.info(f"Layer {idx}: cal_lat = {int(res_cal_l)}, cal_en = {round(res_cal_e, 1)}," + 
                        f" c_P = {round(res_cal_l*res_cal_e/CONST.SCALINGFACTOR, 1)}")
        res_l_eachLayer[idx] = res_latency
        res_e_eachLayer[idx] = res_energy
        res_p_eachLayer[idx] = res_edp
        cal_l += res_cal_l
        cal_e += res_cal_e
    res_l = sum(res_l_eachLayer)
    res_e = sum(res_e_eachLayer)
    res_p = res_l * res_e
                                                # res_energy Unit=nj    = 1e-9j  
                                                # res_latency Unit      = cycle
                                                # Power Unit            = mw
    power = (res_e * CONST.SCALINGFACTOR * 1e-12 * 500 * 1e6 / res_l) * 1e3
    Logger.debug(f"scaling factor: {CONST.SCALINGFACTOR}")
    Logger.critical(f"# # # # # MIP latency={round(res_l,1)}, energy={round(res_e,3)}, EDP={round(res_p,3)}")
    if FLAG.DEBUG_SIMU:
        Logger.critical(f"# # # # # SIM latency={round(cal_l,1)}, energy={round(cal_e,3)}, EDP={round(cal_l*cal_e,3)}")
    Logger.critical(f"# # # # # power={round(power, 3)}mw"+'\n')
    
    # 将列表保存到文件
    if ds_name != 'ds':
        # file_name = 'log/dataflow/' + ds_name + f"_{uuid.uuid4()}.pkl"
        file_name = 'log/dataflow/' + ds_name + ".pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(dataflow, file)
    

    Logger.recover_stdout()



__main__()
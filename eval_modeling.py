# this file is prepared for project 419
# Created by iboxl

import itertools
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad, LoopNest, Mapping
from utils.Tools import *
import gurobipy as gp
from gurobipy import GRB, quicksum
from utils.SolverSP import Solver as Solver
from Simulator.Calculator import calc_ds
from utils.GlobalUT import *
from Architecture.ZigzagAcc import accelerator as acc_zz
from Simulator.Simulax import tranSimulator
from utils.CompatibleZigzag import convert_zz_to_miredo


if __name__ == "__main__":
    Logger.setcfg(setcritical=False, setDebug=False, STD=False, file='511.log', nofile=False)

    import logging
    CONST.FLAG_OPT="Latency"
    Logger.setLevel(logging.DEBUG)
    Logger.info(f'Evaluation in simulator:') 


    acc = CIM_Acc(acc_zz.cores[0])
    #  ['Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7]

    ops = WorkLoad(None, loopDim={'R': 3, 'S': 3, 'C': 64, 'K':64, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1},
                        min_factor=3, max_factor=7)
    temporal_mapping_dic_stationary = \
                    {'O': [[], [('OY', 56), ('FY', 3), ('FX', 3), ('C', 2)], [], [('OX', 7), ('K', 4)]], 
                    'W': [[('OY', 56), ('FY', 3)], [('FX', 3), ('C', 2), ('OX', 7), ('K', 4)], []], 
                    'I': [[], [('OY', 56), ('FY', 3)], [('FX', 3), ('C', 2), ('OX', 7), ('K', 4)], []]}
    spatial_mapping_dict = {'I':[  [('K', 16.0)],  [('C', 32.0)],  [('OX', 8.0)],  [],  [],]          , 
                            'W': [  [],  [('C', 32.0), ('K', 16.0), ('OX', 8.0)],  [],  [],]           ,   
                            'O': [  [('C', 32.0)],  [('K', 16.0)],  [('OX', 8.0)],  [],  [],]    }
    double_buffer_flag = {'O': [False, False, True, True, True], 'W': [False, True, True, True], 'I': [False, False, False, False, True]}
    top_r_loop_size = {'O': [1, 1, 1, 1, 28], 'W': [1, 3, 4, 1], 'I': [1, 1, 1, 1, 1]}

    # ops = WorkLoad(None, loopDim={'R': 3, 'S': 3, 'C': 128, 'K':128, 'P': 28, 'Q': 28, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 1, 'Padding': 1},
    #                min_factor=3, max_factor=7)
    # temporal_mapping_dic_stationary = \
    #                 {'O':  [[], [('OX', 28), ('FX', 3)], [('OY', 28), ('FY', 3), ('C', 2), ('C', 2)], []], 
    #                 'W':  [[('OX', 28), ('FX', 3), ('OY', 28)], [], [('FY', 3), ('C', 2), ('C', 2)]] , 
    #                 'I': [[], [('OX', 28), ('FX', 3)], [('OY', 28), ('FY', 3)], [('C', 2), ('C', 2)]]}
    # spatial_mapping_dict = {'I':[  [('K', 16)],  [('C', 32.0)],  [('K', 8.0)],  [],  [],]         , 
    #                         'W': [   [],  [('K', 128.0), ('C', 32.0)],  [],  [],]          ,   
    #                         'O': [   [('C', 32.0)],  [('K', 16)],  [('K', 8.0)],  [],  [],]   }
    # double_buffer_flag = {'O': [False, False, True, True, True], 'W': [False, True, True, True], 'I': [False, False, True, False, True]}
    # top_r_loop_size = {'O': [1, 1, 1, 1, 1], 'W': [1, 1, 1, 12], 'I': [1, 1, 1, 1, 4]}              

    # ops = WorkLoad(1, loopDim={'R': 1, 'S': 1, 'C': 128, 'K':256, 'P': 14, 'Q': 14, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 2, 'Padding': 0},
                        # min_factor=3, max_factor=7)
                   
    print(ops)
    loops = LoopNest(acc=acc,ops=ops)

    # dataflow = {}
    # dataflow['temporal_mapping']   = [
    # # [ops.dict2Dim('R'), stride, [I_memory,W_memory,O_memory]],
    # # loop[i][2][t]
    #     [ops.dict2Dim('K'), 4,  [2,2,1]],
    #     [ops.dict2Dim('P'), 7,  [2,2,1]],
    #     [ops.dict2Dim('C'), 2,  [2,2,3]],
    #     [ops.dict2Dim('R'), 3,  [2,2,3]],
    #     [ops.dict2Dim('S'), 3,  [4,7,3]],
    #     [ops.dict2Dim('Q'), 56,   [4,7,3]],
    # ]
    # print(cme.temporal_mapping.mapping_dic_stationary)
    # for tm in dataflow['temporal_mapping']:
    #     loops.tm.append(Mapping(tm[0],tm[1],[tm[2][0],tm[2][1],tm[2][2]])) 
    

    # dataflow['spatial_mapping']    = [
    # # sp_dim[mem][t][dim]   ['R', 'S', 'P', 'Q', 'C', 'K']
    #     [['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1]],# -lacehold
    #     [['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1]],
    #     [['-', 1, 1, 8, 1, 1, 1], ['-', 1, 1, 8, 1, 1, 1], ['-', 1, 1, 8, 1, 1, 1]],
    #     [['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1]],
    #     [['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1]],
    #     [['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1], ['-',1, 1, 1, 1, 32, 16]],
    #     [['-', 1, 1, 1, 1, 32, 16], ['-', 1, 1, 1, 1, 1, 1], ['-', 1, 1, 1, 1, 1, 1]],
    #     [['-', 1, 1, 1, 1, 1, 1], ['-',1, 1, 1, 1, 32, 16], ['-', 1, 1, 1, 1, 1, 1]]
    # ]

    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    loops = convert_zz_to_miredo(loops=loops,temporal_mapping_dic=temporal_mapping_dic_stationary, spatial_mapping_dict=spatial_mapping_dict,
                                 double_buffer_flag=double_buffer_flag, top_r_loop_size=top_r_loop_size)
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro

    simu = tranSimulator(acc=acc, ops=ops, dataflow=loops)
    Logger.debug(simu.debugLog())

    # dflag = loops.usr_defined_double_flag
    # for mem in range(1,acc.Num_mem):
    #     print(f"{acc.mem2dict(mem):<13}: I({dflag[mem][0]}), W({dflag[mem][1]}), O({dflag[mem][2]})")

    simu.run()

    simu.LModeling()
    

    
# this file is prepared for project 511
# Created by iboxl

import itertools
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad
from utils.Tools import *
import gurobipy as gp
from gurobipy import GRB, quicksum
from utils.SolverSP import Solver as Solver
from Simulator.Calculator import calc_ds
from utils.GlobalUT import *
import numpy as np
from Architecture.ZigzagAcc import accelerator as acc_zz
   

def checkDataflow(acc:CIM_Acc, ops:WorkLoad, df):
    if len(df['spatial_mapping']) != acc.Num_mem:
        Logger.error(f"Dismatch Spatial_mapping")
        raise ValueError("Error in checkDataflow")
    
    if len(df['double_tag']) != acc.Num_mem+1:
        Logger.error(f"Dismatch Double_tag")
        raise ValueError("Error in checkDataflow")
    

def convert_dflag_to_doubletag(acc:CIM_Acc, ops:WorkLoad, dflag):
    # 定义固定的操作数顺序
    fixed_order = ['I', 'W', 'O']
    num_operations = len(fixed_order)
    
    # 检查 dflag 和 mappingArray 的一致性
    for op_type in fixed_order:
        if op_type in dflag:
            op_index = fixed_order.index(op_type)
            allowed_levels = sum(1 for level in range(acc.Num_mem) if acc.mappingArray[op_index][level] == 1)
            if allowed_levels != len(dflag[op_type]):
                raise ValueError(f"Mismatch in levels count for operation '{op_type}': expected {allowed_levels}, got {len(dflag[op_type])}")
    
    # 初始化 double_tag 矩阵为全 0，大小为 (memory_levels, num_operations)
    double_tag = np.zeros((acc.Num_mem, num_operations), dtype=int)
    # 填充 double_tag 矩阵
    for op_type in fixed_order:
        if op_type in dflag:
            op_index = fixed_order.index(op_type)
            levels = dflag[op_type][::-1]  # 将 dflag 顺序颠倒，从最后一个有效映射开始
            level_counter = 0
            for level_index in range(acc.Num_mem):
                if acc.mappingArray[op_index][level_index] == 1:
                    double_tag[level_index][op_index] = int(levels[level_counter])
                    level_counter += 1
    # for t in range(num_operations):
    #     double_tag[acc.Num_mem,t] = 0
    macro_dflag = [ 0 for _ in range(num_operations)]

    # 使用 vstack 将新行添加到末尾
    double_tag = np.vstack([double_tag, macro_dflag])
    return double_tag


if __name__ == "__main__":
    Logger.setcfg(setcritical=False, setDebug=False, STD=False, file='511.log', nofile=False)

    import logging
    CONST.FLAG_OPT="Latency"
    Logger.setLevel(logging.DEBUG)
    cfg = get_ConfigFile('cim_template.cfg')

    Logger.info(f'Evaluation in simulator:') 


    acc = CIM_Acc(acc_zz.cores[0])
    cfg = get_ConfigFile('cim_template.cfg')
    # ops = WorkLoad(cfg, loopDim={'R': 3, 'S': 3, 'C': 128, 'K':128, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1},
    ops = WorkLoad(cfg, loopDim={'R': 1, 'S': 1, 'C': 128, 'K':256, 'P': 14, 'Q': 14, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 2, 'Padding': 0},
                   min_factor=3, max_factor=7)
    Logger.info(f'R:{ops.R}, S:{ops.S}, C:{ops.C}, K:{ops.K}, P:{ops.P}, Q:{ops.Q}, H:{ops.H}, W:{ops.W}, G:{ops.G}, B:{ops.B}, Stride:{ops.Stride}, Padding:{ops.Padding}')

    #  ['Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7]
    dataflow = {}
    # acc.memSize[4] *= 2
    dataflow['temporal_mapping']   = [
      # [ops.dict2Dim('R'), stride, [I_memory,W_memory,O_memory]],
      # loop[i][2][t]

        [ops.dict2Dim('C'), 2,  [1,1,2]],
        [ops.dict2Dim('C'), 2,  [1,1,2]],
        [ops.dict2Dim('S'), 3,  [2,1,2]],
        [ops.dict2Dim('Q'), 28,  [2,7,2]],
        [ops.dict2Dim('P'), 28,  [4,7,3]],
        [ops.dict2Dim('R'), 3,   [4,7,5]],
    ]
    dataflow['spatial_mapping']    = [
    # sp_dim[mem][t][dim]   ['R', 'S', 'P', 'Q', 'C', 'K']
        [['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1]],# placehold
        [['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1]],
        [['P', 1, 1, 1, 1, 1, 8], ['P', 1, 1, 1, 1, 1, 8], ['P', 1, 1, 1, 1, 1, 8]],
        [['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1]],
        [['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1]],
        [['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1], ['P',1, 1, 1, 1, 32, 16]],
        [['P', 1, 1, 1, 1, 32, 16], ['P', 1, 1, 1, 1, 1, 1], ['P', 1, 1, 1, 1, 1, 1]],
        [['P', 1, 1, 1, 1, 1, 1], ['P',1, 1, 1, 1, 32, 16], ['P', 1, 1, 1, 1, 1, 1]]

    ]
    
    dataflow['double_tag'] = [
        # double_tag[mem][t]
        [],
        [],
        []
    ]
    
    double_buffer_flag = {'O': [False, True, True, True], 'W': [ False, True, True], 'I': [False, True, False, True]}
     
    dataflow['double_tag'] = convert_dflag_to_doubletag(acc,ops,double_buffer_flag)
    print(dataflow['double_tag'])
    # exit()


    checkDataflow(acc=acc, ops=ops, df=dataflow)

    simu_latency, simu_energy = calc_ds(acc=acc, ops=ops, dataflow=dataflow)

    Logger.critical(f"latency={round(simu_latency)}, Energy={round(simu_energy,2)}, EDP={simu_latency * simu_energy / 1e6}")

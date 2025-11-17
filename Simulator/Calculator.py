#  this file is prepared for project 026
#  Created by iboxl

#  Debug dataflow privately ONLY

from utils.Workload import WorkLoad
from Architecture.ArchSpec import CIM_Acc
import math
from utils.GlobalUT import *

def calc_ds(acc:CIM_Acc, ops:WorkLoad, dataflow):
    Logger.debug("# # # inside SIMU Calc: # # #")
    loop = dataflow['temporal_mapping']     # [R,3,[dram,gb,reg]]  [0,3,[0,1,2]]
    spu = dataflow['spatial_mapping']       # sp_dim[mem][t][f] = dim 
    double_tag = dataflow['double_tag']     # double_tag[mem][t]

    scatter_size = {}
    multicast_size = {}
    for i in range(len(loop)):
        print(f"{i}: {ops.dim2Dict[loop[i][0]]}={loop[i][1]:<3}",end="  ")
        for t in range(3):
            print(f"{acc.mem2dict(loop[i][2][t]):>17}|{double_tag[loop[i][2][t]][t]}", end="")
        print("")

    for mem in range(1,acc.Num_mem):
        for t, t_name in enumerate(['I','W','O']):
            scatter_size[mem,t] = 1 * acc.mappingArray[t][mem]
            multicast_size[mem,t] = 1
            if acc.mappingArray[t][mem] == 0:
                continue
            for dim in range(1,ops.Num_dim):
                if spu[mem][t][dim] > 0:
                    if ops.relevance[t][dim] == 1:
                        scatter_size[mem,t] *= spu[mem][t][dim]
                    else:
                        multicast_size[mem,t] *= spu[mem][t][dim]


    # dim_tp[mem][t][dim]   
    dim_tp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
    tileNum = [ [ 1 for t in range(3) ] for mem in range(acc.Num_mem)]
    for i in range(len(loop)):
        for t in range(3):
            for mem in range(1,acc.Num_mem):
                if mem <= loop[i][2][t] and acc.mappingArray[t][mem]==1:
                    dim_tp[mem][t][loop[i][0]] *= loop[i][1] 
                if mem >= loop[i][2][t] and acc.mappingArray[t][mem]==1:
                    tileNum[mem][t] *= loop[i][1]
                if acc.mappingArray[t][mem]==0:
                    tileNum[mem][t] = 0
    
    # dim_sp[mem][t][dim]
    dim_sp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
    for dim in range(1,ops.Num_dim):
        for mem in range(1,acc.Num_mem):
            for t in range(3):
                for m2 in range(1,acc.Num_mem):
                    if mem <= m2 and acc.mappingArray[t][mem]==1:
                        if spu[m2][t][dim] > 0:
                            dim_sp[mem][t][dim] *= spu[m2][t][dim]


 
    memSize = {}
    tmp_dim = [_ for _ in range(ops.Num_dim)]
    for t in range(3):
        memSize[acc.Num_mem, t] = 1
    for mem in range(1,acc.Num_mem):
        t = 0  # Input
        for dim in range(1,ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][t][dim] * dim_tp[mem][t][dim]
        if acc.mappingArray[t][mem]:
            memSize[mem,t] = (tmp_dim[ops.dict2Dim('P')]+tmp_dim[ops.dict2Dim('R')]-1) * (tmp_dim[ops.dict2Dim('Q')]+tmp_dim[ops.dict2Dim('S')]-1) * tmp_dim[ops.dict2Dim('C')]
        else:
            memSize[mem,t] = 0

        t = 1  # Weight
        for dim in range(1,ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][t][dim] * dim_tp[mem][t][dim]
        if acc.mappingArray[t][mem]:
            memSize[mem,t] = tmp_dim[ops.dict2Dim('R')] * tmp_dim[ops.dict2Dim('S')] * tmp_dim[ops.dict2Dim('C')] * tmp_dim[ops.dict2Dim('K')] 
        else:
            memSize[mem,t] = 0

        t = 2  # Output
        for dim in range(1,ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][t][dim] * dim_tp[mem][t][dim]
        if acc.mappingArray[t][mem]:
            memSize[mem,t] = tmp_dim[ops.dict2Dim('P')] * tmp_dim[ops.dict2Dim('Q')] * tmp_dim[ops.dict2Dim('K')] 
        else:
            memSize[mem,t] = 0

    print(f"MemSize:")
    for mem in range(1,acc.Num_mem):
        print(f"{acc.mem2dict(mem):<15}:", end="")
        for t in range(3):
            print(f"{memSize[mem,t]:>6}, {multicast_size[mem,t]}|{scatter_size[mem,t]}", end="   ")
        print("")


    for mem in range(1,acc.Num_mem):
        sum_mem = 0
        for t, t_name in enumerate(['I','W','O']):
            if acc.mappingArray[t][mem]:
                sum_mem += (memSize[mem,t] + memSize[mem,t] * double_tag[mem][t]) * acc.precision[mem,t]
        if sum_mem > acc.memSize[mem]:
            raise ValueError(f"Error #overSize in {acc.mem2dict(mem)}:{acc.memSize[mem]} less than {sum_mem}"
                            +f"\nop I:{memSize[mem,0]*acc.precision[mem,0]} "
                             + f"op W:{memSize[mem,1]*acc.precision[mem,1]}  "
                             + f"op O:{memSize[mem,2]*acc.precision[mem,2]}  ")
            exit()
        else:
            Logger.info(f"Utilization {acc.mem2dict(mem)}({round(sum_mem/acc.memSize[mem]*100,2):>3}%): {sum_mem}/{acc.memSize[mem]}")

    Latency = acc.t_MAC
    stall = {}
    for t, t_name in enumerate(['I','W','O']):
        stall[t] = 0
    for i in range(len(loop)-1,-1,-1):
        for t, t_name in enumerate(['I','W','O']):
            tag_legal = True
            if i == len(loop)-1:
                xMem = True
            else:
                if loop[i][2][t] == loop[i+1][2][t]:
                    xMem = False
                else:
                    xMem = True 
            if ops.relevance[t][loop[i][0]] == 0 and xMem:
                for j in range(i-1,-1,-1):
                    if loop[j][2][t] == loop[i][2][t] and ops.relevance[t][loop[j][0]] == 0:
                        continue
                    elif loop[j][2][t] == loop[i][2][t] and ops.relevance[t][loop[j][0]] == 1:
                        tag_legal = False
                        break
                    else:
                        break
            if tag_legal == False:
                continue
            transTime = math.ceil((memSize[acc.nxtMem[t][loop[i][2][t]],t] * scatter_size[loop[i][2][t],t] * acc.precision[loop[i][2][t],t]) / acc.bw[loop[i][2][t]])
            transCost = max(Latency, transTime) if double_tag[acc.nxtMem[t][loop[i][2][t]]][t] else (Latency+transTime)
            stall[t] = transCost if xMem else Latency
            print(f"in loop({i}) tensor_{t_name}: xMem={xMem}, dflag={double_tag[acc.nxtMem[t][loop[i][2][t]]][t]}, transTime={transTime}, stall={stall[t]}")
        Latency = max(stall[0], stall[1], stall[2]) * loop[i][1]
        Logger.info(f"Latency in loop_{i}: {Latency}")

    res_l = Latency

    # v_transEnergy[mem,t] = v_tileNum[t,mem]*v_memSize[acc.nxtMem[t][mem],t]*v_Sp_scatter[t,mem]*\
                    #                         (acc.cost_r[mem]+acc.cost_w[acc.nxtMem[t][mem]]*v_Sp_multicast[t,mem])+\
                    #                         v_transEnergy[acc.nxtMem[t][mem],t]*v_Sp_scatter[t,mem]*v_Sp_multicast[t,mem]
    v_transEnergy = {}
    for mem in range(1, acc.Num_mem):
        for t, t_name in enumerate(['I','W','O']):
            if mem in [acc.IReg2mem, acc.OReg2mem, acc.Macro2mem] or acc.mappingArray[t][mem] == 0:
                v_transEnergy[mem,t] = 0
            else:
                v_transEnergy[mem,t] = tileNum[mem][t]*memSize[acc.nxtMem[t][mem],t]*scatter_size[mem,t]*\
                                            (acc.cost_r[mem]+acc.cost_w[acc.nxtMem[t][mem]]*multicast_size[mem,t]) * acc.precision[mem,t]
    
    for t, t_name in enumerate(['I','W','O']):
        print(f"{t_name}:")
        for mem in range(1, acc.Num_mem):
            if acc.mappingArray[t][mem] == 0:
                continue
            print(f"{acc.mem2dict(mem)}: {v_transEnergy[mem,t]}")


    v_mac_CIM_once = scatter_size[acc.IReg2mem,0] * scatter_size[acc.OReg2mem,2]
    v_mac_num = round(ops.Num_MAC / v_mac_CIM_once)
    mac_E = acc.cost_ActMacro * v_mac_num

                
            
    trans_E = 0
    for mem in range(1, acc.Num_mem):
        for t, t_name in enumerate(['I','W','O']):
            trans_E += v_transEnergy[mem,t] * acc.Num_instance[mem]

    leakage_E = res_l * acc.leakage_per_cycle
    res_e = (trans_E + mac_E + leakage_E) / 1e3    

    Logger.debug(f"trans_E:{trans_E}, mac_E:{mac_E}, leakage_E:{leakage_E}")


    if FLAG.DEBUG_SIMU:                    #FLAG.DEBUG_SIMU == True:
        print("MemoryHierc|double_tag")
        for i in range(len(loop)):
            print(f"{ops.dim2Dict[loop[i][0]]}: {loop[i][1]}     ", end="")
            for t in range(3):
                print(f"{acc.mem2dict(loop[i][2][t]):>17}|{int(double_tag[loop[i][2][t]][t])} ",end="" )
            print("")
        
        
        print("\n tileNum")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}:")
            for t, t_name in enumerate(['I','W','O']):
                print(f"{t_name}: {tileNum[mem][t]:<6}", end=" ")
            print("")

        print("\n dim_tp")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}:")
            for t, t_name in enumerate(['I','W','O']):
                print(f"{t_name}:", end="")
                for dim in range(1,ops.Num_dim):
                    print(f"{ops.dim2Dict[dim]}:{dim_tp[mem][t][dim]} ",end="")
                print("")

        print("\n dim_sp")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}:")
            for t, t_name in enumerate(['I','W','O']):
                print(f"{t_name}:", end="")
                for dim in range(1,ops.Num_dim):
                    print(f"{ops.dim2Dict[dim]}:{dim_sp[mem][t][dim]} ",end="")
                print("")

        print("TileNum|double_tag")
        for i in range(len(loop)):
            print(f"{ops.dim2Dict[loop[i][0]]}: {loop[i][1]}     ", end="")
            for t in range(3):
                print(f"{tileNum[loop[i][2][t],t]:>17}|{int(double_tag[loop[i][2][t]][t])} ",end="" )
            print("\n")

        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}: ")
            for t, t_name in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem] == 0:
                    continue
                print(f"{t_name}: ",end="")
                for dim in range(1,ops.Num_dim):
                    print(f"{ops.dim2Dict[dim]}:{spu[mem][t][dim]} ",end="")
                print(f"Scatter_diff:{scatter_size[mem,t]:<2} multi_same:{multicast_size[mem,t]:<2} memSize:{memSize[mem,t]:<7} bandwidth:{acc.bw[mem]}")
        

    


    utilize_bandwidth = 0               # 冗余的数据传输造成的偏差？ 如何仅统计有效传输
    utilize_macro = 0
    Logger.info(f"Simu-Latency: {res_l}, Energy: {res_e}, EDP: {res_l * res_e / 1e6}")

    return res_l, res_e 
# this file is prepared for project 511
# Created by iboxl

import os
import psutil
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad
import math
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum, min_
from utils.GlobalUT import *
from utils.UtilsFunction.SolverFunction import *
from utils.UtilsFunction.CostFunction import _Cost_model

class Solver():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad):
        self.acc = acc
        self.ops = ops

        self.model = gp.Model()
        self.model.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
        self.model.setParam('NonConvex', 2)
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('Cuts', 2)
        # self.model.setParam('Threads', os.cpu_count()*0.75)
        self.model.setParam('Threads', psutil.cpu_count(logical=False))
        # self.model.setParam('Threads', 1)
        self.model.setParam('FeasibilityTol', 1e-3)                  # 降低容忍度容易导致求解失败
        self.model.setParam('IntFeasTol', 1e-3)                      # 通过SIMU避免最终结果的差异
        self.model.setParam('Presolve', 2)
        # self.model.setParam('DualReductions', 0)          # dont
        self.model.setParam('ImproveStartTime', CONST.IMPROVESTART)
        self.model.setParam('Heuristics', 0.5)
        # self.model.setParam('BranchDir', -1)
        # self.model.setParam('VarBranch', 1)  
        self.model.setParam('PreQLinearize', 2)
        # self.model.setParam("ScaleFlag", 2)

        # self.model.setParam("FuncMaxVal", 1e4)
        self.model.setParam("ScaleFlag", 2)
        # self.model.setParam("Heuristics", 0.3)
        # self.model.setParam("Presolve", 2)
        self.model.setParam("NumericFocus", 2)
        # model.setParam('MIPGap', 0.025)
        # self.model.setParam("FeasibilityTol", 3e-3)

        # 设置较严格的容差，确保约束生效
        # self.model.setParam('FeasibilityTol', 1e-9)
        # self.model.setParam('IntFeasTol', 1e-9)
        # self.model.setParam('OptimalityTol', 1e-9)


        self.result = {}
        self.dataflow = {}

    def run(self):
        Logger.debug("Start Running MIP Solver")
        m = self.model
        COST = _Cost_model(acc=self.acc, model=self.model, ops=self.ops)
        acc:CIM_Acc = self.acc
        ops:WorkLoad = self.ops
        Num_factors = sum(len(factors) for factors in ops.Factors)
        Num_loops = Num_factors
        MAX_FACTOR = max([item for sublist in ops.Factors for item in sublist])
        MAX_FANOUT = max(acc.fanout)
        MAX_DIM    = max(ops.dim2bound)

        Logger.critical(f"ops.Factors: {ops.Factors}")

        ###########################################################  Variable & Constant & Constraints  ##################################################################

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        indic_TempFactor = gp.tupledict()       # indic_TempFactor[f,p]
        onehot_SpUr2mem = gp.tupledict()      # onehot_SpUr2mem[f,p,t,mem]
        for f, fs in enumerate(ops.Factors):
            if f == 0:
                continue
            for p in range(len(fs)):
                indic_TempFactor[f,p] = m.addVar(vtype=GRB.BINARY, name=f"indicator_TempFactor_({f},{p})")
                for t, t_name in enumerate(['I','W','O']):
                    for mem in range(1, acc.Num_mem):
                        if acc.mappingArray[t][mem] == 0:
                            onehot_SpUr2mem[f,p,t,mem] = 0
                        elif acc.fanout[mem] == 1:
                            onehot_SpUr2mem[f,p,t,mem] = 0
                        else:
                            onehot_SpUr2mem[f,p,t,mem] = m.addVar(vtype=GRB.BINARY, name=f"onehot_SpatialUnroll_2mem_({f},{p},{t})_{mem}")
                    m.addConstr( quicksum(onehot_SpUr2mem[f,p,t,mem] for mem in range(1, acc.Num_mem)) == 1 - indic_TempFactor[f,p], 
                                    name=f"C_factor2_temporal_or_spatial_({f},{p},{t})" )
                    # for mem in range(1, acc.Num_mem):
                    #     m.addConstr(onehot_SpUr2mem[f,p,t,mem] <= acc.mappingArray[t][mem], name=f"sp_unroll_mapping_legal_({f},{p},{t_name}_{mem})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_factors = gp.tupledict()              # v_factors[i,f,p] = [0, 1]
        for f, fs in enumerate(ops.Factors):
            if f == 0:
                continue
            for p in range(len(fs)):
                for i in range(Num_factors):
                    # one-hot for permutation of Dimension_factors
                    v_factors[i,f,p] = m.addVar(vtype=GRB.BINARY, name=f"v_factors_({i},{f},{p})")
                # 每个factor只能出现一次
                m.addConstr( quicksum(v_factors[i,f,p] for i in range(Num_factors)) == 1, name=f"C_factor_Uniqueness_({f},{p})" )
        for i in range(Num_factors):
            #每个idx只能有一个factor
            m.addConstr( quicksum(v_factors[i,f,p] for f, fs in enumerate(ops.Factors) if f>0 for p in range(len(fs))) == 1, name=f"C_idx_Uniqueness_({i})" )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_loop = gp.tupledict()                     # v_loop[3] = 2 每层的loop factor
        for i in range(Num_factors):
            v_loop[i] = m.addVar(lb=0, ub=MAX_FACTOR, vtype=GRB.INTEGER, name=f"v_loop_{i}")
            m.addConstr( v_loop[i] == quicksum(v_factors[i,f,p] * ops.Factors[f][p] for f, fs in enumerate(ops.Factors) if f>0 for p in range(len(fs))), name=f"C_loop_{i}" )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        indic_TempLoop = gp.tupledict()         # indic_TempLoop[i]
        for i in range(Num_factors):
            indic_TempLoop[i] = m.addVar(vtype=GRB.BINARY, name=f"indicator_TempLoop_({i})")
            # m.addConstr( indic_TempLoop[i] == quicksum( indic_TempFactor[f,p]*v_factors[i,f,p] for f, fs in enumerate(ops.Factors) if f>0 for p in range(len(fs)) ))
            for f, fs in enumerate(ops.Factors):
                if f == 0:
                    continue
                for p in range(len(fs)):
                    m.addGenConstrIndicator(v_factors[i,f,p], True, indic_TempLoop[i] == indic_TempFactor[f,p], name=f"C_indicator_TempLoop_({i},{f},{p})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - -  - - - - - - - - - - - - - - -# 
        # loop_idx 和 dimensions 对应关系
        onehot_loop2dim = gp.tupledict()      # onehot_loop2dim[i,dim] = [0,1]
        for i in range(Num_factors):
            for dim in range(1,ops.Num_dim):
                onehot_loop2dim[i,dim] = m.addVar(vtype=GRB.BINARY, name=f"onehot_loop2dim_({i},{dim})")
            m.addConstr( quicksum(onehot_loop2dim[i,dim] for dim in range(1,ops.Num_dim)) == indic_TempLoop[i], name=f"C_loop2dim_onehot_define_{i}")
            for f, fs in enumerate(ops.Factors):
                if f == 0:
                    continue
                m.addConstr( onehot_loop2dim[i,f] == quicksum(v_factors[i,f,p] * indic_TempFactor[f,p] for p in range(len(fs))))
        # loop_idx 和 dimensions 对应关系
        v_loop2dim = gp.tupledict()             # v_loop2dim[i] = [0,Num_dim-1]
        for i in range(Num_factors):
            v_loop2dim[i] = m.addVar(lb=0, ub=ops.Num_dim, vtype=GRB.INTEGER, name=f"v_loop_{i}_2dim")
            m.addConstr( v_loop2dim[i] == quicksum(onehot_loop2dim[i,dim] * dim for dim in range(1,ops.Num_dim)), name=f"C_loop2dim_{i}" )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        # loop_idx 和 memory hierarchy 对应关系
        onehot_loop2mem = gp.tupledict()      # onehot_loop2mem[i,t,mem] = [0,1]
        for i in range(Num_factors):
            for t, t_name in enumerate(['I','W','O']):
                for mem in range(1,acc.Num_mem):
                    if acc.mappingArray[t][mem] == 1:
                        onehot_loop2mem[i,t,mem] = m.addVar(vtype=GRB.BINARY, name=f"onehot_loop2mem_({i},{t_name}_{mem})")
                    else:
                        onehot_loop2mem[i,t,mem] = 0
                        # m.addConstr( onehot_loop2mem[i,t,mem] <= acc.mappingArray[t][mem], name=f"C_onehot_loop2mem_mappingArray_({i},{t_name}_{mem})" )
                m.addConstr( quicksum(onehot_loop2mem[i,t,mem] for mem in range(1,acc.Num_mem)) == indic_TempLoop[i], name=f"C_loop2mem_Uniqueness_({i},{t})" )
        # for t, t_name in enumerate(['I','W','O']):
        #     for mem in range(1,acc.Num_mem):   
        #         if acc.nxtMem[t][mem] != acc.Num_mem:
        #             if acc.mappingArray[t][mem] == 0 or acc.memSize[acc.nxtMem[t][mem]] >= ops.size[t] * acc.precision[acc.nxtMem[t][mem],t]:
        #                 continue
        #         m.addConstr( quicksum(onehot_loop2mem[i,t,mem] for i in range(Num_factors)) >= 1, name=f"NOBypass_({t_name},{mem})" )       #  每个memory level都要被映射至少一次
        
        indic_memUsed = gp.tupledict()        
        for mem in range(1,acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):   
                indic_memUsed[t,mem] = m.addVar(vtype=GRB.BINARY, name=f"indic_memUsed_({t},{mem})")
                m.addGenConstrIndicator(indic_memUsed[t,mem], True, quicksum(onehot_loop2mem[i,t,mem] for i in range(Num_factors)) >= 1, name=f"C_memUsed_({t_name},{mem})_1" )
                m.addGenConstrIndicator(indic_memUsed[t,mem], False, quicksum(onehot_loop2mem[i,t,mem] for i in range(Num_factors)) == 0, name=f"C_memUsed_({t_name},{mem})_2" )
        for t, t_name in enumerate(['I','W','O']):
            indic_memUsed[t,acc.Num_mem] = 1
            for mem in range(1,acc.Num_mem):  
                if acc.mappingArray[t][mem] == 1:
                    m.addConstr(indic_memUsed[t,mem]<=indic_memUsed[t,acc.nxtMem[t][mem]], name=f"C_NOBypass_({t_name},{mem})_alter")

        # loop_idx 和 memory hierarchy 对应关系
        idx2mem = gp.tupledict()             # idx2mem[i,t] = [0-Num_mem-1]
        for t, t_name in enumerate(['I','W','O']):
            for i in range(Num_factors):
                idx2mem[i,t] = m.addVar(lb=0, ub=acc.Num_mem, vtype=GRB.INTEGER, name=f"idx2mem_({i},{t})")
                m.addConstr( idx2mem[i,t] == quicksum(onehot_loop2mem[i,t,mem] * mem for mem in range(1,acc.Num_mem)),name=f"C_idx2mem_({i},{t})_define" ) 
            
        indic_bothLoop_ij = gp.tupledict()   # indic_bothLoop_ij[i,j] = [0,1]
        for i in range(Num_factors):        # j>i cannot merge Loop
            for j in range(i+1, Num_factors):
                indic_bothLoop_ij[i,j] = var_AandB(m, indic_TempLoop[i], indic_TempLoop[j], name=f"C_legalLoop_({i},{j})")
                for t, t_name in enumerate(['I','W','O']):
                    m.addGenConstrIndicator(indic_bothLoop_ij[i,j], True, idx2mem[i,t] <= idx2mem[j,t], name=f"C_MemoryHierarchy_({t_name},{i},{j},{t})_1" )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
         
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_loop2mem = gp.tupledict()         # v_loop2mem[i,t] = [0-Num_mem-1]
        for t, t_name in enumerate(['I','W','O']):
            v_loop2mem[Num_factors, t] = acc.Num_mem
            for i in range(Num_factors):
                v_loop2mem[i,t] = m.addVar(lb=1, ub=acc.Num_mem, vtype=GRB.INTEGER, name=f"v_loop2mem_({i},{t})")
                m.addGenConstrIndicator(indic_TempLoop[i], True, v_loop2mem[i,t] == idx2mem[i,t], name=f"C_MemoryHierarchy_({t_name},{i})_2" )
            for i in range(Num_factors):
                m.addConstr(v_loop2mem[i,t]<=v_loop2mem[i+1,t], name=f"C_MemoryHierarchy_({t_name},{i})_3", )
                m.addGenConstrIndicator(indic_TempLoop[i], False, v_loop2mem[i,t] == v_loop2mem[i+1,t], name=f"C_MemoryHierarchy_({t_name},{i})_4" )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        # indicator for cross Memory of loop_idx 
        indic_xMem = gp.tupledict()             # indic_xMem[i,t] = [0,1]
        for t, t_name in enumerate(['I','W','O']):
            for i in range(Num_factors):
                indic_xMem[i,t] = m.addVar(vtype=GRB.BINARY, name=f"indicator_crossMemory_({i},{t_name})" )           
                m.addGenConstrIndicator(indic_xMem[i,t], False, v_loop2mem[i,t] == v_loop2mem[i+1,t], name=f"C_MemoryHierarchy_({t_name},{i})_5")                        
                m.addGenConstrIndicator(indic_xMem[i,t], True, v_loop2mem[i,t] + 1 <= v_loop2mem[i+1,t], name=f"C_MemoryHierarchy_({t_name},{i})_6")                     
                m.addConstr(indic_xMem[i,t] <= indic_TempLoop[i], name=f"indic_xMem_legal_({i},{t})")
                # for dim in range(1,ops.Num_dim): 
                #     m.addGenConstrIndicator(onehot_loop2dim[i,dim], True, indic_xMem[i,t] <= ops.relevance[t][dim], name=f"C_xxx_({t_name},{i},{dim})")
            indic_xMem[Num_factors,t] = 0
        # indic_ir_xMem = gp.tupledict()
        # for t, t_name in enumerate(['I','W','O']):
        #     for i in range(Num_factors):
        #         indic_ir_xMem[i,t] = m.addVar(vtype=GRB.BINARY, name=f"indic_xMem_for_bottomIR_({i},{t_name})" )
        #         m.addGenConstrIndicator(indic_ir_xMem[i,t], True, 
        #                                 var_AandB(m,A=indic_xMem[i+1,t],B=quicksum(onehot_loop2dim[i,dim]*(1-ops.relevance[t][dim]) for dim in range(1,ops.Num_dim)),
        #                                            name=f"bottomIR_({i},{t})_1" ) == 1, name=f"C_bottomIR_({t_name},{i})_1") 
        #         m.addGenConstrIndicator(indic_ir_xMem[i,t], False, 
        #                                 var_AandB(m,A=indic_xMem[i+1,t],B=quicksum(onehot_loop2dim[i,dim]*(1-ops.relevance[t][dim]) for dim in range(1,ops.Num_dim)),
        #                                            name=f"bottomIR_({i},{t})_2" ) == 0, name=f"C_bottomIR_({t_name},{i})_@")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        # indicator for double buffering of [mem, tensor] 
        indic_double_mem = gp.tupledict()       # indic_double_mem[mem,t]
        for mem in range(1,acc.Num_mem):
            for t, _ in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem] == 1:
                    indic_double_mem[mem,t] = m.addVar(vtype=GRB.BINARY, name=f"v_indicator_double_({mem},{t})")
                else:
                    indic_double_mem[mem,t] = 0
                    # m.addConstr( indic_double_mem[mem,t] <= acc.mappingArray[t][mem], name=f"C_indic_db_mem_legal_({mem},{t})" )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        v_Sp_multicast = gp.tupledict()            # v_Sp_multicast[t,mem]
        v_Sp_scatter = gp.tupledict()              # v_Sp_scatter[t,mem]
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem] == 0:
                    v_Sp_multicast[t,mem] = 0
                    v_Sp_scatter[t,mem] = 0
                elif acc.fanout[mem] == 1:
                    v_Sp_multicast[t,mem] = 1
                    v_Sp_scatter[t,mem] = 1
                else:
                    v_Sp_multicast[t,mem] = m.addVar(lb=0, ub=MAX_FANOUT, vtype=GRB.INTEGER, name=f"v_Sp_multicast_({t},{mem})")
                    v_Sp_scatter[t,mem] = m.addVar(lb=0, ub=MAX_FANOUT, vtype=GRB.INTEGER, name=f"v_Sp_scatter_({t},{mem})")

  
        # # loop_idx 和 操作数op 相关性
        # indic_rel_loop2op = gp.tupledict()            # indic_rel_loop2op[i,t] = [0, 1]           # co-work tileNum
        # for i in range(Num_factors):
        #     for t, t_name in enumerate(['I','W','O']):
        #         indic_rel_loop2op[i,t] = m.addVar(vtype=GRB.BINARY, name=f"v_relevance_({i},{t_name})")
        #         m.addConstr( indic_rel_loop2op[i,t] == quicksum(onehot_loop2dim[i,dim] * ops.relevance[t][dim] for dim in range(1,ops.Num_dim)),
        #                      name=f"C_relevance_loop2op_({i},{t_name})")

       
        
        
     
                
        m.addConstr( v_Sp_multicast[0,acc.Global2mem] == v_Sp_scatter[1,acc.Global2mem], name="Pre_C_1")
        m.addConstr( v_Sp_multicast[1,acc.Global2mem] == v_Sp_scatter[0,acc.Global2mem], name="Pre_C_2")
        m.addConstr( v_Sp_multicast[2,acc.Global2mem] == 1, name="Pre_C_3")
        
        m.addConstr( v_Sp_multicast[1,acc.Macro2mem] == 1, name="Pre_C_4")
        m.addConstr( v_Sp_multicast[0,acc.IReg2mem] == v_Sp_scatter[2,acc.OReg2mem] , name="Pre_C_5")
        m.addConstr( v_Sp_scatter[0,acc.IReg2mem]   == v_Sp_multicast[2,acc.OReg2mem], name="Pre_C_6")
        # m.addConstr( v_Sp_multicast[0,acc.IReg2mem] * v_Sp_multicast[2,acc.OReg2mem] == v_Sp_scatter[1,acc.Macro2mem], name="Pre_C_7")

        m.addConstr( v_Sp_scatter[0,acc.IReg2mem] <= acc.dimX, name="Pre_C_8")
        m.addConstr( v_Sp_scatter[2,acc.OReg2mem] <= acc.dimY, name="Pre_C_9")

        m.addConstr( v_Sp_scatter[1,acc.Macro2mem] >= acc.dimX * acc.dimY * CONST.UTIL_COEFFICIENT, name="Pre_C_X")
        # m.addConstr( v_Sp_scatter[2,acc.OReg2mem] == acc.dimY, name="Pre_C_10")
        # m.addConstr( v_Sp_scatter[0,acc.IReg2mem] == acc.dimX, name="Pre_C_11")
        # m.addConstr( v_Sp_scatter[2,acc.Global2mem] == acc.Num_core, name="Pre_C_12")
        # m.addConstr( quicksum(1 - indic_TempFactor[f, p] for f, fs in enumerate(ops.Factors) if f > 0 for p in range(len(fs))) >= 1, name="C_at_least_one_spatial_factor" )

        for t in range(3):
            indic_double_mem[acc.Num_mem,t] = acc.double_Macro
        # weight reload cost
        m.addConstr( indic_double_mem[acc.Macro2mem,1] == 0 )
        
            #  - - - - - - - - - - - - - - - - - - - Spatial Constraints - - - - - - - - - - - - - - - - - - -
            # Macro-level           K <= D1 | R*S*C  <= D2 * D3 | P-Q <= D3
            # localBuffer-level     Product <= Num_macro
            # GlobalBuffer-level    Product <= Num_core
            # Dram-level            Product <= 1
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # self.dim2Dict = ['R', 'S', 'P', 'Q', 'C', 'K'] 
          

        
        res_latency = m.addVar(lb=1, vtype=GRB.INTEGER,     name="res_latency")
        res_energy  = m.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_energy")
        res_EDP     = m.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_EDP")


        ####################################################################  Execution Performance   #######################################################################
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_dim_loopProduct = gp.tupledict()              # v_dimProduct[i,dim]
        for dim in range(1,ops.Num_dim):
            v_dim_loopProduct[Num_factors,dim] = 1
            for i in range(Num_factors):
                v_dim_loopProduct[i,dim] = m.addVar(lb=0, ub=ops.dim2bound[dim], vtype=GRB.INTEGER, name=f"v_dim_loopProduct_({i},{dim})")
        
        for dim in range(1,ops.Num_dim):
            for i in range(Num_factors):
                m.addGenConstrIndicator(onehot_loop2dim[i,dim], True, 
                                        v_dim_loopProduct[i,dim] == var_mul(m, GRB.INTEGER, A=v_dim_loopProduct[i+1,dim], B=v_loop[i]), name=f"C_loopDim_loopProd_({i},{dim})_1")
                m.addGenConstrIndicator(onehot_loop2dim[i,dim], False, 
                                        v_dim_loopProduct[i,dim] == v_dim_loopProduct[i+1,dim], name=f"C_loopDim_loopProd_({i},{dim})_2")
                # m.addConstr(v_dim_loopProduct[i, dim] ==  var_mulABC(m, GRB.INTEGER, A=onehot_loop2dim[i, dim], B=v_dim_loopProduct[i+1, dim], C=v_loop[i]) + 
                #                                         ((1-onehot_loop2dim[i, dim])*v_dim_loopProduct[i+1, dim] ), name=f"C_loopDim_loopProd_({i},{dim})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_dim_spurProduct = gp.tupledict()              # v_dim_spurProduct[mem,t,f]
        for t, t_name in enumerate(['I','W','O']):
            for f, fs in enumerate(ops.Factors):
                if f == 0:
                    continue
                tmp_spDim = 1
                for mem in range(acc.Num_mem-1,0,-1):
                    if acc.mappingArray[t][mem] == 0:
                        v_dim_spurProduct[mem,t,f] = 0
                    elif acc.fanout[mem] == 1:
                        v_dim_spurProduct[mem,t,f] = tmp_spDim
                    else:
                        # if acc.mappingArray[t][mem] == 1:
                        # v_dim_spurProduct[mem,t,f] = m.addVar(lb=0, ub=1024, vtype=GRB.INTEGER, name=f"v_dim_spurProduct_({mem},{t},{f})")
                        for p in range(len(fs)):
                            tmp_spDim_new = m.addVar(lb=0, ub=ops.dim2bound[dim], vtype=GRB.INTEGER, name=f"tmp_spDim_new_({mem},{t},{f},{p})")
                            m.addGenConstrIndicator(onehot_SpUr2mem[f,p,t,mem], True,  tmp_spDim_new == tmp_spDim * ops.Factors[f][p] )
                            m.addGenConstrIndicator(onehot_SpUr2mem[f,p,t,mem], False, tmp_spDim_new == tmp_spDim )
                            tmp_spDim = tmp_spDim_new
                        # m.addConstr( v_dim_spurProduct[mem,t,f] == tmp_spDim )
                        v_dim_spurProduct[mem,t,f] = tmp_spDim
                v_dim_spurProduct[acc.Num_mem,t,f] = 1
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        Temp_loopProduct = gp.tupledict()           # Temp_loopProduct[mem,t,dim,i]
        for t in range(3):
            for mem in range(1,acc.Num_mem):
                for dim in range(1,ops.Num_dim):
                    for i in range(Num_factors):
                        if acc.mappingArray[t][mem] == 1:
                            Temp_loopProduct[mem,t,dim,i] = m.addVar(lb=0, ub=ops.dim2bound[dim], vtype=GRB.INTEGER, name=f"y_{i}_{mem}_{t}_{dim}")
                            # 当 onehot_loop2mem[i,t,mem] == 1 时，y_i = v_dim_loopProduct[i,dim]
                            m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], True, Temp_loopProduct[mem,t,dim,i] == v_dim_loopProduct[i,dim])
                            # 当 onehot_loop2mem[i,t,mem] == 0 时，y_i = 0
                            m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], False, Temp_loopProduct[mem,t,dim,i] == 0)
                        else:
                            Temp_loopProduct[mem,t,dim,i] = 0

        v_dim_TpUrProduct = gp.tupledict()                # v_dim_TpUrProduct[mem,t,dim]
        for dim in range(1,ops.Num_dim):
            for t, t_name in enumerate(['I','W','O']):
                for mem in range(1,acc.Num_mem):
                    if acc.mappingArray[t][mem] == 1:
                        v_dim_TpUrProduct[mem,t,dim] = m.addVar(lb=0, ub=ops.dim2bound[dim], vtype=GRB.INTEGER, name=f"v_dim_TpUrProduct_({mem},{t},{dim})")
                        # m.addConstr( v_dim_TpUrProduct[mem,t,dim] <= 8 )
                        # for i in range(Num_factors):
                            # m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], True, v_dim_TpUrProduct[mem,t,dim] >= v_dim_loopProduct[i,dim])
                        m.addGenConstrMax(v_dim_TpUrProduct[mem,t,dim], [Temp_loopProduct[mem,t,dim,i] for i in range(Num_factors)], 0.0 )
                        # for i in range(Num_factors):
                        #     m.addConstr(v_dim_TpUrProduct[mem,t,dim] >= v_dim_loopProduct[i,dim]*onehot_loop2mem[i,t,mem], name=f"C_v_dim_TpUrProduct_({mem},{dim},{t},{i})")
                    else:
                        v_dim_TpUrProduct[mem,t,dim] = 0
                v_dim_TpUrProduct[acc.Num_mem,t,dim] = 1
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
                        

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_dim_spur_inMem = gp.tupledict()           # v_dim_spur_inMem[mem,t,dim]
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                for dim in range(1, ops.Num_dim):
                    if acc.mappingArray[t][mem] == 0:
                        v_dim_spur_inMem[mem,t,dim] = 0
                    elif acc.fanout[mem] == 1:
                        v_dim_spur_inMem[mem,t,dim] = 1
                    else:
                        v_dim_spur_inMem[mem,t,dim] = m.addVar(lb=0, ub=ops.dim2bound[dim], vtype=GRB.INTEGER, name=f"v_dim_spur_inMem_({mem},{t},{dim})")
        for dim in range(1, ops.Num_dim):
            # m.addConstr( v_dim_spur_inMem[acc.IReg2mem,0,dim] == v_dim_spur_inMem[acc.Macro2mem,1,dim], name=f"SpUr_consistency_{dim}_1" )
            # m.addConstr( v_dim_spur_inMem[acc.Macro2mem,1,dim] == v_dim_spur_inMem[acc.OReg2mem,2,dim], name=f"SpUr_consistency_{dim}_2" )
            # m.addConstr( v_dim_spur_inMem[acc.Global2mem,0,dim] == v_dim_spur_inMem[acc.Global2mem,1,dim], name=f"SpUr_consistency_{dim}_3" )
            # m.addConstr( v_dim_spur_inMem[acc.Global2mem,1,dim] == v_dim_spur_inMem[acc.Global2mem,2,dim], name=f"SpUr_consistency_{dim}_4" )
            v_dim_spur_inMem[acc.IReg2mem,0,dim] = v_dim_spur_inMem[acc.Macro2mem,1,dim]
            v_dim_spur_inMem[acc.OReg2mem,2,dim] = v_dim_spur_inMem[acc.Macro2mem,1,dim]
            v_dim_spur_inMem[acc.Global2mem,0,dim] = v_dim_spur_inMem[acc.Global2mem,1,dim]
            v_dim_spur_inMem[acc.Global2mem,2,dim] = v_dim_spur_inMem[acc.Global2mem,1,dim]
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                # if acc.fanout[mem] == 1:
                #     continue
                ts_multi = 1
                ts_scatter = 1
                if acc.mappingArray[t][mem] == 1:
                    for dim in range(1, ops.Num_dim):
                        if isinstance(v_dim_spur_inMem[mem,t,dim], gp.Var) or isinstance(v_dim_spurProduct[acc.nxtMem[t][mem],t,dim], gp.Var) or isinstance(v_dim_spurProduct[mem,t,dim], gp.Var) :
                            m.addConstr(v_dim_spur_inMem[mem,t,dim] * v_dim_spurProduct[acc.nxtMem[t][mem],t,dim] == v_dim_spurProduct[mem,t,dim],
                                         name=f"C_dim_spur_inMem_({mem},{t},{dim})")
                        if ops.relevance[t][dim] == 1:
                            ts_new = var_mul(m,GRB.INTEGER,A=ts_scatter, B=v_dim_spur_inMem[mem,t,dim])
                            if isinstance(ts_new, gp.Var):
                                ts_new.VarName = f"ts_multi_new_({mem},{t},{dim})"
                            ts_scatter = ts_new
                        else:             # multicast = scatter
                            ts_new = var_mul(m,GRB.INTEGER,A=ts_multi, B=v_dim_spur_inMem[mem,t,dim])
                            if isinstance(ts_new, gp.Var):
                                ts_new.VarName = f"ts_scatter_new_({mem},{t},{dim})"
                            ts_multi = ts_new
                    m.addConstr( v_Sp_multicast[t,mem] == ts_multi, name=f"C_sp_multi_({t},{mem})" )
                    m.addConstr( v_Sp_scatter[t,mem] == ts_scatter, name=f"C_sp_scatter_({t},{mem})" )
                    m.addConstr( v_Sp_multicast[t,mem] * v_Sp_scatter[t,mem] <= acc.fanout[mem] , name=f"C_fanout_({mem},{t})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_tileNum = gp.tupledict()                  # v_tileNum[t,mem]
        v_dim_tpur_upperMem = gp.tupledict()           # v_dim_spur_inMem[mem,t,dim]
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                for dim in range(1, ops.Num_dim):
                    if acc.mappingArray[t][mem] == 0:
                        v_dim_tpur_upperMem[mem,t,dim] = 0
                    else:
                        # if ops.relevance[t][dim] == 1:
                        v_dim_tpur_upperMem[mem,t,dim] = m.addVar(lb=0, ub=ops.dim2bound[dim], vtype=GRB.INTEGER, name=f"v_dim_tpur_upperMem_({mem},{t},{dim})")
                            
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                ts_rela = 1
                if acc.mappingArray[t][mem] == 1:
                    # v_tileNum[t,mem] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_tileNum_({t},{mem})")
                    for dim in range(1, ops.Num_dim):
                        # if ops.relevance[t][dim] == 1:
                        # if isinstance(v_dim_tpur_upperMem[mem,t,dim], gp.Var):
                        m.addConstr(v_dim_tpur_upperMem[mem,t,dim] * v_dim_TpUrProduct[acc.nxtMem[t][mem],t,dim] == v_dim_loopProduct[0,dim],
                                        name=f"C_dim_tpur_inMem_({mem},{t},{dim})")

                        ts_new = var_mul(m,GRB.INTEGER,A=ts_rela, B=v_dim_tpur_upperMem[mem,t,dim])
                        if isinstance(ts_new, gp.Var):
                            ts_new.VarName = f"ts_rela_new_({mem},{t},{dim})"
                        ts_rela = ts_new
                    # m.addConstr( v_tileNum[t,mem] == ts_rela, name=f"C_tileNum_({mem},{t})")
                    v_tileNum[t,mem] = ts_rela
                    if isinstance(v_tileNum[t,mem], gp.Var):
                        v_tileNum[t,mem].VarName = f"v_tileNum_({t},{mem})"
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        # each memory_utilization of each operator 
        v_memSize = gp.tupledict()              # v_memSize[mem,t] = Integer
        tmp_dim = gp.tupledict()
        for mem in range(1,acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem]:
                    v_memSize[mem,t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_memSize_({mem},{t_name})")
                else:
                    v_memSize[mem,t] = 0
            
            t = 0  # Input
            if acc.mappingArray[t][mem]:
                for f in range(1,ops.Num_dim):
                    tmp_dim[mem,t,f] = var_mul(m,GRB.INTEGER,v_dim_TpUrProduct[mem,t,f], v_dim_spurProduct[mem,t,f])
                m.addConstr( v_memSize[mem,t] == var_mulABC(m,GRB.INTEGER, 
                                                            A=tmp_dim[mem,t,ops.dict2Dim('P')]+tmp_dim[mem,t,ops.dict2Dim('R')]-1, 
                                                            B=tmp_dim[mem,t,ops.dict2Dim('Q')]+tmp_dim[mem,t,ops.dict2Dim('S')]-1,
                                                            C=tmp_dim[mem,t,ops.dict2Dim('C')]),
                             name=f"Set_Mem_{mem}_of_{t}")
            else:
                # m.addConstr( v_memSize[mem,t] == 0, name=f"C_No_mem_{mem}_of_{t}" )
                v_memSize[mem,t] = 0

            t = 1  # Weight
            if acc.mappingArray[t][mem]:
                for f in range(1,ops.Num_dim):
                    tmp_dim[mem,t,f] = var_mul(m,GRB.INTEGER,v_dim_TpUrProduct[mem,t,f], v_dim_spurProduct[mem,t,f])
                m.addConstr( v_memSize[mem,t] == var_mul(m,GRB.INTEGER, 
                                                         A=var_mul(m,GRB.INTEGER,A=tmp_dim[mem,t,ops.dict2Dim('R')], B=tmp_dim[mem,t,ops.dict2Dim('S')]), 
                                                         B=var_mul(m,GRB.INTEGER,A=tmp_dim[mem,t,ops.dict2Dim('C')], B=tmp_dim[mem,t,ops.dict2Dim('K')])), 
                             name=f"Set_Mem_{mem}_of_{t}")
            else:
                # m.addConstr( v_memSize[mem,t] == 0, name=f"C_No_mem_{mem}_of_{t}" )
                v_memSize[mem,t] = 0

            t = 2  # Output
            if acc.mappingArray[t][mem]:
                for f in range(1,ops.Num_dim):
                    tmp_dim[mem,t,f] = var_mul(m,GRB.INTEGER,v_dim_TpUrProduct[mem,t,f], v_dim_spurProduct[mem,t,f])
                m.addConstr( v_memSize[mem,t] == var_mulABC(m,GRB.INTEGER, A=tmp_dim[mem,t,ops.dict2Dim('P')], B=tmp_dim[mem,t,ops.dict2Dim('Q')], C=tmp_dim[mem,t,ops.dict2Dim('K')]), 
                             name=f"Set_Mem_{mem}_of_{t}")
            else:
                # m.addConstr( v_memSize[mem,t] == 0, name=f"C_No_mem_{mem}_of_{t}" )
                v_memSize[mem,t] = 0

        # Memory Capacity Constraints
        for mem in range(1,acc.Num_mem):
            if mem > 0:
                m.addConstr( quicksum( (v_memSize[mem,t] + v_memSize[mem,t]*indic_double_mem[mem,t])
                                            * acc.precision[mem,t] * acc.mappingArray[t][mem] for t in range(3)) <= acc.memSize[mem] )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
    

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_tileSize = gp.tupledict()
        for t, t_name in enumerate(['I','W','O']):
            v_memSize[acc.Num_mem,t] = 1
            for i in range(Num_factors):
                v_tileSize[i,t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_tileSize_({i},{t_name})")
                for mem in range(1,acc.Num_mem):
                    if acc.mappingArray[t][mem] == 1:
                        m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], True, v_tileSize[i,t] == var_mul(m,GRB.INTEGER,A=v_memSize[acc.nxtMem[t][mem],t], B=v_Sp_scatter[t,mem]) )
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_loopTrans_time = gp.tupledict()       # v_loopTrans_time[i,t]
        v_loopStall = gp.tupledict()            # v_loopStall[i,t]
        v_loopConsume = gp.tupledict()          # v_loopConsume[i] 
        tmp_indic_db = gp.tupledict()
        tmp_indicAdd = gp.tupledict()
        v_loopConsume[Num_factors] = acc.t_MAC      # mac cost Computing-in-memory inherent
        for i in range(Num_factors):
            v_loopConsume[i] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_consume_({i},{t})")
        for t, t_name in enumerate(['I','W','O']):
            for i in range(Num_factors):
                v_loopTrans_time[i,t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_loopTrans_time_({i},{t_name})")
                v_loopStall[i,t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"v_stall_({i},{t_name})")
                m.addGenConstrIndicator(indic_TempLoop[i], True, v_loopStall[i,t] >= v_loopTrans_time[i,t], name=f"C_v_loopStall_one_({i},{t})")
                m.addGenConstrIndicator(indic_TempLoop[i], False, v_loopTrans_time[i,t] == 0, name=f"C_v_loopTrans_time_zero_({i},{t})")
                m.addConstr(v_loopStall[i,t] >= v_loopConsume[i+1], name=f"C_v_loopStall_digui_({i},{t})")
            for i in range(Num_factors):
                tmp_indic_db[i,t] = m.addVar(vtype=GRB.BINARY, name=f"tmp_indic_db_({i},{t})" )
                for mem in range(1,acc.Num_mem):
                    if acc.mappingArray[t][mem] == 0:
                        continue
                    m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], True, v_loopTrans_time[i,t] * acc.bw[mem]  >=  v_tileSize[i,t] * acc.precision[mem,t])
                    m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], True, v_loopTrans_time[i,t] * acc.bw[mem] - acc.bw[mem]  <=  v_tileSize[i,t] * acc.precision[mem,t])
                    m.addGenConstrIndicator(onehot_loop2mem[i,t,mem], True, tmp_indic_db[i,t] == 
                                            indic_double_mem[acc.nxtMem[t][mem],t], name=f"C_tmp_indic_db_({i},{t},{mem})")

            for i in range(Num_factors):
                # tmp_xMem = var_AorB(m, A=indic_xMem[i,t], B=indic_ir_xMem[i,t], name=f"C_tmp_indic_xMem_({i},{t})")
                tmp_indicAdd[i,t] = var_AandB(m,indic_xMem[i,t],(1-tmp_indic_db[i,t]), name=f"tmp_indic_db_({i},{t})")
                # tmp_indicAdd[i,t] = var_AandB(m,tmp_xMem,(1-tmp_indic_db[i,t]), name=f"tmp_indic_db_({i},{t})")
                m.addGenConstrIndicator(tmp_indicAdd[i,t], True, v_loopStall[i,t] == var_add(m, GRB.INTEGER, v_loopConsume[i+1], v_loopTrans_time[i,t]), 
                                            name=f"Latency_trans_db_{i}_{t}")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        stall_single_insideLoop = gp.tupledict()
        for i in range(Num_factors):
            stall_single_insideLoop[i] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"tmp_stall_incu_by_xMem_({i})")
            # m.addGenConstrMax(stall_single_insideLoop, [v_loopStall[i,t] for t in range(3)], 0.0 )
            for t in range(3):
                m.addConstr(stall_single_insideLoop[i] >= v_loopStall[i,t], name=f"maxStall_{i}_{t}")
            m.addGenConstrIndicator(indic_TempLoop[i], True, v_loopConsume[i] == var_mul(m, GRB.INTEGER, stall_single_insideLoop[i], v_loop[i]))
            m.addGenConstrIndicator(indic_TempLoop[i], False, v_loopConsume[i] == v_loopConsume[i+1])
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        v_transEnergy = gp.tupledict()          # v_transEnergy[mem,t]
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                if mem in [acc.IReg2mem, acc.OReg2mem, acc.Macro2mem]:
                    v_transEnergy[mem,t] = 0
                elif acc.mappingArray[t][mem] == 0:
                    v_transEnergy[mem,t] = 0
                else:
                    # v_transEnergy[mem,t] = v_tileNum[t,mem]*v_memSize[acc.nxtMem[t][mem],t]*v_Sp_scatter[t,mem]*\
                    #                         (acc.cost_r[mem]+acc.cost_w[acc.nxtMem[t][mem]]*v_Sp_multicast[t,mem])+\
                    #                         v_transEnergy[acc.nxtMem[t][mem],t]*v_Sp_scatter[t,mem]*v_Sp_multicast[t,mem]
                    v_transEnergy[mem,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"v_transEnergy_({t_name},{mem})")
        for mem in range(1, acc.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                if mem in [acc.IReg2mem, acc.OReg2mem, acc.Macro2mem] or acc.mappingArray[t][mem] == 0:
                    continue
                else:
                    transE_single_instance = (acc.cost_r[mem]+acc.cost_w[acc.nxtMem[t][mem]]*v_Sp_multicast[t,mem]) * acc.precision[mem,t]
                    transNum_single = v_tileNum[t,mem]*v_memSize[acc.nxtMem[t][mem],t]
                    m.addConstr(v_transEnergy[mem,t] == var_mulABC(m,vtype=GRB.CONTINUOUS, A=transE_single_instance, B= transNum_single, C=v_Sp_scatter[t,mem]))
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        # Energy Unit = pj
        v_macEnergy = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"v_macEnergy")
        v_staticEnergy = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"v_staticEnergy")

        v_mac_CIM_once = var_mul(model=m, vtype=GRB.INTEGER, A=v_Sp_scatter[0,acc.IReg2mem], B=v_Sp_scatter[2,acc.OReg2mem])
        # v_mac_num * v_mac_CIM_once >= ops.Num_MAC
        v_mac_num = m.addVar(lb=0, ub=ops.Num_MAC, vtype=GRB.INTEGER, name=f"v_mac_num")
        m.addConstr( v_mac_num * v_mac_CIM_once >= ops.Num_MAC )
        # v_macEnergy = mac_cost * v_mac_num

        m.addConstr( v_macEnergy == acc.cost_ActMacro * v_mac_num, name="C_MAC_Energy" )
        m.addConstr( v_staticEnergy == res_latency * acc.leakage_per_cycle, name="C_Static_Leakage_Energy" )


        m.addConstr(res_latency == v_loopConsume[0], name="C_res_latency")

        m.addConstr(res_energy * 1e3 == v_staticEnergy + v_macEnergy + quicksum(v_transEnergy[mem,t] * acc.Num_instance[mem] for t in range(3) for mem in range(1,acc.Num_mem)), name="C_res_energy")

        m.addConstr(res_EDP * 1e6 == res_latency * res_energy, name="C_res_EDP")

        m.update()

        ####################################################################  Set Constraint Flag ###################################################################
        
            
        if CONST.FLAG_OPT=="Latency":
            m.setObjective(res_latency, GRB.MINIMIZE)
        elif CONST.FLAG_OPT=="Energy":
            m.setObjective(res_energy, GRB.MINIMIZE)
        elif CONST.FLAG_OPT=="EDP":
            m.setObjective(res_EDP, GRB.MINIMIZE)
        elif CONST.FLAG_OPT == "Feasible":
            m.setObjective(0)
            m.setParam('MIPFocus', 1)
            m.setParam("ZeroObjNodes",10000)
        else:
            raise ValueError("Illegal Optimization Flag")
        
        # m.setObjective(res_tileSize, GRB.MINIMIZE)

        # m.tune()
        if FLAG.PRESOLVE_SEARCH :
            Logger.debug("PRESOLVE_SEARCH")
            m.setParam("TimeLimit", 300)
            preset_conditions = []

            hc1 = [m.addConstr( v_Sp_scatter[1,acc.Macro2mem] == acc.fanout[acc.Macro2mem]), m.addConstr(v_Sp_scatter[2,acc.Global2mem]==acc.Num_core)]
            # hc1 = [m.addConstr(v_Sp_scatter[1,acc.Macro2mem] >= 48)]
            preset_conditions.append(get_startVar_byConstr(model=m, constraint_hypothesis=hc1))

            set_startVar(model=m, preset_conditions=preset_conditions, res_str='res_l')
        else:
            pass

        m.setParam("TimeLimit", CONST.TIMELIMIT)

        m.update()


        ####################################################################  Optimization    #######################################################################

        m.write("model.lp")
        if FLAG.LOAD_SOLUTION:
            try:
                m.read("MIREDO.sol")
            except ValueError:
                raise ValueError("No MIREDO.sol File")
        start_time = time.time()
        m.optimize()
        end_time = time.time()

        ####################################################################  Debug & Output  #######################################################################

        time_optimization = end_time - start_time
        Logger.critical(f"Optimizing time: {'%.3f' %(time_optimization)}s")

        # for constr in m.getConstrs():               # 确保约束中有变量
        #     if constr.getAttr("Sense") in ['<=', '>=', '='] and constr.getAttr("RHS") is not None:
        #         expr = m.getRow(constr)
        #         if expr.size() == 0:
        #             print(f"约束 {constr.ConstrName} 中没有变量，需要修正。")

        def set_dataflow():
            Logger.critical("set dataflow")
            res_loop = []
            for i in range(Num_factors):
                if indic_TempLoop[i].x == 1:
                    res_loop.append([int(v_loop2dim[i].x),     
                                     int(v_loop[i].x), 
                                     [int(v_loop2mem[i,t].x) for t in range(3)] ])

            sp_dim = [[[1 for _ in range(ops.Num_dim)] for __ in range(3)] for ___ in range(acc.Num_mem)]
            for mem in range(1,acc.Num_mem):
                for t in range(3):
                    if acc.mappingArray[t][mem] == 1:
                        for dim in range(1,ops.Num_dim):
                            sp_dim[mem][t][dim] = 1
                        for f, fs in enumerate(ops.Factors):
                            if f == 0:
                                continue
                            for p in range(len(fs)):
                                if isinstance(onehot_SpUr2mem[f,p,t,mem], gp.Var):
                                    if round(onehot_SpUr2mem[f,p,t,mem].x) == 1:
                                        sp_dim[mem][t][f] *= ops.Factors[f][p]
                                else:
                                    if round(onehot_SpUr2mem[f,p,t,mem]) == 1:
                                        sp_dim[mem][t][f] *= ops.Factors[f][p]
                    else:
                        for dim in range(1,ops.Num_dim):
                            sp_dim[mem][t][dim] = 0
                        

            double_tag = [[1 for _ in range(3)] for __ in range(acc.Num_mem+1)]
            for mem in range(1,acc.Num_mem+1):
                for t in range(3):
                    if mem==acc.Num_mem:
                        double_tag[mem][t] = acc.double_Macro
                        continue
                    if acc.mappingArray[t][mem]:
                        double_tag[mem][t] = indic_double_mem[mem,t].x if mem!=acc.Num_mem else acc.double_Macro
                    else:
                        double_tag[mem][t] = 0

            self.dataflow['temporal_mapping']   = res_loop
            self.dataflow['spatial_mapping']    = sp_dim
            self.dataflow['double_tag']         = double_tag

        if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
            Logger.critical("MIP Solved successfully !!!")
            self.result = [res_latency.x, res_energy.x, res_EDP.x]
            set_dataflow()
            m.write("solution.sol")
            if CONST.FLAG_OPT=="Latency":
                Logger.debug(f"Get best latency= {res_latency.x}")
            elif CONST.FLAG_OPT=="Energy":
                Logger.debug(f"Get best var_energy= {res_energy.x}")
            elif CONST.FLAG_OPT=="EDP":
                Logger.debug(f"Get best var_EDP= {res_EDP.x}")
            else:
                Logger.debug(f"Get simple solution, L={res_latency.x}, E={res_energy.x}")
        elif m.status == GRB.Status.TIME_LIMIT:
            Logger.warning("Solver termination by TIME_LIMIT! Looking for gap solution")
            m.setParam("MIPGap", CONST.GAP_THRESHOLD)
            m.setParam("TimeLimit", CONST.TIMELIMIT_AFTER_TLE)
            m.update()
            m.optimize()
            if m.status == GRB.Status.TIME_LIMIT:
                Logger.critical(f"Solver termination by TIME_LIMIT again! With Final MIP gap: {m.MIPGap * 100:.2f}%")
            if m.SolCount > 0: 
                self.result = [res_latency.x, res_energy.x, res_EDP.x]
                set_dataflow()
                m.write("solution.sol")
                Logger.warning("TimeLimited Solution Found")
                if CONST.FLAG_OPT=="Latency":
                    Logger.debug(f"Get best latency= {res_latency.x}")
                elif CONST.FLAG_OPT=="Energy":
                    Logger.debug(f"Get best var_energy= {res_energy.x}")
                elif CONST.FLAG_OPT=="EDP":
                    Logger.debug(f"Get best var_EDP= {res_EDP.x}")
                else:
                    Logger.debug(f"Get simple solution, L={res_latency.x}, E={res_energy.x}")
            else:
                self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
                Logger.error("TLE---No Feasible Solution Yet")
                m.setParam('IISMethod', 1)
                m.computeIIS()
                m.write("contric.ilp")
                m.write("model.mps")
                Logger.error("TLE---Debug in contric.ilp")
                exit()
        else:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            m.computeIIS()
            m.write("contric.ilp")
            m.write("model.mps")
            Logger.error(f'Model infeasible !!!')
            exit()

        print("\n Loop mapping | indicator-crossMemory")
        for i in range(Num_factors):
            print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}: {round(indic_TempLoop[i].x)}  ",end="")
            for t in range(3):
                print(f"{acc.mem2dict(v_loop2mem[i,t].x):>20}|{round(indic_xMem[i,t].x)},", end=" ")
            print("")

        
        # print("\n Debug Sp Dim in Memory")
        # # m.addConstr(v_loopTrans_time[i,t] * acc.bw[mem]  >=  v_tileSize[i,t] * acc.precision[mem,t])
        # for mem in range(1,acc.Num_mem):
        #     print(f"{acc.mem2dict(mem)}: ")
        #     for t, t_name in enumerate(['I','W','O']):
        #         if acc.mappingArray[t][mem] == 0:
        #             continue
        #         print(f"{t_name}: ")
        #         for dim in range(1,ops.Num_dim):
        #             if acc.nxtMem[t][mem] == acc.Num_mem:
        #                 print(f"{ops.dim2Dict[dim]}: inMem({v_dim_spur_inMem[mem,t,dim].x}) * nxtDim({v_dim_spurProduct[acc.nxtMem[t][mem],t,dim]}) = curDim({v_dim_spurProduct[mem,t,dim].x})", end=" ")
        #             else:
        #                 print(f"{ops.dim2Dict[dim]}: inMem({v_dim_spur_inMem[mem,t,dim].x}) * nxtDim({v_dim_spurProduct[acc.nxtMem[t][mem],t,dim].x}) = curDim({v_dim_spurProduct[mem,t,dim].x})", end=" ")
        #             print("")
        #         print("")
        # exit()
                
        # print("\n Debug Trans Time Cost")
        # # m.addConstr(v_loopTrans_time[i,t] * acc.bw[mem]  >=  v_tileSize[i,t] * acc.precision[mem,t])
        # for i in range(Num_factors):
        #     if indic_TempLoop[i].x == 0:
        #         continue
        #     print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}: {round(indic_TempLoop[i].x)}  ")
        #     for t, t_name in enumerate(['I','W','O']):
        #         print(f"{t_name}: ")
        #         for mem in range(1,acc.Num_mem):
        #             if acc.mappingArray[t][mem] == 0:
        #                 continue
        #             if onehot_loop2mem[i,t,mem].x == 0:
        #                 continue
        #             print(f"{acc.mem2dict(mem):<20}: " ,end="")
        #             print(f"Trans_t:{v_loopTrans_time[i,t].x:>10} * bw:{acc.bw[mem]} >= tileSize:{round(v_tileSize[i,t].x)} * prici:{acc.precision[mem,t]},")
        #     print("")
        # exit()

        print("\n Loop mapping | indicator-crossMemory")
        for i in range(Num_factors):
            if round(indic_TempLoop[i].x) == 0:
                continue
            else:
                print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}: {round(indic_TempLoop[i].x)}  "
                        + f"{acc.mem2dict(v_loop2mem[i,0].x):>20}|{round(indic_xMem[i,0].x)},"
                        + f"{acc.mem2dict(v_loop2mem[i,1].x):>20}|{round(indic_xMem[i,1].x)},"
                        + f"{acc.mem2dict(v_loop2mem[i,2].x):>20}|{round(indic_xMem[i,2].x)}")
            # exit()


        # print("\n Loop2Memory | indicator-crossMemory")
        # for i in range(Num_factors):
        #     print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}: {round(indic_TempLoop[i].x)}  ", end="")
        #     for mem in range(1,acc.Num_mem):
        #         print(f"{onehot_loop2mem[i,0,mem].x}  ",end="")
        #     print("")
            # exit()


        print("\n Double Buffer tag")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem):<20}: " ,end="")
            for t in range(3):
                if acc.mappingArray[t][mem] == 0:
                    print(f"{round(indic_double_mem[mem,t])}   ", end=" ")
                    continue
                print(f"{round(indic_double_mem[mem,t].x)}   ", end=" ")
            print("")

        print("\n tileNum")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem):<20}: " ,end="")
            for t in range(3):
                if acc.mappingArray[t][mem] == 1:
                    print(f"{round(v_tileNum[t,mem].x):<7}   ", end=" ")
                else:
                    print(f"{0:<7}   ", end=" ")
            print("")
            
            
            
        # print("\n onehot_loop2dim")             #  #  #   Check   #  #  #
        # for i in range(Num_factors):
        #     print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]} in {int(v_loop[i].x)}: " ,end="")
        #     for dim in range(1,ops.Num_dim):
        #         print(f"{ops.dim2Dict[dim]}-{onehot_loop2dim[i,dim].x}, ",end="")
        #     print("")
        
        print("\n Spatial Unrolling Dim_factor- [Temporal|spatial]")
        for f, fs in enumerate(ops.Factors):
            if f == 0:
                continue
            print(f"{ops.dim2Dict[f]}: ",end="") 
            for p in range(len(fs)):
                tmp = 0
                for mem in range(1,acc.Num_mem):
                    for t in range(3):
                        if acc.mappingArray[t][mem] == 1:
                            if isinstance(onehot_SpUr2mem[f,p,t,mem], gp.Var):
                                tmp += onehot_SpUr2mem[f,p,t,mem].x
                            else:
                                tmp += onehot_SpUr2mem[f,p,t,mem]
                        else:
                            tmp += 0
                print(f"{ops.Factors[f][p]:>2}-[{int(indic_TempFactor[f,p].x)}|{int(tmp)}]   ", end="")
            print("")

        
        print("\n v_dim_loopProduct")
        for i in range(Num_factors):
            if round(indic_TempLoop[i].x) == 0:
                continue
            else:
                print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}: {round(indic_TempLoop[i].x)}  ", end="")
                for dim in range(1,ops.Num_dim):
                    print(f"{ops.dim2Dict[dim]}:{v_dim_loopProduct[i,dim].x:<5}", end=" ")
                print("")

                
        print("\n v_dim_TpUrProduct")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}: ")
            for t, t_name in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem] == 0:
                    continue
                print(f"{t_name}:", end="")
                for dim in range(1,ops.Num_dim):
                    print(f"{ops.dim2Dict[dim]}:{v_dim_TpUrProduct[mem,t,dim].x if acc.mappingArray[t][mem] else 0:<5}", end=" ")
                print("")

        
        print("\n v_dim_spurProduct")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}: ")
            for t, t_name in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem] == 0:
                    continue
                print(f"{t_name}:", end="")
                for dim in range(1,ops.Num_dim):
                    print(f"{ops.dim2Dict[dim]}:{v_dim_spurProduct[mem,t,dim].x if acc.mappingArray[t][mem] else 0:<5}", end=" ")
                print("")
      
        
        print("\n tmp_dim")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem)}: ")
            for t, t_name in enumerate(['I','W','O']):
                if acc.mappingArray[t][mem] == 1:
                    print(f"{t_name}:", end="")
                    for dim in range(1,ops.Num_dim):
                        print(f"{ops.dim2Dict[dim]}:{tmp_dim[mem,t,dim].x if acc.mappingArray[t][mem] else 0:<5}", end=" ")
                    print("")

        print("\n MemSize Sp-multicast|scatter")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem):<20}: ",end="")
            for t in range(3):
                if acc.mappingArray[t][mem] == 0:
                    print(f"{round(v_memSize[mem,t]):>10},{round(v_Sp_multicast[t,mem]):>3}|{round(v_Sp_scatter[t,mem]):<3} ", end="")
                    continue
                elif acc.fanout[mem]==1:
                    print(f"{round(v_memSize[mem,t].x):>10},{round(v_Sp_multicast[t,mem]):>3}|{round(v_Sp_scatter[t,mem]):<3} ", end="")
                    continue
                print(f"{round(v_memSize[mem,t].x):>10},{round(v_Sp_multicast[t,mem].x):>3}|{round(v_Sp_scatter[t,mem].x):<3} ", end="")
            print("")
        
        print("\n TileSize,(multi|scatter)")
        for mem in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(mem):<20}: ",end="")
            for t in range(3):
                nxt = acc.nxtMem[t][mem]
                if acc.mappingArray[t][mem] == 0:
                    print(f"{round(v_memSize[nxt,t]) if acc.mappingArray[t][mem] else '---':>10},{round(v_Sp_multicast[t,mem]):>3}|{round(v_Sp_scatter[t,mem]):<3} ", end="")
                    continue
                elif acc.fanout[mem] == 1:
                    print(f"{round(v_memSize[nxt,t].x) if acc.mappingArray[t][mem] else '---':>10},{round(v_Sp_multicast[t,mem]):>3}|{round(v_Sp_scatter[t,mem]):<3} ", end="")
                    continue
                if nxt != acc.Num_mem:
                    print(f"{round(v_memSize[nxt,t].x) if acc.mappingArray[t][mem] else '---':>10},{round(v_Sp_multicast[t,mem].x):>3}|{round(v_Sp_scatter[t,mem].x):<3} ", end="")
                else:
                    print(f"{round(v_memSize[nxt,t]) if acc.mappingArray[t][mem] else '---':>10},{round(v_Sp_multicast[t,mem].x):>3}|{round(v_Sp_scatter[t,mem].x):<3} ", end="")
            print("")

        print("\ntileSize & loopTrans_time ")
        for i in range(Num_factors):
            if round(indic_TempLoop[i].x) == 0:
                continue
            else:
                print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}:  {round(indic_TempLoop[i].x)}  " +
                    f"{round(v_tileSize[i,0].x):>8}--{round(v_loopTrans_time[i,0].x) :<13}, "+
                    f"{round(v_tileSize[i,1].x):>8}--{round(v_loopTrans_time[i,1].x) :<13}, "+
                    f"{round(v_tileSize[i,2].x):>8}--{round(v_loopTrans_time[i,2].x) :<13}" )

        print("\nloopStall")
        for i in range(Num_factors):
            if round(indic_TempLoop[i].x) == 0:
                continue
            else:
                print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}:  {round(indic_TempLoop[i].x)}  " +
                    f"{round(v_loopStall[i,0].x) :>8}, "+
                    f"{round(v_loopStall[i,1].x) :>8}, "+
                    f"{round(v_loopStall[i,2].x) :>8}" )
        
        
        print("\nConsume")
        for i in range(Num_factors):
            if round(indic_TempLoop[i].x) == 0:
                continue
            else:
                print(f"{i:<3} for {ops.dim2Dict[int(v_loop2dim[i].x)]}  {int(v_loop[i].x)}:  {round(indic_TempLoop[i].x)}  " +
                    f"{round(v_loopConsume[i].x) :>10}  stall_in_Loop:{stall_single_insideLoop[i].x}" )
       
        if FLAG.DEBUG:
            for i in range(Num_factors):   
                for f, fs in enumerate(ops.Factors):
                    if f == 0:
                        continue
                    for p in range(len(fs)):
                        print(f"{v_loop[f,p,i].x} ",end="")
                print()
            

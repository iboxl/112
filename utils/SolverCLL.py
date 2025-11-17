# this file is prepared for project 419
# Created by iboxl

import os
import psutil
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad, LoopNest, Mapping
import math
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum, min_
from utils.GlobalUT import *
from utils.UtilsFunction.SolverFunction import *
from utils.UtilsFunction.CostFunction import _Cost_model

class Solver():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad, outputdir=None):
        self.acc = acc
        self.ops = ops
        self.outputdir = outputdir

        self.model = gp.Model(name="MIREDO")
        self.model.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
        self.model.setParam('Seed', 419)                               # 随机种子
        # self.model.setParam('NonConvex', 2)
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('Cuts', 2)
        # self.model.setParam('Threads', psutil.cpu_count(logical=False))
        self.model.setParam('Threads', psutil.cpu_count())
        self.model.setParam('FeasibilityTol', 1e-4)                  # 降低容忍度容易导致求解失败
        self.model.setParam('IntFeasTol', 1e-4)                      # 通过SIMU避免最终结果的差异
        self.model.setParam('Presolve', 2)
        # self.model.setParam('DualReductions', 0)          # dont
        self.model.setParam('Heuristics', 0.2)
        # self.model.setParam('BranchDir', -1)
        # self.model.setParam('VarBranch', 1)  
        self.model.setParam('PreQLinearize', 2)
        # self.model.setParam("ScaleFlag", 2)
        self.model.setParam('IntegralityFocus', 1)

        # self.model.setParam("FuncMaxVal", 1e4)
        self.model.setParam("ScaleFlag", 2)                          # 用于调节数值比例问题coefficient range
        self.model.setParam("NumericFocus", 2)
        # self.model.setParam('MIPGap', 0.015)
        
        self.model.setParam('LogFile',os.path.join(self.outputdir, "Solver.log"))

        # 设置较严格的容差，确保约束生效
        # self.model.setParam('FeasibilityTol', 1e-9)
        # self.model.setParam('IntFeasTol', 1e-9)
        # self.model.setParam('OptimalityTol', 1e-9)

        self.result = {}
        self.dataflow = {}

    def run(self):
        Logger.debug("Start Running MIP Solver")
        model = self.model
        COST = _Cost_model(acc=self.acc, model=self.model, ops=self.ops)
        acc:CIM_Acc = self.acc
        ops:WorkLoad = self.ops
        factors = self.ops.Factors
        
        Num_factors = sum(len(fs) for fs in factors[1:])
        MAX_FACTOR = max([item for sublist in factors for item in sublist])
        MAX_SIZE = self.ops.size

        Logger.critical(ops)
        Logger.critical(f"factors: {[f'{ops.dim2Dict[d]}:{factors[d]}' for d in range(1, ops.Num_dim) ]}")

        ###########################################################  Variable & Constant & Constraints  ##################################################################
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        logF = {(d, f): math.log(factors[d][f]) 
                        for d in range(1, ops.Num_dim) 
                        for f in range(len(factors[d]))}

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        indic_factor2SpUr = gp.tupledict()          # indic_factor2SpUr[d,f,u] = {0,1}
        # "spatial_mapping_hint": {"D1": ["K"], "D3": ["K", "OX", "OY"]},
        # ------------------------------- WTD
        mappingRule = {}
        for u in range(acc.Num_SpUr):
            for d in range(1, ops.Num_dim):
                mappingRule[u,d] = 0
        for Dchar in ['K','P','Q']:
            mappingRule[0,ops.dict2Dim(Dchar)] = 1
        for Dchar in ['R','S','C']:
            mappingRule[1,ops.dict2Dim(Dchar)] = 1
        mappingRule[2,ops.dict2Dim("K")] = 1
        # 'spatial_mapping_hint': {'D1': ['K'], 'D3': ['K', 'OX', 'OY'], 'D2': ['B', 'K', 'G', 'OX', 'OY', 'C', 'FX', 'FY']}, 

        for d in range(1, ops.Num_dim):
            if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                for u in range(acc.Num_SpUr):
                    indic_factor2SpUr[d,0,u] = 0
                continue
            for f in range(len(factors[d])):
                for u in range(acc.Num_SpUr):
                    if mappingRule[u,d] == 1:
                        indic_factor2SpUr[d,f,u] = model.addVar(vtype=GRB.BINARY, name=f"indic_factor2SpUr_({ops.dim2Dict[d]},{f},{u})")
                    else:
                        indic_factor2SpUr[d,f,u] = 0
        
        for u in range(acc.Num_SpUr):
            sum_u = 0
            for d in range(1, ops.Num_dim):
                if len(factors[d])==1 and factors[d][0]==1:      # DimSize == 1
                    continue
                if mappingRule[u,d] == 1:
                    for f in range(len(factors[d])):
                        sum_u += logF[d,f]*indic_factor2SpUr[d,f,u]

            if u == 0:
                model.addConstr(sum_u == math.log(acc.SpUnrolling[u]), name=f"C_Spatial_Unrolling_({u})")
            else:
                model.addConstr(sum_u <= math.log(acc.SpUnrolling[u]), name=f"C_Spatial_Unrolling_({u})")
                # model.addConstr(sum_u >= math.log(acc.SpUnrolling[u]) + math.log(0.5), name=f"C_Spatial_Unrolling_({u})_2")
                # if math.prod(math.prod(factors[d]) if mappingRule[u,d] else 1 for d in range(1,ops.Num_dim)) >= acc.SpUnrolling[u] * CONST.UTIL_COEFFICIENT:
                if ops.R <= 5:
                    model.addConstr(sum_u >= math.log(acc.SpUnrolling[u]) + math.log(CONST.UTIL_COEFFICIENT), name=f"C_Spatial_Unrolling_({u})_2")

        par_u = {}
        max_b = max(ops.dim2bound[ops.dict2Dim(Dchar)] for Dchar in ['R','S','P','Q'])
        for u in range(acc.Num_SpUr):
            for Dchar in ['R','S','P','Q']:
                d = ops.dict2Dim(Dchar)
                par_u[u,d] = model.addVar(lb=0, ub=math.log(ops.dim2bound[d]), vtype=GRB.CONTINUOUS, name=f"par_d_({u},{Dchar})")
                model.addConstr(par_u[u,d] == quicksum(logF[d,f]*indic_factor2SpUr[d,f,u] for f in range(len(factors[d]))), name=f"C_par_d_({u},{Dchar})")

            if ops.P + ops.R == ops.Q + ops.S:
                model.addConstr(par_u[u,ops.dict2Dim('P')] + par_u[u,ops.dict2Dim('R')] >= par_u[u,ops.dict2Dim('Q')] + par_u[u,ops.dict2Dim('S')], 
                                name=f"C_Symmetrical_pruning_{u}")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        
        indic_factor2Loop = gp.tupledict()          # indic_factor2Loop[d,f,i] = {0,1}
                                                    
        for i in range(Num_factors):
            for d in range(1, ops.Num_dim):
                if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                    indic_factor2Loop[d,0,i] = 0
                    continue
                for f in range(len(factors[d])):
                    indic_factor2Loop[d,f,i] = model.addVar(vtype=GRB.BINARY, name=f"Indic_factor2Loop_({ops.dim2Dict[d]},{f},{i})")
            
        # # # # # Key Mapping Constraints # # # # # 
        for d in range(1, ops.Num_dim):
            if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                continue
            for f in range(len(factors[d])):
                model.addConstr( quicksum(indic_factor2Loop[d,f,i] for i in range(Num_factors)) + quicksum(indic_factor2SpUr[d,f,u] for u in range(acc.Num_SpUr)) == 1 ,
                                 name=f"C_Uniqueness_FactorMapping_({ops.dim2Dict[d]},{f})")
                
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        # ['PLACEHOLD'0, 'Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7] acc.Num_mem = 8

        indic_factor2Mem = gp.tupledict()           # indic_factor2Mem[d,f,op,m] = {0,1}
        for d in range(1, ops.Num_dim):
            if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                for op, op_name in enumerate(['I','W','O']):
                    for m in range(1,acc.Num_mem):
                        indic_factor2Mem[d,0,op,m] = 0
                continue
            for f in range(len(factors[d])):
                for op, op_name in enumerate(['I','W','O']):
                    for m in range(1,acc.Num_mem):
                        if acc.mappingArray[op][m] == 1:
                            indic_factor2Mem[d,f,op,m] = model.addVar(vtype=GRB.BINARY, name=f"indic_factor2Mem_({ops.dim2Dict[d]},{f},{op_name},{acc.mem2dict(m)})")
                        else:
                            indic_factor2Mem[d,f,op,m] = 0
                    model.addConstr(quicksum(indic_factor2Mem[d,f,op,m] 
                                             for m in range(1,acc.Num_mem) ) == quicksum(indic_factor2Loop[d,f,i] for i in range(Num_factors)),
                                     name=f"C_Uniqueness_indic_factor2Mem_({ops.dim2Dict[d]},{f},{op_name})")

        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('P'),0,0] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('Q'),0,0] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('K'),3,0] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('C'),1,1] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('C'),2,1] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('C'),3,1] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('K'),0,2] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('K'),1,2] == 1)
        # model.addConstr(indic_factor2SpUr[ops.dict2Dim('K'),2,2] == 1)

        # model.addConstr(indic_factor2Mem[ops.dict2Dim('Q'),2,0,1] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('R'),0,0,2] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('P'),1,0,2] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('P'),2,0,2] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('S'),0,0,4] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('C'),0,0,4] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('Q'),1,0,4] == 1)

        # model.addConstr(indic_factor2Mem[ops.dict2Dim('Q'),2,1,2] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('R'),0,1,2] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('P'),1,1,7] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('P'),2,1,7] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('S'),0,1,7] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('C'),0,1,7] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('Q'),1,1,7] == 1)

        # model.addConstr(indic_factor2Mem[ops.dict2Dim('Q'),2,2,1] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('R'),0,2,3] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('P'),1,2,3] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('P'),2,2,3] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('S'),0,2,3] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('C'),0,2,3] == 1)
        # model.addConstr(indic_factor2Mem[ops.dict2Dim('Q'),1,2,3] == 1)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        indic_usedMem = gp.tupledict()            # indic_usedMem[m,op] = {0,1}           
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 1:
                    indic_usedMem[m,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_usedMem_({acc.mem2dict(m)},{op_name})")
                    tmp_var_or = []
                    for d in range(1, ops.Num_dim):
                        if (len(factors[d])==1 and factors[d][0]==1):       # DimSize == 1
                            continue
                        for f in range(len(factors[d])):
                            tmp_var_or.append(indic_factor2Mem[d,f,op,m])
                
                    model.addGenConstrOr(indic_usedMem[m,op], tmp_var_or, name=f'C_Indic_usedMem_({acc.mem2dict(m)},{op_name})')
                else:
                    indic_usedMem[m,op] = 0
        
        indic_usedLoop = gp.tupledict()
        for i in range(Num_factors):
            indic_usedLoop[i] = model.addVar(vtype=GRB.BINARY, name=f"Indic_usedLoop_({i})")
            model.addConstr(indic_usedLoop[i] == quicksum(indic_factor2Loop[d,f,i] for d in range(1, ops.Num_dim) for f in range(len(factors[d]))), 
                            name=f"C_Uniqueness_Indic_factor2Loop_({i})")
        model.addConstr(indic_usedLoop[Num_factors-1] == 1, name=f"C_Monotonicity_indicLoop")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
            
        indic_loop2Mem = gp.tupledict()           # indic_loop2Mem[i,op,m] = {0,1}
        for i in range(Num_factors):
            for op, op_name in enumerate(['I','W','O']):
                for m in range(1,acc.Num_mem):
                    if acc.mappingArray[op][m] == 1:
                        indic_loop2Mem[i,op,m] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)})")
                        for d in range(1, ops.Num_dim):
                            if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                                continue
                            for f in range(len(factors[d])):
                                model.addConstr(indic_loop2Mem[i,op,m]>=indic_factor2Mem[d,f,op,m] + indic_factor2Loop[d,f,i]-1, 
                                                name=f"C_Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                    else:
                        indic_loop2Mem[i,op,m] = 0
                model.addConstr(quicksum(indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)) == indic_usedLoop[i], 
                                name=f"C_Indic_loop2Mem_({i},{op_name})")
                        
        loop2Mem = gp.tupledict()                   # loop2Mem[i,op] = mem
        for op, op_name in enumerate(['I','W','O']):
            if op_name == 'W':
                loop2Mem[Num_factors, op] = acc.Macro2mem
            else:
                loop2Mem[Num_factors, op] = acc.Num_mem
            for i in range(Num_factors):
                loop2Mem[i,op] = model.addVar(lb=1, ub=acc.Num_mem-1, vtype=GRB.INTEGER, name=f"loop2Mem_({i},{op_name})")
                model.addGenConstrIndicator(indic_usedLoop[i], True, loop2Mem[i,op] == quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)),
                                            name=f"C_loop2Mem_({i},{op_name})")
        for i in range(Num_factors-1):
            for op, op_name in enumerate(['I','W','O']):
                model.addConstr(loop2Mem[i,op]<=loop2Mem[i+1,op], name=f"C_Sequence_loop2Mem_({i},{op_name})")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
                            
        loop2Factor = gp.tupledict()                # loop2Factor[i] = X[d,f]
        for i in range(Num_factors):
            loop2Factor[i] = model.addVar(lb=0, ub=MAX_FACTOR, vtype=GRB.INTEGER, name=f"loop2Factor_({i})")
            model.addConstr(loop2Factor[i] == quicksum(indic_factor2Loop[d,f,i] * factors[d][f] 
                                                          for d in range(1, ops.Num_dim) for f in range(len(factors[d]))), name=f"C_loop2Factor_({i})") 
                                                    
        indic_loop2Factor = gp.tupledict()
        primefactor = ops.PrimeFactors
        for i in range(Num_factors):
            for p in range(len(primefactor)):
                indic_loop2Factor[i,p] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Factor_({i},{p})")
            model.addConstr(quicksum(indic_loop2Factor[i,p] for p in range(len(primefactor))) == indic_usedLoop[i])
            model.addConstr(quicksum(primefactor[p] * indic_loop2Factor[i,p] for p in range(len(primefactor))) == loop2Factor[i])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        for op in range(3):
            acc.mappingArray[op].append(1)
            indic_usedMem[acc.Num_mem,op] = 1
        indic_nxtMem = gp.tupledict()
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    continue
                for m1 in range(m+1, acc.Num_mem+1):
                    if acc.mappingArray[op][m1] == 1:
                        indic_nxtMem[m,m1,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_nextMem_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})")
                    else:
                        indic_nxtMem[m,m1,op] = 0
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    continue
                sum_nxt = 0
                for m1 in range(m+1, acc.Num_mem+1):
                    # if m1 == acc.Num_mem:
                    #     sum_nxt += indic_nxtMem[m,m1,op]
                    if acc.mappingArray[op][m1] == 1:
                        for m2 in range(m1+1, acc.Num_mem+1):
                            if acc.mappingArray[op][m2] == 0:
                                continue
                            else:
                                model.addConstr(1 - indic_usedMem[m1,op] + indic_nxtMem[m,m1,op] >= indic_nxtMem[m,m2,op])
                        model.addConstr(indic_nxtMem[m,m1,op] <= indic_usedMem[m1,op])
                        sum_nxt += indic_nxtMem[m,m1,op]
                model.addConstr(sum_nxt == indic_usedMem[m,op])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        indic_doubleMem = gp.tupledict()        # indic_doubleMem[m,op] = {0,1}
        indic_doubleLoop = gp.tupledict()       # indic_doubleloop[i,op] = {0,1}
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.double_config[m][op] == 0:   # or doubleConfig
                    indic_doubleMem[m,op] = 0
                else:
                    indic_doubleMem[m,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleMem_({acc.mem2dict(m)},{op_name})")
                    # model.addConstr(indic_doubleMem[m,op]==0)
        for i in range(Num_factors):
            for op, op_name in enumerate(['I','W','O']):
                indic_doubleLoop[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleLoop_({i},{op_name})")
                tmp_indics = []
                for m in range(1,acc.Num_mem):
                    if acc.mappingArray[op][m] == 1:
                        for m1 in range(m+1, acc.Num_mem):
                            if acc.double_config[m1][op] == 1:
                                tmp_indic = model.addVar(vtype=GRB.BINARY, name=f"tmp_Indic_doubleLoop_({i},{op_name},{acc.mem2dict(m)},{acc.mem2dict(m1)})")
                                model.addGenConstrAnd(tmp_indic, [indic_loop2Mem[i,op,m], indic_nxtMem[m,m1,op], indic_doubleMem[m1,op]],
                                                    name=f"C_tmp_Indic_doubleLoop_({i},{op_name},{acc.mem2dict(m)},{acc.mem2dict(m1)})")
                                tmp_indics.append(tmp_indic)
                model.addGenConstrOr(indic_doubleLoop[i,op], tmp_indics, name=f"C_Indic_doubleLoop_({i},{op_name})")
        

        ####################################################################  Capacity Constraints   #######################################################################

        lg_dimExistMem = gp.tupledict()         # lg_dimExistMem[m,op,d] = dimSize
        for m in range(2,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for d in range(1, ops.Num_dim):
                    if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                        lg_dimExistMem[m,op,d] = 0
                        continue
                    if acc.mappingArray[op][m] == 0 or ops.relevance[op][d] == 0:
                        continue
                    else:    
                        lg_dimExistMem[m,op,d] = model.addVar(lb=0, ub=math.log(ops.dim2bound[d]), vtype=GRB.CONTINUOUS,
                                                            name=f"lg_dimExistMem_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                        dimExistMem = 0
                        for m1 in range(m,acc.Num_mem):
                            if acc.mappingArray[op][m1] == 1:
                                for f in range(len(factors[d])):
                                    dimExistMem += logF[d,f]*indic_factor2Mem[d,f,op,m1]
                            
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUrArray[u,op]:
                                for f in range(len(factors[d])):
                                    dimExistMem += logF[d,f]*indic_factor2SpUr[d,f,u]
                                
                        model.addConstr(lg_dimExistMem[m,op,d] == dimExistMem, 
                                                    name=f"C_sum_lgdimExistMem_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                        
        lg_dimOfTile = gp.tupledict()         # lg_dimOfTile[m,op,d] = dimSize
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for d in range(1, ops.Num_dim):
                    if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                        lg_dimOfTile[m,op,d] = 0
                        continue
                    if acc.mappingArray[op][m] == 0 or ops.relevance[op][d] == 0:
                        continue
                    else:    
                        lg_dimOfTile[m,op,d] = model.addVar(lb=0, ub=math.log(ops.dim2bound[d]), vtype=GRB.CONTINUOUS,
                                                            name=f"lg_dimOfTile_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                        dimOfTile = 0
                        for m1 in range(m+1,acc.Num_mem):               #  m -> m+1
                            if acc.mappingArray[op][m1] == 1:
                                for f in range(len(factors[d])):
                                    dimOfTile += logF[d,f]*indic_factor2Mem[d,f,op,m1]
                            
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUrArray[u,op]:
                                for f in range(len(factors[d])):
                                    dimOfTile += logF[d,f]*indic_factor2SpUr[d,f,u]
                                
                        model.addConstr(lg_dimOfTile[m,op,d] == dimOfTile, 
                                                    name=f"C_sum_lgdimOfTile_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        exp_dataVolume = gp.tupledict()     # exp_dataVolume[m,op]
        for m in range(2,acc.Num_mem):      # Sufficient off-chip capacity [1-Dram]

            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m]:
                exp_dataVolume[m,op] = model.addVar(lb=1, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.INTEGER, name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                indic_sum, sum_volume = 0, 0
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0
                for i_r, r in enumerate(ops.Divisors[ops.dict2Dim('R')]):
                    for i_s, s in enumerate(ops.Divisors[ops.dict2Dim('S')]):
                        for i_c, c in enumerate(ops.Divisors[ops.dict2Dim('C')]):
                            for i_p, p in enumerate(ops.Divisors[ops.dict2Dim('P')]):
                                for i_q, q in enumerate(ops.Divisors[ops.dict2Dim('Q')]):
                                    if ops.Stride >= r:
                                        h = p * r
                                    else:
                                        h = (p-1) * ops.Stride + r
                                    if ops.Stride >= s:
                                        w = q * s
                                    else:
                                        w = (q-1) * ops.Stride + s
                                    # op_volume = h*w*c
                                    op_volume = min(h,ops.H) * min(w,ops.W) *c
                                    if op_volume <= acc.memSize[m] // acc.precision[m,op]:
                                        indic_opVolume = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_opVolume_({acc.mem2dict(m)},{op_name},{i_r},{i_s},{i_c},{i_p},{i_q})")
                                        sum_volume += indic_opVolume*op_volume
                                        sum_r += indic_opVolume*math.log(r)
                                        sum_s += indic_opVolume*math.log(s)
                                        sum_c += indic_opVolume*math.log(c)
                                        sum_p += indic_opVolume*math.log(p)
                                        sum_q += indic_opVolume*math.log(q)
                                        indic_sum += indic_opVolume
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_r == lg_dimExistMem[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicSumR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_s == lg_dimExistMem[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicSumS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_c == lg_dimExistMem[m,op,ops.dict2Dim('C')], name=f"C_Uniqueness_IndicSumC_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimExistMem[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicSumP_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimExistMem[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicSumQ_({acc.mem2dict(m)},{op_name})")
                model.addConstr(exp_dataVolume[m,op] == sum_volume, name=f"sum_dataVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m]:
                exp_dataVolume[m,op] = model.addVar(lb=1, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.INTEGER, name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                indic_sum, sum_volume = 0, 0
                sum_r, sum_c, sum_k, sum_s = 0,0,0,0
                for i_r, r in enumerate(ops.Divisors[ops.dict2Dim('R')]):
                    for i_s, s in enumerate(ops.Divisors[ops.dict2Dim('S')]):
                        for i_c, c in enumerate(ops.Divisors[ops.dict2Dim('C')]):
                            for i_k, k in enumerate(ops.Divisors[ops.dict2Dim('K')]):
                                op_volume = r*s*c*k
                                if op_volume <= acc.memSize[m] // acc.precision[m,op]:
                                    indic_opVolume = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_opVolume_({acc.mem2dict(m)},{op_name},{i_r},{i_s},{i_c},{i_k})")
                                    sum_volume += indic_opVolume*op_volume
                                    sum_r += indic_opVolume*math.log(r)
                                    sum_s += indic_opVolume*math.log(s)
                                    sum_c += indic_opVolume*math.log(c)
                                    sum_k += indic_opVolume*math.log(k)
                                    indic_sum += indic_opVolume
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_r == lg_dimExistMem[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicSumR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_s == lg_dimExistMem[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicSumS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_c == lg_dimExistMem[m,op,ops.dict2Dim('C')], name=f"C_Uniqueness_IndicSumC_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_k == lg_dimExistMem[m,op,ops.dict2Dim('K')], name=f"C_Uniqueness_IndicSumK_({acc.mem2dict(m)},{op_name})")
                model.addConstr(exp_dataVolume[m,op] == sum_volume, name=f"sum_dataVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m]:
                exp_dataVolume[m,op] = model.addVar(lb=1, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.INTEGER, name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                indic_sum, sum_volume = 0, 0
                sum_p, sum_q, sum_k = 0,0,0
                for i_p, p in enumerate(ops.Divisors[ops.dict2Dim('P')]):
                    for i_q, q in enumerate(ops.Divisors[ops.dict2Dim('Q')]):
                        for i_k, k in enumerate(ops.Divisors[ops.dict2Dim('K')]):
                                op_volume = p*q*k
                                if op_volume <= acc.memSize[m] // acc.precision[m,op]:
                                    indic_opVolume = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_opVolume_({acc.mem2dict(m)},{op_name},{i_p},{i_q},{i_k})")
                                    sum_volume += indic_opVolume*op_volume
                                    sum_p += indic_opVolume*math.log(p)
                                    sum_q += indic_opVolume*math.log(q)
                                    sum_k += indic_opVolume*math.log(k)
                                    indic_sum += indic_opVolume
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimExistMem[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicSumP_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimExistMem[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicSumQ_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_k == lg_dimExistMem[m,op,ops.dict2Dim('K')], name=f"C_Uniqueness_IndicSumK_({acc.mem2dict(m)},{op_name})")
                model.addConstr(exp_dataVolume[m,op] == sum_volume, name=f"sum_dataVolume_({acc.mem2dict(m)},{op_name})")

        exp_transVolume = gp.tupledict()    # exp_transVolume[m,op]
        tile_offChip = gp.tupledict()       # tile_offChip[op]
        for m in range(1,acc.Num_mem):      

            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m]:
                exp_transVolume[m,op] = model.addVar(lb=1, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.INTEGER, name=f"exp_transVolume_({acc.mem2dict(m)},{op_name})")
                indic_sum, sum_tile = 0, 0
                lg_sum_tile = 0
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0
                for i_r, r in enumerate(ops.Divisors[ops.dict2Dim('R')]):
                    for i_s, s in enumerate(ops.Divisors[ops.dict2Dim('S')]):
                        for i_c, c in enumerate(ops.Divisors[ops.dict2Dim('C')]):
                            for i_p, p in enumerate(ops.Divisors[ops.dict2Dim('P')]):
                                for i_q, q in enumerate(ops.Divisors[ops.dict2Dim('Q')]):
                                    if ops.Stride >= r:
                                        h = p * r
                                    else:
                                        h = (p-1) * ops.Stride + r
                                    if ops.Stride >= s:
                                        w = q * s
                                    else:
                                        w = (q-1) * ops.Stride + s
                                    op_volume = min(h,ops.H) * min(w,ops.W) *c
                                    if op_volume <= acc.memSize[m] // acc.precision[m,op]:
                                        indic_opVolume = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_tileVolume_({acc.mem2dict(m)},{op_name},{i_r},{i_s},{i_c},{i_p},{i_q})")
                                        sum_tile += indic_opVolume*op_volume
                                        if m == 1:
                                            lg_sum_tile += indic_opVolume*math.log(op_volume)
                                        sum_r += indic_opVolume*math.log(r)
                                        sum_s += indic_opVolume*math.log(s)
                                        sum_c += indic_opVolume*math.log(c)
                                        sum_p += indic_opVolume*math.log(p)
                                        sum_q += indic_opVolume*math.log(q)
                                        indic_sum += indic_opVolume
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_r == lg_dimOfTile[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicTileR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_s == lg_dimOfTile[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicTileS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_c == lg_dimOfTile[m,op,ops.dict2Dim('C')], name=f"C_Uniqueness_IndicTileC_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimOfTile[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicTileP_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimOfTile[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicTileQ_({acc.mem2dict(m)},{op_name})")
                model.addConstr(exp_transVolume[m,op] == sum_tile, name=f"sum_transVolume_({acc.mem2dict(m)},{op_name})")
                if m == 1:
                    tile_offChip[op] = model.addVar(lb=1,ub=math.log(MAX_SIZE[op]),vtype=GRB.CONTINUOUS, name=f"tile_offChip_({op_name})")
                    model.addConstr(tile_offChip[op] == lg_sum_tile)

            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m]:
                exp_transVolume[m,op] = model.addVar(lb=1, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.INTEGER, name=f"exp_transVolume_({acc.mem2dict(m)},{op_name})")
                indic_sum, sum_tile = 0, 0
                lg_sum_tile = 0
                sum_r, sum_c, sum_k, sum_s = 0,0,0,0
                for i_r, r in enumerate(ops.Divisors[ops.dict2Dim('R')]):
                    for i_s, s in enumerate(ops.Divisors[ops.dict2Dim('S')]):
                        for i_c, c in enumerate(ops.Divisors[ops.dict2Dim('C')]):
                            for i_k, k in enumerate(ops.Divisors[ops.dict2Dim('K')]):
                                op_volume = r*s*c*k
                                if op_volume <= acc.memSize[m] // acc.precision[m,op]:
                                    indic_opVolume = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_tileVolume_({acc.mem2dict(m)},{op_name},{i_r},{i_s},{i_c},{i_k})")
                                    sum_tile += indic_opVolume*op_volume
                                    if m == 1:
                                        lg_sum_tile += indic_opVolume*math.log(op_volume)
                                    sum_r += indic_opVolume*math.log(r)
                                    sum_s += indic_opVolume*math.log(s)
                                    sum_c += indic_opVolume*math.log(c)
                                    sum_k += indic_opVolume*math.log(k)
                                    indic_sum += indic_opVolume
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_r == lg_dimOfTile[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicTileR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_s == lg_dimOfTile[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicTileS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_c == lg_dimOfTile[m,op,ops.dict2Dim('C')], name=f"C_Uniqueness_IndicTileC_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_k == lg_dimOfTile[m,op,ops.dict2Dim('K')], name=f"C_Uniqueness_IndicTileK_({acc.mem2dict(m)},{op_name})")
                model.addConstr(exp_transVolume[m,op] == sum_tile, name=f"sum_transVolume_({acc.mem2dict(m)},{op_name})")
                if m == 1:
                    tile_offChip[op] = model.addVar(lb=1,ub=math.log(MAX_SIZE[op]),vtype=GRB.CONTINUOUS, name=f"tile_offChip_({op_name})")
                    model.addConstr(tile_offChip[op] == lg_sum_tile)

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m]:
                exp_transVolume[m,op] = model.addVar(lb=1, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.INTEGER, name=f"exp_transVolume_({acc.mem2dict(m)},{op_name})")
                indic_sum, sum_tile = 0, 0
                lg_sum_tile = 0
                sum_p, sum_q, sum_k = 0,0,0
                for i_p, p in enumerate(ops.Divisors[ops.dict2Dim('P')]):
                    for i_q, q in enumerate(ops.Divisors[ops.dict2Dim('Q')]):
                        for i_k, k in enumerate(ops.Divisors[ops.dict2Dim('K')]):
                                op_volume = p*q*k
                                if op_volume <= acc.memSize[m] // acc.precision[m,op]:
                                    indic_opVolume = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_tileVolume_({acc.mem2dict(m)},{op_name},{i_p},{i_q},{i_k})")
                                    sum_tile += indic_opVolume*op_volume
                                    if m == 1:
                                        lg_sum_tile += indic_opVolume*math.log(op_volume)
                                    sum_p += indic_opVolume*math.log(p)
                                    sum_q += indic_opVolume*math.log(q)
                                    sum_k += indic_opVolume*math.log(k)
                                    indic_sum += indic_opVolume
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimOfTile[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicTileP_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimOfTile[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicTileQ_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_k == lg_dimOfTile[m,op,ops.dict2Dim('K')], name=f"C_Uniqueness_IndicTileK_({acc.mem2dict(m)},{op_name})")
                model.addConstr(exp_transVolume[m,op] == sum_tile, name=f"sum_transVolume_({acc.mem2dict(m)},{op_name})")
                if m == 1:
                    tile_offChip[op] = model.addVar(lb=1,ub=math.log(MAX_SIZE[op]),vtype=GRB.CONTINUOUS, name=f"tile_offChip_({op_name})")
                    model.addConstr(tile_offChip[op] == lg_sum_tile)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        dataVolume = gp.tupledict()
        for m in range(2,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m]:
                    dataVolume[m,op] = var_mul01(model, indic_usedMem[m,op], exp_dataVolume[m,op], f"dataVolume_({acc.mem2dict(m)},{op_name})")
                else:
                    dataVolume[m,op] = 0
        
        sum_util = 0
        for m in range(2,acc.Num_mem):
            tmp_datavolume = 0
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 1:
                    tmp_datavolume += (dataVolume[m,op] + var_mul01(model, indic_doubleMem[m,op], dataVolume[m,op], f"C_mul_{m}_{op}")) * acc.precision[m,op]
                    sum_util += dataVolume[m,op]* (acc.precision[m,op] * m)
                    # dataVolume better than tmp_dataVolume
            model.addConstr( tmp_datavolume <= acc.memSize[m], name=f"C_dataVolume_({acc.mem2dict(m)})" )

        transVolume = gp.tupledict()                # transVolume[m,op] = INTEGER
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m]:
                    transVolume[m,op] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"transVolume_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(transVolume[m,op] == var_mul01(model, indic_usedMem[m,op], exp_transVolume[m,op], f"transVolume_({acc.mem2dict(m)},{op_name})_tmp") * acc.precision[m,op])
                else:
                    transVolume[m,op] = 0

        ####################################################################  Execution Performance   #######################################################################

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Energy - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Latency - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - -#          
        indic_xMem = gp.tupledict()                 # indic_xMem[i,op] = {0,1}
        for i in range(Num_factors):
            for op, op_name in enumerate(['I','W','O']):
                indic_xMem[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_xMem_({i},{op_name})")
                model.addGenConstrIndicator(indic_xMem[i,op], True, loop2Mem[i,op] + 1 <= loop2Mem[i+1,op], name=f"C_xMem_({i},{op_name})_T")
                model.addGenConstrIndicator(indic_xMem[i,op], False, loop2Mem[i,op] == loop2Mem[i+1,op], name=f"C_xMem_({i},{op_name})_F")
                model.addConstr(indic_xMem[i,op]<=indic_usedLoop[i], name=f"C_legality_xMem_({i},{op_name})")

        transfer = gp.tupledict()
        transfer_z = gp.tupledict()
        transfer_z_oneway = gp.tupledict()
        for i in range(Num_factors):
            for op, op_name in enumerate(['I','W','O']):
                # if i < Num_factors-1:
                #     transfer[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"transfer_({i},{op_name})")
                #     for m in range(1,acc.Num_mem):
                #         if acc.mappingArray[op][m]:
                #             model.addGenConstrIndicator(indic_loop2Mem[i,op,m], True, transfer[i,op]*acc.bw[m]==transVolume[m,op], name=f"C_transfer_({i},{op_name},{acc.mem2dict(m)})")
                # else:
                #     transfer[i,op] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"transfer_({i},{op_name})")
                #     for m in range(1,acc.Num_mem):
                #         if acc.mappingArray[op][m]:
                #             model.addGenConstrIndicator(indic_loop2Mem[i,op,m], True, transfer[i,op]*acc.bw[m]>=transVolume[m,op], name=f"C_transfer_({i},{op_name},{acc.mem2dict(m)})")
                transfer[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"transfer_({i},{op_name})")
                for m in range(1,acc.Num_mem):
                    if acc.mappingArray[op][m]:         # WTD
                        model.addGenConstrIndicator(indic_loop2Mem[i,op,m], True, transfer[i,op]*acc.bw[m]>=transVolume[m,op], name=f"C_transfer_({i},{op_name},{acc.mem2dict(m)})")
                        # model.addGenConstrIndicator(indic_loop2Mem[i,op,m], True, transfer[i,op]*acc.bw[m]-acc.bw[m]<=transVolume[m,op], name=f"C_transfer_({i},{op_name},{acc.mem2dict(m)})_2")

                transfer_z[i,op] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"transfer_z_({i},{op_name})")
                if i < Num_factors-1:
                    if op < 2:
                        model.addGenConstrIndicator(indic_xMem[i,op], True, transfer_z[i,op]==transfer[i,op], name=f"test_({i},{op_name})") 
                    else:
                        model.addGenConstrIndicator(indic_xMem[i,op], True, transfer_z[i,op]==transfer[i,op]*2, name=f"test_({i},{op_name})") 
                    model.addGenConstrIndicator(indic_xMem[i,op], False, transfer_z[i,op]==0, name=f"test_({i},{op_name})_F") 
                if op_name == 'O':
                    transfer_z_oneway[i,op] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"transfer_output_oneway_({i},{op_name})")
                    model.addConstr(transfer_z_oneway[i,op] * 2 == transfer_z[i,op], name=f"C_transfer_output_oneway_({i},{op_name})")

        indic_usedLastMem = gp.tupledict()              # indic_usedLastMem[op] = {0,1}
        for op, op_name in enumerate(['I','W','O']):
            indic_usedLastMem[op] = model.addVar(vtype=GRB.BINARY, name=f"indic_usedLastMem_({op_name})")
            model.addGenConstrIndicator(indic_usedLastMem[op], True, loop2Mem[Num_factors-1,op] == acc.lastMem[op])
            model.addGenConstrIndicator(indic_usedLastMem[op], False, loop2Mem[Num_factors-1,op]+1 <= acc.lastMem[op])
            if op < 2:
                model.addConstr(transfer_z[Num_factors-1,op] == (transfer[Num_factors-1,op] + (1-indic_usedLastMem[op])*(2 if op_name=='O' else 1) ), 
                                name=f"C_RegTrans_({op_name})")
            else:
                model.addConstr(transfer_z[Num_factors-1,op] == (transfer[Num_factors-1,op]*2 + (1-indic_usedLastMem[op])*(2 if op_name=='O' else 1) ), 
                                name=f"C_RegTrans_({op_name})")

        latency_cp = gp.tupledict()         # latency_cp[i] = Latency of Critical Path
        latency_consu = gp.tupledict()
        critical_term = gp.tupledict()
        for i in range(Num_factors):
            latency_cp[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"latency_cp_({i})")
            critical_term[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"creitical_({i})")
            for op, op_name in enumerate(['I','W','O']):
                latency_consu[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"latency_consume_({i},{op_name})")
        for op, op_name in enumerate(['I','W','O']):
            critical_term[Num_factors] = acc.t_MAC
            latency_consu[Num_factors,op] = acc.t_MAC


        latency_opN = gp.tupledict()
        latency_maxTL = gp.tupledict()
        exp_critical_term = gp.tupledict()         # latency_cp[i] = Latency of Critical Path
        for i in range(Num_factors):
            exp_critical_term[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"exp_critical_term_({i})")
            for p in range(len(primefactor)):
                model.addGenConstrIndicator(indic_loop2Factor[i,p], True, exp_critical_term[i] == latency_cp[i]*primefactor[p], name=f"C_exp_critical_({i},{p})")
            for op, op_name in enumerate(['I','W','O']):
                latency_opN[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"latency_opN_({i},{op_name})")
                latency_maxTL[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"latency_maxTL_({i},{op_name})")
                model.addConstr(latency_maxTL[i,op] == gp.max_(transfer_z[i,op],latency_consu[i+1,op]), name=f"C_tmp_lmax_({i},{op_name})")
                model.addGenConstrIndicator(indic_doubleLoop[i,op], True,  latency_opN[i,op] == latency_maxTL[i,op], 
                                            name=f"C_latency_opN_({i},{op_name})_T")
                
                model.addGenConstrIndicator(indic_doubleLoop[i,op], False, latency_opN[i,op] == transfer_z[i,op]+latency_consu[i+1,op], 
                                            name=f"C_latency_opN_({i},{op_name})_F")
                
            model.addGenConstrMax(latency_cp[i], [critical_term[i+1]]+[latency_opN[i,op] for op in range(3)], constant=0, name=f"C_latency_cp_({i})")
                
        exp_latency_consu = gp.tupledict()
        latency_transMatters = gp.tupledict()
        for i in range(Num_factors):
            for op, op_name in enumerate(['I','W','O']):
                exp_latency_consu[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"exp_latency_consume_({i},{op_name})")
                latency_transMatters[i,op] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"latency_transMatters_({i},{op_name})")

                tmp_consu = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_consu_({i},{op_name})")
                for p in range(len(primefactor)):
                    model.addGenConstrIndicator(indic_loop2Factor[i,p], True, tmp_consu == latency_cp[i] * (primefactor[p]-1) + latency_consu[i+1,op],
                                                 name=f"C_tmp_consu_({i},{op_name},{p})")
                model.addGenConstrIndicator(indic_xMem[i,op], False, exp_latency_consu[i,op] == tmp_consu)          
                
                
                if op < 2:      # I,W
                    tmp_consu_sing = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_consu_sing_({i},{op_name})")
                    for p in range(len(primefactor)):
                        model.addGenConstrIndicator(indic_loop2Factor[i,p], True, tmp_consu_sing == latency_cp[i] * (primefactor[p]-2) + 2*transfer_z[i,op] + latency_consu[i+1,op])
                    
                    tmp_consu_doub_1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_consu_doub_({i},{op_name})_1")
                    for p in range(len(primefactor)):
                        model.addGenConstrIndicator(indic_loop2Factor[i,p], True, tmp_consu_doub_1 == latency_cp[i] * max(0,primefactor[p]-3) + 2*transfer_z[i,op] + latency_maxTL[i,op])
                    tmp_consu_doub_2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_consu_doub_({i},{op_name})_2")
                    for p in range(len(primefactor)):
                        model.addGenConstrIndicator(indic_loop2Factor[i,p], True, tmp_consu_doub_2 == transfer_z[i,op] * primefactor[p])

                    tmp_vmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_max_double12_({i},{op_name})")
                    model.addGenConstrMax(tmp_vmax, [tmp_consu_doub_1, tmp_consu_doub_2])
                    model.addGenConstrIndicator(indic_doubleLoop[i,op], True, latency_transMatters[i,op] == tmp_vmax)
                    model.addGenConstrIndicator(indic_doubleLoop[i,op], False, latency_transMatters[i,op] == tmp_consu_sing)

                else:           # O
                    tmp_consu_sing = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_consu_sing_({i},{op_name})")
                    for p in range(len(primefactor)):
                        model.addGenConstrIndicator(indic_loop2Factor[i,p], True, tmp_consu_sing == latency_cp[i] * (primefactor[p]-1) + transfer_z[i,op] + latency_consu[i+1,op])
                    
                    tmp_consu_doub_1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_consu_doub_({i},{op_name})_1")
                    
                    tmp_vmax_1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_max_Outdouble_({i},{op_name})_1")
                    tmp_vmax_2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tmp_max_Outdouble_({i},{op_name})_2")
                    model.addGenConstrMax(tmp_vmax_1, [transfer_z_oneway[i,op], latency_cp[i]])
                    model.addGenConstrMax(tmp_vmax_2, [transfer_z_oneway[i,op], latency_consu[i+1,op]])

                    for p in range(len(primefactor)):
                        model.addGenConstrIndicator(indic_loop2Factor[i,p], True, tmp_consu_doub_1 == \
                                                    latency_cp[i] * (primefactor[p]-2) + transfer_z[i,op] + tmp_vmax_1 + tmp_vmax_2)

                    model.addGenConstrIndicator(indic_doubleLoop[i,op], True, latency_transMatters[i,op] == tmp_consu_doub_1)
                    model.addGenConstrIndicator(indic_doubleLoop[i,op], False, latency_transMatters[i,op] == tmp_consu_sing)        

                model.addGenConstrIndicator(indic_xMem[i,op], True, exp_latency_consu[i,op] == latency_transMatters[i,op])

        for i in range(Num_factors):
            model.addGenConstrIndicator(indic_usedLoop[i], True, critical_term[i]==exp_critical_term[i])
            model.addGenConstrIndicator(indic_usedLoop[i], False, critical_term[i]==critical_term[i+1])
            for op, op_name in enumerate(['I','W','O']): 
                model.addGenConstrIndicator(indic_usedLoop[i], True, latency_consu[i,op]==exp_latency_consu[i,op])
                model.addGenConstrIndicator(indic_usedLoop[i], False, latency_consu[i,op]==latency_consu[i+1,op])

            #  - - - - - - - - - - - - - - - - - - - Spatial Constraints - - - - - - - - - - - - - - - - - - -
            # Macro-level           K <= D1 | R*S*C  <= D2 * D3 | P-Q <= D3
            # localBuffer-level     Product <= Num_macro
            # GlobalBuffer-level    Product <= Num_core
            # Dram-level            Product <= 1
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # self.dim2Dict = ['R', 'S', 'P', 'Q', 'C', 'K'] 

        
        res_latency = model.addVar(lb=1, vtype=GRB.INTEGER,     name="res_latency")
        res_energy  = model.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_energy")
        res_EDP     = model.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_EDP")
        model.addConstr(res_latency >= critical_term[0])
        for op in range(3):
            model.addConstr(res_latency >= latency_consu[0,op])
        

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        utliz_all = model.addVar(vtype=GRB.CONTINUOUS, name=f"utilzation_all")
        model.addConstr(utliz_all == sum((indic_usedMem[1,op]*MAX_SIZE[op]) for op in range(3)) )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        
        model.ModelSense = GRB.MINIMIZE

        if FLAG.WEIGHT_STATIONARY:
                
            for Dchar1 in ['R','S','C']:
                d1 = ops.dict2Dim(Dchar1)
                for f1 in range(len(factors[d1])):
                    for Dchar2 in ['P','Q']:
                        d2 = ops.dict2Dim(Dchar2)
                        for f2 in range(len(factors[d2])):
                            model.addConstr( 
                                            quicksum(i*indic_factor2Loop[d1,f1,i] for i in range(Num_factors)) <= 
                                            quicksum(i*indic_factor2Loop[d2,f2,i] for i in range(Num_factors)) + 
                                            (Num_factors+1)*quicksum(indic_factor2SpUr[d2,f2,u] for u in range(acc.Num_SpUr)),
                                             name=f"C_WS_dimSquence_({Dchar1}_{f1}_{Dchar2}_{f2})")
            
            exp_ws = 0

            for Dchar1 in ['R','S','C']:
                d = ops.dict2Dim(Dchar1)
                op, op_name = 1,'W'     # Weight
                if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                    continue
                for f in range(len(factors[d])):
                    exp_ws += logF[d,f]*indic_factor2Mem[d,f,op,acc.Macro2mem]
                    exp_ws += logF[d,f]*indic_factor2SpUr[d,f,1]
            d = ops.dict2Dim('K')
            for f in range(len(factors[d])):
                exp_ws += logF[d,f]*indic_factor2Mem[d,f,op,acc.Macro2mem]
                exp_ws += logF[d,f]*indic_factor2SpUr[d,f,0]
                exp_ws += logF[d,f]*indic_factor2SpUr[d,f,2]
            model.setObjectiveN(-exp_ws, 0, priority=5, name='WeightStationary')    
            env0 = model.getMultiobjEnv(0)                                     
            env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.4)
            env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.4 * 0.1)

            model.setObjectiveN(res_latency, 1, priority=3, name='Latency')    
            env1 = model.getMultiobjEnv(1)                                     
            env1.setParam('TimeLimit', CONST.TIMELIMIT * 0.4)
            env1.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.4 * 0.1)
            model.setObjectiveN(utliz_all, 2, priority=2,
                        reltol=0.0, abstol=0.0, name='Trans-OffChip')
            env2 = model.getMultiobjEnv(2)
            env2.setParam('TimeLimit', CONST.TIMELIMIT * 0.1)

            model.setObjectiveN(-sum_util, 3, priority=1,       # -sum_util = “max sum_util”
                            reltol=0.0, abstol=0.0, name='TemporalInner')
            env3 = model.getMultiobjEnv(3)
            env3.setParam('TimeLimit', CONST.TIMELIMIT * 0.1)
        else:
            model.setObjectiveN(res_latency, 0, priority=3, name='Latency')    
            env0 = model.getMultiobjEnv(0)                                     
            env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.7)
            env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.15)

            model.setObjectiveN(utliz_all, 1, priority=2,
                        reltol=0.0, abstol=0.0, name='Trans-OffChip')
            env1 = model.getMultiobjEnv(1)
            env1.setParam('TimeLimit', CONST.TIMELIMIT * 0.1)

            model.setObjectiveN(-sum_util, 2, priority=1,       # -sum_util = “max sum_util”
                            reltol=0.0, abstol=0.0, name='TemporalInner')
            env2 = model.getMultiobjEnv(2)
            env2.setParam('TimeLimit', CONST.TIMELIMIT * 0.2)
            
            


        
        ####################################################################  Set Constraint Flag ###################################################################
        
        CONST.FLAG_OPT = "Latency"

        

        model.setParam("TimeLimit", CONST.TIMELIMIT)

        ####################################################################  Optimization    #######################################################################

        # FLAG.LOAD_SOLUTION = 1
        # if FLAG.LOAD_SOLUTION:
        #     try:
        #         model.read("MIREDO.sol")
        #         Logger.critical("Load Solution")
        #     except ValueError:
        #         raise ValueError("No MIREDO.sol File")
        model.update()
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        ####################################################################  Debug & Output  #######################################################################

        time_optimization = end_time - start_time
        Logger.critical(f"Optimizing time: {'%.3f' %(time_optimization)}s")

        def set_dataflow():
            Logger.info("# Set Dataflow Loops:")
            loops = LoopNest(acc=acc,ops=ops)
            # loops.tm
            for i in range(Num_factors):
                for d in range(1, ops.Num_dim):
                    if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                        continue
                    for f in range(len(factors[d])):
                        if (round(indic_factor2Loop[d,f,i].x) == 1):
                            Logger.debug(f"{i:<2} for {ops.dim2Dict[d]} in {loop2Factor[i].x:<3}: {[ f'{round(indic_xMem[i,op].x)}|{acc.mem2dict(loop2Mem[i,op].x)}' for op in range(3)]}")
                            
                            loops.tm.append(Mapping(dim=d, 
                                                  dimSize=factors[d][f],
                                                  mem=[round(loop2Mem[i,op].x) for op in range(3)]))
            
            # loops.sm
            for u in range(acc.Num_SpUr):
                for d in range(1, ops.Num_dim):
                    if len(factors[d])==1 and factors[d][0]==1:     # DimSize == 1
                        continue
                    for f in range(len(factors[d])):
                        if mappingRule[u,d] == 1:
                            if (round(indic_factor2SpUr[d,f,u].x) == 1):
                                Logger.debug(f"spfor {ops.dim2Dict[d]} in {factors[d][f]}: {[acc.mem2dict(acc.SpUrArray[u,op]) for op in range(3)]}")
                                loops.sm.append(Mapping(dim=d, 
                                                    dimSize=factors[d][f],
                                                    mem=[acc.SpUrArray[u,op] for op in range(3)]))

            double_tag = [[1 for _ in range(3)] for __ in range(acc.Num_mem+1)]
            for m in range(1,acc.Num_mem+1):
                for op, op_name in enumerate(['I','W','O']):
                    if m==acc.Num_mem:
                        double_tag[m][op] = acc.double_Macro
                        continue
                    if acc.double_config[m][op]:
                        double_tag[m][op] = round(indic_doubleMem[m,op].x)
                    else:
                        double_tag[m][op] = 0
            loops.usr_defined_double_flag = double_tag


            self.dataflow = loops
            
            print("\nDebug in MIP Solver")
            print("dataVolume (Word)")
            for m in range(2,acc.Num_mem):
                print(f"{acc.mem2dict(m):<15}:", end="")
                for op, op_name in enumerate(['I','W','O']):
                    print(f"{(dataVolume[m,op].x if acc.mappingArray[op][m] == 1 else 0):<20}", end="")
                print("")

            print("exp_transVolume") 
            for m in range(1,acc.Num_mem):
                print(f"{acc.mem2dict(m):<15}:", end="")
                for op, op_name in enumerate(['I','W','O']):
                    print(f"{(exp_transVolume[m,op].x if acc.mappingArray[op][m] == 1 else 0):<20}", end="")
                print("")
            print(f"Utilization all: {utliz_all.x}")
            
            print("doubleMem | TransVolume (bit)")
            for m in range(1,acc.Num_mem):
                print(f"{acc.mem2dict(m):<15}:", end="")
                for op, op_name in enumerate(['I','W','O']):
                    tmp = round(transVolume[m,op].x) if acc.mappingArray[op][m] == 1 else 0
                    dtag = round(indic_doubleMem[m,op].x) if acc.double_config[m][op] == 1 else 0
                    print(f"{dtag}|{tmp:<20}", end="")
                print("")

            print("doubleLoop | transfer_time")
            for i in range(Num_factors):
                if indic_usedLoop[i].x < 0.9:
                    continue
                print(f"{i:<3} for {int(loop2Factor[i].x)}:  ", end="")
                for op, op_name in enumerate(['I','W','O']):
                    dtag = round(indic_doubleLoop[i,op].x)
                    print(f"{dtag}|{transfer_z[i,op].x:<10}", end="")
                print(f"Lcp: {round(latency_cp[i].x)}")
            
            print("Latency")
            for i in range(Num_factors):
                if indic_usedLoop[i].x < 0.9:
                    continue
                print(f"{i:<3} for {int(loop2Factor[i].x)}:  ", end="")
                for op, op_name in enumerate(['I','W','O']):
                    print(f"{round(latency_consu[i,op].x):<10}", end="")
                print(f"Critical-term latency: {round(critical_term[i].x)}")

                

        def get_constr_debug(constraintname='test'):
            constr = model.getConstrByName(constraintname)
            debug_str = ""
            if constr is not None:
                # 获取约束对应的行表达式
                row = model.getRow(constr)
                debug_str += f"约束 '{constr.ConstrName}' 中的变量值:\n"
                # 遍历行中的每个变量
                for i in range(row.size()):
                    var = row.getVar(i)
                    coeff = row.getCoeff(i)
                    debug_str += f"---变量 {var.VarName} 的系数为 {coeff}, 解值为 {var.X}\n"
            else:
                debug_str += f"未找到约束 {constraintname}\n"
            Logger.debug(debug_str)
        
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            Logger.critical("MIP Solved successfully !!!")
            self.result = [res_latency.x, res_energy.x, res_EDP.x]
            set_dataflow()
            model.write(os.path.join(self.outputdir, "solution.sol"))
            if CONST.FLAG_OPT=="Latency":
                Logger.debug(f"Get best Latency= {res_latency.x}")
            elif CONST.FLAG_OPT=="Energy":
                Logger.debug(f"Get best Energy= {res_energy.x}")
            elif CONST.FLAG_OPT=="EDP":
                Logger.debug(f"Get best EDP= {res_EDP.x}")
            else:
                Logger.debug(f"Get simple solution, L={res_latency.x}, E={res_energy.x}")
        elif model.status == GRB.Status.TIME_LIMIT:
            # Logger.warning("Solver termination by TIME_LIMIT! Looking for gap solution")
            # model.setParam("MIPGap", CONST.GAP_THRESHOLD)
            # model.setParam("TimeLimit", CONST.TIMELIMIT_AFTER_TLE)
            # model.optimize()
            if model.SolCount > 0: 
                self.result = [res_latency.x, res_energy.x, res_EDP.x]
                set_dataflow()
                model.write(os.path.join(self.outputdir, "solution.sol"))
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
                model.setParam('IISMethod', 2)
                model.computeIIS()
                model.write(os.path.join(self.outputdir, "iis_full.ilp"))
                model.write(os.path.join(self.outputdir, "model.mps"))
                Logger.error("TLE---Debug in contric.ilp")
                # exit()
        else:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            model.setParam("IISMethod", 2) 
            model.computeIIS()
            model.write(os.path.join(self.outputdir, "iis_full.ilp"))
            model.write(os.path.join(self.outputdir, "model.mps"))
            Logger.error(f'Model infeasible !!!')
            exit()


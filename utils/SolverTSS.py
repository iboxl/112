# this file is prepared for project 112
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
from utils.factorization import flexible_factorization
from utils.UtilsFunction.ToolFunction import getDivisors, getUniqueFactors
import copy

class Solver():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad, tu, su, latency_lb, latency_ub, outputdir=None):
        self.acc = copy.deepcopy(acc)
        self.ops = copy.deepcopy(ops)
        self.tu = tu
        self.su = su
        self.lb_latency = latency_lb
        self.ub_latency = max(latency_lb, latency_ub)
        self.outputdir = outputdir

        self.model = gp.Model(name="MIREDO")
        self.model.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
        self.model.setParam('Seed', 112)                               # 随机种子
        # self.model.setParam('NonConvex', 2)
        self.model.setParam('MIPFocus', 3)
        self.model.setParam('Cuts', 2)
        # self.model.setParam('Threads', psutil.cpu_count(logical=False))
        self.model.setParam('Threads', psutil.cpu_count())
        # self.model.setParam('FeasibilityTol', 1e-4)                  # 降低容忍度容易导致求解失败
        # self.model.setParam('IntFeasTol', 1e-4)                      # 通过SIMU避免最终结果的差异
        self.model.setParam('Presolve', 2)
        self.model.setParam('DualReductions', 0)          # dont
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

        # self.model.setParam('FlowCoverCuts', 2)
        # self.model.setParam('Method', 3)
        # self.model.setParam('NoRelHeuristic', 1)

        self.ExpOption = "FuncPieces=-2 FuncPieceError=0.01"
        
        self.model.setParam('LogFile',os.path.join(self.outputdir, "Solver.log"))

        # 设置较严格的容差，确保约束生效
        # self.model.setParam('FeasibilityTol', 1e-9)
        # self.model.setParam('IntFeasTol', 1e-9)
        # self.model.setParam('OptimalityTol', 1e-9)

        self.result = {}
        self.dataflow = {}

        self.maxTrans = [ max([math.ceil(min(ops.size[op] * acc.precision[m,op], acc.memSize[m]) / acc.bw[m]) 
                                for m in range(1, acc.Num_mem) if acc.mappingArray[op][m]] 
                            + [0]) for op in range(3) ]

    def run(self):
        Logger.info('* '*20 + "Start Running MIP Solver" + ' *'*20)
        Logger.critical(f"Most UB is {self.ub_latency}")
        model = self.model
        # COST = _Cost_model(acc=self.acc, model=self.model, ops=self.ops)
        acc:CIM_Acc = self.acc
        ops:WorkLoad = self.ops
        factors = [flexible_factorization(_) for _ in self.tu]

        tempDivisors = [getDivisors(d) for d in self.tu]

        Num_Loops = sum(len(f) for f in factors[1:ops.Num_dim] if f != [1])

        MAX_FACTOR = max([item for sublist in factors for item in sublist])
        MAX_SIZE = self.ops.size

        ###########################################################  Variable & Constant & Constraints  ##################################################################
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        logF = {(d, f): math.log(factors[d][f]) 
                        for d in range(1, ops.Num_dim) 
                        for f in range(len(factors[d]))}

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        
        indic_factor2Loop = gp.tupledict()          # indic_factor2Loop[d,f,i] = {0,1}
                                                    
        for i in range(Num_Loops):
            for d in range(1, ops.Num_dim):
                if factors[d] == [1]:     # DimSize == 1
                    indic_factor2Loop[d,0,i] = 0
                    continue
                for f in range(len(factors[d])):
                    indic_factor2Loop[d,f,i] = model.addVar(vtype=GRB.BINARY, name=f"Indic_factor2Loop_({ops.dim2Dict[d]},{f},{i})")
            
        # # # # # Key Mapping Constraints # # # # # 
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]:     # DimSize == 1
                continue
            for f in range(len(factors[d])):
                model.addConstr( quicksum(indic_factor2Loop[d,f,i] for i in range(Num_Loops)) == 1,
                                 name=f"C_Uniqueness_FactorMapping_({ops.dim2Dict[d]},{f})")
        for i in range(Num_Loops):
            model.addConstr( quicksum(indic_factor2Loop[d,f,i]
                                        for d in range(1, ops.Num_dim)
                                        for f in range(len(factors[d]))) == 1) 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        # ['PLACEHOLD'0, 'Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7] acc.Num_mem = 8

        indic_factor2Mem = gp.tupledict()           # indic_factor2Mem[d,f,op,m] = {0,1}
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]:     # DimSize == 1
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
                                             for m in range(1,acc.Num_mem) ) == 1,
                                     name=f"C_Uniqueness_indic_factor2Mem_({ops.dim2Dict[d]},{f},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        indic_usedMem = gp.tupledict()            # indic_usedMem[m,op] = {0,1}           
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 1:
                    indic_usedMem[m,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_usedMem_({acc.mem2dict(m)},{op_name})")
                    
                    sum_factors_in_mem = quicksum(indic_factor2Mem[d,f,op,m] 
                                                for d in range(1, ops.Num_dim) if factors[d]!=[1]
                                                for f in range(len(factors[d])))

                    model.addConstr(sum_factors_in_mem <= Num_Loops * indic_usedMem[m,op], name=f"C_UsedMem_Trigger_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(indic_usedMem[m,op] <= sum_factors_in_mem, name=f"C_UsedMem_Limit_({acc.mem2dict(m)},{op_name})")
                
                else:
                    indic_usedMem[m,op] = 0
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
            
        indic_loop2Mem = gp.tupledict()           # indic_loop2Mem[i,op,m] = {0,1}
        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                for m in range(1,acc.Num_mem):
                    if acc.mappingArray[op][m] == 1:
                        indic_loop2Mem[i,op,m] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)})")
                        for d in range(1, ops.Num_dim):
                            if factors[d] == [1]:     # DimSize == 1
                                continue
                            for f in range(len(factors[d])):
                                model.addConstr(indic_loop2Mem[i,op,m]>=indic_factor2Mem[d,f,op,m] + indic_factor2Loop[d,f,i]-1, 
                                                name=f"C_Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                    else:
                        indic_loop2Mem[i,op,m] = 0
                model.addConstr(quicksum(indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)) == 1, 
                                name=f"C_Indic_loop2Mem_({i},{op_name})")
                        
        loop2Mem = gp.tupledict()                   # loop2Mem[i,op] = mem
        for op, op_name in enumerate(['I','W','O']):
            if op_name == 'W':
                loop2Mem[Num_Loops, op] = acc.Macro2mem
            else:
                loop2Mem[Num_Loops, op] = acc.Num_mem
            for i in range(Num_Loops):
                loop2Mem[i,op] = model.addVar(lb=1, ub=acc.Num_mem-1, vtype=GRB.INTEGER, name=f"loop2Mem_({i},{op_name})")
                model.addConstr(loop2Mem[i,op] == quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)),
                                            name=f"C_loop2Mem_({i},{op_name})")
        for i in range(Num_Loops-1):
            for op, op_name in enumerate(['I','W','O']):
                model.addConstr(loop2Mem[i,op]<=loop2Mem[i+1,op], name=f"C_Sequence_loop2Mem_({i},{op_name})")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
                            
        loop2Factor = gp.tupledict()                # loop2Factor[i] = X[d,f]
        for i in range(Num_Loops):
            loop2Factor[i] = model.addVar(lb=0, ub=MAX_FACTOR, vtype=GRB.INTEGER, name=f"loop2Factor_({i})")
            model.addConstr(loop2Factor[i] == quicksum(indic_factor2Loop[d,f,i] * factors[d][f] 
                                                          for d in range(1, ops.Num_dim) for f in range(len(factors[d]))), name=f"C_loop2Factor_({i})") 
                                                    
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        for op in range(3):
            acc.mappingArray[op].append(1)
            indic_usedMem[acc.Num_mem,op] = 1
        indic_nxtMem = gp.tupledict()
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    continue

                sum_nxt = 0
                for m1 in range(m+1, acc.Num_mem+1):
                    if acc.mappingArray[op][m1] == 0:
                        indic_nxtMem[m,m1,op] = 0
                        continue
                        
                    indic_nxtMem[m,m1,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_nextMem_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})")

                    model.addConstr(indic_nxtMem[m,m1,op] <= indic_usedMem[m1,op], name=f"C_nxtMem_Valid_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})")
                    for m2 in range(m+1, m1):
                        if acc.mappingArray[op][m2] == 1:
                            model.addConstr(indic_nxtMem[m, m1, op] <= 1 - indic_usedMem[m2, op],
                                            name=f"C_NxtMem_Block_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})_by_({acc.mem2dict(m2)})")
                    
                    sum_nxt += indic_nxtMem[m,m1,op]

                model.addConstr(sum_nxt == indic_usedMem[m,op])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        indic_doubleMem = gp.tupledict()        # indic_doubleMem[m,op] = {0,1}
        indic_doubleLoop = gp.tupledict()       # indic_doubleloop[i,op] = {0,1} 
        indic_feeds_DB = gp.tupledict()         # indic_feeds_DB[m,op] = {0,1}  表示 Memory m 的下一级是否双缓冲（无论是通过 Bypass 还是直接连接）

        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.double_config[m][op] == 0:   # or doubleConfig
                    indic_doubleMem[m,op] = 0
                else:
                    indic_doubleMem[m,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleMem_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(indic_doubleMem[m,op] <= indic_usedMem[m,op], name=f"C_DoubleMem_Valid_{m}_{op_name}")

        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    indic_feeds_DB[m, op] = 0
                    continue
                indic_feeds_DB[m, op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_feeds_DB_({acc.mem2dict(m)},{op_name})")
                
                quad_expr = 0
                for m1 in range(m+1, acc.Num_mem): # 注意边界，假设 acc.Num_mem 是最大索引(如Macro)
                    if acc.mappingArray[op][m1]:
                        if acc.double_config[m1][op] == 1:
                            quad_expr += indic_nxtMem[m,m1,op] * indic_doubleMem[m1,op]
                
                model.addConstr(indic_feeds_DB[m,op] == quad_expr, name=f"C_indic_feeds_DB_{m}_{op_name}")
        
        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                indic_doubleLoop[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleLoop_({i},{op_name})")
                
                quad_expr_loop = 0
                for m in range(1, acc.Num_mem):
                    if acc.mappingArray[op][m]:
                        quad_expr_loop += indic_loop2Mem[i,op,m] * indic_feeds_DB[m,op]
                
                model.addConstr(indic_doubleLoop[i,op] == quad_expr_loop, name=f"C_Indic_doubleLoop_({i},{op_name})")
        

        ####################################################################  Capacity Constraints   #######################################################################

        lg_dimExistMem = gp.tupledict()         # lg_dimExistMem[m,op,d] = dimSize
        for m in range(2,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for d in range(1, ops.Num_dim):
                    if factors[d] == [1]:     # DimSize == 1
                        lg_dimExistMem[m,op,d] = 0
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUrArray[u,op]:
                                lg_dimExistMem[m,op,d] += math.log(self.su[u][d])
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
                                dimExistMem += math.log(self.su[u][d])
                                
                        model.addConstr(lg_dimExistMem[m,op,d] == dimExistMem, 
                                                    name=f"C_sum_lgdimExistMem_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                        
        lg_dimOfTile = gp.tupledict()         # lg_dimOfTile[m,op,d] = dimSize
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for d in range(1, ops.Num_dim):
                    if factors[d] == [1]:     # DimSize == 1
                        lg_dimOfTile[m,op,d] = 0
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUrArray[u,op]:
                                lg_dimOfTile[m,op,d] += math.log(self.su[u][d])
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
                                dimOfTile += math.log(self.su[u][d])
                                
                        model.addConstr(lg_dimOfTile[m,op,d] == dimOfTile, 
                                                    name=f"C_sum_lgdimOfTile_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          

        spur = {}
        for m in range(1,acc.Num_mem):
            for d in range(1, ops.Num_dim):
                for op, op_name in enumerate(['I','W','O']):
                    spur[m,op,d] = 1
                    for u in range(acc.Num_SpUr):
                        if m <= acc.SpUrArray[u,op]:
                            spur[m,op,d] *= self.su[u][d]

        exp_dataVolume = gp.tupledict()     # exp_dataVolume[m,op]
        lg_dataVolume = gp.tupledict()     # lg_dataVolume[m,op]
        for m in range(2,acc.Num_mem):      # Sufficient off-chip capacity [1-Dram]

            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m] == True:
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0
                indic_sum, sum_dim_h = 0, 0
                for i_r, rd in enumerate(tempDivisors[ops.dict2Dim('R')]):
                    for i_p, pd in enumerate(tempDivisors[ops.dict2Dim('P')]):
                        r = rd * spur[m,op,ops.dict2Dim('R')]
                        p = pd * spur[m,op,ops.dict2Dim('P')]
                        if ops.Stride >= r:
                            h = p * r
                        else:
                            h = (p-1) * ops.Stride + r
                        h = min(h,ops.H)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_Height_({acc.mem2dict(m)},{op_name},{i_r},{i_p})")
                        sum_dim_h += indic_dim * math.log(h)
                        sum_r += indic_dim * math.log(r)
                        sum_p += indic_dim * math.log(p)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_Height_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_r == lg_dimExistMem[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicSumR_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_p == lg_dimExistMem[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicSumP_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_r - lg_dimExistMem[m,op,ops.dict2Dim('R')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicSumR_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_p - lg_dimExistMem[m,op,ops.dict2Dim('P')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicSumP_({acc.mem2dict(m)},{op_name})")

                indic_sum, sum_dim_w = 0, 0
                for i_s, sd in enumerate(tempDivisors[ops.dict2Dim('S')]):
                    for i_q, qd in enumerate(tempDivisors[ops.dict2Dim('Q')]):
                        s = sd * spur[m,op,ops.dict2Dim('S')]
                        q = qd * spur[m,op,ops.dict2Dim('Q')]
                        if ops.Stride >= s:
                            w = q * s
                        else:
                            w = (q-1) * ops.Stride + s
                        w = min(w,ops.W)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_Width_({acc.mem2dict(m)},{op_name},{i_s},{i_q})")
                        sum_dim_w += indic_dim * math.log(w)
                        sum_s += indic_dim * math.log(s)
                        sum_q += indic_dim * math.log(q)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_Width_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_s == lg_dimExistMem[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicSumS_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_q == lg_dimExistMem[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicSumQ_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_s - lg_dimExistMem[m,op,ops.dict2Dim('S')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicSumS_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_q - lg_dimExistMem[m,op,ops.dict2Dim('Q')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicSumQ_({acc.mem2dict(m)},{op_name})")

                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], math.log(ops.size[op])),
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimExistMem[m,op,ops.dict2Dim('C')],
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.CONTINUOUS,
                                                         name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                        
            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], math.log(ops.size[op])),
                                                name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K']),
                                name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], math.log(ops.size[op])),
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K']),
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], MAX_SIZE[op]), vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

        transVolume = gp.tupledict()    # exp_transVolume[m,op]
        lg_transVolume = gp.tupledict()     # lg_transVolume[m,op]
        for m in range(1,acc.Num_mem):      

            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m]:
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0

                indic_sum, sum_dim_h = 0, 0
                for i_r, rd in enumerate(tempDivisors[ops.dict2Dim('R')]):
                    for i_p, pd in enumerate(tempDivisors[ops.dict2Dim('P')]):
                        r = rd * spur[m,op,ops.dict2Dim('R')]
                        p = pd * spur[m,op,ops.dict2Dim('P')]
                        if ops.Stride >= r:
                            h = p * r
                        else:
                            h = (p-1) * ops.Stride + r
                        h = min(h,ops.H)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_TileHeight_({acc.mem2dict(m)},{op_name},{i_r},{i_p})")
                        sum_dim_h += indic_dim * math.log(h)
                        sum_r += indic_dim * math.log(r)
                        sum_p += indic_dim * math.log(p)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_Height_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_r == lg_dimOfTile[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicTileR_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_p == lg_dimOfTile[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicTileP_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_r - lg_dimOfTile[m,op,ops.dict2Dim('R')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicTileR_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_p - lg_dimOfTile[m,op,ops.dict2Dim('P')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicTileP_({acc.mem2dict(m)},{op_name})")

                indic_sum, sum_dim_w = 0, 0
                for i_s, sd in enumerate(tempDivisors[ops.dict2Dim('S')]):
                    for i_q, qd in enumerate(tempDivisors[ops.dict2Dim('Q')]):
                        s = sd * spur[m,op,ops.dict2Dim('S')]
                        q = qd * spur[m,op,ops.dict2Dim('Q')]
                        if ops.Stride >= s:
                            w = q * s
                        else:
                            w = (q-1) * ops.Stride + s
                        w = min(w,ops.W)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_TileWidth_({acc.mem2dict(m)},{op_name},{i_s},{i_q})")
                        sum_dim_w += indic_dim * math.log(w)
                        sum_s += indic_dim * math.log(s)
                        sum_q += indic_dim * math.log(q)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_Width_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_s == lg_dimOfTile[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicTileS_({acc.mem2dict(m)},{op_name})")
                # model.addConstr(sum_q == lg_dimOfTile[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicTileQ_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_s - lg_dimOfTile[m,op,ops.dict2Dim('S')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicTileS_({acc.mem2dict(m)},{op_name})")
                model.addRange(sum_q - lg_dimOfTile[m,op,ops.dict2Dim('Q')], -CONST.EPS, CONST.EPS, name=f"C_Uniqueness_IndicTileQ_({acc.mem2dict(m)},{op_name})")

                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], math.log(ops.size[op])),
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimOfTile[m,op,ops.dict2Dim('C')],
                                    name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], math.log(ops.size[op])),
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(acc.memSize[m]//acc.precision[m,op], math.log(ops.size[op])),
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        dataVolume = gp.tupledict()
        for m in range(2,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.shareMemory[m] == True:
                    if acc.mappingArray[op][m]:
                        dataVolume[m,op] = var_mul01(model, indic_usedMem[m,op], exp_dataVolume[m,op], f"dataVolume_({acc.mem2dict(m)},{op_name})")
                    else:
                        dataVolume[m,op] = 0
        
        sum_util = 0
        for m in range(2,acc.Num_mem):
            if acc.shareMemory[m] == True:
                tmp_datavolume = 0
                for op, op_name in enumerate(['I','W','O']):
                    if acc.mappingArray[op][m] == 1:
                        tmp_datavolume += (dataVolume[m,op] + var_mul01(model, indic_doubleMem[m,op], dataVolume[m,op], f"C_mul_{m}_{op}")) * acc.precision[m,op]
                        sum_util += dataVolume[m,op]* (acc.precision[m,op] * m)
                model.addConstr( tmp_datavolume <= acc.memSize[m], name=f"C_dataVolume_({acc.mem2dict(m)})" )
            else:
                for op, op_name in enumerate(['I','W','O']):
                    if acc.mappingArray[op][m] == 1:
                        model.addConstr(lg_dataVolume[m,op] + math.log(2)*indic_doubleMem[m,op] <= math.log(acc.memSize[m])-math.log(acc.precision[m,op]))


        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == False:
                    transVolume[m,op] = 0

        ####################################################################  Execution Performance   #######################################################################

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Energy - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Latency - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - -#          
        indic_xMem = gp.tupledict()                 # indic_xMem[i,op] = {0,1}
        for op, op_name in enumerate(['I','W','O']):
            for i in range(Num_Loops):
                indic_xMem[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_xMem_({i},{op_name})")
                tmp_diff = loop2Mem[i+1,op] - loop2Mem[i,op]
                model.addConstr(tmp_diff >= indic_xMem[i,op], name=f"C_tmp_diff_({i},{op_name})")
                model.addConstr(tmp_diff <= acc.Num_mem * indic_xMem[i,op], name=f"C_tmp_diff_ub_({i},{op_name})")
                
            if op_name == 'W':      # Weight
                model.addConstr(quicksum(indic_xMem[i,op] for i in range(Num_Loops)) <= 2, name=f"max_xMem_{op_name}")
            else:                   # Feature
                model.addConstr(quicksum(indic_xMem[i,op] for i in range(Num_Loops)) <= 4, name=f"max_xMem_{op_name}")
        
        indic_usedLastMem = gp.tupledict()              # indic_usedLastMem[op] = {0,1}
        for op, op_name in enumerate(['I','W','O']):
            indic_usedLastMem[op] = model.addVar(vtype=GRB.BINARY, name=f"indic_usedLastMem_({op_name})")
            tmp_diff = acc.lastMem[op] - loop2Mem[Num_Loops-1,op] 
            model.addConstr(tmp_diff >= 1-indic_usedLastMem[op], name=f"C_tmp_diff_lastMem_({op_name})")
            model.addConstr(tmp_diff <= acc.Num_mem * (1-indic_usedLastMem[op]), name=f"C_tmp_diff_lastMem_ub_({op_name})")

        transfer = gp.tupledict()                       # transfer[i,op]
        lg_transfer = gp.tupledict()
        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                transfer[i,op] = model.addVar(lb=0, ub=self.maxTrans[op], vtype=GRB.CONTINUOUS, name=f"transfer_({i},{op_name})")   #WTD max precision
                lg_transfer[i,op] = model.addVar(lb=0, ub=math.log(self.maxTrans[op]), vtype=GRB.CONTINUOUS, name=f"lg_transfer_({i},{op_name})")   
                
                tmp_rhs_expr = 0
                for m in range(1, acc.Num_mem):
                    if acc.mappingArray[op][m]:
                        tmp_rhs_expr += indic_loop2Mem[i,op,m] * (lg_transVolume[m,op] - math.log(acc.bw[m]) + math.log(acc.precision[m,op]))
                model.addConstr(lg_transfer[i,op] == tmp_rhs_expr, name=f"C_lg_transfer_linear_({i},{op_name})")

                model.addGenConstrExp(xvar=lg_transfer[i,op], yvar=transfer[i,op], options=self.ExpOption, name=f"C_transfer_({i},{op_name})")


        latency_Critical = gp.tupledict()           # latency_cp[i] = Latency of Critical Path
        latency_Process = gp.tupledict()            # latency_Process[i,op] 
        latency_Transfer = gp.tupledict()           # latency_Transfer[i,op]
        for i in range(Num_Loops):
            latency_Critical[i] = model.addVar(lb=acc.t_MAC*math.pow(2,Num_Loops-1-i), ub=self.ub_latency/math.pow(2,i), vtype=GRB.CONTINUOUS, name=f"latency_Critical_({i})")
            for op, op_name in enumerate(['I','W','O']):
                latency_Process[i,op] = model.addVar(lb=acc.t_MAC*math.pow(2,Num_Loops-1-i), ub=self.ub_latency/math.pow(2,i), vtype=GRB.CONTINUOUS, name=f"latency_Process_({i},{op_name})")
                
                if i < Num_Loops-1:
                    latency_Transfer[i,op] = var_mul01(model, indic_xMem[i, op], transfer[i, op], name=f"latency_Transfer_({i},{op_name})", A_ub=self.maxTrans[op])
                else:
                    latency_Transfer[i,op] = model.addVar(lb=0, ub=self.maxTrans[op], vtype=GRB.CONTINUOUS, name=f"latency_Transfer_({i},{op_name})")
                    model.addConstr(latency_Transfer[i,op] == transfer[Num_Loops-1, op] + 1 - indic_usedLastMem[op], name=f"C_RegTrans_({op_name})")
                    latency_Process[Num_Loops,op] = acc.t_MAC


        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):

                tmp_coeff = 2 if op_name == 'O' else 1

                model.addConstr(latency_Critical[i] >= tmp_coeff * latency_Transfer[i,op], name=f"C_latency_cp_transfer_({i},{op_name})")
                model.addConstr(latency_Critical[i] >= latency_Process[i+1,op], name=f"C_latency_cp_process_({i},{op_name})")
                model.addConstr(latency_Critical[i] >= tmp_coeff * latency_Transfer[i,op] + latency_Process[i+1,op] - (self.ub_latency/math.pow(2,i)*indic_doubleLoop[i,op]),
                                 name=f"C_latency_cp_transfer+process_({i},{op_name})")

        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                model.addConstr(latency_Process[i,op] >= latency_Process[i+1,op] + latency_Critical[i] * (loop2Factor[i]-1) - (self.ub_latency/math.pow(2,i) * indic_xMem[i,op]),
                                 name=f"C_latency_process_stationary_({i},{op_name})")
                
                tmp_2TPcpN = 2 * latency_Transfer[i,op] + latency_Process[i+1,op] + latency_Critical[i] * loop2Factor[i]

                if op < 2:      # I,W
                    model.addConstr(latency_Process[i,op] >= tmp_2TPcpN - (2 + indic_doubleLoop[i,op]) * latency_Critical[i], name=f"C_latency_process_({i},{op_name})")
                else:           # O
                    model.addConstr(latency_Process[i,op] >= tmp_2TPcpN - latency_Critical[i], name=f"C_latency_process_({i},{op_name})")

        
        res_latency = model.addVar(lb=self.lb_latency, ub=self.ub_latency, vtype=GRB.CONTINUOUS,     name="res_latency")
        res_energy  = model.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_energy")
        res_EDP     = model.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_EDP")
        for op in range(3):
            model.addConstr(res_latency >= latency_Process[0,op])
        

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        utliz_all = model.addVar(vtype=GRB.CONTINUOUS, name=f"utilzation_all")
        model.addConstr(utliz_all == sum((indic_usedMem[1,op]*MAX_SIZE[op]) for op in range(3)) )

        model.write(os.path.join(self.outputdir, "debug_model.lp"))
        # exit()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        
        model.ModelSense = GRB.MINIMIZE

        model.Params.TimeLimit = 10  # 设置 10 秒硬性限制
        model.Params.MIPFocus = 1 
        model.update()
        model.optimize()
        # 检查第一阶段结果
        if model.SolCount == 0:
            if model.Status == GRB.TIME_LIMIT or model.Status == GRB.INFEASIBLE:
                print("No feasible solution found within 10 seconds, considered unsolvable.")
                print("# # # # # The Scheme has been Determined to be Suboptimal # # # # #")
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            return 1
        
        # 恢复或调整参数
        # model.Params.MIPFocus = 2  # 恢复平衡模式或设为 2 (侧重最优边界)
        model.Params.MIPFocus = 3 

        model.setObjectiveN(res_latency, 0, priority=3, name='Latency')    
        env0 = model.getMultiobjEnv(0)                                     
        env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.7)
        # env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.15)
        env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.7)

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
            for i in range(Num_Loops):
                for d in range(1, ops.Num_dim):
                    if factors[d] == [1]:     # DimSize == 1
                        continue
                    for f in range(len(factors[d])):
                        if (round(indic_factor2Loop[d,f,i].x) == 1):
                            Logger.debug(f"{i:<2} for {ops.dim2Dict[d]} in {loop2Factor[i].x:<3}: {[ f'{round(indic_xMem[i,op].x)}|{acc.mem2dict(loop2Mem[i,op].x)}' for op in range(3)]}")
                            
                            loops.tm.append(Mapping(dim=d, 
                                                    dimSize=factors[d][f],
                                                    mem=[round(loop2Mem[i,op].x) for op in range(3)]))
            
            Logger.debug(self.su)
            for u in range(acc.Num_SpUr):
                for d in range(1, ops.Num_dim):
                    if self.su[u][d] > 1:
                        loops.sm.append(Mapping(dim=d, 
                                                dimSize=self.su[u][d],
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
                return 1
                Logger.error("TLE---No Feasible Solution Yet")
                model.setParam('IISMethod', 2)
                model.computeIIS()
                model.write(os.path.join(self.outputdir, "iis_full.ilp"))
                model.write(os.path.join(self.outputdir, "model.mps"))
                Logger.error("TLE---Debug in contric.ilp")
                exit()
        else:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            return 1
            model.setParam("IISMethod", 2) 
            model.computeIIS()
            model.write(os.path.join(self.outputdir, "iis_full.ilp"))
            model.write(os.path.join(self.outputdir, "model.mps"))
            Logger.error(f'Model infeasible !!!')
            exit()


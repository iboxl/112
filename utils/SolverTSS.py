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
from gurobipy import GRB, quicksum
from utils.GlobalUT import *
from utils.UtilsFunction.SolverFunction import *
from utils.factorization import flexible_factorization
from utils.UtilsFunction.ToolFunction import getDivisors, getUniqueFactors
import copy

class Solver():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad, tu, su, metric_ub, outputdir=None):
        self.acc = copy.deepcopy(acc)
        self.ops = copy.deepcopy(ops)
        self.tu = tu
        self.su = su
        self.metric_ub = metric_ub
        self.outputdir = outputdir

        self.model = gp.Model(name="MIREDO")
        self.model.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
        self.model.setParam('Seed', 112)                               # 随机种子
        self.model.setParam('LogFile',os.path.join(self.outputdir, "Solver.log"))
        
        self.model.setParam('Threads', psutil.cpu_count(logical=False))
        # self.model.setParam('Threads', psutil.cpu_count())

        # self.model.setParam('NonConvex', 2)
        # self.model.setParam('PreQLinearize', 2)

        self.model.setParam('Cuts', -1)
        self.model.setParam('Presolve', 2)
        self.model.setParam('Heuristics', 0.25)
        self.model.setParam("ScaleFlag", 2)

        self.model.setParam("ScaleFlag", 2)                          # 用于调节数值比例问题coefficient range
        self.model.setParam("NumericFocus", 1)
        # self.model.setParam('MIPGap', 0.03)

        self.ExpOption = "FuncPieces=-2 FuncPieceError=0.01"

        self.model.setParam('FeasibilityTol', 1e-6)
        self.model.setParam('IntFeasTol', 1e-6)
        self.model.setParam('OptimalityTol', 1e-6)
        
        self.result = {}
        self.dataflow = {}

        self.FACTORS = [flexible_factorization(_) for _ in self.tu]
        self.spatial_unrolling = [math.prod(col) for col in zip(*su)]
        self.MAX_TRANS = [ max([math.ceil(min(ops.size[op] * acc.precision[m,op], acc.memSize[m]) / acc.bw[m]) / CONST.SCALE_LATENCY
                                for m in range(1, acc.Num_mem) if acc.mappingArray[op][m]] 
                            + [0]) for op in range(3) ]                             # maxTrans[op] = max transfer time for op in {I,W,O}
        
        Logger.critical(f"Best {CONST.FLAG_OPT} metric upper bound is {self.metric_ub}")

    def run(self):
        Logger.info('* '*20 + "Start Running MIP Solver" + ' *'*20)
        model = self.model
        acc:CIM_Acc = self.acc
        ops:WorkLoad = self.ops
        factors = self.FACTORS

        TEMP_DIVISORS = [getDivisors(d) for d in self.tu]

        Num_Loops = sum(len(f) for f in factors[1:ops.Num_dim] if f != [1])

        MAX_SIZE = self.ops.size

        UNIQUE_FACTOR = getUniqueFactors(factors)

        logF = {(d, f): math.log(factors[d][f]) 
                        for d in range(1, ops.Num_dim) 
                        for f in range(len(factors[d]))}
        
        factors_val = [f for fs in factors[1:ops.Num_dim] for f in fs if fs != [1]]
        f_asc = sorted(factors_val)

        MIN_INNER_PROD = {}
        for i in range(Num_Loops):
            MIN_INNER_PROD[i] = int(np.prod(f_asc[:Num_Loops - i]))
        MIN_INNER_PROD[Num_Loops] = 1

        MIN_OUTER_PROD = [1] * (Num_Loops + 1)
        for i in range(1, Num_Loops + 1):
            MIN_OUTER_PROD[i] = MIN_OUTER_PROD[i - 1] * f_asc[i - 1]

        LAT_UNIT = 1 / CONST.SCALE_LATENCY
        t_MAC = acc.t_MAC / CONST.SCALE_LATENCY
        XMAX_TOTAL = {0: min(4, Num_Loops), 1: min(2, Num_Loops), 2: min(4, Num_Loops)}

        TOTAL_TEMPORAL_ITERS = max(MIN_INNER_PROD[0], 1)
        MAX_STAGE_TRANSFER = max((2 if op == 2 else 1) * self.MAX_TRANS[op] for op in range(3))
        UB_offchipBootstrap = max(
            (2 if op == 2 else 1) * MAX_SIZE[op] * acc.precision[acc.Dram2mem, op] / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
            for op in range(3)
        )
        UB_latencySimple = max(
            LAT_UNIT,
            TOTAL_TEMPORAL_ITERS * max(t_MAC, MAX_STAGE_TRANSFER) + UB_offchipBootstrap,
        )
        if CONST.FLAG_OPT == "Latency" and self.metric_ub is not None:
            UB_latencySimple = min(UB_latencySimple, self.metric_ub / CONST.SCALE_LATENCY)
        UB_latencySimple = max(UB_latencySimple, LAT_UNIT)
        UB_latencyLevel = {
            i: max(UB_latencySimple / math.pow(2, i), max(t_MAC, LAT_UNIT))
            for i in range(Num_Loops)
        }

        spur = {}
        UB_dataVolume, UB_lg_dataVolume = {}, {}
        UB_lg_transVolume = {}
        UB_transLatency, UB_lg_transLatency = {}, {}
        LB_lg_transLatency = {}
        for m in range(1, acc.Num_mem):
            for d in range(1, ops.Num_dim):
                for op in range(3):
                    spur[m, op, d] = 1
                    for u in range(acc.Num_SpUr):
                        if m <= acc.SpUr2Mem[u, op]:
                            spur[m, op, d] *= self.su[u][d]

            for op in range(3):
                UB_dataVolumeCap = max(1, min(acc.memSize[m] // acc.precision[m, op], MAX_SIZE[op]))
                UB_dataVolume[m, op] = UB_dataVolumeCap
                UB_lg_dataVolume[m, op] = math.log(UB_dataVolumeCap)
                UB_lg_transVolume[m, op] = UB_lg_dataVolume[m, op]

                UB_transLatencyCap = max(
                    LAT_UNIT,
                    UB_dataVolumeCap * acc.precision[m, op] / acc.bw[m] / CONST.SCALE_LATENCY,
                )
                UB_transLatency[m, op] = UB_transLatencyCap
                UB_lg_transLatency[m, op] = math.log(UB_transLatencyCap)
                LB_lg_transLatency[m, op] = math.log(
                    max(acc.precision[m, op] / acc.bw[m] / CONST.SCALE_LATENCY, 1 / CONST.MAX_POS)
                )

        ###########################################################  Variable & Constant & Constraints  ##################################################################
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        res_latency = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="res_latency")
        res_energy = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="res_energy")
        res_EDP = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="res_EDP")

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

        # symmetry-breaking 对称约束
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]:     # DimSize == 1
                continue
            for f in range(len(factors[d])):
                for f1 in range(f+1, len(factors[d])):
                    if factors[d][f] == factors[d][f1]:   # 相同因子之间才需要加对称约束
                        model.addConstr(quicksum(i * indic_factor2Loop[d,f,i] for i in range(Num_Loops)) <= 
                                        quicksum(i * indic_factor2Loop[d,f1,i] for i in range(Num_Loops)),
                                                name=f"C_SymmetryBreaking_Factor2Loop_({ops.dim2Dict[d]},{f},{f1})") 

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
        for op, op_name in enumerate(['I','W','O']):
            for i in range(Num_Loops):
                for m in range(1,acc.Num_mem):
                    if acc.mappingArray[op][m] == 1:
                        indic_loop2Mem[i,op,m] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)})")
                        for d in range(1, ops.Num_dim):
                            if factors[d] == [1]:
                                continue
                            for f in range(len(factors[d])):
                                model.addConstr(indic_loop2Mem[i,op,m] >= indic_factor2Mem[d,f,op,m] + indic_factor2Loop[d,f,i]-1,
                                                name=f"C_Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                                model.addConstr(indic_loop2Mem[i,op,m] <= 1 - indic_factor2Loop[d,f,i] + indic_factor2Mem[d,f,op,m],
                                                name=f"C_Indic_loop2Mem_Disagg_({i},{op_name},{acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                    else:
                        indic_loop2Mem[i,op,m] = 0

                    model.addConstr(indic_loop2Mem[i,op,m] <= indic_usedMem[m,op], name=f"C_Loop2Mem_Used_({i},{op_name},{acc.mem2dict(m)})")

                model.addConstr(quicksum(indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)) == 1, 
                                name=f"C_Indic_loop2Mem_({i},{op_name})")
                        
        for i in range(Num_Loops-1):
            for op, op_name in enumerate(['I','W','O']):
                model.addConstr(quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)) <=
                                quicksum(m*indic_loop2Mem[i+1,op,m] for m in range(1,acc.Num_mem)),
                                  name=f"C_Sequence_loop2Mem_({i},{op_name})")

        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 1:
                    model.addConstr(indic_usedMem[m,op] <= quicksum(indic_loop2Mem[i,op,m] for i in range(Num_Loops)),
                                    name=f"C_UsedMem_FromLoop2Mem_({acc.mem2dict(m)},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        
        indic_loop2Factor = gp.tupledict()              # indic_loop2Factor[i,p] = {0,1}
        indic_relevantLoop = gp.tupledict()             # indic_relevantLoop[i,op] = {0,1} : whether loop i is relevant to op
        for i in range(Num_Loops):
            for p in range(len(UNIQUE_FACTOR)):
                indic_loop2Factor[i,p] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Factor_({i},{p})")
            model.addConstr(quicksum(indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR))) == 1, name=f"C_Uniqueness_Indic_loop2Factor_({i})")
            model.addConstr(quicksum(UNIQUE_FACTOR[p] * indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR)))
                            ==
                            quicksum(indic_factor2Loop[d,f,i] * factors[d][f] for d in range(1, ops.Num_dim) for f in range(len(factors[d]))),
                             name=f"C_loop2Factor_definition_({i})")
            for op, op_name in enumerate(['I','W','O']):
                indic_relevantLoop[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_relevantLoop_({i},{op_name})")
                model.addConstr(indic_relevantLoop[i,op] == quicksum(ops.relevance[op][d] * indic_factor2Loop[d,f,i] 
                                                                     for d in range(1, ops.Num_dim) for f in range(len(factors[d]))),
                                 name=f"C_relevantLoop_({i},{op_name})")
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
                            quad_expr += var_AandB(model, indic_nxtMem[m,m1,op], indic_doubleMem[m1,op],
                                                    name=f"Quad_Indic_feedsDB_({acc.mem2dict(m)},{op_name})_to_{acc.mem2dict(m1)}")
                
                model.addConstr(indic_feeds_DB[m,op] == quad_expr, name=f"C_indic_feeds_DB_{m}_{op_name}")

        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                indic_doubleLoop[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleLoop_({i},{op_name})")
                
                quad_expr_loop = 0
                for m in range(1, acc.Num_mem):
                    if acc.mappingArray[op][m]:
                        quad_expr_loop += var_AandB(model, indic_loop2Mem[i,op,m], indic_feeds_DB[m,op],
                                                     name=f"Quad_Indic_doubleLoop_({i},{op_name})_{acc.mem2dict(m)}")
                
                model.addConstr(indic_doubleLoop[i,op] == quad_expr_loop, name=f"C_Indic_doubleLoop_({i},{op_name})")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#    
        
        indic_xMem = gp.tupledict()                         # indic_xMem[i,op] = {0,1}
        indic_xMemCarry = gp.tupledict()                    # indic_xMemCarry[i,op] = {0,1}
        for op, op_name in enumerate(['I', 'W', 'O']):
            indic_sameMem, indic_boundaryMem = {}, {}

            for i in range(Num_Loops):
                if i == Num_Loops-1:
                    indic_boundaryMem[i] = 1 - indic_loop2Mem[i,op,acc.Macro2mem] if op_name == 'W' else 1 
                else:
                    indic_boundaryMem[i] = model.addVar(vtype=GRB.BINARY, name=f"Indic_boundaryMem_({i},{op_name})")
                    tmp_diff = quicksum(m*indic_loop2Mem[i+1,op,m] for m in range(1,acc.Num_mem)) - quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem))
                    model.addConstr(tmp_diff >= indic_boundaryMem[i], name=f"C_tmp_diff_({i},{op_name})")
                    model.addConstr(tmp_diff <= acc.Num_mem * indic_boundaryMem[i], name=f"C_tmp_diff_ub_({i},{op_name})")
                indic_sameMem[i] = 1 - indic_boundaryMem[i]

            for i in range(Num_Loops-1, -1, -1):
                '''
                indic_xMemCarry[i, op] = 
                        indic_boundaryMem[i, op] OR (indic_sameMem[i, op] AND indic_xMemCarry[i+1,op] AND (NOT indic_relevantLoop[i+1, op]))
                '''
                if i == Num_Loops-1:
                    indic_xMemCarry[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_xMemCarry_({i},{op_name})")
                    model.addConstr(indic_xMemCarry[i,op] == indic_boundaryMem[i],
                                    name=f"C_xMemCarry_last_({op_name})")
                else:
                    indic_xMemCarry[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_xMemCarry_({i},{op_name})")
                    model.addConstr(indic_xMemCarry[i,op] >= indic_boundaryMem[i], name=f"C_xMemCarry_boundary_({i},{op_name})")
                    model.addConstr(indic_xMemCarry[i,op] <= indic_boundaryMem[i] + indic_xMemCarry[i+1,op],
                                    name=f"C_xMemCarry_next_ub_({i},{op_name})")
                    model.addConstr(indic_xMemCarry[i,op] <= indic_boundaryMem[i] + 1 - indic_relevantLoop[i+1, op],
                                    name=f"C_xMemCarry_irrel_ub_({i},{op_name})")
                    model.addConstr(indic_xMemCarry[i,op] >= indic_sameMem[i] + indic_xMemCarry[i+1,op] - indic_relevantLoop[i+1, op] - 1,
                                    name=f"C_xMemCarry_same_lb_({i},{op_name})")

            for i in range(Num_Loops): 
                '''
                indic_xMem[i, op] = indic_xMemCarry[i, op] AND (indic_relevantLoop[i, op] OR (NOT indic_sameMem[i-1, op]))
                '''       
                if i == 0:
                    indic_xMem[i,op] = indic_xMemCarry[i,op]
                else:
                    indic_xMem[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_xMem_({i},{op_name})")
                    model.addConstr(indic_xMem[i, op] <= indic_xMemCarry[i,op], name=f"C_xMem_carry_({i},{op_name})")
                    model.addConstr(indic_xMem[i, op] <= indic_relevantLoop[i, op] + 1 - indic_sameMem[i-1],
                                    name=f"C_xMem_assignable_ub_({i},{op_name})")
                    model.addConstr(indic_xMem[i, op] >= indic_xMemCarry[i,op] - indic_sameMem[i-1],
                                    name=f"C_xMem_blockstart_lb_({i},{op_name})")
                    model.addConstr(indic_xMem[i, op] >= indic_xMemCarry[i,op] + indic_relevantLoop[i, op] + indic_sameMem[i-1] - 2,
                                    name=f"C_xMem_relevant_lb_({i},{op_name})")
                if op_name == 'W':
                    model.addConstr(indic_xMem[i,op] <= 1 - indic_loop2Mem[i, op, acc.Macro2mem], name=f"C_xMem_macro_({i},{op_name})")
                    # CIM inherent mapping constraint

            model.addConstr(quicksum(indic_xMem[i,op] for i in range(Num_Loops)) <= XMAX_TOTAL[op], name=f"max_xMem_{op_name}")
        
        ####################################################################  Capacity Constraints   #######################################################################

        lg_dimExistMem = gp.tupledict()         # lg_dimExistMem[m,op,d] = dimSize
        for op, op_name in enumerate(['I','W','O']):
            for d in range(1, ops.Num_dim):
                for m in range(2,acc.Num_mem):
                    if acc.mappingArray[op][m] == 0 or ops.relevance[op][d] == 0:
                        continue
                    if factors[d] == [1]:     # DimSize == 1
                        lg_dimExistMem[m,op,d] = 0
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUr2Mem[u,op]:
                                lg_dimExistMem[m,op,d] += math.log(self.su[u][d])
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
                            if m <= acc.SpUr2Mem[u,op]:
                                dimExistMem += math.log(self.su[u][d])
                                
                        model.addConstr(lg_dimExistMem[m,op,d] == dimExistMem, 
                                                    name=f"C_sum_lgdimExistMem_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                lg_dimExistMem[1,op,d] = math.log(ops.dim2bound[d])
                        
        lg_dimOfTile = gp.tupledict()         # lg_dimOfTile[m,op,d] = dimSize
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for d in range(1, ops.Num_dim):
                    if acc.mappingArray[op][m] == 0 or ops.relevance[op][d] == 0:
                        continue
                    if factors[d] == [1]:     # DimSize == 1
                        lg_dimOfTile[m,op,d] = 0
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUr2Mem[u,op]:
                                lg_dimOfTile[m,op,d] += math.log(self.su[u][d])
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
                            if m <= acc.SpUr2Mem[u,op]:
                                dimOfTile += math.log(self.su[u][d])
                                
                        model.addConstr(lg_dimOfTile[m,op,d] == dimOfTile, 
                                                    name=f"C_sum_lgdimOfTile_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          

        exp_dataVolume = gp.tupledict()     # exp_dataVolume[m,op]
        lg_dataVolume = gp.tupledict()     # lg_dataVolume[m,op]
        for op, op_name in enumerate(['I','W','O']):
            lg_dataVolume[1, op] = math.log(MAX_SIZE[op])
            exp_dataVolume[1, op] = MAX_SIZE[op]
        for m in range(2,acc.Num_mem):      # Sufficient off-chip capacity [1-Dram]
            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m] == True:
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0
                indic_sum, sum_dim_h = 0, 0
                for i_r, rd in enumerate(TEMP_DIVISORS[ops.dict2Dim('R')]):
                    for i_p, pd in enumerate(TEMP_DIVISORS[ops.dict2Dim('P')]):
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
                model.addConstr(sum_r == lg_dimExistMem[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicSumR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimExistMem[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicSumP_({acc.mem2dict(m)},{op_name})")

                indic_sum, sum_dim_w = 0, 0
                for i_s, sd in enumerate(TEMP_DIVISORS[ops.dict2Dim('S')]):
                    for i_q, qd in enumerate(TEMP_DIVISORS[ops.dict2Dim('Q')]):
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
                model.addConstr(sum_s == lg_dimExistMem[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicSumS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimExistMem[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicSumQ_({acc.mem2dict(m)},{op_name})")

                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB_lg_dataVolume[m,op],
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimExistMem[m,op,ops.dict2Dim('C')],
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                         name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                        
            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB_lg_dataVolume[m,op],
                                                name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K']),
                                name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB_lg_dataVolume[m,op],
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K']),
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

        lg_transVolume = gp.tupledict()     # lg_transVolume[m,op]
        for m in range(1,acc.Num_mem):      

            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m]:
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0

                indic_sum, sum_dim_h = 0, 0
                for i_r, rd in enumerate(TEMP_DIVISORS[ops.dict2Dim('R')]):
                    for i_p, pd in enumerate(TEMP_DIVISORS[ops.dict2Dim('P')]):
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
                model.addConstr(sum_r == lg_dimOfTile[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicTileR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimOfTile[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicTileP_({acc.mem2dict(m)},{op_name})")

                indic_sum, sum_dim_w = 0, 0
                for i_s, sd in enumerate(TEMP_DIVISORS[ops.dict2Dim('S')]):
                    for i_q, qd in enumerate(TEMP_DIVISORS[ops.dict2Dim('Q')]):
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
                model.addConstr(sum_s == lg_dimOfTile[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicTileS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimOfTile[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicTileQ_({acc.mem2dict(m)},{op_name})")

                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB_lg_transVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimOfTile[m,op,ops.dict2Dim('C')],
                                    name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB_lg_transVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=UB_lg_transVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          

        for m in range(2,acc.Num_mem):
            if acc.shareMemory[m] == True:
                tmp_datavolume = 0
                for op, op_name in enumerate(['I','W','O']):
                    if acc.mappingArray[op][m] == 1:
                        tmp_datavolume += (var_mul01(model, indic_usedMem[m,op], exp_dataVolume[m,op], f"dataVolume_({acc.mem2dict(m)},{op_name})") + 
                                           var_mul01(model, indic_doubleMem[m,op], exp_dataVolume[m,op], f"V_mul_{m}_{op}")) * acc.precision[m,op]
                model.addConstr( tmp_datavolume <= acc.memSize[m], name=f"C_dataVolume_({acc.mem2dict(m)})" )
            else:
                for op, op_name in enumerate(['I','W','O']):
                    if acc.mappingArray[op][m] == 1:
                        model.addConstr(lg_dataVolume[m,op] + math.log(2)*indic_doubleMem[m,op] <= math.log(acc.memSize[m])-math.log(acc.precision[m,op]))

        ####################################################################  Execution Performance   #######################################################################

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Energy - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#                  
        
        count_mac = 1
        count_core = 1
        for d in range(1, ops.Num_dim):
            for f in range(len(factors[d])):
                count_mac *= factors[d][f]
            count_core *= self.su[0][d]
        energy_expr_comp = acc.cost_ActMacro * count_mac * count_core

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#   

        '''
        考虑bottle-ir的情况
        '''
        indic_NotBottleIR = gp.tupledict()                         # indic_NotBottleIR[d,f,op] = {0,1}
        indic_factor2Mem_NotBottleIR = gp.tupledict()              # indic_factor2Mem_NotBottleIR[d,f,op,m] = {0,1}
        for op, op_name in enumerate(['I','W','O']):
            for d in range(1, ops.Num_dim):
                if factors[d] != [1]:
                    for f in range(len(factors[d])):
                        if ops.relevance[op][d] == 1:
                            indic_NotBottleIR[d,f,op] = 1
                            for m in range(1, acc.Num_mem):
                                if acc.mappingArray[op][m] == 0:
                                    continue
                                indic_factor2Mem_NotBottleIR[d,f,op,m] = indic_factor2Mem[d,f,op,m]
                        else:
                            indic_NotBottleIR[d,f,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_NotBottleIR_({ops.dim2Dict[d]},{f},{op_name})")
                            model.addConstr(indic_NotBottleIR[d,f,op] == 1 - quicksum(var_AandB(model, indic_factor2Loop[d,f,i], indic_xMemCarry[i,op],
                                                                                                name=f"Indic_NotBottleIR_src_({ops.dim2Dict[d]},{f},{i},{op_name})")
                                                                                    for i in range(Num_Loops)),
                                            name=f"C_Indic_NotBottleIR_({ops.dim2Dict[d]},{f},{op_name})")
                            for m in range(1, acc.Num_mem):
                                if acc.mappingArray[op][m] == 0:
                                    continue
                                indic_factor2Mem_NotBottleIR[d,f,op,m] = var_AandB(model, indic_factor2Mem[d,f,op,m], indic_NotBottleIR[d,f,op],
                                                                            name=f"Indic_factor2Mem_NotBottleIR_({ops.dim2Dict[d]},{f},{op_name},{acc.mem2dict(m)})")


        def getExpVar(lg_term, lb, ub, name):
            var = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=name)
            model.addGenConstrExp(xvar=lg_term, yvar=var, options=self.ExpOption, name=f"C_{name}")
            return var

        lg_transEnergy_r2L = gp.tupledict()
        lg_transEnergy_w2L = gp.tupledict()
        lg_transEnergy_r2H = gp.tupledict()
        lg_transEnergy_w2H = gp.tupledict()
        energy_perMem, energy_usedMem = gp.tupledict(), gp.tupledict()

        count_ReadOut, count_WriteIn = {}, {}
        indic_factor2Mem_WriteIn = gp.tupledict()
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m] == 0:
                    continue
                count_expr_readOut = 0
                count_expr_writeIn = 0

                for d in range(1, ops.Num_dim):
                    if factors[d] != [1]:
                        for f in range(len(factors[d])):
                            for m1 in range(1,m+1):     # [ 1 ~ m ]             # 本层及以上层级的Loop都会影响 ReadOut 次数
                                if m1 == m:
                                    count_expr_readOut += logF[d,f] * indic_factor2Mem_NotBottleIR[d,f,op,m1]
                                else:
                                    count_expr_readOut += logF[d,f] * indic_factor2Mem[d,f,op,m1]
                            for m1 in range(1,m):       # [ 1 ~ m-1 ]           # 更上层级的Loop才会影响 WriteIn 次数
                                if acc.mappingArray[op][m1] == 0 or (m1, m, op) not in indic_nxtMem:
                                    count_expr_writeIn += logF[d,f] * indic_factor2Mem[d,f,op,m1]
                                    continue
                                
                                cname = f"Indic_factor2Mem_WriteIn_({ops.dim2Dict[d]},{f},{op_name},{acc.mem2dict(m1)},{acc.mem2dict(m)})"
                                indic_factor2Mem_WriteIn[d,f,op,m1,m] = model.addVar(vtype=GRB.BINARY, name=cname)
                                model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] <= indic_factor2Mem[d,f,op,m1],
                                                name=f"C_{cname}_base_ub")
                                model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] >= indic_factor2Mem_NotBottleIR[d,f,op,m1],
                                                name=f"C_{cname}_nb_lb")
                                model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] <= indic_factor2Mem_NotBottleIR[d,f,op,m1] + 1 - indic_nxtMem[m1,m,op],
                                                name=f"C_{cname}_nb_ub")
                                model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] >= indic_factor2Mem[d,f,op,m1] - indic_nxtMem[m1,m,op],
                                                name=f"C_{cname}_base_lb")
                                '''
                                W = A AND ((NOT G) OR N)
                                indic_factor2Mem_WriteIn = indic_factor2Mem AND ((NOT indic_nxtMem) OR indic_NotBottleIR)
                                '''
                                count_expr_writeIn += logF[d,f] * indic_factor2Mem_WriteIn[d,f,op,m1,m]

                    for u in range(acc.Num_SpUr):
                        if m > acc.SpUr2Mem[u,op]:
                            count_expr_readOut += math.log(self.su[u][d])
                            count_expr_writeIn += math.log(self.su[u][d])
                count_ReadOut[m,op] = count_expr_readOut
                count_WriteIn[m,op] = count_expr_writeIn
       

        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m] == 0:
                    continue
                tmp_energy_expr = 0
                count_expr_readOut = count_ReadOut[m,op]
                count_expr_writeIn = count_WriteIn[m,op]
                can_read = m not in [acc.IReg2mem, acc.OReg2mem, acc.Macro2mem] and acc.cost_r[m] > 0
                can_write = m > 1

                # 1) Base read-out energy of this memory level.
                if can_read:
                    lg_transEnergy_r2L[m,op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_r2L_({acc.mem2dict(m)},{op_name})")

                    model.addConstr(lg_transEnergy_r2L[m,op] == math.log(acc.cost_r[m]) + math.log(acc.precision[m,op]) + count_expr_readOut + lg_transVolume[m,op],
                                    name=f"C_lg_transEnergy_r2L_({acc.mem2dict(m)},{op_name})")
                    
                    tmp_energy_expr += getExpVar(lg_term=lg_transEnergy_r2L[m,op], lb=0, ub=GRB.INFINITY, name=f"transEnergy_r_({acc.mem2dict(m)},{op_name})")

                # 2) Base write-in energy of this memory level.
                if can_write:
                    lg_transEnergy_w2L[m,op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_w2L_({acc.mem2dict(m)},{op_name})")
                    
                    model.addConstr(lg_transEnergy_w2L[m,op] == math.log(acc.cost_w[m]) + math.log(acc.precision[m,op]) + count_expr_writeIn + lg_dataVolume[m,op],
                                     name=f"C_lg_transEnergy_w2L_({acc.mem2dict(m)},{op_name})")
                    
                    tmp_energy_expr += getExpVar(lg_term=lg_transEnergy_w2L[m,op], lb=0, ub=GRB.INFINITY, name=f"transEnergy_w_({acc.mem2dict(m)},{op_name})")

                # 3) Output readout for sending upstream memory hierarchy.
                # output往里面写多少次就要读多少次写回 ⬆
                if op_name == 'O' and can_read and m > acc.Dram2mem:
                    lg_transEnergy_r2H[m,op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_r2H_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(lg_transEnergy_r2H[m,op] == math.log(acc.cost_r[m]) + math.log(acc.precision[m,op]) + count_expr_writeIn + lg_dataVolume[m,op],
                                     name=f"C_lg_transEnergy_r2H_({acc.mem2dict(m)},{op_name})")
                    
                    tmp_energy_expr += getExpVar(lg_term=lg_transEnergy_r2H[m,op], lb=0, ub=GRB.INFINITY, name=f"transEnergy_r_b_({acc.mem2dict(m)},{op_name})")

                # 4) Output write-back energy when this level sends psum upward.
                if op_name == 'O':
                    lg_transEnergy_w2H[m,op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_w2H_({acc.mem2dict(m)},{op_name})")

                    model.addConstr(lg_transEnergy_w2H[m,op] == math.log(acc.cost_w[m]) + math.log(acc.precision[m,op]) + count_expr_readOut + lg_transVolume[m,op],
                                     name=f"C_lg_transEnergy_w2H_({acc.mem2dict(m)},{op_name})")
                    
                    tmp_energy_expr += getExpVar(lg_term=lg_transEnergy_w2H[m,op], lb=0, ub=GRB.INFINITY, name=f"transEnergy_w_b_({acc.mem2dict(m)},{op_name})")

                energy_perMem[m,op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"energy_perMem_({acc.mem2dict(m)},{op_name})")
                model.addConstr(energy_perMem[m,op] == tmp_energy_expr, name=f"C_energy_perMem_({acc.mem2dict(m)},{op_name})")

        energy_expr_rw = 0
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    continue
                if m in [acc.Dram2mem, acc.IReg2mem, acc.OReg2mem]:
                    energy_expr_rw += energy_perMem[m,op]
                else:
                    energy_usedMem[m,op] = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                        name=f"energy_usedMem_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrIndicator(indic_usedMem[m,op], False, energy_usedMem[m,op] == 0,
                                                 name=f"C_energy_usedMem_false_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrIndicator(indic_usedMem[m,op], True, energy_usedMem[m,op] == energy_perMem[m,op],
                                                 name=f"C_energy_usedMem_true_({acc.mem2dict(m)},{op_name})")
                    # energy_usedMem[m,op] = var_mul01(model, indic_usedMem[m,op], energy_perMem[m,op],
                    #                                   name=f"energy_usedMem_({acc.mem2dict(m)},{op_name})")
                    energy_expr_rw += energy_usedMem[m,op]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#                 

        energy_expr_leakage = acc.leakage_per_cycle * CONST.SCALE_LATENCY * res_latency

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Latency - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - -#          
              
        transfer = gp.tupledict()                           # transfer[i,op]
        transLatency = gp.tupledict()                       # transLatency[m,op]
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m]:
                    lg_transLatency = model.addVar(lb=LB_lg_transLatency[m,op], ub=UB_lg_transLatency[m,op], vtype=GRB.CONTINUOUS,
                                                    name=f"tmp_lg_transLatency_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(lg_transLatency == lg_transVolume[m,op] + math.log(acc.precision[m,op]) - math.log(acc.bw[m]) - math.log(CONST.SCALE_LATENCY),
                                     name=f"C_lg_transLatency_({acc.mem2dict(m)},{op_name})")
                
                    transLatency[m,op] = model.addVar(lb=0, ub=UB_transLatency[m,op], vtype=GRB.CONTINUOUS, 
                                                       name=f"transLatency_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_transLatency, yvar=transLatency[m,op], options=self.ExpOption, name=f"C_transLatency_({acc.mem2dict(m)},{op_name})")
            for i in range(Num_Loops):
                transfer[i,op] = model.addVar(lb=0, ub=self.MAX_TRANS[op], vtype=GRB.CONTINUOUS, name=f"transfer_({i},{op_name})")
                model.addConstr(transfer[i,op]==quicksum(var_mul01(model, indic_loop2Mem[i,op,m], transLatency[m,op],
                                                                    name=f"tmp_transfer_({i},{op_name})_{acc.mem2dict(m)}")
                                                         for m in range(1, acc.Num_mem) if acc.mappingArray[op][m]),
                                 name=f"C_transfer_({i},{op_name})")
                
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        latency_Critical = gp.tupledict()           # latency_cp[i] = Latency of Critical Path
        latency_Process = gp.tupledict()            # latency_Process[i,op] 
        latency_Transfer = gp.tupledict()           # latency_Transfer[i,op]
        for i in range(Num_Loops):
            latency_Critical[i] = model.addVar(lb=0, ub=UB_latencyLevel[i], vtype=GRB.CONTINUOUS, name=f"latency_Critical_({i})")
            for op, op_name in enumerate(['I','W','O']):
                latency_Process[i,op] = model.addVar(lb=0, ub=UB_latencyLevel[i], vtype=GRB.CONTINUOUS, name=f"latency_Process_({i},{op_name})")
                
                if i < Num_Loops-1:
                    latency_Transfer[i,op] = var_mul01(model, indic_xMem[i, op], transfer[i, op], A_ub=self.MAX_TRANS[op], var_ub=self.MAX_TRANS[op],
                                                        name=f"latency_Transfer_({i},{op_name})")
                else:
                    latency_Transfer[i,op] = model.addVar(lb=0, ub=self.MAX_TRANS[op] + LAT_UNIT, vtype=GRB.CONTINUOUS,
                                                           name=f"latency_Transfer_({i},{op_name})")
                    model.addConstr(latency_Transfer[i,op] == transfer[Num_Loops-1, op] + (1 - indic_usedMem[acc.lastMem[op],op]) / CONST.SCALE_LATENCY,
                                    name=f"C_RegTrans_({op_name})")
                    latency_Process[Num_Loops,op] = t_MAC

        for op, op_name in enumerate(['I','W','O']):
            sum_latency_transfer = quicksum(latency_Transfer[i,op] for i in range(Num_Loops))
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m]:
                    model.addConstr(sum_latency_transfer >= var_mul01(model, indic_usedMem[m,op], transLatency[m,op],
                                                                     name=f"tmp_sum_latencyTransfer_({acc.mem2dict(m)},{op_name})"),
                                    name=f"Cut_sum_latencyTransfer_({acc.mem2dict(m)},{op_name})")

        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):

                tmp_coeff = 2 if op_name == 'O' else 1

                model.addConstr(latency_Critical[i] >= tmp_coeff * latency_Transfer[i,op], name=f"C_latency_cp_transfer_({i},{op_name})")
                model.addConstr(latency_Critical[i] >= latency_Process[i+1,op], name=f"C_latency_cp_process_({i},{op_name})")
                model.addGenConstrIndicator(indic_doubleLoop[i,op], False, 
                                latency_Critical[i] >= tmp_coeff * latency_Transfer[i,op] + latency_Process[i+1,op],
                                 name=f"C_latency_cp_transfer+process_({i},{op_name})")
        
        for i in range(Num_Loops):

            tmp_LxF_expr = quicksum(((UNIQUE_FACTOR[p]-1) * var_mul01(model, indic_loop2Factor[i,p], latency_Critical[i],
                                                                    name=f"tmp_LxF_expr_({i},{p})"))
                                     for p in range(len(UNIQUE_FACTOR)))

            for op, op_name in enumerate(['I','W','O']):
                if op < 2:      # I,W
                    model.addConstr(latency_Process[i,op] >= latency_Process[i+1,op]+tmp_LxF_expr-(2*var_mul01(model, indic_xMem[i,op], latency_Critical[i],
                                                                                                                name=f"tmp01_xMem_LxF_({i},{op_name})")),
                                    name=f"C_latency_process_stationary_({i},{op_name})")

                    model.addConstr(latency_Process[i,op] >= 2*latency_Transfer[i,op]+latency_Process[i+1,op]
                                                                +tmp_LxF_expr-latency_Critical[i]-var_mul01(model, indic_doubleLoop[i,op], latency_Critical[i],
                                                                                                             name=f"tmp01_doubleLoop_LxF_({i},{op_name})"),
                                     name=f"C_latency_process_({i},{op_name})")
                else:           # O
                    model.addConstr(latency_Process[i,op] >= 2*latency_Transfer[i,op]+latency_Process[i+1,op]+tmp_LxF_expr,
                                     name=f"C_latency_process_({i},{op_name})")
                    
                model.addConstr(latency_Process[i,op] >= 2 * latency_Process[i+1,op], name=f"Cut_Hierarchical_Decay_({i},{op_name})")

        model.addConstr(latency_Process[0,2] >= MIN_INNER_PROD[0] * t_MAC +
                        quicksum(2 * MIN_OUTER_PROD[i] * latency_Transfer[i,2] for i in range(Num_Loops)),
                         name="Cut_Output_Transfer_Cascade")
                    
# - - - - - - - - - - - - - - - - - - - - - - - - Dataflow Evaluation Results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        for op, op_name in enumerate(['I','W','O']):
            tmp_coeff = 2 if op_name == 'O' else 1
            offchip_bootstrap = tmp_coeff * MAX_SIZE[op] * acc.precision[acc.Dram2mem,op] / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
            model.addConstr(res_latency >= latency_Process[0,op] + offchip_bootstrap * (1 - indic_loop2Mem[0,op,acc.Dram2mem]),
                             name=f"C_Res_Latency_OffChip_({op})")

        model.addConstr(res_energy >= energy_expr_rw + energy_expr_comp + energy_expr_leakage, name="C_Res_Energy_Summation")
        if CONST.FLAG_OPT == "EDP":
            model.addConstr(res_EDP >= res_latency * res_energy * CONST.SCALINGFACTOR, name="C_Res_EDP_Multiplication")

        match CONST.FLAG_OPT:
            case "Latency":
                model.addConstr(res_latency <= self.metric_ub / CONST.SCALE_LATENCY, name="C_metric_ub_latency")
            case "Energy":
                model.addConstr(res_energy  <= self.metric_ub, name="C_metric_ub_energy")
            case "EDP":
                model.addConstr(res_EDP     <= self.metric_ub / CONST.SCALE_LATENCY, name="C_metric_ub_EDP")
            case _:
                Logger.warning("Undefined Optimization Objective, No Upper Bound Applied.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#            
        Logger.info('* '*20 + "Start Running MIP Solver" + ' *'*20)
        model.ModelSense = GRB.MINIMIZE
        model.Params.TimeLimit = 7  # Determine the feasibility within a 7s time limit.
        model.Params.MIPFocus = 1 
        model.update()
        model.optimize()
        if model.SolCount == 0:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            # ''' Debug IIS
            # model.computeIIS()
            # model.write(os.path.join(self.outputdir, "iis_full.ilp"))
            # model.write(os.path.join(self.outputdir, "model.mps"))
            # exit()
            # '''
            return 1

        ####################################################################  Set Constraint Flag ###################################################################
        model.discardMultiobjEnvs()

        model.setParam('TimeLimit', CONST.TIMELIMIT)
        model.setParam('MIPFocus', CONST.MIPFOCUS)

        match CONST.FLAG_OPT:
            case "Latency":
                model.setObjectiveN(res_latency, 0, priority=3, name='Latency')    
                env0 = model.getMultiobjEnv(0)                                     
                env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.7)
                env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.3)
                env0.setParam('MIPFocus', CONST.MIPFOCUS)
                model.setObjectiveN(res_energy, 1, priority=2, reltol=0.0, abstol=0.0, name='Energy')
                env1 = model.getMultiobjEnv(1)
                env1.setParam('TimeLimit', CONST.TIMELIMIT * 0.3)
                env1.setParam('MIPFocus', CONST.MIPFOCUS)
            case "Energy":
                model.setObjectiveN(res_energy, 0, priority=3, name='Energy')    
                env0 = model.getMultiobjEnv(0)                                     
                env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.7)
                env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.3)
                env0.setParam('MIPFocus', CONST.MIPFOCUS)
                model.setObjectiveN(res_latency, 1, priority=2, reltol=0.0, abstol=0.0, name='Latency')
                env1 = model.getMultiobjEnv(1)
                env1.setParam('TimeLimit', CONST.TIMELIMIT * 0.3)
                env1.setParam('MIPFocus', CONST.MIPFOCUS)
            case "EDP":
                model.setObjective(res_EDP, GRB.MINIMIZE)
                model.setParam('TimeLimit', CONST.TIMELIMIT)
                model.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.3)
                model.setParam('MIPFocus', CONST.MIPFOCUS)
            case _:
                model.setObjective(0)
                model.setParam("TimeLimit", CONST.TIMELIMIT * 0.1)
        ####################################################################  Optimization    #######################################################################

        # FLAG.LOAD_SOLUTION = 1
        # if FLAG.LOAD_SOLUTION:
        #     try:
        #         model.read("MIREDO.sol")
        #         Logger.critical("Load Solution")
        #     except ValueError:
        #         raise ValueError("No MIREDO.sol File")

        model.setParam('TimeLimit', CONST.TIMELIMIT)
        model.update()
        model.write(os.path.join(self.outputdir, "debug_model.lp"))

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
                            loop2Mem = [0,0,0]
                            for m in range(1,acc.Num_mem):
                                for op in range(3):
                                    if acc.mappingArray[op][m] == 0:
                                        continue
                                    if round(indic_loop2Mem[i,op,m].x) == 1:
                                        loop2Mem[op] = m
                            loops.tm.append(Mapping(dim=d, 
                                                    dimSize=factors[d][f],
                                                    mem=[loop2Mem[op] for op in range(3)]))
            
            Logger.debug(self.su)
            for u in range(acc.Num_SpUr):
                for d in range(1, ops.Num_dim):
                    if self.su[u][d] > 1:
                        loops.sm.append(Mapping(dim=d, 
                                                dimSize=self.su[u][d],
                                                mem=[acc.SpUr2Mem[u,op] for op in range(3)]))

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

                # model.Params.TimeLimit = 10  # 设置 10 秒硬性限制

        if model.SolCount > 0:
            Logger.critical("MIP Solved successfully !!!")
            solved_latency = res_latency.x * CONST.SCALE_LATENCY
            solved_energy = res_energy.x
            if CONST.FLAG_OPT == "EDP":
                solved_edp = res_EDP.x * CONST.SCALE_LATENCY
            else:
                solved_edp = solved_latency * solved_energy * CONST.SCALINGFACTOR
            self.result = [solved_latency, solved_energy, solved_edp]
            set_dataflow()
            model.write(os.path.join(self.outputdir, "solution.sol"))
            match CONST.FLAG_OPT:
                case "Latency":
                    Logger.debug(f"Get best Latency= {solved_latency}")
                case "Energy":
                    Logger.debug(f"Get best Energy= {solved_energy}")
                case "EDP":
                    Logger.debug(f"Get best EDP= {solved_edp}")
                case _:
                    Logger.debug(f"Get simple solution, L={solved_latency}, E={solved_energy}")
        else:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            return 1
            model.setParam("IISMethod", 2) 
            model.computeIIS()
            model.write(os.path.join(self.outputdir, "iis_full.ilp"))
            model.write(os.path.join(self.outputdir, "model.mps"))
            Logger.error(f'Model infeasible !!!')
            exit()

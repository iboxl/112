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
        self.MAX_TRANS = [ max([math.ceil(min(ops.size[op] * acc.precision[m,op], acc.memSize[m]) / acc.bw[m]) 
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

        MAX_FACTOR = max([item for sublist in factors for item in sublist])
        MAX_SIZE = self.ops.size

        UNIQUE_FACTOR = getUniqueFactors(factors)

        logF = {(d, f): math.log(factors[d][f]) 
                        for d in range(1, ops.Num_dim) 
                        for f in range(len(factors[d]))}
        
        # ---------------- 体系结构感知的严密上下界推导 ---------------- #

        factors_val = [f for fs in factors[1:ops.Num_dim] for f in fs if fs != [1]]
        f_asc, f_desc = sorted(factors_val), sorted(factors_val, reverse=True)

        MIN_INNER_PROD, MAX_INNER_PROD = {}, {}
        for i in range(Num_Loops):
            MIN_INNER_PROD[i] = int(np.prod(f_asc[:Num_Loops - i]))
            MAX_INNER_PROD[i] = int(np.prod(f_desc[:Num_Loops - i]))
        MIN_INNER_PROD[Num_Loops], MAX_INNER_PROD[Num_Loops] = 1, 1

        MIN_OUTER_PROD = [1] * (Num_Loops + 1)
        for i in range(1, Num_Loops + 1):
            MIN_OUTER_PROD[i] = MIN_OUTER_PROD[i - 1] * f_asc[i - 1]

        spur = {}
        for m in range(1, acc.Num_mem):
            for d in range(1, ops.Num_dim):
                for op, op_name in enumerate(['I','W','O']):
                    spur[m,op,d] = 1
                    for u in range(acc.Num_SpUr):
                        if m <= acc.SpUr2Mem[u,op]:
                            spur[m,op,d] *= self.su[u][d]

        LB_dataVolume,      UB_dataVolume = {}, {}
        LB_lg_dataVolume,   UB_lg_dataVolume = {}, {}
        LB_transLatency,    UB_transLatency = {}, {}
        LB_lg_transLatency, UB_lg_transLatency = {}, {}
        for m in range(1, acc.Num_mem):
            min_dataVolume = [0,0,0]
            r_min = spur[m,0,ops.dict2Dim('R')]
            s_min = spur[m,0,ops.dict2Dim('S')]
            p_min = spur[m,0,ops.dict2Dim('P')]
            q_min = spur[m,0,ops.dict2Dim('Q')]

            h_min = (p_min * r_min) if ops.Stride >= r_min else ((p_min - 1) * ops.Stride + r_min)
            w_min = (q_min * s_min) if ops.Stride >= s_min else ((q_min - 1) * ops.Stride + s_min)

            min_dataVolume[0] = max(1, min(h_min, ops.H) * min(w_min, ops.W) * spur[m,0,ops.dict2Dim('C')])
            min_dataVolume[1] = max(1, spur[m,1,ops.dict2Dim('R')] * spur[m,1,ops.dict2Dim('S')] *
                                       spur[m,1,ops.dict2Dim('C')] * spur[m,1,ops.dict2Dim('K')])
            min_dataVolume[2] = max(1, spur[m,2,ops.dict2Dim('P')] * spur[m,2,ops.dict2Dim('Q')] *
                                       spur[m,2,ops.dict2Dim('K')])

            for op, op_name in enumerate(['I','W','O']):
                UB_dataVolume[m,op] = min(acc.memSize[m] // acc.precision[m,op], MAX_SIZE[op])
                LB_dataVolume[m,op] = min(min_dataVolume[op], UB_dataVolume[m,op])
                LB_lg_dataVolume[m,op] = math.log(LB_dataVolume[m,op])
                UB_lg_dataVolume[m,op] = math.log(UB_dataVolume[m,op])
                LB_transLatency[m,op] = LB_dataVolume[m,op] * acc.precision[m,op] / acc.bw[m]
                UB_transLatency[m,op] = min(MAX_SIZE[op] * acc.precision[m,op], acc.memSize[m]) / acc.bw[m]
                LB_lg_transLatency[m,op] = math.log(LB_transLatency[m,op])
                UB_lg_transLatency[m,op] = math.log(UB_transLatency[m,op])

        UB_Process, LB_Process = {}, {}
        UB_TransferRaw, LB_TransferRaw = {}, {}
        UB_TransferActive, LB_TransferActive = {}, {}
        UB_Critical, LB_Critical = {}, {}

        t_MAC = acc.t_MAC
        c_coeff = [1, 1, 2]
        XMAX_TOTAL = {0: min(4, Num_Loops), 1: min(2, Num_Loops), 2: min(4, Num_Loops)}

        for op in range(3):
            LB_TransferRaw[op] = min(LB_transLatency[m,op] for m in range(1, acc.Num_mem) if acc.mappingArray[op][m])
        for i in range(Num_Loops):
            for op in range(3):
                UB_TransferRaw[i, op] = self.MAX_TRANS[op]
                LB_TransferActive[i, op] = LB_TransferRaw[op]

        UB_ProcessBase, UB_CriticalBase = {}, {}
        for op in range(3):
            UB_ProcessBase[Num_Loops, op] = t_MAC

        for i in range(Num_Loops - 1, -1, -1):
            num_inner = Num_Loops - i
            inner_f = f_desc[:num_inner]
            P_cur = {op: float(t_MAC) for op in range(3)}
            for k in range(num_inner - 1, -1, -1):
                F = inner_f[k]
                C = max(c_coeff[op] * self.MAX_TRANS[op] + P_cur[op] for op in range(3))
                P_new = {}
                for op in range(3):
                    if op < 2:
                        P_new[op] = max(P_cur[op] + (F - 1) * C,
                                        2 * self.MAX_TRANS[op] + P_cur[op] + max(0, F - 2) * C,
                                        2 * P_cur[op])
                    else:
                        P_new[op] = max(P_cur[op] + (F - 1) * C,
                                        2 * self.MAX_TRANS[op] + P_cur[op] + (F - 1) * C,
                                        2 * P_cur[op])
                P_cur = P_new
            for op in range(3):
                UB_ProcessBase[i, op] = math.ceil(P_cur[op])

        self.lb_latency = max(MIN_INNER_PROD[0] * t_MAC, 1)

        offchip_max = [c_coeff[op] * MAX_SIZE[op] * acc.precision[acc.Dram2mem, op] / acc.bw[acc.Dram2mem]
                       for op in range(3)]
        computed_ub = max(UB_ProcessBase[0, op] + offchip_max[op] for op in range(3))
        if CONST.FLAG_OPT == "Latency" and self.metric_ub < computed_ub:
            self.ub_latency = self.metric_ub
        else:
            self.ub_latency = computed_ub

        if Num_Loops > 0:
            min_output_transfer = min(LB_transLatency[m,2] for m in range(1, acc.Num_mem) if acc.mappingArray[2][m])
            self.lb_latency = max(self.lb_latency,
                                  MIN_INNER_PROD[0] * t_MAC + 2 * MIN_OUTER_PROD[Num_Loops - 1] * min_output_transfer)

        for i in range(Num_Loops):
            hierarchy_p_ub = math.ceil(self.ub_latency / max(MIN_OUTER_PROD[i], 1))
            for op in range(3):
                UB_ProcessBase[i, op] = min(UB_ProcessBase[i, op], hierarchy_p_ub)

        for i in range(Num_Loops):
            from_process = math.ceil(
                max(c_coeff[op] * self.MAX_TRANS[op] + UB_ProcessBase[i + 1, op] for op in range(3))
            )
            from_hierarchy = math.ceil(self.ub_latency / max(MIN_OUTER_PROD[i + 1], 1))
            UB_CriticalBase[i] = min(from_process, from_hierarchy)

        def run_process_ub_dp(critical_cap):
            process_ub = {}
            for op in range(3):
                process_ub[Num_Loops, op] = t_MAC
            for start in range(Num_Loops - 1, -1, -1):
                process_dp = {}
                for op in range(3):
                    for b in range(XMAX_TOTAL[op] + 1):
                        process_dp[Num_Loops, op, b] = t_MAC
                for i in range(Num_Loops - 1, start - 1, -1):
                    F = f_desc[i - start]
                    for op in range(3):
                        for b in range(XMAX_TOTAL[op] + 1):
                            stay_branch = process_dp[i + 1, op, b] + (F - 1) * critical_cap[i]
                            hier_branch = 2 * process_dp[i + 1, op, b]
                            best = max(stay_branch, hier_branch)
                            if b > 0:
                                trans_branch = 2 * UB_TransferRaw[i, op] + process_dp[i + 1, op, b - 1]
                                if op < 2:
                                    trans_branch += max(0, F - 2) * critical_cap[i]
                                else:
                                    trans_branch += (F - 1) * critical_cap[i]
                                best = max(best, trans_branch)
                            process_dp[i, op, b] = math.ceil(best)
                hierarchy_p_ub = math.ceil(self.ub_latency / max(MIN_OUTER_PROD[start], 1))
                for op in range(3):
                    process_ub[start, op] = min(math.ceil(process_dp[start, op, XMAX_TOTAL[op]]), hierarchy_p_ub)
            return process_ub

        def tighten_active_transfer_and_critical(process_ub, critical_cap):
            transfer_active = {}
            critical_ub = {}
            for i in range(Num_Loops):
                critical_terms = []
                for op in range(3):
                    transfer_cap = UB_TransferRaw[i, op] + (1 if i == Num_Loops - 1 else 0)
                    transfer_cap = min(transfer_cap, math.ceil(critical_cap[i] / c_coeff[op]))
                    transfer_active[i, op] = max(LB_TransferActive[i, op], math.ceil(transfer_cap))
                    critical_terms.append(c_coeff[op] * transfer_active[i, op] + process_ub[i + 1, op])
                critical_ub[i] = min(math.ceil(max(critical_terms)),
                                     math.ceil(self.ub_latency / max(MIN_OUTER_PROD[i + 1], 1)))
            return transfer_active, critical_ub

        UB_Process = run_process_ub_dp(UB_CriticalBase)
        UB_TransferActive, UB_CriticalMid = tighten_active_transfer_and_critical(UB_Process, UB_CriticalBase)

        UB_Process = run_process_ub_dp(UB_CriticalMid)
        UB_TransferActive, UB_Critical = tighten_active_transfer_and_critical(UB_Process, UB_CriticalMid)

        computed_ub = max(UB_Process[0, op] + offchip_max[op] for op in range(3))
        if CONST.FLAG_OPT == "Latency" and self.metric_ub < computed_ub:
            self.ub_latency = self.metric_ub
        else:
            self.ub_latency = min(self.ub_latency, computed_ub)

        for i in range(Num_Loops):
            hierarchy_p_ub = math.ceil(self.ub_latency / max(MIN_OUTER_PROD[i], 1))
            hierarchy_c_ub = math.ceil(self.ub_latency / max(MIN_OUTER_PROD[i + 1], 1))
            for op in range(3):
                UB_Process[i, op] = min(UB_Process[i, op], hierarchy_p_ub)
            UB_Critical[i] = min(UB_Critical[i], hierarchy_c_ub)
            for op in range(3):
                transfer_cap = UB_TransferRaw[i, op] + (1 if i == Num_Loops - 1 else 0)
                transfer_cap = min(transfer_cap, math.ceil(UB_Critical[i] / c_coeff[op]))
                UB_TransferActive[i, op] = max(LB_TransferActive[i, op], math.ceil(transfer_cap))

        for op in range(3):
            LB_Process[Num_Loops, op] = t_MAC
        for i in range(Num_Loops):
            for op in range(3):
                LB_Process[i, op] = MIN_INNER_PROD[i] * t_MAC
            LB_Critical[i] = MIN_INNER_PROD[i + 1] * t_MAC
        if Num_Loops > 0:
            LB_Critical[Num_Loops - 1] = max(LB_Critical[Num_Loops - 1],
                                             max(c_coeff[op] * LB_TransferRaw[op] for op in range(3)))

        for op in range(3):
            assert LB_TransferRaw[op] <= min(UB_TransferRaw[i, op] for i in range(Num_Loops)), f"Transfer raw bound inversion on op {op}"
        for i in range(Num_Loops):
            assert LB_Critical[i] <= UB_Critical[i], f"Critical bound inversion at loop {i}"
            for op in range(3):
                assert LB_Process[i, op] <= UB_Process[i, op], f"Process bound inversion at loop {i}, op {op}"
                assert LB_TransferActive[i, op] <= UB_TransferActive[i, op], f"Active transfer bound inversion at loop {i}, op {op}"
        
        #######################################################################################################################################

        # Logger.info(f"Operand-specific Tight DP Bounds LB: {self.lb_latency}, UB: {self.ub_latency}")
        # for i in range(Num_Loops):
        #     print(f"loop {i}: Critical LB-[ {min_inner_prod[i+1] * acc.t_MAC} ] UB-[ {LB_Critical[i]}-{UB_Critical[i]} ]")
        #     pstr = "Latency in "
        #     for op, op_name in enumerate(['I','W','O']):
        #         pstr += f"{op_name}: Process-[{LB_Process[i,op]}-{UB_Process[i,op]}] ,TransRaw-[{LB_TransferRaw[op]}-{UB_TransferRaw[i,op]}] "
        #     print(pstr)
        # exit()

        ###########################################################  Variable & Constant & Constraints  ##################################################################
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        res_latency = model.addVar(lb=self.lb_latency, ub=self.ub_latency, vtype=GRB.CONTINUOUS,     name="res_latency")
        res_energy  = model.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_energy")
        res_EDP     = model.addVar(lb=1, vtype=GRB.CONTINUOUS,  name="res_EDP")

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
        
        indic_loop2Factor = gp.tupledict()
        for i in range(Num_Loops):
            for p in range(len(UNIQUE_FACTOR)):
                indic_loop2Factor[i,p] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Factor_({i},{p})")
            model.addConstr(quicksum(indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR))) == 1, name=f"C_Uniqueness_Indic_loop2Factor_({i})")
            model.addConstr(quicksum(UNIQUE_FACTOR[p] * indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR)))
                            ==
                            quicksum(indic_factor2Loop[d,f,i] * factors[d][f] for d in range(1, ops.Num_dim) for f in range(len(factors[d]))),
                             name=f"C_loop2Factor_definition_({i})")
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

                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume[m,op], ub=UB_lg_dataVolume[m,op],
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimExistMem[m,op,ops.dict2Dim('C')],
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=LB_dataVolume[m,op], ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                         name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                        
            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume[m,op], ub=UB_lg_dataVolume[m,op],
                                                name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K']),
                                name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=LB_dataVolume[m,op], ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=self.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume[m,op], ub=UB_lg_dataVolume[m,op],
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K']),
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=LB_dataVolume[m,op], ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
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

                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume[m,op], ub=UB_lg_dataVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimOfTile[m,op,ops.dict2Dim('C')],
                                    name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume[m,op], ub=UB_lg_dataVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume[m,op], ub=UB_lg_dataVolume[m,op],
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
        

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Latency - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - -#          
        indic_xMem = gp.tupledict()                 # indic_xMem[i,op] = {0,1}
        for op, op_name in enumerate(['I','W','O']):
            for i in range(Num_Loops):
                indic_xMem[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_xMem_({i},{op_name})")
                if i == Num_Loops-1:
                    if op_name == 'W':
                        tmp_diff = acc.Macro2mem - quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem))
                    else:
                        tmp_diff = acc.Num_mem - quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem))
                else:
                    tmp_diff = quicksum(m*indic_loop2Mem[i+1,op,m] for m in range(1,acc.Num_mem)) - quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem))
                model.addConstr(tmp_diff >= indic_xMem[i,op], name=f"C_tmp_diff_({i},{op_name})")
                model.addConstr(tmp_diff <= acc.Num_mem * indic_xMem[i,op], name=f"C_tmp_diff_ub_({i},{op_name})")
                
            if op_name == 'W':      # Weight
                model.addConstr(quicksum(indic_xMem[i,op] for i in range(Num_Loops)) <= 2, name=f"max_xMem_{op_name}")
            else:                   # Feature
                model.addConstr(quicksum(indic_xMem[i,op] for i in range(Num_Loops)) <= 4, name=f"max_xMem_{op_name}")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        transfer = gp.tupledict()                           # transfer[i,op]
        transLatency = gp.tupledict()                       # transLatency[m,op]
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m]:
                    lg_transLatency = model.addVar(lb=LB_lg_transLatency[m,op], ub=UB_lg_transLatency[m,op], vtype=GRB.CONTINUOUS,
                                                    name=f"tmp_lg_transLatency_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(lg_transLatency == lg_transVolume[m,op] + math.log(acc.precision[m,op]) - math.log(acc.bw[m]),
                                     name=f"C_lg_transLatency_({acc.mem2dict(m)},{op_name})")
                
                    transLatency[m,op] = model.addVar(lb=LB_transLatency[m,op], ub=UB_transLatency[m,op], vtype=GRB.CONTINUOUS, 
                                                       name=f"transLatency_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_transLatency, yvar=transLatency[m,op], options=self.ExpOption, name=f"C_transLatency_({acc.mem2dict(m)},{op_name})")
            for i in range(Num_Loops):
                transfer[i,op] = model.addVar(lb=LB_TransferRaw[op], ub=UB_TransferRaw[i,op], vtype=GRB.CONTINUOUS, name=f"transfer_({i},{op_name})")
                model.addConstr(transfer[i,op]==quicksum(var_mul01(model, indic_loop2Mem[i,op,m], transLatency[m,op],
                                                                    name=f"tmp_transfer_({i},{op_name})_{acc.mem2dict(m)}")
                                                         for m in range(1, acc.Num_mem) if acc.mappingArray[op][m]),
                                 name=f"C_transfer_({i},{op_name})")
                
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        latency_Critical = gp.tupledict()           # latency_cp[i] = Latency of Critical Path
        latency_Process = gp.tupledict()            # latency_Process[i,op] 
        latency_Transfer = gp.tupledict()           # latency_Transfer[i,op]
        for i in range(Num_Loops):
            latency_Critical[i] = model.addVar(lb=LB_Critical[i], ub=UB_Critical[i], vtype=GRB.CONTINUOUS, name=f"latency_Critical_({i})")
            for op, op_name in enumerate(['I','W','O']):
                latency_Process[i,op] = model.addVar(lb=LB_Process[i,op], ub=UB_Process[i,op], vtype=GRB.CONTINUOUS, name=f"latency_Process_({i},{op_name})")
                
                if i < Num_Loops-1:
                    latency_Transfer[i,op] = var_mul01(model, indic_xMem[i, op], transfer[i, op], var_ub=UB_TransferActive[i,op],
                                                        name=f"latency_Transfer_({i},{op_name})")
                else:
                    latency_Transfer[i,op] = model.addVar(lb=LB_TransferActive[i,op], ub=UB_TransferActive[i,op], vtype=GRB.CONTINUOUS,
                                                           name=f"latency_Transfer_({i},{op_name})")
                    model.addConstr(latency_Transfer[i,op] == transfer[Num_Loops-1, op] + 1 - indic_usedMem[acc.lastMem[op],op] , name=f"C_RegTrans_({op_name})")
                    latency_Process[Num_Loops,op] = acc.t_MAC

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

        model.addConstr(latency_Process[0,2] >= MIN_INNER_PROD[0] * acc.t_MAC +
                        quicksum(2 * MIN_OUTER_PROD[i] * latency_Transfer[i,2] for i in range(Num_Loops)),
                        name="Cut_Output_Transfer_Cascade")
                    
# - - - - - - - - - - - - - - - - - - - - - - - - Dataflow Evaluation Results - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        for op, op_name in enumerate(['I','W','O']):
            tmp_coeff = 2 if op_name == 'O' else 1
            offchip_bootstrap = tmp_coeff * MAX_SIZE[op] * acc.precision[acc.Dram2mem,op] / acc.bw[acc.Dram2mem]
            model.addConstr(res_latency >= latency_Process[0,op] + offchip_bootstrap * (1 - indic_loop2Mem[0,op,acc.Dram2mem]),
                             name=f"C_Res_Latency_OffChip_({op})")

        # model.addConstr(res_energy >= energy_expr_rw + energy_expr_comp + energy_expr_leakage, name="C_Res_Energy_Summation")

        # model.addConstr(res_EDP >= res_latency * res_energy * CONST.SCALINGFACTOR, name="C_Res_EDP_Multiplication")

        match CONST.FLAG_OPT:
            case "Latency":
                model.addConstr(res_latency <= self.metric_ub, name="C_metric_ub_latency")
            case "Energy":
                model.addConstr(res_energy  <= self.metric_ub, name="C_metric_ub_energy")
            case "EDP":
                model.addConstr(res_EDP     <= self.metric_ub, name="C_metric_ub_EDP")
            case _:
                Logger.warning("Undefined Optimization Objective, No Upper Bound Applied.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          
        assert CONST.FLAG_OPT== "Latency", "This version only supports Latency Optimization, Please set FLAG_OPT to Latency and try again."

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

        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            Logger.critical("MIP Solved successfully !!!")
            self.result = [res_latency.x, res_energy.x, res_EDP.x]
            set_dataflow()
            model.write(os.path.join(self.outputdir, "solution.sol"))
            match CONST.FLAG_OPT:
                case "Latency":
                    Logger.debug(f"Get best Latency= {res_latency.x}")
                case "Energy":
                    Logger.debug(f"Get best Energy= {res_energy.x}")
                case "EDP":
                    Logger.debug(f"Get best EDP= {res_EDP.x}")
                case _:
                    Logger.debug(f"Get simple solution, L={res_latency.x}, E={res_energy.x}")
        else:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            return 1
            model.setParam("IISMethod", 2) 
            model.computeIIS()
            model.write(os.path.join(self.outputdir, "iis_full.ilp"))
            model.write(os.path.join(self.outputdir, "model.mps"))
            Logger.error(f'Model infeasible !!!')
            exit()

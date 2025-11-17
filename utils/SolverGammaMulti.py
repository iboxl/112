# this file is prepared for project 026
# Created by iboxl

from Architecture.Accelerator import CIM_acc
from utils.Workload import Operands
import math
import gurobipy as gp
from gurobipy import GRB, quicksum, min_
from utils.GlobalUT import *
from utils.UtilsFunction.SolverFunction import *
from utils.UtilsFunction.CostFunction import _Cost_model

def Solver_block_gamma(acc:CIM_acc, ops:Operands):
    Logger.debug("use Solver-block")
    m = gp.Model()
    cost = _Cost_model(acc=acc, model=m, ops=ops)
    m.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
    m.setParam('NonConvex', 2)
    m.setParam('MIPFocus', 1)
    m.setParam('Cuts', 2)
    m.setParam('Threads', 112)
    m.setParam('FeasibilityTol', 5e-3)                  # 降低容忍度容易导致求解失败
    m.setParam('IntFeasTol', 5e-3)                      # 通过SIMU避免最终结果的差异
    m.setParam('Presolve', 2)
    # m.setParam('ImproveStartTime', 15)
    m.setParam('Heuristics', 0.5)
    m.setParam('BranchDir', -1)
    m.setParam('VarBranch', 1)
    m.setParam('PreQLinearize', 2)
    m.setParam("TimeLimit", CONST.TIMELIMIT)
    if CONST.FLAG_OPT == 0:
        m.setParam('MIPFocus', 1)
        m.setParam("ZeroObjNodes",10000)

####################################################################  Variable & Constant  ##################################################################
    MAX_BLOCK_N = min(CONST.MAX_BLOCK_N, math.ceil(ops.dim_N / (acc.macro.column // ops.weight.bitwidth)))
    MAX_BLOCK_M = min(CONST.MAX_BLOCK_M, math.ceil(ops.dim_M / (acc.core.size_output_buffer//(ops.output.bitwidth*acc.macro.column // ops.weight.bitwidth))))
    # Logger.debug(f"max_block_n: {MAX_BLOCK_N}, max_block_m: {MAX_BLOCK_M}")

    alpha = m.addVars(MAX_BLOCK_N, lb=0, ub=acc.num_core, vtype=GRB.INTEGER, name='alpha')
    tag_block_n = m.addVars(MAX_BLOCK_N, vtype=GRB.BINARY, name="tag")
    num_tag_n = tag_constraint(model=m, tag=tag_block_n, num=MAX_BLOCK_N, name="num_tag_n")
    beta = m.addVars(MAX_BLOCK_N, lb=0, ub=acc.num_core, vtype=GRB.INTEGER, name="beta")
    gamma = m.addVars(MAX_BLOCK_N, lb=0, ub=acc.num_core, vtype=GRB.INTEGER, name="gamma")

    if ops.multi > 1:
        num_parallel = m.addVar(lb=1, ub=acc.num_core, vtype=GRB.INTEGER, name="num_parallel")
        num_core_separate = m.addVar(lb=1, ub=acc.num_core, vtype=GRB.INTEGER, name="num_core_separate")
        num_separate_times = m.addVar(lb=1, ub=ops.multi, vtype=GRB.INTEGER, name="num_separate_times")
        m.addConstr(num_parallel * num_core_separate <= acc.num_core)
        m.addConstr(num_separate_times * num_parallel >= ops.multi)
        if FLAG.ROW_STATIONARY and ops.dim_N <=  acc.macro.column // ops.weight.bitwidth:
            m.addConstr(num_parallel == acc.num_core)
            m.addConstr(num_core_separate == 1)
    else:
        num_parallel = 1
        num_core_separate = acc.num_core
        num_separate_times = 1

    num_block_m = m.addVars(MAX_BLOCK_N, lb=1, ub=MAX_BLOCK_M, vtype=GRB.INTEGER, name=f"block_m")

    dim_block_n = m.addVars(MAX_BLOCK_N, lb=0, ub=ops.dim_N, vtype=GRB.INTEGER, name="dim_block_n")
    dim_block_k = m.addVars(MAX_BLOCK_N, lb=0, ub=ops.dim_K, vtype=GRB.INTEGER, name="dim_block_k")
    dim_block_m = m.addVars(MAX_BLOCK_N, lb=0, ub=ops.dim_M, vtype=GRB.INTEGER, name="dim_block_m")

    dim_macro_para = m.addVars(MAX_BLOCK_N, lb=0, ub=min(ops.dim_N, acc.macro.column // ops.weight.bitwidth), vtype=GRB.INTEGER, name="dim_macro_para")
    dim_compartment_acc=m.addVars(MAX_BLOCK_N, lb=0, ub=max(1, math.ceil(ops.dim_K / acc.macro.compartment)), vtype=GRB.INTEGER, name="dim_compartment_acc")
    dim_compartment_para = m.addVars(MAX_BLOCK_N, lb=0, ub=max(1, acc.macro.cell), vtype=GRB.INTEGER, name="dim_compartment_para")

    if (not FLAG.BLOCK_M) or FLAG.ROW_STATIONARY:
        m.remove(num_block_m)
        m.remove(dim_block_m)
        dim_block_m = [ops.dim_M for i in range(MAX_BLOCK_N)]
        num_block_m = [1 for i in range(MAX_BLOCK_N)]
    else:
        m.addConstrs(dim_block_m[i] * num_block_m[i] >= ops.dim_M for i in range(MAX_BLOCK_N))

    if (not FLAG.GAMMA) or FLAG.ROW_STATIONARY:
        m.remove(gamma)
        m.remove(dim_block_k)
        dim_block_k = [ops.dim_K for i in range(MAX_BLOCK_N)]
        gamma = [1 for i in range(MAX_BLOCK_N)]
    else:
        m.addConstrs(dim_block_k[i] * gamma[i] >= ops.dim_K for i in range(MAX_BLOCK_N))
    
    bandwidth_in_seprate = m.addVars(MAX_BLOCK_N, lb=0, ub=acc.bandwidth_g2ib, vtype=GRB.CONTINUOUS)
    bandwidth_out_seprate = m.addVars(MAX_BLOCK_N, lb=0, ub=acc.bandwidth_ob2g, vtype=GRB.CONTINUOUS)

####################################################################  Constraints    #######################################################################

    for i in range(MAX_BLOCK_N):        # num core constraint
        abc_core = var_mulABC(model=m, vtype=GRB.INTEGER, A=alpha[i], B=beta[i], C=gamma[i])
        abc_core.VarName = f"abc_core_{i}"
        m.addGenConstrIndicator(tag_block_n[i], True, abc_core <= num_core_separate, name=f"num_core_constraint_{i}")
        m.addConstr(abc_core>=tag_block_n[i], name="constriant_tag_{i}")

    for i in range(MAX_BLOCK_N):
        tmp_para = var_eq(model=m, vtype=GRB.INTEGER, expr=dim_macro_para[i] * dim_compartment_para[i])
        m.addConstr(dim_block_n[i] == alpha[i] * tmp_para)
        m.addConstr(dim_compartment_acc[i]*gamma[i]>=math.ceil(ops.dim_K/acc.macro.compartment))

    m.addConstrs(bandwidth_in_seprate[i] *  var_mul(model=m, vtype=GRB.INTEGER, A=gamma[i], B=num_parallel) == acc.bandwidth_g2ib for i in range(MAX_BLOCK_N))
    m.addConstrs(bandwidth_out_seprate[i] * var_mul(model=m, vtype=GRB.INTEGER, A=gamma[i], B=num_parallel) == acc.bandwidth_ob2g for i in range(MAX_BLOCK_N))
        
    m.addConstr(quicksum(dim_block_n[i] * tag_block_n[i] for i in range(MAX_BLOCK_N)) >= ops.dim_N)
    m.update()

    time_cost_n = m.addVars(MAX_BLOCK_N, vtype=GRB.CONTINUOUS)
    energy_cost_n = m.addVars(MAX_BLOCK_N, vtype=GRB.CONTINUOUS)
    for i in range(MAX_BLOCK_N):

        tmp_weight_num = var_mul(model=m, vtype=GRB.INTEGER, A=alpha[i], B=dim_compartment_para[i])
        t_weight_1st = var_mul(model=m, vtype=GRB.CONTINUOUS, A=tmp_weight_num, B=num_parallel) * \
                        var_min_const(model=m, vtype=GRB.INTEGER, B=dim_block_k[i], const=acc.macro.compartment * acc.macro.cell) 
        t_weight_all = var_mul(model=m, vtype=GRB.CONTINUOUS, A=tmp_weight_num, B=num_parallel) * dim_block_k[i]# *                                                        # 和dim_n无关

        v_computation = var_AmulBeqC(model=m, vtype=GRB.CONTINUOUS, A=dim_compartment_para[i], 
                                     B = var_min_const(model=m, vtype=GRB.INTEGER, B=dim_block_k[i], const=acc.macro.compartment)*beta[i])
        v_computation.VarName = f"v_computation_{i}"
        v_stalling = m.addVar(lb=1, vtype=GRB.CONTINUOUS, name=f"v_stalling_{i}")

        if FLAG.INPUT_STATIONARY:
            tag_input_stationary = m.addVar(vtype=GRB.BINARY) 
            ibuffer_size = var_mul(model=m, vtype=GRB.CONTINUOUS, A=beta, B=acc.core.size_input_buffer)
            input_data_size = var_mul(model=m, vtype=GRB.CONTINUOUS, A=dim_block_m[i], B=dim_block_k[i]) * ops.input.bitwidth #dim_block_m * ops.dim_K * ops.input.bitwidth
            m.addGenConstrIndicator(tag_input_stationary, True, ibuffer_size>=input_data_size)
            m.addGenConstrIndicator(tag_input_stationary, False, ibuffer_size<=input_data_size-1)
            m.addGenConstrIndicator(tag_input_stationary, False, v_stalling<=bandwidth_in_seprate[i])
        else:
            m.addConstr(v_stalling == var_min(model=m, vtype=GRB.CONTINUOUS, A=v_computation, B=bandwidth_in_seprate[i]))
        t_input = var_AmulBeqC(model=m, vtype=GRB.CONTINUOUS, A=v_stalling, B=dim_block_m[i] * dim_block_k[i] * ops.input.bitwidth)
        if FLAG.DEBUG_PER_LAYER_DETAIL:
            t_i = m.addVar(vtype=GRB.CONTINUOUS, name=f"t_input_{i}")
            m.addConstr(t_i==t_input)
        output_data_all = dim_block_m[i] * dim_block_n[i] * ops.output.bitwidth

        # t_input 已经包含了计算和载入的时间 （考虑了输入带宽）

        output_buffer_capacity = alpha[i] * beta[i] * acc.core.size_output_buffer
        output_psum_overSize = m.addVar(lb=0, ub=ops.dim_M*ops.dim_N*ops.output.bitwidth, vtype=GRB.INTEGER, name=f"output_overSize_{i}")
        tag_overSize, var_output_data_all, var_output_buffer_capacity = var_AgeB(model=m, A=output_data_all, B=output_buffer_capacity)
        tag_overSize.VarName = f"tag_overSize_{i}"
        m.addGenConstrIndicator(tag_overSize, True, output_psum_overSize==var_output_data_all-var_output_buffer_capacity)
        m.addGenConstrIndicator(tag_overSize, False, output_psum_overSize==0)
        
        tag_small_k,_1,_2 = var_AleB(model=m, A=dim_block_k[i], B=acc.macro.compartment * acc.macro.cell)     # 不需要保存Psum
        tag_small_k.VarName = f"tag_small_k_{i}"
        m.addGenConstrIndicator(tag_small_k, True, var_mul(model=m, vtype=GRB.INTEGER, A=dim_compartment_para[i], B=dim_compartment_acc[i])<=acc.macro.cell)
        m.addGenConstrIndicator(tag_small_k, False, dim_compartment_para[i]==1)

        t_out_1 = var_AmulBeqC(model=m, vtype=GRB.CONTINUOUS, A=bandwidth_out_seprate[i], B=var_output_data_all)
        t_out_1.VarName = f"t_out_1_{i}"
        e_merge_SIMD_once_1 = 0

        t_s1 = var_AmulBeqC(model=m, vtype=GRB.CONTINUOUS, A=bandwidth_out_seprate[i], B=output_psum_overSize * dim_compartment_acc[i])           # Psum反复输出
        t_s1.VarName = f"t_s1_{i}"
        t_s2 = var_AmulBeqC(model=m, vtype=GRB.CONTINUOUS, A=bandwidth_out_seprate[i], B=var_output_buffer_capacity)         # 最终累加结果输出
        t_s2.VarName = f"t_s2_{i}"
        # W.T.D t_SIMD 并行于t_out和 ？ 
        t_out_overSize = exp_add(model=m, vtype=GRB.CONTINUOUS, A=t_s1, B=t_s2)   
        t_out_inSize = var_AmulBeqC(model=m, vtype=GRB.CONTINUOUS, A=bandwidth_out_seprate[i], B=var_output_data_all)
        t_out_2 = m.addVar(vtype=GRB.CONTINUOUS)
        m.addGenConstrIndicator(tag_overSize, True, t_out_2 == t_out_overSize)
        m.addGenConstrIndicator(tag_overSize, False, t_out_2 == t_out_inSize)
        e_merge_SIMD_once_2 = cost.mergePsum_simd(dataSize=output_psum_overSize, num_psums=dim_compartment_acc[i])

        t_out = m.addVar(vtype=GRB.CONTINUOUS, name= f"t_out_{i}")
        e_merge_SIMD_once = m.addVar(vtype=GRB.CONTINUOUS, name=f"e_merge_SIMD_once_{i}")
        m.addGenConstrIndicator(tag_small_k, True, t_out == t_out_1)
        m.addGenConstrIndicator(tag_small_k, False, t_out == t_out_2)
        m.addGenConstrIndicator(tag_small_k, True, e_merge_SIMD_once == e_merge_SIMD_once_1)
        m.addGenConstrIndicator(tag_small_k, False, e_merge_SIMD_once == e_merge_SIMD_once_2)

        t_in = var_eq(model=m, vtype=GRB.CONTINUOUS, expr=t_input + t_weight_all - t_weight_1st)
        t_in.VarName = f"t_in_{i}"
        t_overlap = var_max(model=m, vtype=GRB.CONTINUOUS, A=t_out, B=t_in)
        t_overlap.VarName = f"t_overlap_{i}"

        e_load_input = cost.load_input(dataShape=var_mul(model=m, vtype=GRB.INTEGER, A=dim_block_m[i], B=dim_block_k[i]), alpha=alpha[i])           # weight shift for input stationary
        e_load_weight = cost.load_weight(dataShape=(dim_block_k[i]*dim_block_n[i]), alpha=alpha[i], beta=beta[i], para=dim_compartment_para[i], dim_k=dim_block_k[i])   # num_block_m
        e_computation = cost.mac(alpha=alpha[i], acc=dim_compartment_acc[i], para=dim_compartment_para[i], dim_m=dim_block_m[i], dim_k=dim_block_k[i])
        e_addTree = cost.addTree(acc=dim_compartment_acc[i], para=dim_compartment_para[i], alpha=alpha[i])
        
        e_mm_once = e_load_input + e_load_weight + e_computation + e_addTree 

        if FLAG.DEBUG_PER_LAYER_DETAIL:
            e_i = m.addVar(vtype=GRB.CONTINUOUS, name=f"e_load_input_{i}")
            e_c = m.addVar(vtype=GRB.CONTINUOUS, name=f"e_compute_{i}")
            e_a = m.addVar(vtype=GRB.CONTINUOUS, name=f"e_addTree_{i}")
            e_w = m.addVar(vtype=GRB.CONTINUOUS, name=f"e_load_weight_{i}")
            m.addConstr(e_i==e_load_input)
            m.addConstr(e_c==e_computation)
            m.addConstr(e_a==e_addTree)
            m.addConstr(e_w==e_load_weight)

        e_merge_intraCore_overSize = cost.mergePsum_intra(dataSize=var_output_buffer_capacity, num_psums=dim_compartment_acc[i])
        e_merge_intraCore_inSize = cost.mergePsum_intra(dataSize=var_output_data_all, num_psums=dim_compartment_acc[i])
        e_merge_intraCore_once = m.addVar(vtype=GRB.CONTINUOUS)
        e_merge_intraCore_once.VarName = f"e_merge_intraCore_once_{i}"
        m.addGenConstrIndicator(tag_overSize, True, e_merge_intraCore_once == e_merge_intraCore_overSize)
        m.addGenConstrIndicator(tag_overSize, False, e_merge_intraCore_once == e_merge_intraCore_inSize)

        t_block = var_add(model=m, vtype=GRB.CONTINUOUS, A=t_overlap, B=t_weight_1st)
        t_block_g = var_mul(model=m, vtype=GRB.CONTINUOUS, A=t_block, B=gamma[i])
        # t_merge_global = 
        m.addConstr(time_cost_n[i] == t_block_g * num_block_m[i])
        e_block_g = var_mul(model=m, vtype=GRB.CONTINUOUS, A=e_mm_once + e_merge_intraCore_once + e_merge_SIMD_once, B=gamma[i]) 

        e_merge_SIMD_global = cost.operation_simd_global(dataSize=var_mul(model=m, vtype=GRB.CONTINUOUS, A=dim_block_m[i], B=dim_block_n[i])*ops.output.bitwidth, 
                                                  num_psums=gamma[i])
        if FLAG.DEBUG_PER_LAYER_DETAIL:
            e_g = m.addVar(vtype=GRB.CONTINUOUS, name=f"e_merge_SIMD_global_{i}")
            m.addConstr(e_g==e_merge_SIMD_global)

        e_single_ab = m.addVar(vtype=GRB.CONTINUOUS, name=f"energy_singe_alpha&beta_{i}")
        m.addConstr(e_single_ab * CONST.SCALINGFACTOR == e_block_g + e_merge_SIMD_global)
        
        e_abc = var_mul(model=m, vtype=GRB.CONTINUOUS, A=e_single_ab, B=num_block_m[i])
        m.addConstr(energy_cost_n[i] == e_abc)

    res_latency_separate = m.addVar(vtype=GRB.CONTINUOUS)
    res_energy_separate_dynamic = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(res_latency_separate==quicksum(time_cost_n[i] * tag_block_n[i] for i in range(MAX_BLOCK_N)))
    m.addConstr(res_energy_separate_dynamic==quicksum(energy_cost_n[i] * tag_block_n[i] for i in range(MAX_BLOCK_N)))

    res_latency = m.addVar(vtype=GRB.CONTINUOUS, name="res_l")
    res_energy = m.addVar(vtype=GRB.CONTINUOUS, name="res_e")
    res_EDP = m.addVar(vtype=GRB.CONTINUOUS, name="res_p")
    m.addConstr(res_latency == res_latency_separate * num_separate_times)              
    m.addConstr(res_energy == res_energy_separate_dynamic * ops.multi + (res_latency * acc.leakage_per_cycle / CONST.SCALINGFACTOR) )
    m.addConstr(res_EDP * CONST.SCALINGFACTOR == res_energy * res_latency)


####################################################################  Set Constraint Flag ###################################################################
    RS_constraint = []
    if FLAG.ROW_STATIONARY:
        RS_constraint.append(m.addConstrs(dim_compartment_para[i]==1 for i in range(MAX_BLOCK_N)))
        # m.addConstrs(alpha[i]>=alpha[i+1] for i in range(MAX_BLOCK_N-1))
        max_alpha = m.addVars(MAX_BLOCK_N, vtype=GRB.BINARY, name="max_alpha")
        for i in range(MAX_BLOCK_N):
            RS_constraint.append(m.addGenConstrIndicator(max_alpha[i], True, alpha[i]==acc.num_core))
            RS_constraint.append(m.addGenConstrIndicator(max_alpha[i], True, dim_macro_para[i]==min(ops.dim_N, acc.macro.column // ops.weight.bitwidth)))
        RS_constraint.append(m.addConstr(quicksum(max_alpha[i] for i in range(MAX_BLOCK_N))>=num_tag_n-1))
        if ops.dim_N % (acc.num_core * min(ops.dim_N, acc.macro.column // ops.weight.bitwidth))==0:
            RS_constraint.append(m.addConstr(quicksum(max_alpha[i] for i in range(MAX_BLOCK_N))==num_tag_n))
            RS_constraint.append(m.addConstr(num_tag_n==(ops.dim_N / (acc.num_core * min(ops.dim_N, acc.macro.column // ops.weight.bitwidth)))))
        else:
            RS_constraint.append(m.addConstr(quicksum(max_alpha[i] for i in range(MAX_BLOCK_N))==num_tag_n-1))
            RS_constraint.append(m.addConstr(num_tag_n==math.ceil(ops.dim_N / (acc.num_core * min(ops.dim_N, acc.macro.column // ops.weight.bitwidth)))))
        RS_constraint.append(m.setObjective(res_EDP, GRB.MINIMIZE))
    elif CONST.FLAG_OPT==1:
        m.setObjective(res_latency, GRB.MINIMIZE)
    elif CONST.FLAG_OPT==2:
        m.setObjective(res_energy, GRB.MINIMIZE)
    elif CONST.FLAG_OPT==3:
        m.setObjective(res_EDP, GRB.MINIMIZE)
    else:
        m.setObjective(0)
    # m.tune()

    m.update()
    # presolve by constraint
    if (CONST.FLAG_OPT != 0) and (FLAG.ROW_STATIONARY == False) and FLAG.PRESOLVE_SEARCH:
        m.setParam("TimeLimit", 5)
        preset_conditions = []
        hc_r = RS_constraint
        preset_conditions.append(get_startVar_byConstr(model=m, constraint_hypothesis=hc_r))

        hc1 = [m.addConstrs((alpha[i]<=1 for i in range(MAX_BLOCK_N))), m.addConstrs(gamma[i]<=1 for i in range(MAX_BLOCK_N))]
        preset_conditions.append(get_startVar_byConstr(model=m, constraint_hypothesis=hc1))

        hc2 = [m.addConstrs((beta[i]<=1 for i in range(MAX_BLOCK_N))), m.addConstrs(gamma[i]<=1 for i in range(MAX_BLOCK_N))]
        preset_conditions.append(get_startVar_byConstr(model=m, constraint_hypothesis=hc2))

        hc5 = [m.addConstrs((alpha[i]<=1 for i in range(MAX_BLOCK_N))), m.addConstrs(beta[i]<=1 for i in range(MAX_BLOCK_N))]
        preset_conditions.append(get_startVar_byConstr(model=m, constraint_hypothesis=hc5))

        if CONST.FLAG_OPT==1:
            set_startVar(model=m, preset_conditions=preset_conditions, res_str='res_l')
        elif CONST.FLAG_OPT==2:
            set_startVar(model=m, preset_conditions=preset_conditions, res_str='res_e')
        elif CONST.FLAG_OPT==3:
            set_startVar(model=m, preset_conditions=preset_conditions, res_str='res_p')
    else:
        pass
    m.setParam("TimeLimit", CONST.TIMELIMIT)

####################################################################  Optimization    #######################################################################

    m.optimize()

####################################################################  Debug & Output  #######################################################################

    if FLAG.DEBUG_PER_LAYER_DETAIL:
        Logger.debug(f"tag_n: {num_tag_n.x}") 
        if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL or m.status == GRB.Status.TIME_LIMIT:
            for i in range(int(num_tag_n.x)):
                Logger.debug(f"   alpha_{i}: {round(alpha[i].x, 2)}, beta: {round(beta[i].x, 2)}, gamma: {round((gamma[i].x), 2) if isinstance(gamma[i], gp.Var) else gamma[i]}, " \
                            + f"num_parallel: {num_parallel.x if ops.multi>1 else num_parallel}, num_core_separate: {num_core_separate.x if ops.multi>1 else num_core_separate}\n" \
                + f"   dim_n: {round(dim_block_n[i].x, 2)}, comp_para: {round(dim_compartment_para[i].x, 2)}, comp_acc: {round(dim_compartment_acc[i].x, 2)} "\
                + f" num_block_m: {round(num_block_m[i].x, 2) if isinstance(num_block_m[i], gp.Var) else num_block_m[i]}")
                Logger.debug(f"      latency: {round(time_cost_n[i].x, 2)}, t_s1: {round(m.getVarByName(f't_s1_{i}').x, 2)}, t_s2: {round(m.getVarByName(f't_s2_{i}').x, 2)}"
                )
                Logger.debug(f"      bandwidth_in_sep: {bandwidth_in_seprate[i].x}, bandwidth_out_sep: {bandwidth_out_seprate[i].x}")
                Logger.debug(f"      t_out_1: {round(m.getVarByName(f't_out_1_{i}').x, 2)}, t_out: {round(m.getVarByName(f't_out_{i}').x, 2)}," \
                            + f" t_in: {round(m.getVarByName(f't_in_{i}').x, 2)}, t_input: {round(m.getVarByName(f't_input_{i}').x,1)}, t_overlap: {round(m.getVarByName(f't_overlap_{i}').x,1)}\n" \
                            + f"      v_computation: {m.getVarByName(f'v_computation_{i}').x}, v_stall: {m.getVarByName(f'v_stalling_{i}').x}" \
                )
                Logger.debug(f"      e_load_input: {round(m.getVarByName(f'e_load_input_{i}').x, 2)}, e_load_weight: {round(m.getVarByName(f'e_load_weight_{i}').x, 2)}")
                Logger.debug(f"      e_compute: {round(m.getVarByName(f'e_compute_{i}').x, 2)}, e_addTree: {round(m.getVarByName(f'e_addTree_{i}').x, 2)},"\
                                +f" e_merge_SIMD_global: {round(m.getVarByName(f'e_merge_SIMD_global_{i}').x, 2)}")
                Logger.debug(f"      output_overSize: {round(m.getVarByName(f'output_overSize_{i}').x, 2)}" \
                            + f", e_merge_intraCore_once: {round(m.getVarByName(f'e_merge_intraCore_once_{i}').x, 2)}"\
                            + f", e_merge_SIMD_once: {round(m.getVarByName(f'e_merge_SIMD_once_{i}').x, 2)}" \
                            )
            return m    
        else:
            # m.computeIIS()
            # m.write("contric.ilp")
            Logger.error(f'model infeasible with Dim: {ops.dim_M} {ops.dim_K} {ops.dim_N}')
            m.dispose()
            return CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS
    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
        if CONST.FLAG_OPT==1:
            Logger.debug(f"get best latency= {res_latency.x}")
        elif CONST.FLAG_OPT==2:
            Logger.debug(f"get best var_energy= {res_energy.x}")
        elif CONST.FLAG_OPT==3:
            Logger.debug(f"get best var_EDP= {res_EDP.x}")
        else:
            Logger.debug("get simple solution")

        Logger.debug(f"tag_n: {num_tag_n.x}") 
        for i in range(int(round(num_tag_n.x))):
            Logger.debug(f"   alpha_{i}: {alpha[i].x}, beta:{beta[i].x},  dim_n: {dim_block_n[i].x}" + \
                            f" compartment_para_{i}: {dim_compartment_para[i].x}, ")
            if FLAG.BLOCK_M: 
                Logger.debug(f"   num_block_m: {num_block_m[i].x}")
        return m
    elif m.status == GRB.Status.TIME_LIMIT:
        Logger.warning("Solver termination by TIME_LIMIT! Looking for gap solution")
        m.setParam("MIPGap", CONST.GAP_THRESHOLD)
        m.setParam("TimeLimit", CONST.TIMELIMIT_AFTER_TLE)
        m.update()
        m.optimize()
        if m.status == GRB.Status.TIME_LIMIT:
            Logger.critical(f"Solver termination by TIME_LIMIT again! With Final MIP gap: {m.MIPGap * 100:.2f}%")
        if m.SolCount > 0: 
            Logger.debug(f"tag_n: {num_tag_n.x}") 
            for i in range(int(num_tag_n.x)):
                Logger.debug(f"   alpha_{i}: {alpha[i].x}, beta:{beta[i].x},  dim_n: {dim_block_n[i].x}" + \
                            f" compartment_para_{i}: {dim_compartment_para[i].x}, ")
                if FLAG.BLOCK_M: 
                    Logger.debug(f"   num_block_m: {num_block_m[i].x}")
            return m
        else:
            Logger.error("No Feasible Solution with Dim: {ops.dim_M} {dim_K} {ops.dim_N} Yet")
    else:
        # m.computeIIS()
        # m.write("contric.ilp")
        Logger.error(f'model infeasible with Dim: {ops.dim_M} {ops.dim_K} {ops.dim_N}')
        m.dispose()
        return CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS



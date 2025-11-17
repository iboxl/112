#  this file is prepared for project 026
#  Created by iboxl

#  Debug dataflow privately ONLY

from utils.Workload import Operands
from Architecture.Accelerator import CIM_acc
import math
from utils.GlobalUT import *

def analy_ds(acc:CIM_acc, ops:Operands, num_tag:int, alpha:[], beta:[], gamma:[], 
            dim_acc:[], dim_para:[], dim_macro_para:[], num_block_m:[], num_parallel, num_separate_times):
    print(f"# # # inside SIMU Analyzer: {ops.dim_M} {ops.dim_K} {ops.dim_N} * {ops.multi}# # #")
    compartment = acc.macro.compartment
    cell = acc.macro.cell
    column = acc.macro.column
    latency = [0 for i in range(num_tag)]
    energy = [0 for i in range(num_tag)]
    load_input = [0 for i in range(num_tag)]
    load_weight = [0 for i in range(num_tag)]
    mvm = [0 for i in range(num_tag)]
    merge = [0 for i in range(num_tag)]
    store = [0 for i in range(num_tag)]
    simd = [0 for i in range(num_tag)]
    for i in range(num_tag):
        alpha[i] = round(alpha[i],0) 
        beta[i] = round(beta[i], 0)
        gamma[i] = round(gamma[i], 0)
        dim_acc[i] = round(dim_acc[i], 0)
        dim_para[i] = round(dim_para[i], 0)
        dim_macro_para[i] = round(dim_macro_para[i], 0)
        num_block_m[i] = round(num_block_m[i], 0)
    num_parallel = round(num_parallel, 0) 
    num_separate_times = round(num_separate_times, 0)

    for i in range(num_tag):
        print(f"alpha:{round(alpha[i],1)}, beta:{round(beta[i],1)}, gamma:{round(gamma[i],1)}, num_block_m:{round(num_block_m[i],1)}"  
                    +f", acc={round(dim_acc[i],1)}, para={round(dim_para[i],1)}, macro_para={round(dim_macro_para[i],1)}"
                    +f", dim_n={alpha[i] * dim_para[i] * dim_macro_para[i]}")
    
    debug_t_load = [0 for i in range(num_tag)]
    debug_t_compute = [0 for i in range(num_tag)]
    debug_t_store = [0 for i in range(num_tag)]
    debug_t_weight = [0 for i in range(num_tag)]
    cost_input = [0 for i in range(num_tag)]
    cost_weight = [0 for i in range(num_tag)]
    cost_mvm = [0 for i in range(num_tag)]
    cost_store = [0 for i in range(num_tag)]
    cost_merge = [0 for i in range(num_tag)]
    cost_simd_cal = [0 for i in range(num_tag)]
    cost_simd_access = [0 for i in range(num_tag)]
    cost_leakage = [0 for i in range(num_tag)]
    
    for i in range(num_tag):
        assert dim_macro_para[i] * ops.weight.bitwidth <= acc.macro.column
        assert alpha[i]*beta[i]*gamma[i] <= acc.num_core / num_parallel
        
        dim_m = math.ceil(ops.dim_M / num_block_m[i])
        dim_k = math.ceil(ops.dim_K / gamma[i])
        dim_n = alpha[i] * dim_para[i] * dim_macro_para[i] 
        # Logger.debug(f"m={round(dim_m,1)}, k={round(dim_k,1)}, n={round(dim_n,1)}")

        bandwidth_in  = acc.bandwidth_g2ib / gamma[i] / num_parallel
        bandwidth_out = acc.bandwidth_ob2g / gamma[i] / num_parallel
        # Logger.debug(f"bandwidth in={round(bandwidth_in,1)}, out={round(bandwidth_out,1)}")
        
        t_load_input = dim_m * dim_k * ops.input.bitwidth / bandwidth_in
        t_load_weight = dim_k * dim_para[i] * alpha[i] * gamma[i] * num_parallel
        # Logger.debug(f"t_weight={t_load_weight}")
        t_load_weight_1st = min(dim_k, cell*compartment) * dim_para[i] * alpha[i] * gamma[i] * num_parallel  
        # Logger.debug(f"t_weight_1st={t_load_weight_1st}")

        t_compute = (dim_m * dim_k * ops.input.bitwidth) / (min(dim_k, compartment)*beta[i]/dim_para[i])

        # t_input = max(t_load_input, t_compute)
        t_load = t_load_input + t_load_weight
        # Logger.debug(f"t_load={t_load}, t_compute={t_compute}")
        t_input = max(t_load_input, t_compute)
        t_in = t_input + t_load_weight - t_load_weight_1st
        out_size = dim_m*dim_n*ops.output.bitwidth
        obuf_size = alpha[i]*beta[i]*acc.core.size_output_buffer
        # Logger.debug(f"out_size={round(out_size,1)}, buf_size={round(obuf_size,1)}, overSize={round(max(0, (out_size - obuf_size)),1)}")                                    # check
        if dim_k <= compartment * cell:
            t_out = out_size / bandwidth_out
        else :
            t_out_global = max(0, (out_size - obuf_size)) * dim_acc[i] / bandwidth_out
            t_out_local = min(out_size, obuf_size) / bandwidth_out
            # Logger.debug(f"t_out local={round(t_out_local,1)}, global={round(t_out_global,1)}")                 # check
            t_out = t_out_local + t_out_global

        t_overlap = max(t_in, t_out)
        debug_t_load[i] = t_load
        debug_t_compute[i] = t_compute
        debug_t_store[i] = t_out
        debug_t_weight[i] = t_load_weight

        latency[i] = (t_load_weight_1st + t_overlap) * gamma[i] * num_block_m[i]
        # Logger.debug(f"t_in={round(t_in,1)}, t_out={round(t_out,1)}")
        
    for i in range(num_tag):
        dim_m = math.ceil(ops.dim_M / num_block_m[i])
        dim_k = math.ceil(ops.dim_K / gamma[i])
        dim_n = alpha[i] * dim_para[i] * dim_macro_para[i] 
        
        e_load_input_r = dim_m * dim_k * ops.input.bitwidth * acc.energy_r_gb
        e_load_input_w = dim_m * dim_k * alpha[i] * ops.input.bitwidth * acc.core.energy_iBuffer_w
        en_load_input = e_load_input_r + e_load_input_w
        # Logger.debug(f"e_load_input: {en_load_input}")                         # check

        e_load_weight_r = dim_k * dim_n * ops.weight.bitwidth * acc.energy_r_gb
        e_load_weight_w = dim_k * dim_para[i] * alpha[i] * acc.macro.energy_w_per_row
        en_load_weight = e_load_weight_r + e_load_weight_w * beta[i]
        # Logger.debug(f"e_load_weight: {e_load_weight}")                         # check

        num_acc = dim_acc[i]*dim_para[i]*alpha[i]
        e_compute_input = dim_m*dim_k*alpha[i]*ops.input.bitwidth*acc.core.energy_iBuffer_r
        e_compute_periph = num_acc*acc.macro.energy_r_periph_per_acc
        e_compute_mm = num_acc * acc.macro.energy_compute_MM_per_acc
        e_compute_idle = (dim_acc[i]*compartment-dim_k) * acc.macro.energy_compute_MM_per_row
        e_addTree = num_acc * acc.core.energy_addTree_per_acc
        
        en_compute = e_compute_input + e_compute_periph + e_compute_mm + e_addTree - e_compute_idle
        # Logger.debug(f"e_compute: {en_compute}")                         # check
        # Logger.debug(f"e_adden_addTree: {en_addTree}")                         # check

        out_size = dim_m*dim_n*ops.output.bitwidth
        obuf_size = alpha[i]*beta[i]*acc.core.size_output_buffer
        en_merge_intra = min(out_size, obuf_size) * (dim_acc[i]-1) * \
                            (acc.core.energy_oBuffer_r + acc.core.energy_merge + acc.core.energy_oBuffer_w) + \
                            min(out_size, obuf_size) * acc.energy_w_gb
        # Logger.debug(f"e_emerge_intra_once: {e_merge_intra}")                         # check
        en_merge_simd = max(0, (out_size - obuf_size)) * dim_acc[i] * acc.energy_w_gb + \
                       max(0, (out_size - obuf_size)) * (dim_acc[i]-1) * \
                       ((acc.energy_r_gb*2) + acc.simd.energy_per_bit_byOperation + acc.energy_w_gb)
        en_simd_global = dim_m*dim_n*ops.output.bitwidth * (gamma[i]-1) * \
                    ((acc.energy_r_gb*2) + acc.simd.energy_per_bit_byOperation + acc.energy_w_gb)
        if dim_k <= compartment * cell:
            indi_k = 0
            en_merge_simd = 0
        else:
            indi_k = 1 
        # Logger.debug(f"e_emerge_simd_once: {e_merge_simd}")                         # check

        en_all_block = (en_load_input + en_compute + en_load_weight+en_merge_intra+en_merge_simd) * gamma[i] + en_simd_global
        
        # Logger.debug(f"e_simd_global: {en_simd_global}")                         # check

        energy_dyna = en_all_block*num_block_m[i]
        energy[i] = energy_dyna

        cost_input[i] = en_load_input * gamma[i] * num_block_m[i]
        cost_weight[i] = en_load_weight * gamma[i] * num_block_m[i]
        cost_store[i] = (min(out_size, obuf_size) + max(0, (out_size - obuf_size)) * dim_acc[i] * indi_k) * acc.energy_w_gb * gamma[i] * num_block_m[i]
        cost_mvm[i] = en_compute * gamma[i] * num_block_m[i]
        cost_merge[i] = min(out_size, obuf_size) * (dim_acc[i]-1) * \
                            (acc.core.energy_oBuffer_r + acc.core.energy_merge + acc.core.energy_oBuffer_w) * gamma[i] * num_block_m[i]
        size_simd_process = (max(0, (out_size - obuf_size)) * (dim_acc[i]-1) * gamma[i] * indi_k + dim_m*dim_n*ops.output.bitwidth * (gamma[i]-1)) * num_block_m[i]
        cost_simd_cal[i] = size_simd_process * acc.simd.energy_per_bit_byOperation 
        cost_simd_access[i] = size_simd_process * ((acc.energy_r_gb*2) + acc.energy_w_gb)
        if dim_k <= compartment * cell:
            en_merge_simd = 0
        cost_leakage[i] = latency[i] * acc.leakage_per_cycle
        
        
    res_l = sum(latency) * num_separate_times
    res_e = sum(energy) * ops.multi + res_l * acc.leakage_per_cycle

    
    utilize_bandwidth = 0               # 冗余的数据传输造成的偏差？ 如何仅统计有效传输
    utilize_macro = 0
    ecost = {
    "load_input" : sum(cost_input) / CONST.SCALINGFACTOR,
    "load_weight" : sum(cost_weight) / CONST.SCALINGFACTOR,
    "mvm ": sum(cost_mvm) / CONST.SCALINGFACTOR,
    "merge" : sum(cost_merge) / CONST.SCALINGFACTOR,
    "store" : sum(cost_store) / CONST.SCALINGFACTOR,
    "simd_cal" : sum(cost_simd_cal) / CONST.SCALINGFACTOR,
    "simd_access" : sum(cost_simd_access) / CONST.SCALINGFACTOR,
    "leakage" : sum(cost_leakage) / CONST.SCALINGFACTOR,
    }
    if FLAG.DEBUG_SIMU:
        for i in range(num_tag):
            print(f"latency={latency[i]}: load={debug_t_load[i]}, compute={debug_t_compute[i]}, store={debug_t_store[i]}")

    # Logger.info(f"calc latency: {res_l}, energy: {res_e}")
    print("# "*10)
    return res_l, res_e / CONST.SCALINGFACTOR, ecost
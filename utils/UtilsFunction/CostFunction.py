# this file is prepared for project 026
# Created by iboxl

from Architecture.Accelerator import CIM_acc
from utils.Workload import Operands
from gurobipy import GRB
import gurobipy as gp
from utils.UtilsFunction.SolverFunction import *

class _Cost_model():
    
    gamma = 1           # W.T.D.

    def __init__(self, acc:CIM_acc, model:gp.Model, ops:Operands):
        self.acc = acc
        self.m = model
        self.ops = ops
    
    # dataSize Unit should be [bit]
    def simd(self, dataSize):
        cost_r = dataSize * self.acc.energy_r_gb
        cost_c = dataSize // self.acc.simd.vector_width * self.acc.simd.energy_per_operation
        cost_w = dataSize * self.acc.energy_w_gb
        return cost_r + cost_c + cost_w
    
    def load_input(self, dataShape, alpha=1):                    # dataSize: input_data[m,k]     # multicast communication
        dataSize = dataShape * self.ops.input.bitwidth
        cost_r = dataSize * self.acc.energy_r_gb 
        cost_w = dataSize * self.acc.core.energy_iBuffer_w
        return cost_r + cost_w * alpha
    
    def load_weight(self, dataShape, alpha, beta, para, dim_k):             # dim_k = dim_K * compartment_para     # 和 num_block_m 相关
        dataSize = dataShape * self.ops.weight.bitwidth
        cost_r = dataSize * self.acc.energy_r_gb
        cost_w = var_mulABC(model=self.m, vtype=GRB.INTEGER, A=dim_k, B=para, C=alpha) * self.acc.macro.energy_w_per_row
        return cost_r + var_mul(model=self.m, vtype=GRB.CONTINUOUS, A=cost_w, B=beta)
    
    def mac(self, alpha, acc, para, dim_m, dim_k):
        num_acc = var_mulABC(model=self.m, vtype=GRB.INTEGER, A=acc, B=para, C=alpha)

        cost_input_read = var_mul(model=self.m, vtype=GRB.INTEGER, A=dim_m, B=dim_k)* alpha * self.ops.input.bitwidth * self.acc.core.energy_iBuffer_r 
        cost_input_perp = num_acc * self.acc.macro.energy_r_periph_per_acc

        # cost_get_weight_per_acc = self.acc.macro.compartment * self.acc.macro.energy_r_per_row 
        # cost_get_weight = num_acc * cost_get_weight_per_acc 

        cost_computation_mm = num_acc * self.acc.macro.energy_compute_MM_per_acc

        idle_row = self.m.addVar(lb=0, ub=self.acc.macro.compartment, vtype=GRB.INTEGER)
        self.m.addConstr(idle_row+dim_k==acc*self.acc.macro.compartment)

        # 按行可能闲置 不考虑"列"闲置能耗 不考虑更复杂的情况
        cost_idel_row = idle_row * self.acc.macro.energy_compute_MM_per_row * self.acc.macro.Idle_coefficient

        return cost_input_read + cost_input_perp + cost_computation_mm - cost_idel_row

    # def mac_input_staionary(self, r, c):
    #     # 不必要-缺失
    #     return 0
    
    def addTree(self, acc, para, alpha):                                    # dataSize: output_data[M, N] * output_bitwidth
        num_acc = var_mulABC(model=self.m, vtype=GRB.INTEGER, A=acc, B=para, C=alpha)
        cost = num_acc * self.acc.core.energy_addTree_per_acc
        return cost
    
    def mergePsum_intra(self, dataSize, num_psums):
        # 每个output需要产生num_psums个部分和 —— num_psums-1次写入读出 num_psums-1次累加
        dataSize_all = dataSize * (num_psums-1)

        cost_merge = dataSize_all * (self.acc.core.energy_oBuffer_r + self.acc.core.energy_merge + self.acc.core.energy_oBuffer_w)
        cost_store = dataSize * self.acc.energy_w_gb

        energy_merge = self.m.addVar(vtype=GRB.CONTINUOUS)
        self.m.addConstr(energy_merge == cost_merge + cost_store)
        return energy_merge  
    
    def mergePsum_simd(self, dataSize, num_psums):
        # 每个output需要产生num_psums个部分和 —— num_psums次GB传输 num_psums-1 次累加
        dataSize_all = dataSize * num_psums

        cost_send = dataSize_all * self.acc.energy_w_gb

        cost_merge_each = (self.acc.energy_r_gb*2) + self.acc.simd.energy_per_bit_byOperation + self.acc.energy_w_gb
        merge_times_all = dataSize * (num_psums-1)

        energy_merge = self.m.addVar(vtype=GRB.CONTINUOUS)
        self.m.addConstr(energy_merge == cost_send + merge_times_all * cost_merge_each)
        return energy_merge
    
    # store包含在merge里了

    def operation_simd_global(self, dataSize, num_psums):
        # 每个数需要执行num_psums-1次操作
        cost_merge_each = (self.acc.energy_r_gb*2) + self.acc.simd.energy_per_bit_byOperation + self.acc.energy_w_gb
        merge_times_all = dataSize * (num_psums-1)

        energy_merge = self.m.addVar(vtype=GRB.CONTINUOUS)
        self.m.addConstr(energy_merge == merge_times_all * cost_merge_each)
        return energy_merge
    
    #   W.T.D    Post process energy

    def store_output(self, dataSize):
        cost_r = dataSize * self.acc.core.energy_oBuffer_r 
        cost_w = dataSize * self.acc.energy_w_gb 
        return cost_r + cost_w
    
    def store_psum(self, dataSize):
        cost = dataSize * self.acc.energy_w_gb
        return cost
   

# this file is prepared for project 511
# Created by iboxl


from zigzag.classes.hardware.architecture.core import Core as core_zz
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
import math
from utils.cacti.EvalCacti import cacti_power, dram_static

def convert_mapping_to_next(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    # 初始化结果矩阵，大小与输入矩阵相同，初值为-1
    nxtMem = [[-1 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        last_valid_layer = -1
        for col in range(cols):
            # 只处理有效映射位置
            if matrix[row][col] == 1:
                # 找到下一个有效层
                next_layer = -1
                for next_col in range(col + 1, cols):
                    if matrix[row][next_col] == 1:
                        next_layer = next_col
                        break
                # 如果没有下一个有效层，则设置为层数总数
                if next_layer == -1:
                    next_layer = cols
                nxtMem[row][col] = next_layer
                last_valid_layer = col
    return nxtMem

def find_lastMem_index(mapArray, idx):
    count = 0
    for i in range(len(mapArray) - 1, -1, -1):
        if mapArray[i] == 1:
            count += 1
            if count == idx:
                return i
    return -1

def find_FirstMem_index(mapArray, idx):
    count = 0
    for i in range(1,len(mapArray)):
        if mapArray[i] == 1:
            count += 1
            if count == idx:
                return i
    return -1

class CIM_Acc():                         # WTD. init CIM_Acc from zigzag | Yaml | ...
    def __init__(self, accSpec:core_zz):
        # self.leakage_per_cycle = 28.824
        print("Create CIM Acc")
        self._mem2dict = ['-P-']
        self.memSize = [-1] 
        self.bw     = [-1]
        self.cost_r = [-1]
        self.cost_w = [-1]
        self.mappingArray = [[-1],[-1],[-1]]      # mappingArray[op][mem]    # memory allow which operand mapping 


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # WTD. init CIM_Acc from zigzag | Yaml | ...           Below is convert func
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        for mem in accSpec.memory_level_list_in_order[::-1]:
            mem:MemoryLevel
            inst:MemoryInstance = mem.memory_instance
            self._mem2dict.append(inst.name)
            self.bw.append(inst.r_bw)
            self.cost_r.append(inst.r_cost / inst.r_bw)
            self.cost_w.append(inst.w_cost / inst.w_bw)
            for t, t_name in enumerate(['I1','I2','O']):
                if t_name in mem.operands:
                    self.mappingArray[t].append(1)
                else:
                    self.mappingArray[t].append(0)
            self.memSize.append(inst.size)
        
        self.Num_mem = len(self._mem2dict)      # Num_mem = mem2dict + 1 [0,1,2,....], index from 1 to avoid using 0 rep either (None or Mem_1)
        
        self.cost_r.append(0)
        self.cost_w.append(0)

        self.IReg2mem  = find_lastMem_index(self.mappingArray[0], 1)
        self.Macro2mem = find_lastMem_index(self.mappingArray[1], 1)
        self.OReg2mem  = find_lastMem_index(self.mappingArray[2], 1)
        # self.Global2mem= find_FirstMem_index(self.mappingArray[0], 2) 
        self.Global2mem = self._mem2dict.index("Global_buffer")

        self.lastMem = {}
        self.lastMem[0] = self.IReg2mem
        self.lastMem[1] = self.Macro2mem
        self.lastMem[2] = self.OReg2mem
        
        self.double_Macro = 0 
        self.double_config = [[0 for op in range(3)] for m in range(self.Num_mem)]
        for m in range(1, self.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if self.mappingArray[op][m] == 1:
                    self.double_config[m][op] = 1
        self.double_config[self.Macro2mem][1] = self.double_Macro
        self.double_config[self.IReg2mem][0] = 0
        self.double_config[self.OReg2mem][0] = 0

        self.shareMemory = [False for m in range(self.Num_mem)]
        for m in range(1, self.Num_mem):
            opShared = 0
            for op, op_name in enumerate(['I','W','O']):
                opShared += self.mappingArray[op][m]
            if opShared > 1:
                self.shareMemory[m] = True
            elif opShared == 1:
                self.shareMemory[m] = True
            else:
                raise ValueError(f"Memory {self._mem2dict[m]} have no operands.")

            
        self.nxtMem = convert_mapping_to_next(self.mappingArray)

        MacroSpec = accSpec.operational_array.unit
        
        precision_op = {
            'I': MacroSpec.hd_param["input_precision"],
            'W': MacroSpec.hd_param["weight_precision"],
            'O': MacroSpec.hd_param["input_precision"],
        }       # Feature Precision
        self.precision = {}
        for mem in range(1,self.Num_mem):
            for t, t_name in enumerate(['I','W','O']):
                self.precision[mem,t] = precision_op[t_name]

        self.precision[self.OReg2mem, 2] = MacroSpec.hd_param["input_precision"] + MacroSpec.hd_param["weight_precision"] 
        # self.precision[find_lastMem_index(self.mappingArray[2], 2) , 2] = MacroSpec.hd_param["input_precision"] + MacroSpec.hd_param["weight_precision"] 
               
        

        self.Num_core = MacroSpec.nb_of_banks
        
        assert(MacroSpec.hd_param["input_bit_per_cycle"] == 1)

        
        self.dimX = MacroSpec.bl_dim_size                               # IN-parallel
        self.dimY = MacroSpec.wl_dim_size                               # OUT-parallel
        # self.fanout = ['PLACEHOLD', 1, self.Num_core, 1, 1, self.dimX*self.dimY, self.dimX*self.dimY, self.dimX*self.dimY]    
        self.fanout = [[1 for op in range(3)] for i in range(self.Num_mem)]
        #  ['Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7]
        self.fanout[self.Global2mem][0] = self.Num_core
        self.fanout[self.Global2mem][1] = self.Num_core*self.dimX * self.dimY
        self.fanout[self.Global2mem][2] = self.Num_core
        self.fanout[3][2] = self.dimY
        self.fanout[4][0] = self.dimX
        self.fanout[self.IReg2mem][2] = self.dimY
        self.fanout[self.OReg2mem][1] = self.dimX

        self.SpUnrolling = [self.Num_core, self.dimX, self.dimY]
        self.Num_SpUr = len(self.SpUnrolling)

        self.Num_compartment = MacroSpec.hd_param["group_depth"]
        
        # self.unrollingArray = {}
        # for op in range(3):
        #     for m in range(1,self.Num_mem):
        #         for u in range(self.Num_SpUr):
        #             self.unrollingArray[m,op,u] = 0
        #     self.unrollingArray[self.Global2mem,op,0] = 1
        # self.unrollingArray[self.IReg2mem,0,1] = 1

        self.SpUrArray = {}     # SpUrArray[u,op] = m
        self.SpUrArray[0,0] = self.Global2mem
        self.SpUrArray[0,1] = self.Global2mem
        self.SpUrArray[0,2] = self.Global2mem

        self.SpUrArray[1,0] = 4
        self.SpUrArray[1,1] = self.Global2mem
        self.SpUrArray[1,2] = self.OReg2mem

        self.SpUrArray[2,0] = self.IReg2mem
        self.SpUrArray[2,1] = self.Global2mem
        self.SpUrArray[2,2] = 3


        # self.bw[self.IReg2mem] *= self.dimX
        # self.memSize[self.IReg2mem] *= self.dimX 
        # self.bw[self.OReg2mem] *= self.dimY 
        # self.memSize[self.OReg2mem] *= self.dimY 
        # self.bw[self.Macro2mem] *= (self.dimX * self.dimY)
        # self.memSize[self.Macro2mem] *= ( self.dimX * self.dimY  )  # * MacroSpec.hd_param["group_depth"]  ArchZZ already calc

        self.t_MAC = math.ceil(precision_op['I'] / MacroSpec.hd_param["input_bit_per_cycle"])

        # self.bw[self.IReg2mem] = int(self.bw[self.IReg2mem] / self.t_MAC)
        # self.bw[self.OReg2mem] = int(self.bw[self.OReg2mem] / self.t_MAC)
        # self.bw[self.Macro2mem] = int(self.bw[self.Macro2mem] / self.t_MAC)

        # self.Num_instance = [1 for _ in range(self.Num_mem)]    # self.Num_instance[1] = 1
        # self.Num_instance.append(0)
        # for mem in range(1,self.Num_mem):
        #     for t in range(3):
        #         self.Num_instance[self.nxtMem[t][mem]] = self.fanout[mem]
        # print(self.fanout)
        # print(self.Num_instance)

        # != energy_precharging
        precision_Psum = precision_op['I']+ precision_op['W'] + math.ceil(math.log2(self.dimX))
        num_bit_MAC_in_Macro = MacroSpec.hd_param["weight_precision"] * MacroSpec.hd_param["input_bit_per_cycle"] * self.dimX * self.dimY
        num_bit_ADD_in_addTree = self.dimX * (precision_op['I']+1) - (precision_op['I'] + math.ceil(math.log2(precision_op['I'])) + 1) * self.dimY

        CIMArray_energy = MacroSpec.logic_unit.get_1b_multiplier_energy() * num_bit_MAC_in_Macro * self.t_MAC
        Adder_tree_energy = MacroSpec.logic_unit.get_1b_multiplier_energy() * num_bit_ADD_in_addTree * self.t_MAC
        Merge_energy = (MacroSpec.logic_unit.get_1b_adder_energy() + MacroSpec.logic_unit.get_1b_reg_energy()) * precision_Psum * self.t_MAC * self.dimY
        
        # fj=1e-15      # pj=1e-12     # nj=1e-9     # µj=1e-6
        # Energy Unit = pj
        self.cost_ActMacro = CIMArray_energy + Adder_tree_energy + Merge_energy

        # ['PLACEHOLD'0, 'Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7] acc.Num_mem = 8
        self.leakage_per_cycle = 0
        self.leakage_per_cycle += dram_static(capacity_bytes=(self.memSize[1]/8), bus_width_bits=self.bw[1])[1]
        self.leakage_per_cycle += cacti_power(capacity_bytes=(self.memSize[2]/8), bitwidth_bits=self.bw[2])[2]
        self.leakage_per_cycle += cacti_power(capacity_bytes=(self.memSize[3]/8), bitwidth_bits=self.bw[3])[2] * self.Num_core
        self.leakage_per_cycle += cacti_power(capacity_bytes=(self.memSize[4]/8), bitwidth_bits=self.bw[4])[2] * self.Num_core
        self.leakage_per_cycle += cacti_power(capacity_bytes=(self.memSize[7]/8) * self.dimX * self.dimY, 
                                              bitwidth_bits=self.dimY*MacroSpec.hd_param["weight_precision"])[2] * self.Num_core

    def mem2dict(self,x):
        if x>=0 and x<self.Num_mem:
            return self._mem2dict[int(x)]
        elif x == self.Num_mem:
            return 'opMAC'
        else:
            return '---'
       
# hd_param = {
#         "pe_type":              "in_sram_computing",     # for in-memory-computing. Digital core for different values.
#         "imc_type":             "digital",  # "digital" or "analog"
#         "input_precision":      8,          # activation precision expected in the hardware
#         "weight_precision":     8,          # weight precision expected in the hardware
#         "input_bit_per_cycle":  1,          # nb_bits of input/cycle/PE
#         "group_depth":          4,          # group depth in each PE
#         "wordline_dimension": "D1",         # hardware dimension where wordline is (corresponds to the served dimension of output #-input-# regs)
#         "bitline_dimension": "D2",          # hardware dimension where bitline is (corresponds to the served dimension of intput #-output-# regs)
#         "enable_cacti":         True,       # use CACTI to estimated cell array area cost (cell array exclude build-in logic part)
#         # Energy of writing weight. Required when enable_cacti is False.
#         # "w_cost_per_weight_writing": 0.08,  # [OPTIONAL] unit: pJ/weight.
#     }

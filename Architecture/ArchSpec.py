# this file is prepared for project 511
# Created by iboxl

from utils.ZigzagUtils import ensure_zigzag_submodule_on_path

ensure_zigzag_submodule_on_path()
from zigzag.classes.hardware.architecture.core import Core as core_zz
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.memory_level import MemoryLevel
from zigzag.classes.hardware.architecture.operational_array import OperationalArray
import math
from utils.Cacti_wrapper.EvalCacti import cacti_power, dram_static
from utils.GlobalUT import Logger as Logger

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
        Logger.critical("Create CIM Acc")
        self.source_spec = None  # legacy path 无 Spec 追溯；from_spec 会覆盖
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
        self.Dram2mem   = 1
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
                self.shareMemory[m] = False
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
        for mem in range(1,self.Num_mem+1):
            for t, t_name in enumerate(['I','W','O']):
                self.precision[mem,t] = precision_op[t_name]

        self.precision[self.OReg2mem, 2] = MacroSpec.hd_param["input_precision"] + MacroSpec.hd_param["weight_precision"]
        # self.precision[find_lastMem_index(self.mappingArray[2], 2) , 2] = MacroSpec.hd_param["input_precision"] + MacroSpec.hd_param["weight_precision"]
        self.precision_final = precision_op['O']
        self.precision_psum = MacroSpec.hd_param["input_precision"] + MacroSpec.hd_param["weight_precision"]
               
        

        self.Num_core = MacroSpec.nb_of_banks
        
        assert(MacroSpec.hd_param["input_bit_per_cycle"] == 1)

        
        self.dimX = MacroSpec.bl_dim_size                               # IN-parallel
        self.dimY = MacroSpec.wl_dim_size                               # OUT-parallel

        self.SpUnrolling = [self.Num_core, self.dimX, self.dimY]
        self.Num_SpUr = len(self.SpUnrolling)

        '''
        ------------- mappingRule ------------------ WTD, Dim-Group always 0 / mapping configuration
        'spatial_mapping_hint': {'D1': ['K'], 'D3': ['K', 'OX', 'OY'], 'D2': ['B', 'K', 'G', 'OX', 'OY', 'C', 'FX', 'FY']}, 
        '''
        self.mappingRule = [
            # self.dim2Dict = ['-','R', 'S', 'P', 'Q', 'C', 'K', 'G']
            [0, 0, 0, 1, 1, 0, 1, 1],  # 轴0允许展开P, Q, K, G
            [0, 1, 1, 0, 0, 1, 0, 0],  # 轴1允许展开R, S, C
            [0, 0, 0, 0, 0, 0, 1, 0]   # 轴2允许展开K
        ]

        '''
        Not use self.Num_compartment, memSize has considered compartment in zigzag (self.memSize.append(inst.size)). Dimension can be mapped into macro in temporal loop 
        '''
        # self.Num_compartment = MacroSpec.hd_param["group_depth"]
        
        self.SpUr2Mem = {}                          # SpUr2Mem[u,op] = m
                                                    # SpUr2Mem 表示的是Multicast/Unicast的源地址-存储层次，在这个层次的数据量是Unrolling展开前的
        self.SpUr2Mem[0,0] = self.Global2mem
        self.SpUr2Mem[0,1] = self.Global2mem
        self.SpUr2Mem[0,2] = self.Global2mem

        self.SpUr2Mem[1,0] = 4
        self.SpUr2Mem[1,1] = self.Global2mem
        self.SpUr2Mem[1,2] = self.OReg2mem

        self.SpUr2Mem[2,0] = self.IReg2mem
        self.SpUr2Mem[2,1] = self.Global2mem
        self.SpUr2Mem[2,2] = 3


        self.minBW = [1e9, 1e9, 1e9]
        for m in range(1,self.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if self.mappingArray[op][m]:
                    self.minBW[op] = min(self.minBW[op], self.bw[m])

        self.t_MAC = math.ceil(precision_op['I'] / MacroSpec.hd_param["input_bit_per_cycle"])

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

        """
        Energy Unit: pJ ---> nJ
        """
        def pj_to_nj(value):
            if isinstance(value, list):
                return [x * 1e-3 for x in value]
            return value * 1e-3
        
        self.cost_ActMacro, self.leakage_per_cycle, self.cost_r, self.cost_w = map(
            pj_to_nj, [self.cost_ActMacro, self.leakage_per_cycle, self.cost_r, self.cost_w]
        )
        
    def mem2dict(self,x):
        if x>=0 and x<self.Num_mem:
            return self._mem2dict[int(x)]
        elif x == self.Num_mem:
            return 'opMAC'
        else:
            return '---'

    @classmethod
    def from_spec(cls, spec) -> "CIM_Acc":
        """从 HardwareSpec 构造 CIM_Acc，字段布局与 __init__ 严格一致。

        单位约定：cost_ActMacro / cost_r / cost_w 最终为 nJ 单位（遵循 pj_to_nj），
        leakage_per_cycle 取 spec.leakage_per_cycle_nJ（已是 nJ，不再二次转换）。
        """
        Logger.critical("Create CIM Acc (from_spec)")
        self = cls.__new__(cls)
        self._mem2dict = ['-P-']
        self.memSize = [-1]
        self.bw = [-1]
        self.cost_r = [-1]
        self.cost_w = [-1]
        self.mappingArray = [[-1], [-1], [-1]]

        I = spec.macro.precision.I
        W = spec.macro.precision.W
        psum = spec.macro.precision.psum
        O_final = spec.macro.precision.O_final

        op_name_to_idx = {'I': 0, 'W': 1, 'O': 2}

        for level in spec.memory_hierarchy:
            self._mem2dict.append(level.name)
            self.bw.append(level.r_bw_bits_per_cycle)
            self.cost_r.append(level.r_cost_per_bit_pJ)
            self.cost_w.append(level.w_cost_per_bit_pJ)
            self.memSize.append(level.size_bits)
            for t_name in ['I', 'W', 'O']:
                t = op_name_to_idx[t_name]
                self.mappingArray[t].append(1 if t_name in level.operands else 0)

        self.Num_mem = len(self._mem2dict)
        self.cost_r.append(0)
        self.cost_w.append(0)

        self.IReg2mem  = find_lastMem_index(self.mappingArray[0], 1)
        self.Macro2mem = find_lastMem_index(self.mappingArray[1], 1)
        self.OReg2mem  = find_lastMem_index(self.mappingArray[2], 1)
        self.Dram2mem  = 1
        self.Global2mem = self._mem2dict.index("Global_buffer")

        self.lastMem = {0: self.IReg2mem, 1: self.Macro2mem, 2: self.OReg2mem}

        self.double_Macro = 0
        self.double_config = [[0 for _ in range(3)] for _ in range(self.Num_mem)]
        for m in range(1, self.Num_mem):
            for op in range(3):
                if self.mappingArray[op][m] == 1:
                    self.double_config[m][op] = 1
        self.double_config[self.Macro2mem][1] = self.double_Macro
        self.double_config[self.IReg2mem][0] = 0
        self.double_config[self.OReg2mem][0] = 0

        self.shareMemory = [False for _ in range(self.Num_mem)]
        for m in range(1, self.Num_mem):
            opShared = sum(self.mappingArray[op][m] for op in range(3))
            if opShared > 1:
                self.shareMemory[m] = True
            elif opShared == 1:
                self.shareMemory[m] = False
            else:
                raise ValueError(f"Memory {self._mem2dict[m]} have no operands.")

        self.nxtMem = convert_mapping_to_next(self.mappingArray)

        self.precision = {}
        for mem in range(1, self.Num_mem + 1):
            self.precision[mem, 0] = I
            self.precision[mem, 1] = W
            self.precision[mem, 2] = O_final
        self.precision[self.OReg2mem, 2] = psum
        self.precision_final = O_final
        self.precision_psum = psum

        self.Num_core = spec.cores
        assert spec.macro.input_bit_per_cycle == 1

        self.dimX = spec.macro.dimX
        self.dimY = spec.macro.dimY

        self.SpUnrolling = [self.Num_core, self.dimX, self.dimY]
        self.Num_SpUr = len(self.SpUnrolling)

        dim_names = ['_', 'R', 'S', 'P', 'Q', 'C', 'K', 'G']
        dim_idx_map = {n: i for i, n in enumerate(dim_names)}
        self.mappingRule = []
        for axis in spec.macro.spatial_axes:
            row = [0] * len(dim_names)
            for loop in axis.allowed_loops:
                row[dim_idx_map[loop]] = 1
            self.mappingRule.append(row)

        self.SpUr2Mem = {}
        mem_name_to_idx = {name: i for i, name in enumerate(self._mem2dict)}
        for ai, axis in enumerate(spec.macro.spatial_axes):
            for op_name, mem_name in axis.source_memory_per_operand.items():
                self.SpUr2Mem[ai, op_name_to_idx[op_name]] = mem_name_to_idx[mem_name]

        self.minBW = [1e9, 1e9, 1e9]
        for m in range(1, self.Num_mem):
            for op in range(3):
                if self.mappingArray[op][m]:
                    self.minBW[op] = min(self.minBW[op], self.bw[m])

        self.t_MAC = math.ceil(I / spec.macro.input_bit_per_cycle)

        mult_1b = spec.macro.logic_energies_pJ.mult_1b
        adder_1b = spec.macro.logic_energies_pJ.adder_1b
        reg_1b = spec.macro.logic_energies_pJ.reg_1b

        precision_Psum = I + W + math.ceil(math.log2(self.dimX))
        num_bit_MAC_in_Macro = W * spec.macro.input_bit_per_cycle * self.dimX * self.dimY
        num_bit_ADD_in_addTree = self.dimX * (I + 1) - (I + math.ceil(math.log2(I)) + 1) * self.dimY

        CIMArray_energy = mult_1b * num_bit_MAC_in_Macro * self.t_MAC
        Adder_tree_energy = mult_1b * num_bit_ADD_in_addTree * self.t_MAC
        Merge_energy = (adder_1b + reg_1b) * precision_Psum * self.t_MAC * self.dimY

        cost_ActMacro_pJ = CIMArray_energy + Adder_tree_energy + Merge_energy

        def pj_to_nj(value):
            if isinstance(value, list):
                return [x * 1e-3 for x in value]
            return value * 1e-3

        self.cost_ActMacro = pj_to_nj(cost_ActMacro_pJ)
        self.cost_r = pj_to_nj(self.cost_r)
        self.cost_w = pj_to_nj(self.cost_w)
        self.leakage_per_cycle = spec.leakage_per_cycle_nJ

        # 追溯 spec 来源，供 downstream（ZigZag baseline adapter、cache key）使用
        self.source_spec = spec

        return self
       
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

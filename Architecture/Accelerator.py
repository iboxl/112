# this file is prepared for project 026
# Created by iboxl

from Config.PowerConfig import PowerClass_STD as PowerClass
# # # # # Variable Representations # # # # # 

### hardware architecture

class CIM_macro():
    def __init__(self, config, powerclass:PowerClass):
        self.column = config.getint("Macro", "column") 
        self.compartment = config.getint("Macro", "compartment") 
        self.cell = config.getint("Macro", "cell") 

        self.size = self.column * self.cell * self.compartment
        
        self.energy_w_per_row = powerclass.energy_w_macro_per_row

        self.energy_compute_MM_per_acc = powerclass.energy_computation_per_acc     
        self.energy_r_periph_per_acc = powerclass.energy_peripherals_per_acc        

        self.energy_compute_MM_per_row = self.energy_compute_MM_per_acc / self.compartment

        # self.energy_shiftAdd = config.getint("Workload", "weight_bit_width") * config.getfloat("Macro", "energy_shiftAdd_per_bit")           # per operation 需要input_bw次 shiftAdd

        self.Idle_coefficient = config.getfloat("Macro", "Idle_coefficient")

        # self.leakeay

class CIM_core():
    def __init__(self, config, powerclass:PowerClass):
        self.size_input_buffer = config.getint("Core", "size_input_buffer") * 8 * 1024 
        self.size_output_buffer = config.getint("Core", "size_output_buffer") * 8 * 1024 

        self.energy_iBuffer_r = powerclass.energy_r_lib_per_bit
        self.energy_iBuffer_w = powerclass.energy_w_lib_per_bit

        self.energy_oBuffer_r = powerclass.energy_r_lob_per_bit 
        self.energy_oBuffer_w = powerclass.energy_w_lob_per_bit

        # 整个macro的X个compartment一次完整累加的开销
        self.energy_addTree_per_acc = powerclass.energy_addTree_per_acc

        self.energy_merge = powerclass.energy_merge_per_bit   # just add     # per bit

class SIMD_core():
    def __init__(self, config, powerclass:PowerClass):
        self.vector_width = config.getint("SIMD", "VectorWidth") 
        self.cyc_computation = config.getint("SIMD", "cycle_computation") 
        self.energy_per_bit_byOperation = powerclass.energy_simd_op_per_bit              #  W.T.D. :  different cost for different operation

class CIM_acc():
    def __init__(self, config):
        self.powerclass = PowerClass(config)
        self.macro = CIM_macro(config, self.powerclass)                                  # Heterogeneous macro W.T.D.
        self.core = CIM_core(config, self.powerclass)
        self.simd = SIMD_core(config, self.powerclass)
        
        self.num_core = config.getint("Accelerator", "num_core")  

        self.size_global_iBuffer = config.getint("Accelerator", "size_global_iBuffer") * 8 * 1024 
        self.size_global_oBuffer = config.getint("Accelerator", "size_global_oBuffer") * 8 * 1024 

        # unit = bits/cycle
        self.bandwidth_g2ib = config.getint("Accelerator", "bandwidth_global_to_iBuffer") 
        self.bandwidth_ob2g = config.getint("Accelerator", "bandwidth_oBuffer_to_global")  

        self.energy_r_gb = self.powerclass.energy_r_gb_per_bit          # per bit 
        self.energy_w_gb = self.powerclass.energy_w_gb_per_bit          # per bit

        self.leakage_per_cycle = self.powerclass.leakage
        
        """         W.T.D.        """
        # [SIMD core]
        # self.enengy_SIMD_acc = config.get("SIMD", "energy_SIMD_acc")
        # self.speed_SIMD

        self._mem2dict = [
            'Off-chip_Memory',  # 0
            'Global_buffer',    # 1
            'Input_buffer',     # 2
            'Output_buffer',    # 3
            'IReg',             # 4
            'OReg',             # 5
            'Macro'             # 6
        ]            

        self.Num_mem = len(self._mem2dict)

        self.IReg2mem = 4
        self.OReg2mem = 5
        self.Macro2mem = 6


        self.double_Macro = 0           # 1 or 0 double Macro supporting

        self.mappingArray = [
            [1,1,1,0,1,0,0],
            [1,1,0,0,0,0,1],
            [1,1,0,1,0,1,0]
        ]
        
        self.nxtMem = [ # 9 = -1
            [1, 2, 4,-1, 7,-1,-1],
            [1, 6,-1,-1,-1,-1, 7],
            [1, 3,-1, 5,-1, 7,-1]
        ]

        self.op2mem = [[],[],[]]             # op2mem[W] = [0,1,6]
        for t, _ in enumerate(['I','W','O']):
            for mem in range(self.Num_mem):
                if self.mappingArray[t][mem] == 1:
                    self.op2mem[t].append(mem)
        
        # self.tileSize = [ size * 1024 for size in [1,1024,32,32,0.25,0.25,0.25]]
        # self.tileSize = [ size * 1024 for size in [1,512,32,32,0.5,0.5,2]]
        # self.tileSize = [1,102400,8192,8192,32,32,2048] 
        self.memSize = [1,1024*512,8192,8192,32,32,2048] 

        # self.bw     = [32,    32,    32,     128,    128,    256,    1024]
        # self.bw     = [32,    3200,    3200,     1280,    1280,    2560,    1024]
        self.bw     = [1,    4,    4,     4,    1024,    1024,    1024]
        # self.bw     = [1,    0.04,    0.04,     0.04,    0.1024,    0.1024,    0.1024]
        self.cost_r = [0.013, 0.143, 0.0113, 0.037,  7.929,  7.929,  164.39]
        self.cost_w = [0.023, 0.177, 0.0136, 0.027,  10.354, 10.354, 108.0]

        self.fanout = [1,8,1,1,32*16,32*16,32*16]

    
    def mem2dict(self,x):
        return self._mem2dict[int(x)]




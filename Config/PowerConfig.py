# this file is prepared for project 026
# Created by iboxl

# size unit = Byte
# bandwidth unit = Bytes/cycle
# fj=1e-15      # pj=1e-12     # nj=1e-9     # µj=1e-6
# Energy Unit = pj
# frequency = 500M hz
from utils.GlobalUT import *

class PowerClass_PRE():
    # finfet 14 0.7V
    def __init__(self):
        ###########################################################
        self.buf_read_2kB_256 = 1.789
        self.buf_write_2kB_256 = 1.828
        self.buf_leakage_2kB_256 = 0.253

        self.buf_read_4kB_256 = 1.875
        self.buf_write_4kB_256 = 2.009
        self.buf_leakage_4kB_256 = 0.3

        self.buf_read_8kB_256 = 2.022
        self.buf_write_8kB_256 = 2.348
        self.buf_leakage_8kB_256 = 0.391

        self.buf_read_16kB_256 = 2.278
        self.buf_write_16kB_256 =  2.497
        self.buf_leakage_16kB_256 = 0.556

        self.buf_read_32kB_256 = 2.538
        self.buf_write_32kB_256 = 3.092
        self.buf_leakage_32kB_256 = 0.879
        ###########################################################
        self.gbuf_read_512kB_32 = 1.513
        self.gbuf_write_512kB_32 = 1.316
        self.gbuf_leakage_512kB_32 = 10.214

        self.gbuf_read_512kB_64 = 2.499
        self.gbuf_write_512kB_64 = 2.104 
        self.gbuf_leakage_512kB_64 = 10.016

        self.gbuf_read_512kB_128 = 3.747
        self.gbuf_write_512kB_128 = 3.785
        self.gbuf_leakage_512kB_128 = 10.176

        self.gbuf_read_512kB_256 = 6.5 
        self.gbuf_write_512kB_256 = 8.342
        self.gbuf_leakage_512kB_256 = 10.227

        self.gbuf_read_512kB_512 =  11.385
        self.gbuf_write_512kB_512 = 15.069
        self.gbuf_leakage_512kB_512 = 10.229

        self.gbuf_read_1024kB_32 = 1.842
        self.gbuf_write_1024kB_32 =  1.644
        self.gbuf_leakage_1024kB_32 = 20.379

        self.gbuf_read_1024kB_64 = 3.164
        self.gbuf_write_1024kB_64 = 2.517
        self.gbuf_leakage_1024kB_64 =  19.758

        self.gbuf_read_1024kB_128 =  5.177
        self.gbuf_write_1024kB_128 = 4.387
        self.gbuf_leakage_1024kB_128 = 19.784

        self.gbuf_read_1024kB_256 = 8.219
        self.gbuf_write_1024kB_256 = 8.294
        self.gbuf_leakage_1024kB_256 = 20.106

        self.gbuf_read_1024kB_512 =  15.87
        self.gbuf_write_1024kB_512 = 23.313
        self.gbuf_leakage_1024kB_512 = 19.947
        ###########################################################
        self.macro_leakage_32kB_64 = 0.698

        self.macro_leakage_16kB_64 = 0.395
        
        self.macro_leakage_8kB_64 = 0.218
        
        self.macro_leakage_4kB_64 = 0.137
        
        self.macro_leakage_2kB_64 = 0.089
        
        self.macro_leakage_1kB_64 = 0.064

        self.macro_read_32kB_128 = 1.406
        self.macro_write_32kB_128 = 2.017
        self.macro_leakage_32kB_128 =  0.761

        self.macro_read_16kB_128 =  1.176
        self.macro_write_16kB_128 =  1.453
        self.macro_leakage_16kB_128 = 0.456
        
        self.macro_read_8kB_128 = 1.015
        self.macro_write_8kB_128 = 1.125
        self.macro_leakage_8kB_128 = 0.277
        
        self.macro_read_4kB_128 = 0.869
        self.macro_write_4KB_128 = 1.032
        self.macro_leakage_4kB_128 =0.185
        
        self.macro_read_2kB_128 = 0.782
        self.macro_write_2kB_128 = 0.85
        self.macro_leakage_2kB_128 = 0.134
        
        self.macro_read_1kB_128 =  0.748
        self.macro_write_1kB_128 = 0.767
        self.macro_leakage_1kB_128 = 0.109
        
        self.macro_read_32kB_256 = 2.538
        self.macro_write_32kB_256 =  3.092
        self.macro_leakage_32kB_256 = 0.879

        self.macro_read_16kB_256 = 2.278
        self.macro_write_16kB_256 =  2.497
        self.macro_leakage_16kB_256 = 0.556
        
        self.macro_read_8kB_256 = 2.022
        self.macro_write_8kB_256 = 2.348
        self.macro_leakage_8kB_256 =  0.391
        
        self.macro_read_4kB_256 = 1.875
        self.macro_write_4kB_256 = 2.009
        self.macro_leakage_4kB_256 = 0.3
        
        self.macro_read_2kB_256 = 1.789
        self.macro_write_2kB_256 = 1.828
        self.macro_leakage_2kB_256 =  0.253
        
        self.macro_read_1kB_256 = 1.712
        self.macro_write_1kB_256 = 1.749
        self.macro_leakage_1kB_256 = 0.201

        # macro_write_per_bit = 0.3726 / 4          # 28
        self.macro_write_per_bit = 0.140874 / 4          # 14
        ###########################################################
        self.peripherals_read_comp16_8 = 0.450
        self.peripherals_read_comp16_4 = 0.265
        
        self.peripherals_read_comp32_8 = 0.989
        self.peripherals_read_comp32_4 = 0.582
        
        self.peripherals_read_comp64_8 = 2.37
        self.peripherals_read_comp64_4 = 1.40
        
        self.peripherals_read_comp128_8 = 5.46
        self.peripherals_read_comp128_4 = 3.21

        # peripherals_leakage_comp16_8 =  0
        
        ###########################################################
        # compute_mm_per_bit = 0.00164          # 28
        compute_mm_per_bit = 0.00062           # 14

        adder_16 = 0.073
        adder_32 = 0.145
        ###########################################################
        treeAdd_comp16_4 = 0.366
        treeAdd_comp16_8 = 0.648

        treeAdd_comp32_4 = 1.38
        treeAdd_comp32_8 = 1.35

        treeAdd_comp64_4 = 1.58
        treeAdd_comp64_8 = 2.76

        treeAdd_comp128_4 = 3.20
        treeAdd_comp128_8 = 5.60

        treeAdd_leakage = 0 
        ###########################################################
        shiftAdd_4 = 0.342
        shiftAdd_8 = 0.356

    def get(self, key):
        return getattr(self, f"{key}", None)

class PowerClass_STD():
    # finfet 14 0.7V
    def __init__(self, config):
        ###########################################################
        buf_read_2kB_256 = 1.789
        buf_write_2kB_256 = 1.828
        buf_leakage_2kB_256 = 0.253

        buf_read_4kB_256 = 1.875
        buf_write_4kB_256 = 2.009
        buf_leakage_4kB_256 = 0.3

        buf_read_8kB_256 = 2.022
        buf_write_8kB_256 = 2.348
        buf_leakage_8kB_256 = 0.391

        buf_read_16kB_256 = 2.278
        buf_write_16kB_256 =  2.497
        buf_leakage_16kB_256 = 0.556

        buf_read_32kB_256 = 2.538
        buf_write_32kB_256 = 3.092
        buf_leakage_32kB_256 = 0.879
        ###########################################################
        gbuf_read_512kB_32 = 1.513
        gbuf_write_512kB_32 = 1.316
        gbuf_leakage_512kB_32 = 10.214

        gbuf_read_512kB_64 = 2.499
        gbuf_write_512kB_64 = 2.104 
        gbuf_leakage_512kB_64 = 10.016

        gbuf_read_512kB_128 = 3.747
        gbuf_write_512kB_128 = 3.785
        gbuf_leakage_512kB_128 = 10.176

        gbuf_read_512kB_256 = 6.5 
        gbuf_write_512kB_256 = 8.342
        gbuf_leakage_512kB_256 = 10.227

        gbuf_read_512kB_512 =  11.385
        gbuf_write_512kB_512 = 15.069
        gbuf_leakage_512kB_512 = 10.229

        gbuf_read_1024kB_32 = 1.842
        gbuf_write_1024kB_32 =  1.644
        gbuf_leakage_1024kB_32 = 20.379

        gbuf_read_1024kB_64 = 3.164
        gbuf_write_1024kB_64 = 2.517
        gbuf_leakage_1024kB_64 =  19.758

        gbuf_read_1024kB_128 =  5.177
        gbuf_write_1024kB_128 = 4.387
        gbuf_leakage_1024kB_128 = 19.784

        gbuf_read_1024kB_256 = 8.219
        gbuf_write_1024kB_256 = 8.294
        gbuf_leakage_1024kB_256 = 20.106

        gbuf_read_1024kB_512 =  15.87
        gbuf_write_1024kB_512 = 23.313
        gbuf_leakage_1024kB_512 = 19.947
        ###########################################################
        macro_leakage_32kB_64 = 0.698

        macro_leakage_16kB_64 = 0.395
        
        macro_leakage_8kB_64 = 0.218
        
        macro_leakage_4kB_64 = 0.137
        
        macro_leakage_2kB_64 = 0.089
        
        macro_leakage_1kB_64 = 0.064

        macro_read_32kB_128 = 1.406
        macro_write_32kB_128 = 2.017
        macro_leakage_32kB_128 =  0.761

        macro_read_16kB_128 =  1.176
        macro_write_16kB_128 =  1.453
        macro_leakage_16kB_128 = 0.456
        
        macro_read_8kB_128 = 1.015
        macro_write_8kB_128 = 1.125
        macro_leakage_8kB_128 = 0.277
        
        macro_read_4kB_128 = 0.869
        macro_write_4KB_128 = 1.032
        macro_leakage_4kB_128 =0.185
        
        macro_read_2kB_128 = 0.782
        macro_write_2kB_128 = 0.85
        macro_leakage_2kB_128 = 0.134
        
        macro_read_1kB_128 =  0.748
        macro_write_1kB_128 = 0.767
        macro_leakage_1kB_128 = 0.109
        
        macro_read_32kB_256 = 2.538
        macro_write_32kB_256 =  3.092
        macro_leakage_32kB_256 = 0.879

        macro_read_16kB_256 = 2.278
        macro_write_16kB_256 =  2.497
        macro_leakage_16kB_256 = 0.556
        
        macro_read_8kB_256 = 2.022
        macro_write_8kB_256 = 2.348
        macro_leakage_8kB_256 =  0.391
        
        macro_read_4kB_256 = 1.875
        macro_write_4kB_256 = 2.009
        macro_leakage_4kB_256 = 0.3
        
        macro_read_2kB_256 = 1.789
        macro_write_2kB_256 = 1.828
        macro_leakage_2kB_256 =  0.253
        
        macro_read_1kB_256 = 1.712
        macro_write_1kB_256 = 1.749
        macro_leakage_1kB_256 = 0.201

        # macro_write_per_bit = 0.3726 / 4          # 28
        macro_write_per_bit = 0.140874 / 4          # 14
        ###########################################################
        peripherals_read_comp16_8 = 0.450
        peripherals_read_comp16_4 = 0.265
        
        peripherals_read_comp32_8 = 0.989
        peripherals_read_comp32_4 = 0.582
        
        peripherals_read_comp64_8 = 2.37
        peripherals_read_comp64_4 = 1.40
        
        peripherals_read_comp128_8 = 5.46
        peripherals_read_comp128_4 = 3.21

        # peripherals_leakage_comp16_8 =  0
        
        ###########################################################
        # compute_mm_per_bit = 0.00164          # 28
        compute_mm_per_bit = 0.00062           # 14

        adder_16 = 0.073
        adder_32 = 0.145
        ###########################################################
        treeAdd_comp16_4 = 0.366
        treeAdd_comp16_8 = 0.648

        treeAdd_comp32_4 = 1.38
        treeAdd_comp32_8 = 1.35

        treeAdd_comp64_4 = 1.58
        treeAdd_comp64_8 = 2.76

        treeAdd_comp128_4 = 3.20
        treeAdd_comp128_8 = 5.60

        treeAdd_leakage = 0 
        ###########################################################
        shiftAdd_4 = 0.342
        shiftAdd_8 = 0.356
        ###########################################################
        num_core = config.getint("Accelerator", "num_core") 
        input_bw = config.getint("Workload", "input_bit_width")
        weight_bw = config.getint("Workload", "weight_bit_width")
        compartment_num = config.getint("Macro", "compartment") 
        cell_num = config.getint("Macro", "cell")
        column_num = config.getint("Macro", "column") 
        bandwidth_r_gb = config.getint("Accelerator", "bandwidth_global_to_iBuffer")            # bit 
        bandwidth_w_gb = config.getint("Accelerator", "bandwidth_oBuffer_to_global")            # bit
        size_global_iBuffer = config.getint("Accelerator", "size_global_iBuffer")               # K-Byte
        size_global_oBuffer = config.getint("Accelerator", "size_global_oBuffer")               # K-Byte
        size_local_ibuffer = config.getint("Core", "size_input_buffer")                         # K-Byte
        size_local_obuffer = config.getint("Core", "size_output_buffer")                        # K-Byte
        macro_size = int(compartment_num * cell_num * column_num / (8 * 1024))                       # K-Byte

        # 使用locals()或globals()获取变量值
        energy_addTree_tree = locals().get(f"treeAdd_comp{compartment_num}_{weight_bw}") * input_bw             # per 32 * 8bit * 8bit
        energy_addTree_shift = locals().get(f"shiftAdd_{weight_bw}") * (input_bw-1)        # per 32 * 8bit * 8bit
        self.energy_addTree_per_acc = (energy_addTree_tree + energy_addTree_shift) * (column_num//weight_bw)

        self.energy_merge_per_bit = (adder_16 if (input_bw==4 and weight_bw==4) else adder_32) / 32
        self.energy_simd_op_per_bit = (adder_16 if (input_bw==4 and weight_bw==4) else adder_32) / 32           # W.T.D.

        self.energy_r_gb_per_bit = locals().get(f"gbuf_read_{size_global_iBuffer}kB_{bandwidth_r_gb}") / bandwidth_r_gb
        self.energy_w_gb_per_bit = locals().get(f"gbuf_write_{size_global_oBuffer}kB_{bandwidth_w_gb}") / bandwidth_w_gb

        self.energy_r_lib_per_bit = locals().get(f"buf_read_{size_local_ibuffer}kB_256") / 256  
        self.energy_w_lib_per_bit = locals().get(f"buf_write_{size_local_ibuffer}kB_256") / 256  

        self.energy_r_lob_per_bit = locals().get(f"buf_read_{size_local_obuffer}kB_256") / 256 
        self.energy_w_lob_per_bit = locals().get(f"buf_write_{size_local_obuffer}kB_256") / 256  

        # self.energy_w_macro_per_row = locals().get(f"macro_write_{macro_size}kB_{column_num}")
        self.energy_w_macro_per_row = macro_write_per_bit * column_num

        self.energy_computation_per_acc = compute_mm_per_bit * column_num * compartment_num

        self.energy_peripherals_per_acc = locals().get(f"peripherals_read_comp{compartment_num}_{input_bw}")

        leakage_core = locals().get(f"buf_leakage_{size_local_ibuffer}kB_256") + locals().get(f"buf_leakage_{size_local_obuffer}kB_256")
        leakage_addTree = treeAdd_leakage * (column_num//weight_bw)
        leakage_macro = locals().get(f"macro_leakage_{macro_size}kB_{column_num}")
        leakage_global = locals().get(f"gbuf_leakage_{size_global_iBuffer}kB_{bandwidth_r_gb}") + locals().get(f"gbuf_leakage_{size_global_oBuffer}kB_{bandwidth_w_gb}")

        self.leakage = (leakage_core + leakage_macro + leakage_addTree) * num_core + leakage_global
        Logger.info(f'leakage is {self.leakage}')

class PowerClass_LP():
    # finfet 14 low Power DVS 0.3
    def __init__(self, config):
        ###########################################################
        buf_read_2kB_256 = 0.738
        buf_write_2kB_256 = 0.741
        buf_leakage_2kB_256 = 0.044

        buf_read_4kB_256 = 0.767
        buf_write_4kB_256 = 0.785
        buf_leakage_4kB_256 = 0.051

        buf_read_8kB_256 = 0.825
        buf_write_8kB_256 = 0.873
        buf_leakage_8kB_256 = 0.065

        buf_read_16kB_256 = 0.895
        buf_write_16kB_256 =  0.910
        buf_leakage_16kB_256 = 0.091
        ###########################################################
        gbuf_read_512kB_32 = 0.692
        gbuf_write_512kB_32 = 0.555
        gbuf_leakage_512kB_32 = 1.662

        gbuf_read_512kB_64 = 1.123
        gbuf_write_512kB_64 = 0.847 
        gbuf_leakage_512kB_64 = 1.63

        gbuf_read_512kB_128 = 1.837
        gbuf_write_512kB_128 = 1.641 
        gbuf_leakage_512kB_128 = 1.656

        gbuf_read_512kB_256 = 3.492 
        gbuf_write_512kB_256 = 3.433
        gbuf_leakage_512kB_256 = 1.664

        gbuf_read_512kB_512 = 6.333 
        gbuf_write_512kB_512 = 6.215 
        gbuf_leakage_512kB_512 = 1.664
        ###########################################################
        macro_read_32kB_128 = 0.538
        macro_write_32kB_128 = 0.601
        macro_leakage_32kB_128 = 0.124

        macro_read_16kB_128 = 0.406
        macro_write_16kB_128 =  0.432
        macro_leakage_16kB_128 = 0.075
        
        macro_read_8kB_128 = 0.335
        macro_write_8kB_128 =  0.343
        macro_leakage_8kB_128 = 0.047
        
        macro_read_4kB_128 = 0.3
        macro_write_4kB_128 = 0.323
        macro_leakage_4kB_128 = 0.034
        
        macro_read_2kB_128 = 0.268
        macro_write_2kB_128 = 0.277
        macro_leakage_2kB_128 = 0.025
        
        macro_read_1kB_128 = 0.252
        macro_write_1kB_128 = 0.254
        macro_leakage_1kB_128 = 0.021
        
        macro_read_32kB_256 = 1.034
        macro_write_32kB_256 = 1.087
        macro_leakage_32kB_256 = 0.144

        macro_read_16kB_256 = 0.895
        macro_write_16kB_256 = 0.91
        macro_leakage_16kB_256 = 0.091
        
        macro_read_8kB_256 = 0.826
        macro_write_8kB_256 = 0.873
        macro_leakage_8kB_256 =  0.065
        
        macro_read_4kB_256 = 0.767
        macro_write_4kB_256 = 0.785
        macro_leakage_4kB_256 = 0.051
        
        macro_read_2kB_256 = 0.738
        macro_write_2kB_256 = 0.742
        macro_leakage_2kB_256 = 0.044
        
        macro_read_1kB_256 = 0.712
        macro_write_1kB_256 = 0.721
        macro_leakage_1kB_256 = 0.032
        
        macro_write_per_bit = 0.3726 / 4
        ###########################################################
        peripherals_read_comp16_8 = 0.000450
        peripherals_read_comp16_4 = 0.000265
        
        peripherals_read_comp32_8 = 0.000989
        peripherals_read_comp32_4 = 0.000582
        
        peripherals_read_comp64_8 = 0.00237
        peripherals_read_comp64_4 = 0.00140
        
        peripherals_read_comp128_8 = 0.00546
        peripherals_read_comp128_4 = 0.00321

        # peripherals_leakage_comp16_8 =  0
        
        ###########################################################
        compute_mm_per_bit = 0.00154

        adder_16 = 7.32e-05
        adder_32 = 14.46e-05
        ###########################################################
        treeAdd_comp16_4 = 3.66e-04
        treeAdd_comp16_8 = 6.48e-04

        treeAdd_comp32_4 = 1.38e-03
        treeAdd_comp32_8 = 1.35e-03

        treeAdd_comp64_4 = 1.58e-03
        treeAdd_comp64_8 = 2.76e-03

        treeAdd_comp128_4 = 3.20e-03
        treeAdd_comp128_8 = 5.60e-03

        treeAdd_leakage = 0 
        ###########################################################
        shiftAdd_4 = 3.42e-4
        shiftAdd_8 = 3.56e-4
        ###########################################################
        num_core = config.getint("Accelerator", "num_core") 
        input_bw = config.getint("Workload", "input_bit_width")
        weight_bw = config.getint("Workload", "weight_bit_width")
        compartment_num = config.getint("Macro", "compartment") 
        cell_num = config.getint("Macro", "cell")
        column_num = config.getint("Macro", "column") 
        bandwidth_r_gb = config.getint("Accelerator", "bandwidth_global_to_iBuffer")            # bit 
        bandwidth_w_gb = config.getint("Accelerator", "bandwidth_oBuffer_to_global")            # bit
        size_global_iBuffer = config.getint("Accelerator", "size_global_iBuffer")               # K-Byte
        size_global_oBuffer = config.getint("Accelerator", "size_global_oBuffer")               # K-Byte
        size_local_ibuffer = config.getint("Core", "size_input_buffer")                         # K-Byte
        size_local_obuffer = config.getint("Core", "size_output_buffer")                        # K-Byte
        macro_size = int(compartment_num * cell_num * column_num / (8 * 1024))                       # K-Byte

        # 使用locals()或globals()获取变量值
        energy_addTree_tree = locals().get(f"treeAdd_comp{compartment_num}_{weight_bw}") * input_bw             # per 32 * 8bit * 8bit
        energy_addTree_shift = locals().get(f"shiftAdd_{weight_bw}") * (input_bw-1)        # per 32 * 8bit * 8bit
        self.energy_addTree_per_acc = (energy_addTree_tree + energy_addTree_shift) * (column_num//weight_bw)

        self.energy_merge_per_bit = (adder_16 if (input_bw==4 and weight_bw==4) else adder_32) / 32
        self.energy_simd_op_per_bit = (adder_16 if (input_bw==4 and weight_bw==4) else adder_32) / 32           # W.T.D.

        self.energy_r_gb_per_bit = locals().get(f"gbuf_read_{size_global_iBuffer}kB_{bandwidth_r_gb}") / bandwidth_r_gb
        self.energy_w_gb_per_bit = locals().get(f"gbuf_write_{size_global_oBuffer}kB_{bandwidth_w_gb}") / bandwidth_w_gb

        self.energy_r_lib_per_bit = locals().get(f"buf_read_{size_local_ibuffer}kB_256") / 256  
        self.energy_w_lib_per_bit = locals().get(f"buf_write_{size_local_ibuffer}kB_256") / 256  

        self.energy_r_lob_per_bit = locals().get(f"buf_read_{size_local_obuffer}kB_256") / 256 
        self.energy_w_lob_per_bit = locals().get(f"buf_write_{size_local_obuffer}kB_256") / 256  

        self.energy_w_macro_per_row = macro_write_per_bit * column_num

        self.energy_computation_per_acc = compute_mm_per_bit * column_num * compartment_num

        self.energy_peripherals_per_acc = locals().get(f"peripherals_read_comp{compartment_num}_{input_bw}")

        leakage_core = locals().get(f"buf_leakage_{size_local_ibuffer}kB_256") + locals().get(f"buf_leakage_{size_local_obuffer}kB_256")
        leakage_addTree = treeAdd_leakage * (column_num//weight_bw)
        leakage_macro = locals().get(f"macro_leakage_{macro_size}kB_{column_num}")
        leakage_global = locals().get(f"gbuf_leakage_{size_global_iBuffer}kB_{bandwidth_r_gb}") + locals().get(f"gbuf_leakage_{size_global_oBuffer}kB_{bandwidth_w_gb}")

        self.leakage = (leakage_core + leakage_macro + leakage_addTree) * num_core + leakage_global

class PowerClass_CMOS():
    # cmos 14
    def __init__(self, config):
        ###########################################################
        buf_read_2kB_256 = 7.41021
        buf_write_2kB_256 = 7.42743
        buf_leakage_2kB_256 = 0.6796944

        buf_read_4kB_256 = 7.89569
        buf_write_4kB_256 = 7.72948 
        buf_leakage_4kB_256 = 0.923152

        buf_read_8kB_256 = 8.43833 
        buf_write_8kB_256 = 8.3925 
        buf_leakage_8kB_256 = 1.37292

        buf_read_16kB_256 = 9.49011 
        buf_write_16kB_256 = 9.68506 
        buf_leakage_16kB_256 = 2.279808
        ###########################################################
        gbuf_read_512kB_32 = 7.79159
        gbuf_write_512kB_32 = 6.33697
        gbuf_leakage_512kB_32 = 50.14312

        gbuf_read_512kB_64 = 12.3393
        gbuf_write_512kB_64 = 10.9933 
        gbuf_leakage_512kB_64 = 52.72392

        gbuf_read_512kB_128 = 21.5176
        gbuf_write_512kB_128 = 19.5059 
        gbuf_leakage_512kB_128 = 52.4516

        gbuf_read_512kB_256 = 41.0499 
        gbuf_write_512kB_256 = 38.8387 
        gbuf_leakage_512kB_256 = 52.02824

        gbuf_read_512kB_512 = 75.2166 
        gbuf_write_512kB_512 = 77.4218 
        gbuf_leakage_512kB_512 = 50.83248
        ###########################################################
        macro_read_16kB_128 = 4.18703
        macro_write_16kB_128 = 4.16732
        macro_leakage_16kB_128 = 1.923856
        
        macro_read_8kB_128 = 3.25039
        macro_write_8kB_128 = 3.34448
        macro_leakage_8kB_128 = 1.096776
        
        macro_read_4kB_128 = 2.71223
        macro_write_4kB_128 = 2.68593
        macro_leakage_4kB_128 = 0.6126144
        
        macro_read_2kB_128 = 2.43447
        macro_write_2kB_128 = 2.34797
        macro_leakage_2kB_128 = 0.3745864
        
        macro_read_1kB_128 = 2.18038
        macro_write_1kB_128 = 2.18729
        macro_leakage_1kB_128 = 0.2440808
        
        macro_read_16kB_256 = 9.49011
        macro_write_16kB_256 = 9.68506
        macro_leakage_16kB_256 = 2.279808
        
        macro_read_8kB_256 = 8.43833
        macro_write_8kB_256 = 8.3925
        macro_leakage_8kB_256 = 1.37292
        
        macro_read_4kB_256 = 7.89569
        macro_write_4kB_256 = 7.72948
        macro_leakage_4kB_256 = 0.923152
        
        macro_read_2kB_256 = 7.41021
        macro_write_2kB_256 = 7.42743
        macro_leakage_2kB_256 = 0.6796944
        
        macro_read_1kB_256 = 6.6477
        macro_write_1kB_256 = 6.9486
        macro_leakage_1kB_256 = 0.4433754

        macro_write_per_bit = 0.3726 / 4
        ###########################################################
        peripherals_read_comp16_8 = 0.000450
        peripherals_read_comp16_4 = 0.000265
        
        peripherals_read_comp32_8 = 0.000989
        peripherals_read_comp32_4 = 0.000582
        
        peripherals_read_comp64_8 = 0.00237
        peripherals_read_comp64_4 = 0.00140
        
        peripherals_read_comp128_8 = 0.00546
        peripherals_read_comp128_4 = 0.00321

        # peripherals_leakage_comp16_8 =  0
        
        ###########################################################
        compute_mm_per_bit = 0.00154

        adder_16 = 7.32e-05
        adder_32 = 14.46e-05
        ###########################################################
        treeAdd_comp16_4 = 3.66e-04
        treeAdd_comp16_8 = 6.48e-04

        treeAdd_comp32_4 = 1.38e-03
        treeAdd_comp32_8 = 1.35e-03

        treeAdd_comp64_4 = 1.58e-03
        treeAdd_comp64_8 = 2.76e-03

        treeAdd_comp128_4 = 3.20e-03
        treeAdd_comp128_8 = 5.60e-03

        treeAdd_leakage = 0 
        ###########################################################
        shiftAdd_4 = 3.42e-4
        shiftAdd_8 = 3.56e-4
        ###########################################################
        num_core = config.getint("Accelerator", "num_core") 
        input_bw = config.getint("Workload", "input_bit_width")
        weight_bw = config.getint("Workload", "weight_bit_width")
        compartment_num = config.getint("Macro", "compartment") 
        cell_num = config.getint("Macro", "cell")
        column_num = config.getint("Macro", "column") 
        bandwidth_r_gb = config.getint("Accelerator", "bandwidth_global_to_iBuffer")            # bit 
        bandwidth_w_gb = config.getint("Accelerator", "bandwidth_oBuffer_to_global")            # bit
        size_global_iBuffer = config.getint("Accelerator", "size_global_iBuffer")               # K-Byte
        size_global_oBuffer = config.getint("Accelerator", "size_global_oBuffer")               # K-Byte
        size_local_ibuffer = config.getint("Core", "size_input_buffer")                         # K-Byte
        size_local_obuffer = config.getint("Core", "size_output_buffer")                        # K-Byte
        macro_size = int(compartment_num * cell_num * column_num / (8 * 1024))                       # K-Byte

        # 使用locals()或globals()获取变量值
        energy_addTree_tree = locals().get(f"treeAdd_comp{compartment_num}_{weight_bw}") * input_bw             # per 32 * 8bit * 8bit
        energy_addTree_shift = locals().get(f"shiftAdd_{weight_bw}") * (input_bw-1)        # per 32 * 8bit * 8bit
        self.energy_addTree_per_acc = (energy_addTree_tree + energy_addTree_shift) * (column_num//weight_bw)

        self.energy_merge_per_bit = (adder_16 if (input_bw==4 and weight_bw==4) else adder_32) / 32
        self.energy_simd_op_per_bit = (adder_16 if (input_bw==4 and weight_bw==4) else adder_32) / 32           # W.T.D.

        self.energy_r_gb_per_bit = locals().get(f"gbuf_read_{size_global_iBuffer}kB_{bandwidth_r_gb}") / bandwidth_r_gb
        self.energy_w_gb_per_bit = locals().get(f"gbuf_write_{size_global_oBuffer}kB_{bandwidth_w_gb}") / bandwidth_w_gb

        self.energy_r_lib_per_bit = locals().get(f"buf_read_{size_local_ibuffer}kB_256") / 256  
        self.energy_w_lib_per_bit = locals().get(f"buf_write_{size_local_ibuffer}kB_256") / 256  

        self.energy_r_lob_per_bit = locals().get(f"buf_read_{size_local_obuffer}kB_256") / 256 
        self.energy_w_lob_per_bit = locals().get(f"buf_write_{size_local_obuffer}kB_256") / 256  

        self.energy_w_macro_per_row = macro_write_per_bit * column_num

        self.energy_computation_per_acc = compute_mm_per_bit * column_num * compartment_num

        self.energy_peripherals_per_acc = locals().get(f"peripherals_read_comp{compartment_num}_{input_bw}")

        leakage_core = locals().get(f"buf_leakage_{size_local_ibuffer}kB_256") + locals().get(f"buf_leakage_{size_local_obuffer}kB_256")
        leakage_addTree = treeAdd_leakage * (column_num//weight_bw)
        leakage_macro = locals().get(f"macro_leakage_{macro_size}kB_{column_num}")
        leakage_global = locals().get(f"gbuf_leakage_{size_global_iBuffer}kB_{bandwidth_r_gb}") + locals().get(f"gbuf_leakage_{size_global_oBuffer}kB_{bandwidth_w_gb}")

        self.leakage = (leakage_core + leakage_macro + leakage_addTree) * num_core + leakage_global


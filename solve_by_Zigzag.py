# import sys
# import os
# base_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
# simulator_path = os.path.join(base_path, 'Simulator')  # 指向 Simulator 目录的路径

# if simulator_path not in sys.path:
#     sys.path.insert(0, simulator_path)

# from zigzag.classes.io.accelerator.parser import AcceleratorParser

# # arch = "zigzag.inputs.examples.hardware.Dimc"
# arch = "Architecture.511"

# a = AcceleratorParser(arch)
# a.run()

# print()
# for _ in range(1):
    # name = a.accelerator.cores[0].memory_hierarchy.mem_level_list[3].name
    # assert name == "local_sram_iBuf"

# print(a.accelerator.cores[0].memory_hierarchy.mem_level_list[3].memory_instance.r_bw)
# print(a.accelerator.cores[0].memory_hierarchy.mem_level_list[3].memory_instance.r_cost)
# print(a.accelerator.cores[0].memory_hierarchy.mem_level_list[3].memory_instance.area)
# print(a.accelerator.cores[0].memory_level_w_cost)
# print(a.accelerator.cores[0].memory_level_r_cost)

# exit()
import time

from zigzag.api import get_hardware_performance_zigzag
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph
import pickle

def print_sp_mappings(mappings):
    print('*********** Spatial Mapping ***********')
    for key in ['I', 'W', 'O']:  # 确保按 I, W, O 的顺序打印
        if key in mappings:
            print(f"{key}):[", end='')
            for mapping in mappings[key]:
                if mapping:  # 检查映射是否为空
                    formatted_mapping = ', '.join([f"('{k}', {v})" for k, v in mapping])
                    print(f"  [{formatted_mapping}],", end='')
                else:
                    print("  [],", end='')
            print(']') 

# model = f"conv_6_layer2_0_conv2_3_3_28_28_128_128.onnx"
# model = f"conv_4_layer1_1_conv2_3_3_56_56_64_64.onnx"
model = f"conv_8_layer2_1_conv1_3_3_28_28_128_128.onnx"
# model = f"mobilenetv2.onnx"
workload = f"model/Resnet18/{model}"
# workload = f"model/alexNet/{model}"

arch = "Architecture.ZigzagAcc"

# mapping = "zigzag.inputs.examples.mapping.default_imc"
mapping = "Config.zigzag_mapping"

test_name = "test"

start_time = time.time()

energy, latency, cme = get_hardware_performance_zigzag(
    workload = workload,
    accelerator = arch,
    mapping = mapping,
    # opt="latency",
    opt="EDP",      
    dump_filename_pattern=f"zzz_outputs/{test_name}.json",
    pickle_filename=f"zzz_outputs/{test_name}.pickle"
)

print(f"energy = {energy:.3e} pj, latency = {latency:.3e} clks, EDP = {energy * latency:.3e}")
print('\n' + '# '*20 +'\n')


# Load in the pickled list of CMEs
with open(f"zzz_outputs/{test_name}.pickle", 'rb') as fp:
    cme_for_all_layers = pickle.load(fp)


visualize_memory_hierarchy_graph(cme_for_all_layers[0].accelerator.cores[0].memory_hierarchy)

for cme in cme_for_all_layers:
    print_sp_mappings(cme.spatial_mapping.mapping_dict_origin)
    print(f"user_spatial_mapping: {cme.layer.user_spatial_mapping}")
    mapped_group_depth = 1
    for (loop_name, loop_size) in cme.temporal_mapping.mapping_dic_origin[cme.layer.constant_operands[0]][0]:
        if loop_name in cme.layer.operand_loop_dim[cme.layer.constant_operands[0]]['r']:
            mapped_group_depth *= loop_size
    print(f"Mapped_group_depth): {mapped_group_depth}", end="")
    # print(f"wl_dim_size: {cme.accelerator.cores[0].operational_array.unit.wl_dim_size}")
    # print(f"bl_dim_size: {cme.accelerator.cores[0].operational_array.unit.bl_dim_size}")
    # print(f"nb_of_banks: {cme.accelerator.cores[0].operational_array.unit.nb_of_banks}")

    # print(f"tm_maping: {cme.temporal_mapping.mapping_dic_origin[cme.layer.constant_operands[0]]}")
    # print(f"in cell tm_maping: {cme.temporal_mapping.mapping_dic_origin[cme.layer.constant_operands[0]][0]}")

    # print(f"allowed_mem_updat_cycle: {cme.allowed_mem_updat_cycle}")
    # print(f"real_data_trans_cycle: {cme.real_data_trans_cycle}")
    # print(f"memory_word_access: {cme.memory_word_access}")

    # print(f"arch_level: {cme.spatial_mapping.arch_level}")
    # print(f"unroll_size_r: {cme.spatial_mapping.unroll_size_r}")
    # print(f"unroll_size_ir: {cme.spatial_mapping.unroll_size_ir}")
    # print(f"unroll_size_total: {cme.spatial_mapping.unroll_size_total}")
    # print(f"unit_count: {cme.spatial_mapping.unit_count}")
    # print(f"ir: {cme.layer.operand_loop_dim[cme.layer.constant_operands[0]]['ir']}")
    # print(f"r: {cme.layer.operand_loop_dim[cme.layer.constant_operands[0]]['r']}")
    
    print_mapping(cme)
    print(f"double buffer flag: {cme.double_buffer_true}")

    print("Energy Breakdown")
    print(f"mem_energy_breakdown: {cme.mem_energy_breakdown}")
    print(f"MAC_energy_breakdown: {cme.MAC_energy_breakdown}")
    print(f"mem_energy: {cme.mem_energy}")
    print(f"MAC_energy: {cme.MAC_energy}")
    print(f"energy_total: {cme.energy_total}")
    print(f"mem_utili_individual: {cme.mem_utili_individual}")
    for layer_op in cme.layer.operand_list:
        print(f"layer_op:{layer_op}")
        for mem_lv in range(cme.active_mem_level[layer_op]):
                    print(f"mem_utilization:{cme.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1]/cme.mem_size_dict[cme.layer_op_to_mem_op[layer_op]][mem_lv]} =  "
                        +f"data:{cme.mapping.data_bit_per_level_unrolled[layer_op][mem_lv + 1]} "
                        +f"/ memSize:{cme.mem_size_dict[cme.layer_op_to_mem_op[layer_op]][mem_lv]} ")
    
    # print(f"ideal_cycle: {cme.ideal_cycle}")
    # print(f"SS_comb: {cme.SS_comb}")
    # print(f"data_loading_cycle: {cme.data_loading_cycle}")
    # print(f"data_offloading_cycle: {cme.data_offloading_cycle}")

    # print(f"area_total: {cme.area_total}")
    # print(f"imc_area: {cme.imc_area}")
    # print(f"mem_area: {cme.mem_area}")
    # print(f"mem_area_breakdown: {cme.mem_area_breakdown}")

    # for mem in cme.accelerator.get_core(0).memory_hierarchy.mem_level_list:
    #     print(f"{mem.memory_instance}:")
    #     print(mem.memory_instance.size)

    print(f"data_bit_per_level_unrolled")
    print(cme.mapping.data_bit_per_level_unrolled)
    print(f"r_loop_size_cabl2)")
    print(cme.mapping.r_loop_size_cabl2)
            
    print("data_precision_dict")
    print(cme.mapping.data_precision_dict)
    print("temporal_mapping.top_r_loop_size")
    print(cme.temporal_mapping.top_r_loop_size)
    print("mapping.effective_data_bit")
    print(cme.mapping.effective_data_bit )
    print('- '* 30)

    print("r_loop_size_per_level2")
    # for op in cme.mapping.operand_list:
    #     print(f"op:{op}")
    #     for lv in range(cme.mapping.spatial_mapping.arch_level[op]):
    #         print(cme.mapping.r_loop_size_per_level2[op][0 : lv + 1])

    print("combined_mapping_dict_1s2t_reform")
    # print(cme.mapping.combined_mapping_dict_1s1t_reform)
    # print(cme.mapping.combined_mapping_dict_1s2t_reform)


    print(f"* * *mem_utili_individual:")
    print(cme.mem_utili_individual)
    print(f"* * *mem_utili_shared:")
    print(cme.mem_utili_shared)
    print(f"* * *effective_mem_utili_individual:")
    print(cme.effective_mem_utili_individual)
    print(f"* * *effective_mem_utili_shared:")
    print(cme.effective_mem_utili_shared)

    for operand in cme.temporal_mapping.operand_list:
        print(f"op:{operand}-")
        for level, current_level_loops in enumerate(
            cme.temporal_mapping.mapping_dic_stationary[operand]
        ):
            print(f" level:{level}, current_level_loops:{current_level_loops}")

    # for op in cme.mapping.operand_list:
    #     print(f"tensor-{op}")
    #     for lv in range(cme.mapping.spatial_mapping.arch_level[op]):
    #         print(f"{lv}")
    #         for lp_type, lp_dim in cme.mapping.combined_mapping_dict_1s2t_reform[op][lv]:
    #             print(f"type:{lp_type}, dim:{lp_dim}")
        

    print('\n' + '# '*20 +'\n')
    print(f"energy = {cme.energy_total:.3e} pj, latency = {cme.latency_total2:.3e} clks, EDP = {cme.energy_total * cme.latency_total2:.3e}")


end_time = time.time()
print(f"Solving time cost: {round(end_time-start_time, 3)}s")
# this file is prepared for project 511
# Created by iboxl

import pickle
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.classes.stages import *
from zigzag.classes.cost_model.cost_model_for_sram_imc import CostModelEvaluationForIMC
from zigzag.classes.mapping.spatial.spatial_mapping import SpatialMapping
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.classes.io.accelerator.parser import AcceleratorParser


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

# with open(f"debugFile/pickleFile/CostModel_zigzag.pickle", 'rb') as fp:         # original dataflow from Zigzag
with open(f"debugFile/pickleFile/Arch511.pickle", 'rb') as fp:         # original dataflow from Zigzag
    ds = pickle.load(fp)[0]

tag_eval_DS = True
# tag_eval_DS = False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
if tag_eval_DS :
    accelerator_parser = AcceleratorParser("Architecture.ZigzagAcc")

    accelerator_parser.run()
    ds.accelerator = accelerator_parser.get_accelerator()               # without RemoveUnusedMemoryStage, so attention to Archi-spec.

    layer_attrs = {
        'equation': 'O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]', 
        'loop_dim_size': {'B': 1, 'K': 64, 'G': 1, 'OX': 56, 'OY': 56, 'C': 64, 'FX': 3, 'FY': 3}, 
        'pr_loop_dim_size': {'IX': 56, 'IY': 56}, 'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'], 
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8}, 
        'constant_operands': ['W'], 'core_allocation': 0, 'temporal_ordering': None, 
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}, 'spatial_mapping': None, 
        'spatial_mapping_hint': {'D1': ['K'], 'D3': ['K', 'OX', 'OY'], 'D2': ['B', 'K', 'G', 'OX', 'OY', 'C', 'FX', 'FY']}, 
        'operand_source': {'I': []}, 
        'padding': {'IY': (1, 1), 'IX': (1, 1)}
        # 'padding': {'IY': (0, 0), 'IX': (0, 0)}
    }
    ds.layer = LayerNode(0, layer_attrs=layer_attrs, node_name="evaluation_node", type=None)

# # # # # # #Zigzag# # # # # # # # # # # # # # # # # # # # # # # # #Zigzag # # # # # # # # # # # # # # # # # # # # # # #Zigzag# # # # # # # # # # # # # # 
    ds.spatial_mapping = SpatialMapping(spatial_mapping_dict={
                            'I':[  [('K', 16.0)],  [('C', 32.0)],  [('OX', 8.0)],  [],  [],]          , 
                            'W': [  [],  [('C', 32.0), ('K', 16.0), ('OX', 8.0)],  [],  [],]           ,   
                            'O': [  [('C', 32.0)],  [('K', 16.0)],  [('OX', 8.0)],  [],  [],]    },       layer_node=ds.layer)
    ds.spatial_mapping_int = ds.spatial_mapping        
    ds.layer.user_spatial_mapping = {'D1': ('K', 16), 'D2': ('C', 32), 'D3':  ('OX', 8)}

    ds.temporal_mapping = TemporalMapping(temporal_mapping_dict={
                            'O':     [[], [('OY', 56), ('FY', 3), ('FX', 3), ('C', 2)], [], [('OX', 7), ('K', 4)]]             , 
                            'W':   [[('OY', 56), ('FY', 3)], [('FX', 3), ('C', 2), ('OX', 7), ('K', 4)], []]              , 
                            'I':   [[], [('OY', 56), ('FY', 3)], [('FX', 3), ('C', 2), ('OX', 7), ('K', 4)], []]             },        layer_node=ds.layer)
# # # # # #Zigzag# # # # # # # # # # # # # # # # # # # # # # Zigzag # # # # # # # # # # # # # Zigzag # # # # # # # # # # # # # # # # # # 

# # # # # # # MIREDO# # # # # # # # # # # # # # # MIREDO # # # # # # # # # # # # # # # # # MIREDO# # # # # # # # # # # # # 
    ds.spatial_mapping = SpatialMapping(spatial_mapping_dict={
                            'I': [  [('K', 16.0)],  [('C', 8),('FY', 3)],  [('OX', 8)],  [],  [],]         , 
                            'W': [   [],  [('K', 16),('C', 8),('FY', 3), ('OX', 8)],  [],  [],]             ,   
                            'O': [   [('C', 8),('FY', 3)],  [('K', 16.0)],  [('OX', 8)],  [],  [],]      },       layer_node=ds.layer)
    ds.spatial_mapping_int = ds.spatial_mapping        
    ds.layer.user_spatial_mapping = {'D1': ('K', 16), 'D2': (('C', 8),('FY', 3)), 'D3': (('OX', 8))}

    ds.temporal_mapping = TemporalMapping(temporal_mapping_dict={
                            'I':  [[], [('FX', 3),('OX', 7),('OY', 7)], [], [('OX', 8),('C', 8),('K', 4)]]           , 
                            'W':  [[('FX', 3),('OX', 7),('OY', 7)], [('OX', 8),('C', 8),('K', 4)], []]        , 
                            'O':  [[('FX', 3)], [], [('OX', 7),('OY', 7),('OX', 8),('C', 8),('K', 4)], []]          },        layer_node=ds.layer)  
# # # # # # # MIREDO # # # # # # # # # # # # # # # # MIREDO # # # # # # # # # # # # # # # # # # # MIREDO# # # # # # # # # # # # # # 

    ds.access_same_data_considered_as_no_access = False

cme = CostModelEvaluationForIMC(accelerator=ds.accelerator, layer=ds.layer, spatial_mapping=ds.spatial_mapping, 
                                spatial_mapping_int=ds.spatial_mapping_int, temporal_mapping=ds.temporal_mapping, 
                                access_same_data_considered_as_no_access=ds.access_same_data_considered_as_no_access)

print_sp_mappings(cme.spatial_mapping.mapping_dict_origin)
print(f"user_spatial_mapping: {cme.layer.user_spatial_mapping}")
mapped_group_depth = 1
for (loop_name, loop_size) in cme.temporal_mapping.mapping_dic_origin[cme.layer.constant_operands[0]][0]:
    if loop_name in cme.layer.operand_loop_dim[cme.layer.constant_operands[0]]['r']:
        mapped_group_depth *= loop_size
print(f"Mapped_group_depth): {mapped_group_depth}", end="")

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

print(f"r_loop_size_cabl2)")
print(cme.mapping.r_loop_size_cabl2)
           
print("data_precision_dict")
print(cme.mapping.data_precision_dict)
print("temporal_mapping.top_r_loop_size")
print(cme.temporal_mapping.top_r_loop_size)
print("temporal_mapping.top_ir_loop_size")
print(cme.temporal_mapping.top_ir_loop_size)
print(f"data_bit_per_level_unrolled")
print(cme.mapping.data_bit_per_level_unrolled)
print("mapping.effective_data_bit")                     # effective_data_bit = data_bit_unrolled // top_r_size
print(cme.mapping.effective_data_bit )
print(f"iteration_each_level for-loops")      # iteration_each_level only counts for the current level for-loops
print(cme.mapping.temporal_mapping.cycle_cabl_level) 
print(f"data_trans_amount")
for layer_op in cme.layer.operand_list:
    print(f"{layer_op}: ")
    for mem_lv in range(cme.mapping_int.mem_level[layer_op]):
        print(cme.mapping.unit_mem_data_movement[layer_op][mem_lv].data_trans_amount_per_period)
    print("")
print('- '* 30)

# print("r_loop_size_per_level2")
# for op in cme.mapping.operand_list:
#     print(f"op:{op}")
#     for lv in range(cme.mapping.spatial_mapping.arch_level[op]):
#         print(cme.mapping.r_loop_size_per_level2[op][0 : lv + 1])

# print("combined_mapping_dict_1s2t_reform")
# for op in cme.mapping.operand_list:
#     print(f"op:{op}")
#     for lv in range(cme.mapping.spatial_mapping.arch_level[op]):
#         print(lv)
#         for lp_type, lp_dim in cme.mapping.combined_mapping_dict_1s2t_reform[op][lv]:
#             print(f"[{lp_type}:{lp_dim}]")
# print("relevancy_table[op]")
# for op in cme.mapping.operand_list:
#     print(f"op:{op}- ",end="")
#     print(cme.mapping.layer_node.operand_loop_dim_reform[op])

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

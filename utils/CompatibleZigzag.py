# this file is prepared for project 419
# Created by iboxl

from utils.Workload import WorkLoad, LoopNest, Mapping
from Architecture.ArchSpec import CIM_Acc
import numpy as np
import copy
import math
from typing import Dict, List, Tuple
from collections import defaultdict


def convert_ZZ_dflag_to_doubleflag(acc:CIM_Acc, ops:WorkLoad, cme_dflag):
    # 定义固定的操作数顺序
    fixed_order = ['I', 'W', 'O']
    num_operations = len(fixed_order)
    
    dflag = {}
    for op in fixed_order:          # """ Zigzag start with False for each operand at the lowest arch level (MAC array level) """
        dflag[op] = cme_dflag[op][1:]
    
    # 检查 dflag 和 mappingArray 的一致性
    for op_type in fixed_order:
        if op_type in dflag:
            op_index = fixed_order.index(op_type)
            allowed_levels = sum(1 for level in range(acc.Num_mem) if acc.mappingArray[op_index][level] == 1)
            if allowed_levels != len(dflag[op_type]):
                raise ValueError(f"Mismatch in levels count for operation '{op_type}': expected {allowed_levels}, got {len(dflag[op_type])}")
    
    # 初始化 double_tag 矩阵为全 0，大小为 (memory_levels, num_operations)
    double_tag = np.zeros((acc.Num_mem, num_operations), dtype=int)
    # 填充 double_tag 矩阵
    for op_type in fixed_order:
        if op_type in dflag:
            op_index = fixed_order.index(op_type)
            levels = dflag[op_type][::-1]  # 将 dflag 顺序颠倒，从最后一个有效映射开始
            level_counter = 0
            for level_index in range(acc.Num_mem):
                if acc.mappingArray[op_index][level_index] == 1:
                    double_tag[level_index][op_index] = int(levels[level_counter])
                    level_counter += 1
    # for t in range(num_operations):
    #     double_tag[acc.Num_mem,t] = 0
    macro_dflag = [ 0 for _ in range(num_operations)]

    # 使用 vstack 将新行添加到末尾
    double_tag = np.vstack([double_tag, macro_dflag])
    return double_tag

def process_top_r(ori_tm_dict, cme_top_r_loop):
    """
    输入：
      ori_tm_dict: dict，键为操作数名称，对应值为一个层次列表，每一层是一个列表，里面存放形如 (维度名称, 大小) 的元组。
      top_r_loop: dict，键与 A 相同，对应值为一个转移关系列表，每个整数与 A 中对应层次一一对应，
         表示：从该层尾部开始，凡是映射中数值小于等于该整数的，都应转移到上一层（即索引+1处）。
         注意：如果转移目标所在层原本为空，则直接插入到该层；如果已处于最高层，则不转移。
         
    输出：
      new_tm_dict: dict，与 A 同样结构，经过转移关系 B 调整后的映射关系。
    """
    top_r_loop = {}
    for op in ['I', 'W', 'O']:          # """ Zigzag start with False for each operand at the lowest arch level (MAC array level) """
        top_r_loop[op] = cme_top_r_loop[op][1:]

    new_tm_dict = copy.deepcopy(ori_tm_dict)
    for key in new_tm_dict:
        layers = new_tm_dict[key]
        thresholds = top_r_loop[key]
        num_layers = len(layers)
        # 处理每一层（除了最高层）
        for i in range(num_layers - 1):
            # 阈值为1时不转移
            if thresholds[i] == 1:
                continue
            product = 1
            count = 0
            # 从当前层尾部开始累积映射中大小的乘积
            for mapping in reversed(layers[i]):
                product *= mapping[1]
                count += 1
                if product == thresholds[i]:
                    break
                if product > thresholds[i]:
                    count = 0
                    break
            # 若满足条件，则进行转移
            if product == thresholds[i] and count > 0:
                # 从当前层尾部取出连续 count 个映射
                tail = layers[i][-count:]
                layers[i] = layers[i][:-count]
                # 确定转移目标层：从直接上层开始查找非空层
                dest = i + 1
                while dest < num_layers - 1 and len(layers[dest]) == 0:
                    dest += 1
                # 将转移的映射插入到目标层的前端，保持顺序（自下向上）
                layers[dest] = tail + layers[dest]
    return new_tm_dict

from collections import deque
def convert_ZZMP_to_loopMP(
    mapping_dict: Dict[str, List[List[Tuple[str, int]]]],
    mappingArray: List[List[int]],
    mappingList: List[Mapping],
    *,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),   # mappingArray 的行序
    dim_code = None                # 维度名字 → 编号
) -> List[Mapping]:
    """
    将 temporal mapping (mapping_dict + mappingArray) 转成 loop-level 列表。

    Parameters
    ----------
    mapping_dict : {'I'|'W'|'O' : List[List[(dim, size)]]}
        每个操作数在各层展开的因子，**层序 = 外 → 内**。
    mappingArray : List[List[int]]
        array_row_order = ('I','W','O') 时：
        第 0 行对应 I 1 行 W-2 行 O每个 1 表示该层选用的存储级编号。
        行里可含占位 -1 读取时会被直接跳过。
    mappingList  : List[Mapping]
        结果累加到此列表并返回（保持原引用）。
    array_row_order : Tuple[str, ...], default ('I','W','O')
        mappingArray 各行对应的操作数顺序。
    dim_code : Dict[str,int] | None
        维度字符串 → 编号；缺省时直接保留字符串。

    Returns
    -------
    Same list object with新 Mapping 追加完毕。
    """
    # ---------------- 0. 参数 / 默认 -----------------
    if dim_code is None:
        dim_code = {"FX": 1, "FY": 2, "OX": 3, "OY": 4, "C": 5, "K": 6, "G":7}

    if len(mappingArray) != len(array_row_order):
        raise ValueError("mappingArray 行数必须等于 array_row_order 长度")

    # ---------------- 1. 统计 (op,tup) → deque[inner…outer] -------
    level_map: Dict[str, Dict[Tuple[str, int], deque[int]]] = {
        op: defaultdict(deque) for op in array_row_order
    }

    for row_idx, op in enumerate(array_row_order):
        layers = mapping_dict[op]                       # 外 → 内
        # 取出行中所有“1”出现的位置（跳过 -1）
        pos = [col for col, v in enumerate(mappingArray[row_idx]) if v == 1]

        if len(pos) != len(layers):
            raise ValueError(
                f"{op}: 1 的数量 {len(pos)} ≠ 层数 {len(layers)}"
                "请检查 mappingArray / mapping_dict"
            )

        rev_pos = list(reversed(pos))                  # 外层 → 对应的存储级
        for layer_idx, layer in enumerate(layers):     # 外 → 内
            mem_level = rev_pos[layer_idx]             # 正确的列号
            for tup in layer:                          # 该层所有因子
                level_map[op][tuple(tup)].appendleft(mem_level)
                # appendleft 保证 popleft 先弹“最内层”

    # ---------------- 2. 生成全局遍历次序 (基于 I)，内 → 外 -------
    order: List[Tuple[str, int]] = []
    for layer_idx in reversed(range(len(mapping_dict["I"]))):        # 内 → 外
        for tup in reversed(mapping_dict["I"][layer_idx]):           # 右 → 左
            order.append(tuple(tup))

    # ---------------- 3. 依次输出 Mapping ------------------------
    for tup in order:
        levels = []
        for op in array_row_order:                   # I, W, O 顺序
            dq = level_map[op].get(tup)
            levels.append(dq.popleft() if dq else None)

        mappingList.append(
            Mapping(
                dim=dim_code.get(tup[0], tup[0]),
                dimSize=round(tup[1]),
                mem=levels
            )
        )
    return mappingList

def fix_all_memHierarchy(acc:CIM_Acc, tm:list[Mapping]):
    tm_first = tm[0]
    tm_last = tm[-1]
    if (tm_first.mem[0]!=1) or (tm_first.mem[1]!=1) or (tm_first.mem[2]!=1):
        tm.insert(0, Mapping(tm_first.dim, 1, [1, 1, 1]))
    # if (tm_last.mem[0] != acc.IReg2mem) or (tm_last.mem[1]!=acc.Macro2mem) or (tm_last.mem[2]!=acc.OReg2mem):
    #     tm.append(Mapping(tm_last.dim, 1, [acc.IReg2mem, acc.Macro2mem, acc.OReg2mem]))
    return tm

def normalize_spatial_mapping(mapping):
    """
    规范化空间映射字典（无基准键版本）

    1. 对所有键，找出同名循环变量的“最细拆分”（出现次数最多的那一方）
       并将其它键中同变量的合并因子在**原所在层**内按该拆分展开；
       只在该变量原本是“单因子且值等于乘积”时才展开，防止误改已拆分结构。
    2. 尾部裁剪：仅当 *全部* 键的最末层同时为空列表 `[]` 时，
       统一删除这一层（连续空层也只删除最外层一层）。
    3. 不修改输入对象，返回深拷贝。
    """
    mp = copy.deepcopy(mapping)           # 深拷贝以免破坏原数据

    # ---------- 1. 统计每个变量最细拆分（出现次数最多的那一键） ----------
    canonical: Dict[str, List[float]] = {}
    for layers in mp.values():
        counts: Dict[str, List[float]] = {}
        for fl in layers:
            for v, val in fl:
                counts.setdefault(v, []).append(float(val))
        for v, seq in counts.items():
            if len(seq) > len(canonical.get(v, [])):
                canonical[v] = seq[:]                  # 记录该变量最细拆分

    # ---------- 2. 对每个键展开合并因子（保持在原层内） ----------
    for layers in mp.values():
        # 记录该键内每个变量出现的层和索引
        occ: Dict[str, List[Tuple[int, int, float]]] = {}
        for depth, fl in enumerate(layers):
            for idx, (v, val) in enumerate(fl):
                occ.setdefault(v, []).append((depth, idx, float(val)))

        # 逐变量处理
        for var, finest in canonical.items():
            if len(finest) <= 1:          # 该变量本就只有单因子，无需展开
                continue
            fin_prod = math.prod(finest)

            # 若该键里该变量只有一次出现且值恰好等于乘积 → 需展开
            positions = occ.get(var, [])
            if len(positions) == 1:
                depth, idx, cur_val = positions[0]
                if math.isclose(cur_val, fin_prod):
                    layer = layers[depth]

                    # ① 删除原合并因子
                    layer.pop(idx)

                    # ② 在同一位置插入展开因子（保持相对顺序）
                    for offset, f in enumerate(finest):
                        layer.insert(idx + offset, (var, f))

    # ---------- 3. 统一删除一层公共尾部空列表（若存在） ----------
    if mp and all(l and l[-1] == [] for l in mp.values()):
        for k in mp:
            mp[k] = mp[k][:-1]

    return mp

def convert_zz_to_miredo(loops:LoopNest, cme=None):
    
    temporal_mapping_dic=cme.temporal_mapping.mapping_dic_origin
    spatial_mapping_dict=cme.spatial_mapping_int.mapping_dict_origin
    double_buffer_flag=cme.double_buffer_true

    # top_r_loop_size=cme.temporal_mapping.top_r_loop_size
    # loops.tm = convert_ZZMP_to_loopMP(mapping_dict = process_top_r(ori_tm_dict=temporal_mapping_dic, cme_top_r_loop=top_r_loop_size),
    loops.tm = convert_ZZMP_to_loopMP(mapping_dict = temporal_mapping_dic,
                                       mappingArray = loops.acc.mappingArray, 
                                       mappingList = loops.tm)
    
    # loops.tm = fix_all_memHierarchy(acc=loops.acc, tm=loops.tm)
    
    loops.sm = convert_ZZMP_to_loopMP(mapping_dict = normalize_spatial_mapping(spatial_mapping_dict),
                                       mappingArray = loops.acc.mappingArray, 
                                       mappingList = loops.sm)
    print(double_buffer_flag)
    loops.usr_defined_double_flag = convert_ZZ_dflag_to_doubleflag(loops.acc, loops.ops, double_buffer_flag)
    return loops

# def compare_ops_cme(ops:WorkLoad, cme):
#     loop_dim = cme.layer.loop_dim_size
#     if len(loop_dim) < 6:
#         return False
#     return (
#         ops.R == loop_dim['FX'] and
#         ops.S == loop_dim['FY'] and
#         ops.P == loop_dim['OX'] and
#         ops.Q == loop_dim['OY'] and
#         ops.C == loop_dim['C']  and
#         ops.K == loop_dim['K']
#     ) 

def compare_ops_cme(loopDim, cme):
    cme_dim = cme.layer.loop_dim_size
    if len(cme_dim) < 6:
        return False
    return (
        loopDim['R'] == cme_dim['FX'] and
        loopDim['S'] == cme_dim['FY'] and
        loopDim['P'] == cme_dim['OX'] and
        loopDim['Q'] == cme_dim['OY'] and
        loopDim['C'] == cme_dim['C']  and
        loopDim['K'] == cme_dim['K']  and
        loopDim['G'] == cme_dim['G']
    ) 

def get_dim_from_cme(cme):
    # Not Use
    dims = cme.layer.loop_dim_size
    loop_dim = f"{dims['FX']}_{dims['FY']}_{dims['OX']}_{dims['OY']}_{dims['C']}_{dims['K']}"
    if len(loop_dim) < 6:
        return False
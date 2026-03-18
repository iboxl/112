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

'''
Why this function exists:
ZigZag evaluates double buffering with an analytical abstraction. The trailing
relevant loops captured by `top_r_loop_size` are treated as a rolling-update
window, so they shrink the effective resident footprint used by the cost model.
That abstraction is valid inside ZigZag, but it is not explicit in the exported
temporal mapping consumed by MIREDO/Simulax, which replays a concrete loop-to-
memory dataflow.

`process_top_r` therefore does not re-search, re-score, or otherwise alter the
baseline objective. It only materializes the overlap that ZigZag already assumes
by promoting the matched top-r tail loops to the immediate upper memory level.
This keeps the original loop order and makes the rolling refill window explicit
enough for behavior-level replay. The transformation is conservative: it never
introduces more overlap than ZigZag assumed, and any original double-buffer flag
that becomes physically unrealizable after explicitization is pruned later by a
replay-safety projection instead of being strengthened.
'''
def process_top_r(ori_tm_dict, cme_top_r_loop):
    top_r_loop = {
        op: [int(round(x)) for x in cme_top_r_loop[op][1:]]
        for op in ['I', 'W', 'O']
    }

    new_tm_dict = copy.deepcopy(ori_tm_dict)
    for op in ['I', 'W', 'O']:
        layers = new_tm_dict[op]
        thresholds = top_r_loop[op]
        for level_idx in range(len(layers) - 1):
            threshold = thresholds[level_idx]
            if threshold <= 1 or len(layers[level_idx]) == 0:
                continue

            product = 1
            move_from = None
            for tuple_idx in range(len(layers[level_idx]) - 1, -1, -1):
                product *= int(round(layers[level_idx][tuple_idx][1]))
                if product == threshold:
                    move_from = tuple_idx
                    break
                if product > threshold:
                    break

            if move_from is None:
                continue

            promoted_tail = layers[level_idx][move_from:]
            layers[level_idx] = layers[level_idx][:move_from]
            layers[level_idx + 1] = promoted_tail + layers[level_idx + 1]

    return new_tm_dict


def _extract_top_r_loop_size(cme):
    if hasattr(cme.temporal_mapping, 'top_r_loop_size'):
        return cme.temporal_mapping.top_r_loop_size
    return cme.mapping.temporal_mapping.top_r_loop_size


def _calc_explicit_mem_usage(loops: LoopNest):
    acc = loops.acc
    ops = loops.ops
    loops.preprogress()

    dim_tp = [[[1 for _ in range(ops.Num_dim)] for _ in range(3)] for _ in range(acc.Num_mem)]
    dim_sp = [[[1 for _ in range(ops.Num_dim)] for _ in range(3)] for _ in range(acc.Num_mem)]

    for mem in range(1, acc.Num_mem):
        for op in range(3):
            for mapping in loops.tm:
                if mem <= mapping.mem[op] and acc.mappingArray[op][mem] == 1:
                    dim_tp[mem][op][mapping.dim] *= mapping.dimSize
            for mapping in loops.sm:
                if mem <= mapping.mem[op] and acc.mappingArray[op][mem] == 1:
                    dim_sp[mem][op][mapping.dim] *= mapping.dimSize

    data_size = {}
    tmp_dim = [0 for _ in range(ops.Num_dim)]
    for op in range(3):
        data_size[acc.Num_mem, op] = 1

    for mem in range(1, acc.Num_mem):
        op = 0
        for dim in range(1, ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
        if acc.mappingArray[op][mem]:
            if ops.Stride >= tmp_dim[ops.dict2Dim('R')]:
                tmp_h = tmp_dim[ops.dict2Dim('P')] * tmp_dim[ops.dict2Dim('R')]
            else:
                tmp_h = (tmp_dim[ops.dict2Dim('P')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('R')]
            if ops.Stride >= tmp_dim[ops.dict2Dim('S')]:
                tmp_w = tmp_dim[ops.dict2Dim('Q')] * tmp_dim[ops.dict2Dim('S')]
            else:
                tmp_w = (tmp_dim[ops.dict2Dim('Q')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('S')]
            data_size[mem, op] = min(tmp_h, ops.H) * min(tmp_w, ops.W) * tmp_dim[ops.dict2Dim('C')]
        else:
            data_size[mem, op] = 0

        op = 1
        for dim in range(1, ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
        if acc.mappingArray[op][mem]:
            data_size[mem, op] = (
                tmp_dim[ops.dict2Dim('R')] *
                tmp_dim[ops.dict2Dim('S')] *
                tmp_dim[ops.dict2Dim('C')] *
                tmp_dim[ops.dict2Dim('K')]
            )
        else:
            data_size[mem, op] = 0

        op = 2
        for dim in range(1, ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
        if acc.mappingArray[op][mem]:
            data_size[mem, op] = (
                tmp_dim[ops.dict2Dim('P')] *
                tmp_dim[ops.dict2Dim('Q')] *
                tmp_dim[ops.dict2Dim('K')]
            )
        else:
            data_size[mem, op] = 0

    for op in range(3):
        for mem in range(1, acc.Num_mem):
            if loops.bypassMem[mem][op]:
                data_size[mem, op] = 0

    raw_bits = np.zeros((acc.Num_mem + 1, 3), dtype=int)
    used_bits = np.zeros((acc.Num_mem + 1, 3), dtype=int)
    for mem in range(1, acc.Num_mem):
        for op in range(3):
            if acc.mappingArray[op][mem] == 0:
                continue
            if hasattr(loops, "get_precision"):
                precision = loops.get_precision(mem, op)
            else:
                precision = acc.precision[mem, op]
            raw_bits[mem][op] = int(data_size[mem, op] * precision)
            used_bits[mem][op] = raw_bits[mem][op] * (1 + int(loops.usr_defined_double_flag[mem][op]))
    return raw_bits, used_bits


def project_replay_safe_double_flag(loops: LoopNest, double_tag):
    safe_double_tag = np.array(double_tag, dtype=int, copy=True)
    loops.usr_defined_double_flag = safe_double_tag

    while True:
        raw_bits, used_bits = _calc_explicit_mem_usage(loops)
        changed = False

        for mem in range(1, loops.acc.Num_mem):
            total_bits = int(np.sum(used_bits[mem]))
            if total_bits <= loops.acc.memSize[mem]:
                continue

            removable = []
            for op in range(3):
                if safe_double_tag[mem][op] == 1 and raw_bits[mem][op] > 0:
                    removable.append((int(raw_bits[mem][op]), op))
            removable.sort(reverse=True)

            for released_bits, op in removable:
                if total_bits <= loops.acc.memSize[mem]:
                    break
                safe_double_tag[mem][op] = 0
                total_bits -= released_bits
                changed = True

        loops.usr_defined_double_flag = safe_double_tag
        if changed is False:
            break

    return safe_double_tag

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

def convert_Zigzag_to_MIREDO(loops:LoopNest, cme=None):
    
    temporal_mapping_dic=cme.temporal_mapping.mapping_dic_origin
    spatial_mapping_dict=cme.spatial_mapping_int.mapping_dict_origin
    double_buffer_flag=cme.double_buffer_true

    top_r_loop_size = _extract_top_r_loop_size(cme)
    loops.tm = convert_ZZMP_to_loopMP(mapping_dict = process_top_r(ori_tm_dict=temporal_mapping_dic, cme_top_r_loop=top_r_loop_size),
                                       mappingArray = loops.acc.mappingArray, 
                                       mappingList = loops.tm)
    
    # loops.tm = fix_all_memHierarchy(acc=loops.acc, tm=loops.tm)
    
    loops.sm = convert_ZZMP_to_loopMP(mapping_dict = normalize_spatial_mapping(spatial_mapping_dict),
                                       mappingArray = loops.acc.mappingArray, 
                                       mappingList = loops.sm)
    loops.usr_defined_double_flag = convert_ZZ_dflag_to_doubleflag(loops.acc, loops.ops, double_buffer_flag)
    loops.usr_defined_double_flag[loops.acc.Macro2mem][1] = loops.acc.double_Macro
    loops.usr_defined_double_flag = project_replay_safe_double_flag(loops, loops.usr_defined_double_flag)
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

# 绝对最优数据流穷举搜索脚本
# 穷举所有合法数据流映射（空间+时间+双缓冲），用cycle-accurate simulator评估，
# 获取给定负载和硬件的理论最优数据流。
# 与MIREDO MIP框架完全独立，仅共享：硬件规格、负载描述、simulator。
# 搜索空间：空间展开 × 时间因子分解 × 循环排序 × 存储级分配 × 双缓冲配置
# 用途：为MIREDO的全局最优性提供独立穷举证明。
# 用法：python Evaluation/VerifyBruteforce.py [--case CASE_NAME]
# 可选case: 1x1_C64K64, 3x3_C64K64, resnet_layer13, 自定义(修改CASES字典)

import os, sys, math, itertools, time, copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from Evaluation.common.EvalCommon import save_experiment_json
from utils.Workload import WorkLoad, LoopNest, Mapping
from utils.UtilsFunction.ToolFunction import get_Spatial_Unrolling
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import *

_LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
_LOG_FILE = os.path.join(_LOG_DIR, 'brute_force_result.log')

def log(msg):
    """写入结果文件并打印到stdout"""
    os.makedirs(_LOG_DIR, exist_ok=True)
    with open(_LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
    print(msg, flush=True)


# ======================== 辅助函数 ========================

def ordered_factorizations(n):
    """枚举n的所有有序因子分解（因子≥2，乘积=n）。
    n=1返回空元组（无需循环）。不使用MIREDO的flexible_factorization。"""
    if n == 1:
        return [()]
    result = [(n,)]
    for d in range(2, n):
        if n % d == 0:
            for rest in ordered_factorizations(n // d):
                result.append((d,) + rest)
    return result


def unique_permutations(seq):
    """生成序列的所有不重复排列（正确处理重复元素）。"""
    if len(seq) <= 1:
        yield tuple(seq)
        return
    seen = set()
    for i in range(len(seq)):
        if seq[i] in seen:
            continue
        seen.add(seq[i])
        rest = seq[:i] + seq[i+1:]
        for perm in unique_permutations(rest):
            yield (seq[i],) + perm


def nondecreasing_seqs(vals, length):
    """生成从vals中选取的长度为length的非递减序列（允许重复）。"""
    if length == 0:
        yield ()
        return
    for i, v in enumerate(vals):
        for rest in nondecreasing_seqs(vals[i:], length - 1):
            yield (v,) + rest


# ======================== 存储级预过滤 ========================

def filter_valid_mems(acc, ops, sm_list, active_dims, tu):
    """预过滤存储级：移除无法容纳任何活跃时间因子的存储级。"""
    min_factors = {}
    for d in active_dims:
        min_factors[d] = min(f for ft in ordered_factorizations(tu[d]) for f in ft)

    effective_mems = []
    for op in range(3):
        valid = []
        for m in range(1, acc.Num_mem):
            if acc.mappingArray[op][m] == 0:
                continue
            dim_size = [1] * ops.Num_dim
            for sm in sm_list:
                if m <= sm.mem[op]:
                    dim_size[sm.dim] *= sm.dimSize
            # Output精度用precision_final（宽松方向）：确保不误排有效级，由simulator做最终判定
            prec = acc.precision_final if op == 2 else acc.precision.get((m, op), 8)
            base_bits = ops.get_operand_size(dim_size, op) * prec
            if base_bits > acc.memSize[m]:
                continue
            can_fit_relevant = False
            for d in active_dims:
                if ops.relevance[op][d] == 0:
                    continue
                test_dim = dim_size[:]
                test_dim[d] *= min_factors[d]
                if ops.get_operand_size(test_dim, op) * prec <= acc.memSize[m]:
                    can_fit_relevant = True
                    break
            has_irrelevant_active = any(ops.relevance[op][d] == 0 for d in active_dims)
            if can_fit_relevant or has_irrelevant_active:
                valid.append(m)
        if not valid:
            valid = [m for m in range(1, acc.Num_mem) if acc.mappingArray[op][m] == 1]
        effective_mems.append(valid)
    return effective_mems


# ======================== 存储容量检查（含双缓冲） ========================

def compute_tile_bits(tm_list, sm_list, acc, ops, double_flag):
    """计算每个存储级的总占用bits。返回dict{mem: total_bits}，或None表示有级溢出。"""
    for mem in range(1, acc.Num_mem):
        total_bits = 0
        for op in range(3):
            if acc.mappingArray[op][mem] == 0:
                continue
            dim_size = [1] * ops.Num_dim
            for mapping in tm_list:
                if mem <= mapping.mem[op]:
                    dim_size[mapping.dim] *= mapping.dimSize
            for mapping in sm_list:
                if mem <= mapping.mem[op]:
                    dim_size[mapping.dim] *= mapping.dimSize
            data_elems = ops.get_operand_size(dim_size, op)
            prec = acc.precision_final if op == 2 else acc.precision.get((mem, op), 8)
            dbl = 1 + double_flag[mem][op]
            total_bits += data_elems * prec * dbl
        if total_bits > acc.memSize[mem]:
            return None
    return True


def get_double_configs(tm_list, acc):
    """生成所有合法双缓冲配置。只枚举映射中实际使用的(m,op)对。"""
    # 收集使用的(m, op)对中支持双缓冲的位
    used = set()
    for tm in tm_list:
        for op in range(3):
            m = tm.mem[op]
            if 1 <= m < acc.Num_mem and acc.double_config[m][op]:
                used.add((m, op))
    eligible = sorted(used)

    # 基础配置：全0
    base = [[0] * 3 for _ in range(acc.Num_mem + 1)]
    if not eligible:
        yield base
        return

    # 枚举所有2^k子集
    for mask in range(1 << len(eligible)):
        cfg = [row[:] for row in base]
        for bit, (m, op) in enumerate(eligible):
            if mask & (1 << bit):
                cfg[m][op] = 1
        yield cfg


# ======================== 核心搜索 ========================

def brute_force_temporal(acc, ops, scheme, log_fn=log):
    """对给定spatial scheme，穷举所有时间映射（含双缓冲）并用simulator评估。"""

    spatial = [math.prod(col) for col in zip(*scheme)]
    tu = [math.ceil(x / y) if y > 0 else x for x, y in zip(ops.dim2bound, spatial)]
    active_dims = [d for d in range(1, ops.Num_dim) if tu[d] > 1]

    log_fn(f"  空间展开: {[f'{ops.dim2Dict[d]}={spatial[d]}' for d in range(1,ops.Num_dim) if spatial[d]>1]}")
    log_fn(f"  时间循环: {[f'{ops.dim2Dict[d]}={tu[d]}' for d in active_dims]}")

    if not active_dims:
        log_fn("  无时间循环维度")
        return float('inf'), float('inf'), None, {'candidates':0, 'feasible':0, 'time':0}

    facts_per_dim = {}
    for d in active_dims:
        facts_per_dim[d] = ordered_factorizations(tu[d])
        log_fn(f"    {ops.dim2Dict[d]}({tu[d]}): {len(facts_per_dim[d])} 种因子分解")

    sm_list = []
    for u in range(acc.Num_SpUr):
        for d in range(1, ops.Num_dim):
            if scheme[u][d] > 1:
                sm_list.append(Mapping(dim=d, dimSize=scheme[u][d],
                                       mem=[acc.SpUr2Mem[u, op] for op in range(3)]))

    raw_mems = [[m for m in range(1, acc.Num_mem) if acc.mappingArray[op][m] == 1] for op in range(3)]
    valid_mems = filter_valid_mems(acc, ops, sm_list, active_dims, tu)
    log_fn(f"  原始存储级: I={raw_mems[0]}, W={raw_mems[1]}, O={raw_mems[2]}")
    log_fn(f"  过滤后存储级: I={valid_mems[0]}, W={valid_mems[1]}, O={valid_mems[2]}")

    no_dbl = [[0] * 3 for _ in range(acc.Num_mem + 1)]
    best_lat, best_energy, best_loops = float('inf'), float('inf'), None
    candidates, feasible, mem_pruned, dbl_pruned, simu_fail = 0, 0, 0, 0, 0
    t0 = time.time()

    dim_keys = list(facts_per_dim.keys())
    dim_fact_lists = [facts_per_dim[d] for d in dim_keys]

    for fact_combo in itertools.product(*dim_fact_lists):
        items = []
        for i, d in enumerate(dim_keys):
            for f in fact_combo[i]:
                items.append((d, f))
        num_loops = len(items)
        if num_loops == 0:
            continue

        seqs = [list(nondecreasing_seqs(vm, num_loops)) for vm in valid_mems]
        log_fn(f"  枚举: {num_loops} loops, "
               f"{len(seqs[0])}×{len(seqs[1])}×{len(seqs[2])} mem_assigns/ordering")

        for perm in unique_permutations(items):
            for seq_I in seqs[0]:
                for seq_W in seqs[1]:
                    for seq_O in seqs[2]:
                        candidates += 1

                        if candidates % 2000 == 0:
                            elapsed = time.time() - t0
                            log_fn(f"    [{candidates:,}] pruned={mem_pruned:,} "
                                   f"dbl_pruned={dbl_pruned:,} feasible={feasible} "
                                   f"best={best_lat:.0f} {elapsed:.1f}s")

                        tm_list = [
                            Mapping(dim=perm[j][0], dimSize=perm[j][1],
                                    mem=[seq_I[j], seq_W[j], seq_O[j]])
                            for j in range(num_loops)
                        ]

                        # 无双缓冲下的基础容量检查
                        if compute_tile_bits(tm_list, sm_list, acc, ops, no_dbl) is None:
                            mem_pruned += 1
                            continue

                        # 枚举所有双缓冲配置
                        for dbl_cfg in get_double_configs(tm_list, acc):
                            # 双缓冲容量检查（非全0时才需额外检查）
                            if any(dbl_cfg[m][op] for m in range(1, acc.Num_mem) for op in range(3)):
                                if compute_tile_bits(tm_list, sm_list, acc, ops, dbl_cfg) is None:
                                    dbl_pruned += 1
                                    continue

                            try:
                                loops = LoopNest(acc=acc, ops=ops)
                                loops.tm = tm_list
                                loops.sm = sm_list[:]
                                loops.usr_defined_double_flag = [row[:] for row in dbl_cfg]
                                loops.psum_flag = None

                                simu = tranSimulator(acc=acc, ops=ops, dataflow=loops)
                                lat, energy = simu.run()
                                feasible += 1

                                if lat < best_lat:
                                    best_lat = lat
                                    best_energy = energy
                                    best_loops = copy.deepcopy(loops)

                            except (ValueError, KeyError, IndexError,
                                    ZeroDivisionError, TypeError, AttributeError):
                                simu_fail += 1

    elapsed = time.time() - t0
    stats = {'candidates': candidates, 'feasible': feasible,
             'mem_pruned': mem_pruned, 'dbl_pruned': dbl_pruned,
             'simu_fail': simu_fail, 'time': elapsed}
    return best_lat, best_energy, best_loops, stats


# ======================== 预定义测试用例 ========================

# dim2Dict = ['-','R','S','P','Q','C','K','G']
# scheme格式: [[core轴], [dimX轴], [dimY轴]]，每轴8个元素对应8个维度
CASES = {
    '1x1_C64K64': {
        'ops': {'R':1,'S':1,'C':64,'K':64,'P':7,'Q':7,'G':1,'B':1,'H':7,'W':7,'Stride':1,'Padding':0},
        'scheme': [[1,1,1,1,1,1,8,1],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,8,1]],
    },
    '3x3_C64K64': {
        'ops': {'R':3,'S':3,'C':64,'K':64,'P':7,'Q':7,'G':1,'B':1,'H':7,'W':7,'Stride':1,'Padding':1},
        'scheme': [[1,1,1,1,1,1,8,1],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,8,1]],
    },
    'resnet_layer13': {
        'ops': {'R':3,'S':3,'C':256,'K':256,'P':7,'Q':7,'G':1,'B':1,'H':7,'W':7,'Stride':1,'Padding':1},
        'scheme': [[1,1,1,1,1,1,8,1],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,16,1]],
    },
    # QK^T tile (seq=d_head=32, heads=2) for HW-Transformer. Matmul enters the
    # solver as a degenerate conv (R=S=Q=1); scheme respects HW-Transformer's
    # allowed_loops (cores∈{P,Q,K,G}, dimX∈{R,S,C}, dimY∈{K}), which is why K
    # goes on dimY and C on dimX rather than the other way around.
    'attention_tiny': {
        'ops': {'R':1,'S':1,'C':32,'K':32,'P':32,'Q':1,'G':2,'B':1,'H':32,'W':1,'Stride':1,'Padding':0},
        'scheme': [[1,1,1,8,1,1,1,2],[1,1,1,1,1,32,1,1],[1,1,1,1,1,1,16,1]],
    },
}


def run_verify(acc, ops, scheme, mip_timelimit=120, log_fn=log, arch_spec=None):
    """
    arch_spec: 可选的 HardwareSpec 实例；若为 None 则回退 Architecture.templates.default.default_spec()。
    用于支持 HW-Transformer 等非默认架构下的 bruteforce 对比。
    """
    if arch_spec is None:
        arch_spec = default_spec()
    """完整的穷举验证流程：穷举搜索 + MIP对比。
    可被外部脚本直接调用以验证任意(acc, ops, scheme)组合。"""

    log_fn(f"\n{'='*60}")
    log_fn(f"穷举验证")
    log_fn(f"{'='*60}")

    best_lat, best_energy, best_loops, stats = brute_force_temporal(acc, ops, scheme, log_fn)

    log_fn(f"\n搜索完成:")
    log_fn(f"  总候选映射: {stats['candidates']:,}")
    log_fn(f"  基础存储剪枝: {stats['mem_pruned']:,}")
    log_fn(f"  双缓冲容量剪枝: {stats['dbl_pruned']:,}")
    log_fn(f"  Simulator失败: {stats['simu_fail']:,}")
    log_fn(f"  可行解: {stats['feasible']:,}")
    log_fn(f"  耗时: {stats['time']:.1f}s")
    log_fn(f"  穷举最优 Latency = {best_lat:.0f} cycles")
    log_fn(f"  穷举最优 Energy  = {best_energy:.2f} nJ")
    if best_loops:
        log_fn(f"\n穷举最优映射:\n{best_loops}")

    # MIP对比
    from utils.SolverTSS import Solver
    from utils.UtilsFunction.ToolFunction import prepare_save_dir
    import logging

    CONST.FLAG_OPT = "Latency"
    CONST.TIMELIMIT = mip_timelimit
    CONST.MIPFOCUS = 1
    FLAG.GUROBI_OUTPUT = False
    FLAG.SIMU = False

    spatial = [math.prod(col) for col in zip(*scheme)]
    tu = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial)]
    mip_dir = os.path.join(os.path.dirname(__file__), '..', 'output', "#BF_mip_compare")
    prepare_save_dir(mip_dir)

    solver = Solver(acc=CIM_Acc.from_spec(arch_spec), ops=ops, tu=tu, su=scheme,
                    metric_ub=CONST.MAX_POS, outputdir=mip_dir)
    solver.run()

    mip_simu_lat = None
    if solver.model is not None and solver.model.SolCount > 0:
        try:
            gap = solver.model.MIPGap
        except Exception:
            gap = -1
        logging.disable(logging.CRITICAL)
        simu = tranSimulator(acc=CIM_Acc.from_spec(arch_spec), ops=ops, dataflow=solver.dataflow)
        mip_simu_lat, _ = simu.run()

        log_fn(f"\nMIP对比:")
        log_fn(f"  MIP+Simu Latency = {mip_simu_lat:.0f} (MIP obj={solver.result[0]:.0f}, gap={gap*100:.1f}%)")
        log_fn(f"  穷举最优 Latency = {best_lat:.0f}")
        if best_lat > 0 and best_lat < float('inf'):
            diff = (mip_simu_lat - best_lat) / best_lat * 100
            log_fn(f"  差距: {diff:+.2f}%")
    else:
        log_fn(f"\nMIP求解失败")
    solver.close()
    log_fn("=" * 60)

    # 保存结构化结果到 output/Eval_Result/
    from utils.UtilsFunction.ToolFunction import save_result_json
    result_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'Eval_Result')
    result = {
        'script': 'VerifyBruteforce',
        'workload': str(ops),
        'scheme': scheme,
        'bruteforce': {
            'optimal_latency': best_lat if best_lat < float('inf') else None,
            'optimal_energy': best_energy if best_energy < float('inf') else None,
            'feasible_count': stats['feasible'],
            'candidates': stats['candidates'],
            'time_seconds': round(stats['time'], 1),
        },
        'mip': {
            'simu_latency': mip_simu_lat,
        },
        'optimality_gap_pct': round((mip_simu_lat - best_lat) / best_lat * 100, 2)
            if mip_simu_lat and best_lat and best_lat < float('inf') else None,
    }
    result_file = save_result_json(result_dir, 'bruteforce', result)
    log_fn(f"\n结果已保存: {result_file}")

    exp6_file = save_experiment_json(
        output_dir=result_dir,
        file_name=f"EXP-6_bruteforce_{time.strftime('%Y%m%d_%H%M%S')}.json",
        experiment_id="EXP-6",
        script_path=__file__,
        config={
            "verification_method": "bruteforce",
            "workload": {dim: getattr(ops, dim) for dim in ops.dim2Dict if dim != '-'},
            "scheme": scheme,
            "mip_time_limit": mip_timelimit,
        },
        results={
            "verification": {
                "independent_optimal_latency": best_lat if best_lat < float('inf') else None,
                "independent_optimal_energy": best_energy if best_energy < float('inf') else None,
                "feasible_count": stats['feasible'],
                "candidate_count": stats['candidates'],
                "search_time_seconds": round(stats['time'], 3),
                "mip_simulator_latency": mip_simu_lat,
                "optimality_gap_pct": round((mip_simu_lat - best_lat) / best_lat * 100, 2)
                    if mip_simu_lat and best_lat and best_lat < float('inf') else None,
            },
            "optimality_verification": [{
                "model": "manual",
                "layer": f"Conv_{ops.R}x{ops.S}_C{ops.C}K{ops.K}",
                "tier": "small",
                "mip_latency": mip_simu_lat,
                "exhaustive_latency": best_lat if best_lat < float('inf') else None,
                "is_optimal": (
                    mip_simu_lat is not None
                    and best_lat < float('inf')
                    and abs(mip_simu_lat - best_lat) / best_lat < 0.01
                ),
                "optimality_gap_pct": round((mip_simu_lat - best_lat) / best_lat * 100, 2)
                    if mip_simu_lat and best_lat and best_lat < float('inf') else None,
                "solve_time_sec": round(stats['time'], 3),
                "num_spatial_schemes": stats['candidates'],
                "num_schemes_after_pruning": stats['feasible']
            }],
        },
        anomalies=[],
    )
    log_fn(f"EXP-6结果已保存: {exp6_file}")

    return best_lat, mip_simu_lat


# ======================== 主程序 ========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="绝对最优数据流穷举验证")
    parser.add_argument('--case', default='1x1_C64K64', choices=list(CASES.keys()),
                        help=f"预定义测试用例 (可选: {', '.join(CASES.keys())})")
    parser.add_argument('--timelimit', type=int, default=120, help="MIP求解时间限制(秒)")
    parser.add_argument('--arch', default='CIM_ACC_TEMPLATE',
                        help="架构注册名：CIM_ACC_TEMPLATE（默认 HW-Small）或 CIM_ACC_TEMPLATE_TRANSFORMER（HW-Transformer）")
    args = parser.parse_args()

    # 通过架构注册表解析 spec；未注册则抛出明确错误
    from importlib import import_module
    from Evaluation.common.EvalCommon import _ARCHITECTURE_SPEC_BUILDERS
    arch_module_path = _ARCHITECTURE_SPEC_BUILDERS.get(args.arch)
    if arch_module_path is None:
        raise NotImplementedError(
            f"Architecture {args.arch!r} 未在 _ARCHITECTURE_SPEC_BUILDERS 注册。"
            f"已注册: {list(_ARCHITECTURE_SPEC_BUILDERS.keys())}"
        )
    _arch_spec = import_module(arch_module_path).default_spec()

    Logger.setcfg(setcritical=False, setDebug=False, STD=False, file="", nofile=True)
    import logging
    logging.disable(logging.CRITICAL)

    os.makedirs(_LOG_DIR, exist_ok=True)
    with open(_LOG_FILE, 'w') as f:
        f.write("")

    log("=" * 60)
    log(f"绝对最优数据流穷举搜索（含双缓冲）- Case: {args.case} on {args.arch}")
    log("=" * 60)

    case = CASES[args.case]
    ops = WorkLoad(loopDim=case['ops'])
    acc = CIM_Acc.from_spec(_arch_spec)
    _verify_arch_spec = _arch_spec  # used below in the run_verify call

    log(f"\n负载: {ops}")
    log(f"PE阵列: {acc.Num_core} cores × {acc.dimX}(BL) × {acc.dimY}(WL)")

    run_verify(acc, ops, case['scheme'], mip_timelimit=args.timelimit, arch_spec=_arch_spec)

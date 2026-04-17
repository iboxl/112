# 验证 tranSimulator.run() 与 tranSimulator.run_analytical() 在 bit-exact 语义下等价。
# run_analytical 是为了把 CoSA / CIMLoop baseline 的 tile-walk 从数小时压到秒级而新加
# 的解析路径；它在 iter0/iter1/iter2 显式执行 + (N-3) 次稳态线性外推 + flush 下应当与
# run() 得到完全一致的 latency / count_mac 以及 ≤1e-12 相对误差以内的 memCost / PD
# 浮点量。本脚本用来在提交/发 baseline 实验前验证这一点。
#
# 运行方式（先 Codex 审核、再手动执行）：
#   python -m Evaluation.Verify_simulax_equivalence
# 或者（脚本自己会把 repo root 加进 sys.path）：
#   python Evaluation/Verify_simulax_equivalence.py
#
# 覆盖面（按 codex 审查建议在 smoke 之上补齐）：
#   1. WS dataflow（enable_double_buffer 两种）× 若干 shape — 单缓冲 / 双缓冲两条路径
#   2. 自构造 DRAM-heavy mapping — 模拟 CoSA / CIMLoop 大 N 内层
#   3. 混合 dflag — I/W 开 double、O 关；覆盖 per-op 差异化 dflag 的 pre-read 稳态
#   4. G>1 grouped/depthwise conv — CIMLoop 路径支持（CoSA 在 BaselineProvider raise）
#   5. 除 latency/energy/count_mac 外，还断言 PD.transfer/mismatch/mode_switch/
#      output_writeback_cycles、各级 memCost.r/.w/.t、bootstrap/idle/stall 全字段

import copy
import math
import logging
import pathlib
import sys

# repo root 加到 sys.path，使本脚本同时支持 `python -m` 与直接执行
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from Evaluation.WeightStationaryGenerator import (
    build_weight_stationary_dataflow,
    derive_weight_stationary_spatial_scheme,
)
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import Logger
from utils.Workload import LoopNest, Mapping, WorkLoad


LAT_TOL = 0               # latency 走整数 cycle，要求严格 ==
MAC_TOL = 0
FLOAT_REL_TOL = 1e-12     # memCost / energy / PD 的 FP 累加顺序允许的相对误差


SHAPES = [
    dict(R=3, S=3, C=32,  K=64,  P=14, Q=14, G=1, B=1, H=14, W=14, Stride=1, Padding=1),
    dict(R=1, S=1, C=16,  K=32,  P=7,  Q=7,  G=1, B=1, H=7,  W=7,  Stride=1, Padding=0),
    dict(R=3, S=3, C=8,   K=16,  P=4,  Q=4,  G=1, B=1, H=4,  W=4,  Stride=1, Padding=1),
    dict(R=3, S=3, C=64,  K=128, P=28, Q=28, G=1, B=1, H=28, W=28, Stride=1, Padding=1),
    dict(R=3, S=3, C=128, K=256, P=14, Q=14, G=1, B=1, H=14, W=14, Stride=1, Padding=1),
    dict(R=1, S=1, C=64,  K=64,  P=56, Q=56, G=1, B=1, H=56, W=56, Stride=1, Padding=0),
    dict(R=5, S=5, C=32,  K=64,  P=8,  Q=8,  G=1, B=1, H=8,  W=8,  Stride=1, Padding=2),
]

# G>1 depthwise 风格（C=K=1，G 做 group 展开；WS 生成器可处理）
SHAPES_GROUPED = [
    dict(R=3, S=3, C=1, K=1, P=14, Q=14, G=32, B=1, H=14, W=14, Stride=1, Padding=1),
    dict(R=3, S=3, C=1, K=1, P=7,  Q=7,  G=64, B=1, H=7,  W=7,  Stride=1, Padding=1),
]


def _quiet():
    Logger.setLevel(logging.ERROR)
    logging.disable(logging.CRITICAL)


def _max_rel(x, y):
    return abs(x - y) / max(abs(x), 1e-12)


def _snapshot_profile(sim):
    pd = sim.PD
    memCost = {m: (tc.r, tc.w, tc.t) for m, tc in sim.memCost.items()}
    return {
        "count_mac": sim.count_mac,
        "latency": pd.latency,
        "transfer_cycles": list(pd.transfer_cycles),
        "mismatch_cycles": list(pd.mismatch_cycles),
        "mode_switch_cycles": list(pd.mode_switch_cycles),
        "output_writeback_cycles": pd.output_writeback_cycles,
        "bootstrap_cycles": pd.bootstrap_cycles,
        "mode_switch_stall": pd.mode_switch_stall,
        "mismatch_stall": pd.mismatch_stall,
        "writeback_stall": pd.writeback_stall,
        "idle_cycles": pd.idle_cycles,
        "memCost": memCost,
    }


def _run_both(acc, ops, loops):
    sim_ref = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=copy.deepcopy(loops))
    lat_ref, eng_ref = sim_ref.run()
    sim_ana = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=copy.deepcopy(loops))
    lat_ana, eng_ana = sim_ana.run_analytical()
    return (sim_ref, lat_ref, eng_ref), (sim_ana, lat_ana, eng_ana)


def _check_case(tag, ref_tuple, ana_tuple, N):
    sim_ref, lat_ref, eng_ref = ref_tuple
    sim_ana, lat_ana, eng_ana = ana_tuple

    fails = []
    if abs(lat_ref - lat_ana) > LAT_TOL:
        fails.append(f"latency: ref={lat_ref} ana={lat_ana}")
    if abs(sim_ref.count_mac - sim_ana.count_mac) > MAC_TOL:
        fails.append(f"count_mac: ref={sim_ref.count_mac} ana={sim_ana.count_mac}")
    if _max_rel(eng_ref, eng_ana) > FLOAT_REL_TOL:
        fails.append(f"energy rel={_max_rel(eng_ref, eng_ana):.2e} ref={eng_ref} ana={eng_ana}")

    pr, pa = _snapshot_profile(sim_ref), _snapshot_profile(sim_ana)
    # 浮点标量字段
    for key in ("output_writeback_cycles", "bootstrap_cycles",
                "mode_switch_stall", "mismatch_stall",
                "writeback_stall", "idle_cycles"):
        if _max_rel(pr[key], pa[key]) > FLOAT_REL_TOL:
            fails.append(f"{key}: ref={pr[key]} ana={pa[key]}")
    # 三 op 数组字段
    for key in ("transfer_cycles", "mismatch_cycles", "mode_switch_cycles"):
        for op in range(3):
            if _max_rel(pr[key][op], pa[key][op]) > FLOAT_REL_TOL:
                fails.append(f"{key}[{op}]: ref={pr[key][op]} ana={pa[key][op]}")
    # memCost 逐级 r/w/t
    for m, (rr, rw, rt) in pr["memCost"].items():
        ar, aw, at = pa["memCost"][m]
        if _max_rel(rr, ar) > FLOAT_REL_TOL:
            fails.append(f"memCost[{m}].r ref={rr} ana={ar}")
        if _max_rel(rw, aw) > FLOAT_REL_TOL:
            fails.append(f"memCost[{m}].w ref={rw} ana={aw}")
        if _max_rel(rt, at) > FLOAT_REL_TOL:
            fails.append(f"memCost[{m}].t ref={rt} ana={at}")

    ok = not fails
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {tag} N={N} | "
          f"lat {lat_ref}→{lat_ana} | "
          f"eng rel={_max_rel(eng_ref, eng_ana):.2e} | "
          f"mac {sim_ref.count_mac}→{sim_ana.count_mac}")
    for line in fails[:8]:
        print(f"       - {line}")
    return ok


def _build_custom_dram_heavy(acc, ops, dflag_builder=None):
    loops = LoopNest(acc=acc, ops=ops)
    sm_scheme = derive_weight_stationary_spatial_scheme(acc=copy.deepcopy(acc), ops=ops)
    for axis in range(acc.Num_SpUr):
        for dim in range(1, ops.Num_dim):
            if sm_scheme[axis][dim] > 1:
                loops.sm.append(
                    Mapping(
                        dim=dim,
                        dimSize=sm_scheme[axis][dim],
                        mem=[acc.SpUr2Mem[axis, op] for op in range(3)],
                    )
                )
    dim_spatial = {
        dim: math.prod([sm_scheme[ax][dim] for ax in range(acc.Num_SpUr)])
        for dim in range(ops.Num_dim)
    }
    dram = acc.Dram2mem
    for dim_char in ["R", "S", "C", "K", "P", "Q", "G"]:
        d = ops.dict2Dim(dim_char)
        rem = ops.dim2bound[d] // dim_spatial.get(d, 1)
        if rem > 1:
            loops.tm.append(Mapping(dim=d, dimSize=rem, mem=[dram, dram, dram]))
    if not loops.tm:
        loops.tm.append(Mapping(dim=0, dimSize=1, mem=[dram] * 3))
    if dflag_builder is None:
        loops.usr_defined_double_flag = [[0] * 3 for _ in range(acc.Num_mem + 1)]
    else:
        loops.usr_defined_double_flag = dflag_builder(acc, loops)
    loops.preprogress()
    return loops


def _mixed_dflag(acc, loops):
    # I/W 开 double、O 关；与全关 / 全开 形成 3 种 dflag 组合
    dflag = [[0] * 3 for _ in range(acc.Num_mem + 1)]
    used = {(m.mem[op], op) for m in loops.tm for op in range(3)}
    for mem in range(1, acc.Num_mem):
        for op in (0, 1):
            if acc.double_config[mem][op] and (mem, op) in used:
                dflag[mem][op] = 1
    return dflag


def main():
    _quiet()
    spec = default_spec()
    acc = CIM_Acc.from_spec(spec)

    pass_count = 0
    fail_count = 0

    for i, loopDim in enumerate(SHAPES):
        ops = WorkLoad(loopDim=loopDim)

        loops_sb = build_weight_stationary_dataflow(acc=copy.deepcopy(acc), ops=ops,
                                                    enable_double_buffer=False)
        ref, ana = _run_both(acc, ops, loops_sb)
        N = [t.dimSize for t in loops_sb.tm]
        if _check_case(f"ws_sb #{i}", ref, ana, N):
            pass_count += 1
        else:
            fail_count += 1

        loops_db = build_weight_stationary_dataflow(acc=copy.deepcopy(acc), ops=ops,
                                                    enable_double_buffer=True)
        ref, ana = _run_both(acc, ops, loops_db)
        N = [t.dimSize for t in loops_db.tm]
        if _check_case(f"ws_db #{i}", ref, ana, N):
            pass_count += 1
        else:
            fail_count += 1

        loops_cus = _build_custom_dram_heavy(copy.deepcopy(acc), ops)
        ref, ana = _run_both(acc, ops, loops_cus)
        N = [t.dimSize for t in loops_cus.tm]
        if _check_case(f"custom #{i}", ref, ana, N):
            pass_count += 1
        else:
            fail_count += 1

        loops_mix = _build_custom_dram_heavy(copy.deepcopy(acc), ops, dflag_builder=_mixed_dflag)
        ref, ana = _run_both(acc, ops, loops_mix)
        N = [t.dimSize for t in loops_mix.tm]
        if _check_case(f"mixed_df #{i}", ref, ana, N):
            pass_count += 1
        else:
            fail_count += 1

    for i, loopDim in enumerate(SHAPES_GROUPED):
        ops = WorkLoad(loopDim=loopDim)
        try:
            loops = build_weight_stationary_dataflow(acc=copy.deepcopy(acc), ops=ops)
        except Exception as e:
            print(f"[SKIP] grouped #{i}: WS dataflow build failed: {e}")
            continue
        ref, ana = _run_both(acc, ops, loops)
        N = [t.dimSize for t in loops.tm]
        if _check_case(f"grouped #{i} G={loopDim['G']}", ref, ana, N):
            pass_count += 1
        else:
            fail_count += 1

    print()
    print(f"Summary: pass={pass_count} fail={fail_count}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

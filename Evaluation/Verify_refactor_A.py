# 静态字段一致性测试（Phase 3）
# 验证 CIM_Acc(old) ↔ CIM_Acc.from_spec(new) 公开字段逐字段相等
#   整数/字符串/tuple：严格相等
#   浮点标量/列表：|diff| < 1e-9
#   dict：按 key 递归比较
#
# 退出码：0 = 全过；非 0 = 有差异

from __future__ import annotations

import math
import sys
from typing import Any

from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
from Evaluation.Zigzag_imc.zigzag_adapter import to_zigzag_accelerator


TOL = 1e-9


_MISSING = object()  # sentinel — 区分字段缺失和值为 None


def _is_strict_numeric(x) -> bool:
    # bool 是 int 的子类；显式排除，避免 True==1 被当成数值通过
    return type(x) is int or type(x) is float


def _iter_diffs(old: Any, new: Any, path: str):
    # 返回差异字符串列表；空 = 无差异
    diffs = []

    # 严格类型检查：bool 与 int 不互通
    if type(old) is not type(new):
        if _is_strict_numeric(old) and _is_strict_numeric(new):
            if not math.isclose(old, new, rel_tol=0, abs_tol=TOL):
                diffs.append(f"{path}: numeric {old!r} vs {new!r}")
            return diffs
        diffs.append(f"{path}: type mismatch {type(old).__name__} vs {type(new).__name__}")
        return diffs

    if type(old) is float:
        if not math.isclose(old, new, rel_tol=0, abs_tol=TOL):
            diffs.append(f"{path}: float {old!r} vs {new!r}  diff={new-old!r}")
    elif type(old) in (int, str, bool) or old is None:
        if old != new:
            diffs.append(f"{path}: scalar {old!r} vs {new!r}")
    elif isinstance(old, (list, tuple)):
        if len(old) != len(new):
            diffs.append(f"{path}: length {len(old)} vs {len(new)}")
            return diffs
        for i, (a, b) in enumerate(zip(old, new)):
            diffs += _iter_diffs(a, b, f"{path}[{i}]")
    elif isinstance(old, dict):
        keys_old = set(old.keys())
        keys_new = set(new.keys())
        if keys_old != keys_new:
            diffs.append(f"{path}: dict keys mismatch old-only={keys_old-keys_new} new-only={keys_new-keys_old}")
        for k in keys_old & keys_new:
            diffs += _iter_diffs(old[k], new[k], f"{path}[{k!r}]")
    else:
        if old != new:
            diffs.append(f"{path}: other {old!r} vs {new!r}")
    return diffs


# 对比的 public 字段清单（非 _mem2dict 这类内部）
PUBLIC_FIELDS = [
    "Num_mem",
    "Num_core",
    "Num_SpUr",
    "dimX",
    "dimY",
    "t_MAC",
    "IReg2mem",
    "Macro2mem",
    "OReg2mem",
    "Dram2mem",
    "Global2mem",
    "lastMem",
    "double_Macro",
    "double_config",
    "shareMemory",
    "nxtMem",
    "mappingArray",
    "mappingRule",
    "SpUnrolling",
    "SpUr2Mem",
    "minBW",
    "memSize",
    "bw",
    "cost_r",
    "cost_w",
    "cost_ActMacro",
    "leakage_per_cycle",
    "precision",
    "precision_final",
    "precision_psum",
    "_mem2dict",
]


def main() -> int:
    print("[Verify_A] building old CIM_Acc via legacy __init__(core) from adapter-built accelerator")
    old = CIM_Acc(to_zigzag_accelerator(default_spec()).cores[0])

    print("[Verify_A] building new CIM_Acc.from_spec(default_spec())")
    new = CIM_Acc.from_spec(default_spec())

    all_diffs = []
    for field_name in PUBLIC_FIELDS:
        old_v = getattr(old, field_name, _MISSING)
        new_v = getattr(new, field_name, _MISSING)
        if old_v is _MISSING and new_v is _MISSING:
            all_diffs.append(f"{field_name}: field missing on both sides")
            continue
        if old_v is _MISSING:
            all_diffs.append(f"{field_name}: missing on old, new={new_v!r}")
            continue
        if new_v is _MISSING:
            all_diffs.append(f"{field_name}: missing on new, old={old_v!r}")
            continue
        all_diffs += _iter_diffs(old_v, new_v, field_name)

    # 自动覆盖检查：列出 old/new 都存在但未包含在 PUBLIC_FIELDS 中的 public 属性
    # source_spec 为 provenance 字段（legacy=None, from_spec=<spec>），非物理硬件，不参与比对
    SKIP_AUTO = {"source_spec"}
    def _public_attrs(obj):
        return {k for k in vars(obj).keys() if not k.startswith("_") and k not in SKIP_AUTO}
    auto_old = _public_attrs(old)
    auto_new = _public_attrs(new)
    missing_in_list = (auto_old & auto_new) - set(PUBLIC_FIELDS)
    # _mem2dict 是 _ 开头故 _public_attrs 已过滤，但我们已显式加入 PUBLIC_FIELDS
    if missing_in_list:
        all_diffs.append(
            f"PUBLIC_FIELDS 未覆盖这些 public 属性: {sorted(missing_in_list)}"
        )
    only_old = auto_old - auto_new
    only_new = auto_new - auto_old
    if only_old:
        all_diffs.append(f"属性仅在 old 上存在: {sorted(only_old)}")
    if only_new:
        all_diffs.append(f"属性仅在 new 上存在: {sorted(only_new)}")

    # mem2dict() method check
    for i in range(old.Num_mem + 1):
        if old.mem2dict(i) != new.mem2dict(i):
            all_diffs.append(f"mem2dict({i}): {old.mem2dict(i)!r} vs {new.mem2dict(i)!r}")

    if all_diffs:
        print(f"[Verify_A] FAIL — {len(all_diffs)} diff(s):")
        for d in all_diffs:
            print("  -", d)
        return 1
    print(f"[Verify_A] PASS — all {len(PUBLIC_FIELDS)} fields match (tol {TOL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

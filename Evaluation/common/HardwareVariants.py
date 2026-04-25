# Sensitivity 硬件变体：输入输出均为 HardwareSpec
# 调用方（RunSensitivity.py）从 spec 派生 CIM_Acc（via CIM_Acc.from_spec）与 ZigZag acc（via
# zigzag_adapter.to_zigzag_accelerator），保证两侧同源。
# Leakage / memory cost 在参数变化后通过 CACTI/DRAM-static 重跑重建。

from __future__ import annotations

from dataclasses import replace
from typing import List

from Architecture.HardwareSpec import HardwareSpec, MemoryLevelSpec
from utils.Cacti_wrapper.EvalCacti import cacti_power, dram_static


def _mem_index(spec: HardwareSpec, name: str) -> int:
    for i, m in enumerate(spec.memory_hierarchy):
        if m.name == name:
            return i
    raise KeyError(f"Memory level {name!r} not in spec")


def _recompute_memory_cost_pJ(spec: HardwareSpec, mem_name: str) -> MemoryLevelSpec:
    """重跑 CACTI（或 DRAM 常量），返回更新后的 MemoryLevelSpec 副本。
    tech_node 从 spec.macro.tech_params.tech_node 取，使变体与 default spec 的 r/w 数字同源。
    """
    level = spec.memory_by_name(mem_name)
    size_bytes = max(1, int(level.size_bits / 8))
    bw = max(8, int(level.r_bw_bits_per_cycle))
    if mem_name == "Dram":
        # DRAM 在 legacy 固定为 7.91 pJ/bit
        r_pb = 7.91
        w_pb = 7.91
    else:
        read_pj, write_pj, _, _ = cacti_power(
            tech_node=spec.macro.tech_params.tech_node,
            capacity_bytes=size_bytes,
            bitwidth_bits=bw,
        )
        r_pb = read_pj / bw
        w_pb = write_pj / bw
    return replace(level, r_cost_per_bit_pJ=r_pb, w_cost_per_bit_pJ=w_pb)


def _leakage_per_cycle_nJ(spec: HardwareSpec) -> float:
    """累加 leakage：DRAM + Global + (Output/Input_buffer) × cores + Macro(SRAM) × cores。
    OReg/IReg 不计入。SRAM 部分走本地 CACTI，tech_node 取自 spec，与 r/w 能耗同源。
    """
    tech = spec.macro.tech_params.tech_node

    def _sram_leak_pJ(mem_name: str) -> float:
        level = spec.memory_by_name(mem_name)
        _, _, leak_pJ, _ = cacti_power(
            tech_node=tech,
            capacity_bytes=max(1, int(level.size_bits / 8)),
            bitwidth_bits=max(8, int(level.r_bw_bits_per_cycle)),
        )
        return leak_pJ

    # DRAM
    dram = spec.memory_by_name("Dram")
    _, dram_leak_pJ = dram_static(capacity_bytes=dram.size_bits / 8, bus_width_bits=dram.r_bw_bits_per_cycle)
    total_pJ = dram_leak_pJ

    # Global
    total_pJ += _sram_leak_pJ("Global_buffer")

    # Output / Input buffer (per core)
    total_pJ += _sram_leak_pJ("Output_buffer") * spec.cores
    total_pJ += _sram_leak_pJ("Input_buffer") * spec.cores

    # Macro cell array: 按 legacy 口径用 size × dimX × dimY 的容量、dimY × W 的 bitwidth
    macro = spec.memory_by_name("Macro")
    W = spec.macro.precision.W
    _, _, macro_leak_pJ, _ = cacti_power(
        tech_node=tech,
        capacity_bytes=(macro.size_bits / 8) * spec.macro.dimX * spec.macro.dimY,
        bitwidth_bits=max(8, spec.macro.dimY * W),
    )
    total_pJ += macro_leak_pJ * spec.cores

    return total_pJ * 1e-3  # pJ → nJ


def _replace_memory_level(spec: HardwareSpec, mem_name: str, **updates) -> HardwareSpec:
    """返回一份 spec 副本，对 memory_hierarchy 中名为 mem_name 的层做 `replace(**updates)`。"""
    new_mems = []
    for m in spec.memory_hierarchy:
        if m.name == mem_name:
            new_mems.append(replace(m, **updates))
        else:
            new_mems.append(m)
    return replace(spec, memory_hierarchy=new_mems)


def _on_chip_buffer_names(spec: HardwareSpec) -> List[str]:
    # 非 Dram / 非 Reg 的片上 SRAM：Global_buffer / Output_buffer / Input_buffer
    return ["Global_buffer", "Output_buffer", "Input_buffer"]


def set_core_count(spec: HardwareSpec, count: int) -> HardwareSpec:
    count = max(1, int(count))
    new_axes = []
    for a in spec.macro.spatial_axes:
        if a.name == "cores":
            new_axes.append(replace(a, size=count))
        else:
            new_axes.append(a)
    new_macro = replace(spec.macro, spatial_axes=new_axes)
    new_spec = replace(spec, cores=count, macro=new_macro)
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_gbuf_capacity_kb(spec: HardwareSpec, gbuf_kb: int) -> HardwareSpec:
    gbuf_kb = max(1, int(gbuf_kb))
    target_gbuf_bits = gbuf_kb * 8 * 1024
    old_gbuf = spec.memory_by_name("Global_buffer")
    factor = target_gbuf_bits / max(1, old_gbuf.size_bits)

    new_spec = spec
    for name in _on_chip_buffer_names(spec):
        old = new_spec.memory_by_name(name)
        new_size = max(8, int(round(old.size_bits * factor)))
        new_spec = _replace_memory_level(new_spec, name, size_bits=new_size)
        updated_level = _recompute_memory_cost_pJ(new_spec, name)
        new_spec = _replace_memory_level(
            new_spec, name,
            r_cost_per_bit_pJ=updated_level.r_cost_per_bit_pJ,
            w_cost_per_bit_pJ=updated_level.w_cost_per_bit_pJ,
        )
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_gbuf_bandwidth(spec: HardwareSpec, bw_bits_per_cycle: int) -> HardwareSpec:
    bw = max(8, int(bw_bits_per_cycle))
    old = spec.memory_by_name("Global_buffer")
    new_spec = _replace_memory_level(
        spec, "Global_buffer",
        r_bw_bits_per_cycle=bw,
        w_bw_bits_per_cycle=bw,
    )
    updated = _recompute_memory_cost_pJ(new_spec, "Global_buffer")
    new_spec = _replace_memory_level(
        new_spec, "Global_buffer",
        r_cost_per_bit_pJ=updated.r_cost_per_bit_pJ,
        w_cost_per_bit_pJ=updated.w_cost_per_bit_pJ,
    )
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_compartment_depth(spec: HardwareSpec, depth: int) -> HardwareSpec:
    depth = int(depth)
    if depth < 1:
        raise ValueError(f"compartment_depth ({depth}) must be >= 1.")
    W = spec.macro.precision.W
    new_macro_bits = depth * W
    old_macro = spec.memory_by_name("Macro")
    old_depth = max(1, spec.macro.compartment_depth)

    new_macro = replace(spec.macro, compartment_depth=depth)
    new_spec = replace(spec, macro=new_macro)
    new_spec = _replace_memory_level(new_spec, "Macro", size_bits=new_macro_bits)

    scale = depth / old_depth
    new_w_cost = old_macro.w_cost_per_bit_pJ * scale
    new_spec = _replace_memory_level(
        new_spec, "Macro",
        w_cost_per_bit_pJ=new_w_cost,
    )

    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def build_hardware_variant(spec: HardwareSpec, parameter: str, value) -> HardwareSpec:
    if parameter == "core_count":
        return set_core_count(spec, int(value))
    if parameter == "buffer_capacity":
        return set_gbuf_capacity_kb(spec, int(value))
    if parameter == "gbuf_core_bw":
        return set_gbuf_bandwidth(spec, int(value))
    if parameter == "compartment_depth":
        return set_compartment_depth(spec, int(value))
    raise ValueError(f"Unsupported sensitivity parameter: {parameter}")

DEFAULT_SWEEPS = {
    "core_count": [4, 8, 16, 32],
    "buffer_capacity": [64, 128, 256, 512],
    "gbuf_core_bw": [64, 128, 256, 512],
    "compartment_depth": [1, 2, 4, 8, 16],
}

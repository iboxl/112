# Sensitivity 硬件变体：输入输出均为 HardwareSpec
# 调用方（RunSensitivity.py）从 spec 派生 CIM_Acc（via CIM_Acc.from_spec）与 ZigZag acc（via
# zigzag_adapter.to_zigzag_accelerator），保证两侧同源。
# Leakage / memory cost 在参数变化后通过 CACTI/DRAM-static 重跑重建。

from __future__ import annotations

import copy
import math
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
    """重跑 CACTI（或 DRAM 常量），返回更新后的 MemoryLevelSpec 副本。"""
    level = spec.memory_by_name(mem_name)
    size_bytes = max(1, int(level.size_bits / 8))
    bw = max(8, int(level.r_bw_bits_per_cycle))
    if mem_name == "Dram":
        # DRAM 在 legacy 固定为 7.91 pJ/bit
        r_pb = 7.91
        w_pb = 7.91
    else:
        read_pj, write_pj, _, _ = cacti_power(capacity_bytes=size_bytes, bitwidth_bits=bw)
        r_pb = read_pj / bw
        w_pb = write_pj / bw
    return replace(level, r_cost_per_bit_pJ=r_pb, w_cost_per_bit_pJ=w_pb)


def _leakage_per_cycle_nJ(spec: HardwareSpec) -> float:
    """对齐 legacy CIM_Acc.__init__ 的 leakage 累加逻辑：
    DRAM + Global + (Output/Input_buffer) × cores + Macro(SRAM) × cores。
    OReg/IReg 不计入（遵循 legacy 行为）。
    """
    def _sram_leak_pJ(mem_name: str) -> float:
        level = spec.memory_by_name(mem_name)
        _, _, leak_pJ, _ = cacti_power(
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


def scale_core_count(spec: HardwareSpec, factor: float) -> HardwareSpec:
    new_cores = max(1, int(round(spec.cores * factor)))
    # 更新 cores 轴的 size
    new_axes = []
    for a in spec.macro.spatial_axes:
        if a.name == "cores":
            new_axes.append(replace(a, size=new_cores))
        else:
            new_axes.append(a)
    new_macro = replace(spec.macro, spatial_axes=new_axes)
    new_spec = replace(spec, cores=new_cores, macro=new_macro)
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def scale_buffer_capacity(spec: HardwareSpec, factor: float) -> HardwareSpec:
    new_spec = spec
    for name in _on_chip_buffer_names(spec):
        old = new_spec.memory_by_name(name)
        new_size = max(8, int(math.ceil(old.size_bits * factor)))
        new_spec = _replace_memory_level(new_spec, name, size_bits=new_size)
        updated_level = _recompute_memory_cost_pJ(new_spec, name)
        new_spec = _replace_memory_level(
            new_spec, name,
            r_cost_per_bit_pJ=updated_level.r_cost_per_bit_pJ,
            w_cost_per_bit_pJ=updated_level.w_cost_per_bit_pJ,
        )
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def scale_global_buffer_bandwidth(spec: HardwareSpec, factor: float) -> HardwareSpec:
    old = spec.memory_by_name("Global_buffer")
    new_bw = max(8, int(math.ceil(old.r_bw_bits_per_cycle * factor)))
    new_spec = _replace_memory_level(
        spec, "Global_buffer",
        r_bw_bits_per_cycle=new_bw,
        w_bw_bits_per_cycle=new_bw,
    )
    updated = _recompute_memory_cost_pJ(new_spec, "Global_buffer")
    new_spec = _replace_memory_level(
        new_spec, "Global_buffer",
        r_cost_per_bit_pJ=updated.r_cost_per_bit_pJ,
        w_cost_per_bit_pJ=updated.w_cost_per_bit_pJ,
    )
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def scale_macro_input_precision(spec: HardwareSpec, input_bits: int) -> HardwareSpec:
    input_bits = int(input_bits)
    W = spec.macro.precision.W
    new_precision = replace(
        spec.macro.precision,
        I=input_bits,
        psum=input_bits + W,
        O_final=input_bits,
    )
    new_macro = replace(spec.macro, precision=new_precision)
    new_spec = replace(spec, macro=new_macro)
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def build_hardware_variant(spec: HardwareSpec, parameter: str, value) -> HardwareSpec:
    if parameter == "core_count":
        return scale_core_count(spec, value)
    if parameter == "buffer_capacity":
        return scale_buffer_capacity(spec, value)
    if parameter == "gbuf_core_bw":
        return scale_global_buffer_bandwidth(spec, value)
    if parameter == "macro_input_precision":
        return scale_macro_input_precision(spec, int(value))
    raise ValueError(f"Unsupported sensitivity parameter: {parameter}")

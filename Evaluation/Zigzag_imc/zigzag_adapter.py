# HardwareSpec → zigzag.Accelerator 适配器
#
# 只走 ZigZag 公开接口（MemoryInstance / MemoryHierarchy / ImcArray / Core / Accelerator）。
# 不 patch submodule 内部。r_cost / w_cost 通过构造器显式注入，绕过 CACTI auto extraction。
# ImcArray 构造时 enable_cacti=True（vendored ZigZag `enable_cacti=False` 分支 raise）；
# ZigZag 自算的 macro area 仅供其内部 bookkeeping，MIREDO 不读取。

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List

from utils.ZigzagUtils import ensure_zigzag_submodule_on_path, zigzag_submodule_root

ensure_zigzag_submodule_on_path()
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.ImcArray import ImcArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

from Architecture.HardwareSpec import HardwareSpec, MemoryLevelSpec


_OPERAND_SPEC_TO_ZIGZAG = {"I": "I1", "W": "I2", "O": "O"}


@contextmanager
def _zigzag_runtime_cwd():
    prev_cwd = os.getcwd()
    os.chdir(zigzag_submodule_root())
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _build_tech_param(spec: HardwareSpec) -> Dict[str, float]:
    tp = spec.macro.tech_params
    return {
        "tech_node": tp.tech_node,
        "vdd": tp.vdd,
        "nd2_cap": tp.nd2_cap,
        "xor2_cap": tp.xor2_cap,
        "dff_cap": tp.dff_cap,
        "nd2_area": tp.nd2_area,
        "xor2_area": tp.xor2_area,
        "dff_area": tp.dff_area,
        "nd2_dly": tp.nd2_dly,
        "xor2_dly": tp.xor2_dly,
    }


def _build_hd_param(spec: HardwareSpec) -> Dict[str, Any]:
    macro_level = spec.memory_by_name("Macro")
    group_depth = macro_level.size_bits // spec.macro.precision.W
    return {
        "pe_type": "in_sram_computing",
        "imc_type": "digital",
        "input_precision": spec.macro.precision.I,
        "weight_precision": spec.macro.precision.W,
        "input_bit_per_cycle": spec.macro.input_bit_per_cycle,
        "group_depth": group_depth,
        "wordline_dimension": "D1",
        "bitline_dimension": "D2",
        "enable_cacti": True,
    }


def _build_dimensions(spec: HardwareSpec) -> Dict[str, int]:
    # D1 = wordline = dimY, D2 = bitline = dimX, D3 = cores
    return {
        "D1": spec.macro.dimY,
        "D2": spec.macro.dimX,
        "D3": spec.cores,
    }


def _port_alloc_for_operand(op_spec: str) -> Dict[str, Any]:
    # I (→I1) / W (→I2)：单向 fh+tl；O：双向 fh+tl+fl+th
    if op_spec in ("I", "W"):
        return {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None}
    if op_spec == "O":
        return {"fh": "rw_port_1", "tl": "rw_port_1", "fl": "rw_port_1", "th": "rw_port_1"}
    raise ValueError(f"Unknown operand {op_spec!r}")


def _build_memory_instance(level: MemoryLevelSpec) -> MemoryInstance:
    kwargs: Dict[str, Any] = dict(
        name=level.name,
        size=level.size_bits,
        r_bw=level.r_bw_bits_per_cycle,
        w_bw=level.w_bw_bits_per_cycle,
        r_cost=level.r_cost_per_bit_pJ * level.r_bw_bits_per_cycle,
        w_cost=level.w_cost_per_bit_pJ * level.w_bw_bits_per_cycle,
        area=level.area_mm2,
        r_port=level.ports.r,
        w_port=level.ports.w,
        rw_port=level.ports.rw,
        latency=level.r_latency_cycles,
    )
    if level.min_r_granularity_bits is not None:
        kwargs["min_r_granularity"] = level.min_r_granularity_bits
    if level.min_w_granularity_bits is not None:
        kwargs["min_w_granularity"] = level.min_w_granularity_bits
    return MemoryInstance(**kwargs)


def _served_dimensions_for_zigzag(level: MemoryLevelSpec):
    sd = level.served_dimensions_zigzag
    if isinstance(sd, str):
        return sd  # "all"
    if not sd:
        return set()
    return {tuple(t) for t in sd}


def to_zigzag_accelerator(spec: HardwareSpec, acc_name: str = "CIM_ACC_TEMPLATE") -> Accelerator:
    tech_param = _build_tech_param(spec)
    hd_param = _build_hd_param(spec)
    dimensions = _build_dimensions(spec)

    with _zigzag_runtime_cwd():
        imc_array = ImcArray(tech_param, hd_param, dimensions)

    mem_hierarchy = MemoryHierarchy(operational_array=imc_array)

    # ZigZag add_memory 顺序：内 → 外。Spec 存储为外 → 内，故反向迭代。
    for level in reversed(spec.memory_hierarchy):
        instance = _build_memory_instance(level)
        operands_zz = tuple(_OPERAND_SPEC_TO_ZIGZAG[op] for op in level.operands)
        port_alloc = tuple(_port_alloc_for_operand(op) for op in level.operands)
        served = _served_dimensions_for_zigzag(level)
        mem_hierarchy.add_memory(
            memory_instance=instance,
            operands=operands_zz,
            port_alloc=port_alloc,
            served_dimensions=served,
        )

    core = Core(0, imc_array, mem_hierarchy)
    return Accelerator(acc_name, {core})

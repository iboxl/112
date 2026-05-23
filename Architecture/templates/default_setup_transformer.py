# HW-Transformer default for the 2026-05-13 production rerun.
#
# Differences from templates/transformer.py (the legacy HW-Transformer kept as
# CIM_ACC_TEMPLATE_TRANSFORMER for backward comparison):
#   - Output_buffer: 16 KB → 32 KB per core (1:2 ratio with 16 KB IBuf,
#                                            matching 8b activation / 16b psum width)
#   - Input_buffer:  16 KB unchanged
#   - Dram BW:       256 bit/cyc unchanged (already LPDDR5x level)
#
# Geometry (16 cores, 64×32 macro, 4 MB GBuf) unchanged — HW-Transformer is
# already HAMMER-anchored. The only design-principle gap closed here is
# OBuf = 2× IBuf to absorb the 16-bit psum width.

from __future__ import annotations

from dataclasses import replace

from Architecture.HardwareSpec import HardwareSpec
from Architecture.templates.transformer import _build_transformer_spec_dict


def _build_default_setup_transformer_spec_dict() -> dict:
    spec = _build_transformer_spec_dict()
    for mem in spec["memory_hierarchy"]:
        if mem["name"] == "Output_buffer":
            mem["size_bits"] = 32 * 1024 * 8  # 32 KB per core
            # bandwidth unchanged

    spec["metadata"] = dict(spec["metadata"])
    spec["metadata"]["notes"] = (
        "CIM_ACC_DEFAULT_SETUP_TRANSFORMER (2026-05-13): 16-core digital SRAM CIM, "
        "28nm, I=W=8b psum=16b, 64x32 macro, 4 MB GBuf, 16 KB IBuf / 32 KB OBuf per core, "
        "4 GB DRAM @256 bit/cyc. Only OBuf differs from CIM_ACC_TEMPLATE_TRANSFORMER "
        "(16→32 KB per core, 1:2 ratio matching 8b/16b width asymmetry)."
    )
    return spec


def default_spec() -> HardwareSpec:
    """CIM_ACC_DEFAULT_SETUP_TRANSFORMER entry. CACTI energies refreshed for
    Output_buffer (capacity changed); other SRAM levels reuse legacy values."""
    from Evaluation.common.HardwareVariants import (
        _recompute_memory_cost_pJ,
        _leakage_per_cycle_nJ,
    )

    spec = HardwareSpec.from_dict(_build_default_setup_transformer_spec_dict())

    cacti_levels = ("Global_buffer", "Input_buffer", "Output_buffer")
    refreshed = []
    for m in spec.memory_hierarchy:
        if m.name in cacti_levels:
            updated = _recompute_memory_cost_pJ(spec, m.name)
            refreshed.append(replace(
                m,
                r_cost_per_bit_pJ=updated.r_cost_per_bit_pJ,
                w_cost_per_bit_pJ=updated.w_cost_per_bit_pJ,
            ))
        else:
            refreshed.append(m)
    spec = replace(spec, memory_hierarchy=refreshed)

    spec = replace(spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(spec))

    return spec

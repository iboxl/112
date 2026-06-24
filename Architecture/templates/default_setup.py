# CNN production-rerun default (CIM_ACC_DEFAULT_SETUP).
#
# Differences from templates/default.py (the legacy 0420 baseline, kept as
# CIM_ACC_TEMPLATE for backward comparison):
# FROZEN SRAM-CIM operating point (2026-05-18, final — supersedes the
# transient 0518 8/8/64 symmetric variant; matches RERUN_GUIDE §3):
#   - Input_buffer:  8 KB per core
#   - Output_buffer: 16 KB per core  (2x IBuf: 8b activation in / 16b psum out
#                                     width asymmetry)
#   - Dram BW:       64 bit/cycle    (LPDDR4-class single channel)
#   Presented as one point on the §5.5.1 buffer / DRAM-BW sensitivity curve
#   (not cherry-picked); framed honestly: dataflow-optimization value scales
#   with memory pressure. This is the LAST hardware change — frozen regardless
#   of results.
#
# All other fields (8 cores, 32×16×8 macro, 256 KB GBuf @128 bit/cyc,
# 28 nm CACTI, INT8/INT8/INT16, OReg/IReg/Macro registers) unchanged.
#
# CACTI energies for the two changed SRAM levels (Input_buffer, Output_buffer)
# are auto-recomputed in default_spec() below via _recompute_memory_cost_pJ.
# The unchanged Global_buffer reuses the legacy CACTI values from default.py.

from __future__ import annotations

import copy
from dataclasses import replace

from Architecture.HardwareSpec import HardwareSpec
from Architecture.templates.default import _DEFAULT_SPEC_DICT


def _build_default_setup_spec_dict() -> dict:
    spec = copy.deepcopy(_DEFAULT_SPEC_DICT)

    overrides = {
        "Dram":          {"r_bw_bits_per_cycle": 64, "w_bw_bits_per_cycle": 64},
        "Input_buffer":  {"size_bits": 8 * 1024 * 8,   "r_bw_bits_per_cycle": 128, "w_bw_bits_per_cycle": 128},
        "Output_buffer": {"size_bits": 16 * 1024 * 8,  "r_bw_bits_per_cycle": 128, "w_bw_bits_per_cycle": 128},
    }
    for mem in spec["memory_hierarchy"]:
        if mem["name"] in overrides:
            mem.update(overrides[mem["name"]])

    spec["metadata"] = dict(spec["metadata"])
    spec["metadata"]["notes"] = (
        "CIM_ACC_DEFAULT_SETUP (frozen 2026-05-18): 8-core digital SRAM CIM, 28nm, "
        "I=W=8b psum=16b, 32x16x8 macro, 256 KB GBuf, 8 KB IBuf / 16 KB OBuf per core, "
        "1 GB DRAM @64 bit/cyc (LPDDR4-class single channel)."
    )
    return spec


def default_spec() -> HardwareSpec:
    """CIM_ACC_DEFAULT_SETUP entry. CACTI energies are refreshed for the
    Input/Output_buffer levels whose capacity changed; Global_buffer reuses the
    legacy CACTI values inherited from _DEFAULT_SPEC_DICT."""
    # Delayed import to avoid circular dependency
    # (HardwareVariants imports HardwareSpec and the CACTI wrapper).
    from Evaluation.common.HardwareVariants import (
        _recompute_memory_cost_pJ,
        _leakage_per_cycle_nJ,
    )

    spec = HardwareSpec.from_dict(_build_default_setup_spec_dict())

    # Only the two capacity-changed SRAM levels need CACTI re-run.
    cacti_levels = ("Input_buffer", "Output_buffer")
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

    # Total leakage recomputed against new memory hierarchy.
    spec = replace(spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(spec))

    return spec

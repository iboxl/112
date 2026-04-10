import copy
import math

from utils.Cacti_wrapper.EvalCacti import cacti_power, dram_static


def clone_accelerator(acc):
    return copy.deepcopy(acc)


def _local_buffer_levels(acc):
    input_buffer_mem = max(
        mem for mem in range(acc.Global2mem + 1, acc.IReg2mem)
        if acc.mappingArray[0][mem] == 1
    )
    output_buffer_mem = max(
        mem for mem in range(acc.Global2mem + 1, acc.OReg2mem)
        if acc.mappingArray[2][mem] == 1
    )
    return input_buffer_mem, output_buffer_mem


def _on_chip_buffer_levels(acc):
    first_terminal_mem = min(acc.OReg2mem, acc.IReg2mem, acc.Macro2mem)
    input_buffer_mem, output_buffer_mem = _local_buffer_levels(acc)
    on_chip_buffers = list(range(acc.Dram2mem + 1, first_terminal_mem))
    assert on_chip_buffers == sorted([acc.Global2mem, output_buffer_mem, input_buffer_mem])
    return on_chip_buffers


def _recompute_memory_cost(acc, mem):
    mem_name = acc.mem2dict(mem)
    if mem_name == "Dram":
        dram_pj, _ = dram_static(capacity_bytes=(acc.memSize[mem] / 8), bus_width_bits=acc.bw[mem])
        per_access_pj = 7.91 * acc.bw[mem]
        acc.cost_r[mem] = per_access_pj * 1e-3 / acc.bw[mem]
        acc.cost_w[mem] = per_access_pj * 1e-3 / acc.bw[mem]
        return dram_pj

    read_pj, write_pj, _, _ = cacti_power(
        capacity_bytes=max(1, int(acc.memSize[mem] / 8)),
        bitwidth_bits=max(8, int(acc.bw[mem])),
    )
    acc.cost_r[mem] = read_pj * 1e-3 / acc.bw[mem]
    acc.cost_w[mem] = write_pj * 1e-3 / acc.bw[mem]
    return None


def _recompute_leakage(acc):
    input_buffer_mem, output_buffer_mem = _local_buffer_levels(acc)
    assert acc.Dram2mem < acc.Global2mem < output_buffer_mem < acc.OReg2mem
    assert acc.Dram2mem < acc.Global2mem < input_buffer_mem < acc.IReg2mem < acc.Macro2mem < acc.Num_mem

    # Dram/global are shared once; local input/output buffers and the macro array are replicated per core.
    leakage = 0.0
    _, dram_leak = dram_static(capacity_bytes=(acc.memSize[acc.Dram2mem] / 8), bus_width_bits=acc.bw[acc.Dram2mem])
    leakage += dram_leak
    leakage += cacti_power(capacity_bytes=(acc.memSize[acc.Global2mem] / 8), bitwidth_bits=acc.bw[acc.Global2mem])[2]
    leakage += cacti_power(capacity_bytes=(acc.memSize[output_buffer_mem] / 8), bitwidth_bits=acc.bw[output_buffer_mem])[2] * acc.Num_core
    leakage += cacti_power(capacity_bytes=(acc.memSize[input_buffer_mem] / 8), bitwidth_bits=acc.bw[input_buffer_mem])[2] * acc.Num_core
    weight_precision = max(1, acc.precision_psum - acc.precision_final)
    leakage += cacti_power(
        capacity_bytes=(acc.memSize[acc.Macro2mem] / 8) * acc.dimX * acc.dimY,
        bitwidth_bits=max(8, acc.dimY * weight_precision),
    )[2] * acc.Num_core
    acc.leakage_per_cycle = leakage * 1e-3


def scale_core_count(acc, factor):
    variant = clone_accelerator(acc)
    variant.Num_core = max(1, int(round(acc.Num_core * factor)))
    variant.SpUnrolling[0] = variant.Num_core
    _recompute_leakage(variant)
    return variant


def scale_buffer_capacity(acc, factor):
    variant = clone_accelerator(acc)
    for mem in _on_chip_buffer_levels(variant):
        variant.memSize[mem] = max(8, int(math.ceil(acc.memSize[mem] * factor)))
        _recompute_memory_cost(variant, mem)
    _recompute_leakage(variant)
    return variant


def scale_global_buffer_bandwidth(acc, factor):
    variant = clone_accelerator(acc)
    variant.bw[variant.Global2mem] = max(8, int(math.ceil(acc.bw[variant.Global2mem] * factor)))
    _recompute_memory_cost(variant, variant.Global2mem)
    _recompute_leakage(variant)
    return variant


def scale_macro_input_precision(acc, input_bits):
    variant = clone_accelerator(acc)
    old_input_bits = variant.precision_final
    weight_precision = max(1, variant.precision_psum - variant.precision_final)

    for mem in range(1, variant.Num_mem + 1):
        variant.precision[mem, 0] = input_bits
        variant.precision[mem, 2] = input_bits

    variant.precision_final = input_bits
    variant.precision_psum = input_bits + weight_precision
    variant.precision[variant.OReg2mem, 2] = variant.precision_psum
    variant.t_MAC = input_bits
    variant.cost_ActMacro = variant.cost_ActMacro * (input_bits / max(1, old_input_bits))
    _recompute_leakage(variant)
    return variant


def build_hardware_variant(acc, parameter, value):
    if parameter == "core_count":
        return scale_core_count(acc, value)
    if parameter == "buffer_capacity":
        return scale_buffer_capacity(acc, value)
    if parameter == "gbuf_core_bw":
        return scale_global_buffer_bandwidth(acc, value)
    if parameter == "macro_input_precision":
        return scale_macro_input_precision(acc, int(value))
    raise ValueError(f"Unsupported sensitivity parameter: {parameter}")

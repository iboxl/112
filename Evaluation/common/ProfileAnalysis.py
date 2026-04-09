def stall_decomposition(profile):
    compute_cycles = float(getattr(profile, "macLatency", 0))
    mode_switch_stall = float(getattr(profile, "mode_switch_stall", 0))
    mismatch_stall = float(getattr(profile, "mismatch_stall", 0))
    writeback_stall = float(getattr(profile, "writeback_stall", 0))
    total_latency = float(getattr(profile, "latency", 0))
    idle_cycles = float(
        getattr(
            profile,
            "idle_cycles",
            max(0.0, total_latency - compute_cycles - mode_switch_stall - mismatch_stall - writeback_stall),
        )
    )
    return {
        "compute_cycles": compute_cycles,
        "mode_switch_stall": mode_switch_stall,
        "mismatch_stall": mismatch_stall,
        "writeback_stall": writeback_stall,
        "idle_cycles": idle_cycles,
        "total_latency": total_latency,
    }


def dominant_stall_type(profile):
    decomposition = stall_decomposition(profile)
    ranked = {
        "compute_bound": decomposition["compute_cycles"],
        "mode_switch": decomposition["mode_switch_stall"],
        "mismatch": decomposition["mismatch_stall"],
        "psum_writeback": decomposition["writeback_stall"],
        "idle": decomposition["idle_cycles"],
    }
    return max(ranked.items(), key=lambda item: item[1])[0]


def summarize_double_buffer_decisions(dataflow):
    decisions = {}
    for mem in range(1, dataflow.acc.Num_mem):
        mem_name = dataflow.acc.mem2dict(mem)
        flags = dataflow.usr_defined_double_flag[mem]
        decisions[mem_name] = {
            op_name: bool(flags[op])
            for op, op_name in enumerate(["I", "W", "O"])
            if dataflow.acc.mappingArray[op][mem]
        }
    return decisions


def summarize_memory_residency(dataflow):
    summary = {}
    for op, op_name in enumerate(["I", "W", "O"]):
        seen = []
        for mapping in dataflow.tm:
            mem_name = dataflow.acc.mem2dict(mapping.mem[op])
            if mem_name not in seen:
                seen.append(mem_name)
        summary[op_name] = " -> ".join(seen)
    return summary


def dominant_tiling_pattern(dataflow, max_terms=4):
    spatial_terms = [
        f"{dataflow.ops.dim2Dict[mapping.dim]}x{mapping.dimSize}"
        for mapping in dataflow.sm
    ][:max_terms]
    temporal_terms = [
        f"{dataflow.ops.dim2Dict[mapping.dim]}x{mapping.dimSize}"
        for mapping in dataflow.tm[:max_terms]
    ]
    spatial_desc = ", ".join(spatial_terms) if spatial_terms else "none"
    temporal_desc = ", ".join(temporal_terms) if temporal_terms else "none"
    return f"spatial[{spatial_desc}] temporal_outer[{temporal_desc}]"


def summarize_dataflow_decisions(dataflow):
    return {
        "double_buffer_decisions": summarize_double_buffer_decisions(dataflow),
        "memory_residency": summarize_memory_residency(dataflow),
        "dominant_tiling_pattern": dominant_tiling_pattern(dataflow),
    }

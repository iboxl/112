def stall_decomposition(profile):
    """Decompose total latency into event categories.

    IMPORTANT: The four event fields (compute_cycles, mode_switch_stall,
    mismatch_stall, writeback_stall) are NOT mutually exclusive — events
    can overlap in cycle space. Their sum can exceed total_latency.

    unattributed_cycles = max(0, total_latency - compute_cycles - sum(stalls))
    is the residual after clamping at zero. Callers must NOT stack the four
    event fields as exclusive cycle slices (geometry error).

    See stall_breakdown.md §三 finding 1.
    """
    compute_cycles = float(getattr(profile, "macLatency", 0))
    mode_switch_stall = float(getattr(profile, "mode_switch_stall", 0))
    mismatch_stall = float(getattr(profile, "mismatch_stall", 0))
    writeback_stall = float(getattr(profile, "writeback_stall", 0))
    total_latency = float(getattr(profile, "latency", 0))
    unattributed_cycles = float(
        getattr(
            profile,
            "unattributed_cycles",
            max(0.0, total_latency - compute_cycles - mode_switch_stall - mismatch_stall - writeback_stall),
        )
    )
    return {
        "compute_cycles": compute_cycles,
        "mode_switch_stall": mode_switch_stall,
        "mismatch_stall": mismatch_stall,
        "writeback_stall": writeback_stall,
        "unattributed_cycles": unattributed_cycles,
        "total_latency": total_latency,
    }


def dominant_stall_type(profile):
    decomposition = stall_decomposition(profile)
    ranked = {
        "compute_bound": decomposition["compute_cycles"],
        "mode_switch": decomposition["mode_switch_stall"],
        "mismatch": decomposition["mismatch_stall"],
        "psum_writeback": decomposition["writeback_stall"],
        "unattributed": decomposition["unattributed_cycles"],
    }
    return max(ranked.items(), key=lambda item: item[1])[0]


def event_cycle_intensity(profile) -> dict:
    """Event-cycle intensity per category for grouped (NOT stacked) bars.

    Returns absolute cycles AND % of total_latency for each event class.
    The four event categories are NON-ADDITIVE; do not stack.
    """
    decomp = stall_decomposition(profile)
    total = decomp["total_latency"] or 1.0
    fields = ["compute_cycles", "mode_switch_stall", "mismatch_stall",
              "writeback_stall", "unattributed_cycles"]
    return {
        "events": {
            f: {"cycles": decomp[f], "pct_of_total": decomp[f] / total * 100.0}
            for f in fields
        },
        "total_latency": decomp["total_latency"],
        "non_additive": True,
    }


def macrowait_decomposition(acc, ops, dataflow) -> dict | None:
    """Exact ADDITIVE macro-wait stall decomposition for one dataflow.

    Re-simulates `dataflow` and charges every macro-idle cycle to the single
    operand the macro waits on. Unlike stall_decomposition / event_cycle_intensity,
    whose mode_switch / mismatch / writeback counters overlap in cycle space and
    are NON-additive, the four terms here are mutually exclusive and sum exactly
    to latency:

        latency = compute + stall_input + stall_weight + stall_output

    `acc` and `ops` are the accelerator and workload the dataflow was generated
    against. Returns None when no dataflow is available (e.g. a framework
    fallback). The simulator import is deferred so callers using only the
    post-processing metrics do not pay it.
    """
    if dataflow is None:
        return None
    from Evaluation.common.MacroWaitSim import run_macrowait
    return run_macrowait(acc, ops, dataflow)


def utilization_metrics(profile) -> dict:
    """Spatial: count_mac × t_MAC / (peak_mac_per_cycle × total_latency).
    Temporal: macLatency / total_latency.

    Both report % of cycles spent in MAC compute, from two angles:
    spatial uses peak parallel-MAC capacity as denominator; temporal
    uses sequential cycle count.
    """
    total = float(getattr(profile, "latency", 0)) or 1.0
    mac_lat = float(getattr(profile, "macLatency", 0))
    n_mac = float(getattr(profile, "count_mac", 0))
    peak = float(getattr(profile, "peak_mac_per_cycle", 0))
    # Note: no t_MAC field on ProfilingDetail; spatial util as
    # macLatency / (peak × total) since macLatency = count_mac × t_MAC
    spatial = (mac_lat / (peak * total)) if peak > 0 else None
    return {
        "spatial_util_pct": spatial * 100.0 if spatial is not None else None,
        "temporal_util_pct": (mac_lat / total) * 100.0,
        "total_latency": total,
        "mac_latency": mac_lat,
    }


def memory_traffic_metrics(profile, dataflow=None, acc=None) -> dict:
    """Per-level memory traffic and reload counters.

    Returns:
      per_level: {mem_idx: {"read_bytes": ..., "write_bytes": ...}}
      psum_traffic_bytes: int (operand 'O' transfer cycles; downstream fig converts)
      weight_reload_count: int (loop-carried weight reloads in dataflow.tm)
      traffic_per_mac_bytes: float (sum read+write / count_mac)
    """
    per_level = {}
    bytes_r = list(getattr(profile, "bytes_read", []))
    bytes_w = list(getattr(profile, "bytes_written", []))
    transfer_cyc = list(getattr(profile, "transfer_cycles", [0, 0, 0]))
    n_mem = max(len(bytes_r), len(bytes_w))
    for m in range(n_mem):
        per_level[m] = {
            "read_bytes": bytes_r[m] if m < len(bytes_r) else None,
            "write_bytes": bytes_w[m] if m < len(bytes_w) else None,
        }
    # psum traffic = output operand transfer cycles (operand index 2 = O)
    # bytes = transfer_cycles × BW (bytes/cycle) — BW per level differs;
    # report cycles here; downstream figure converts with acc.bw if available.
    psum_traffic_bytes = None
    if len(transfer_cyc) > 2:
        # report as cycles; downstream converts to bytes
        psum_traffic_bytes = transfer_cyc[2]
    weight_reload_count = _count_weight_reloads_neutral(dataflow) if dataflow is not None else None
    total_bytes = sum((b or 0) for b in bytes_r) + sum((b or 0) for b in bytes_w)
    n_mac = float(getattr(profile, "count_mac", 0)) or 1.0
    return {
        "per_level": per_level,
        "psum_traffic_bytes": psum_traffic_bytes,
        "weight_reload_count": weight_reload_count,
        "traffic_per_mac_bytes": total_bytes / n_mac if n_mac > 0 else None,
    }


def _count_weight_reloads_neutral(dataflow) -> int:
    """Count loop-carried weight reloads in temporal mapping.

    A reload occurs when a weight-relevant loop crosses an outer
    memory boundary in dataflow.tm. Framework-agnostic: any LoopNest
    with .tm field works.
    """
    if not hasattr(dataflow, "tm"):
        return None
    count = 0
    prev_mem = None
    for mapping in dataflow.tm:
        if hasattr(mapping, "mem") and len(mapping.mem) > 1:
            cur_mem = mapping.mem[1]  # operand 1 = W (weight)
            if prev_mem is not None and cur_mem != prev_mem:
                count += 1
            prev_mem = cur_mem
    return count


def mapping_decision_summary(dataflow, acc=None) -> dict:
    """Observable mapping decisions in framework-agnostic vocabulary.

    Field names use NEUTRAL terms — no §4 variable names (β, δ, φ, ψ).
    """
    if dataflow is None:
        return {k: None for k in [
            "reload_trigger_boundary", "psum_residency_level",
            "double_buffer_pairs", "operand_residency_path",
            "dominant_tiling_pattern",
        ]}
    return {
        "reload_trigger_boundary": _reload_boundary_neutral(dataflow, acc),
        "psum_residency_level": _psum_residency_neutral(dataflow, acc),
        "double_buffer_pairs": summarize_double_buffer_decisions(dataflow),
        "operand_residency_path": summarize_memory_residency(dataflow),
        "dominant_tiling_pattern": dominant_tiling_pattern(dataflow),
    }


def _reload_boundary_neutral(dataflow, acc) -> str:
    """Outermost loop level holding W (weight). Framework-agnostic."""
    if not hasattr(dataflow, "tm") or not dataflow.tm:
        return None
    # Find outermost mapping whose mem[1] (W) is highest mem level
    max_mem = max((m.mem[1] for m in dataflow.tm if hasattr(m, "mem") and len(m.mem) > 1),
                  default=None)
    if max_mem is None:
        return None
    if acc is not None and hasattr(acc, "memName"):
        return acc.memName[max_mem] if max_mem < len(acc.memName) else f"L{max_mem}"
    return f"L{max_mem}"


def _psum_residency_neutral(dataflow, acc) -> str:
    """Outermost level holding O (output). For MIREDO this is dataflow.psum_flag.
    For others, derived from O memory path; if all at innermost, mark 'implicit'.
    """
    if hasattr(dataflow, "psum_flag") and dataflow.psum_flag:
        # MIREDO native flag
        psum_levels = [m for m, flag in dataflow.psum_flag.items() if flag]
        if psum_levels:
            level = max(psum_levels)
            if acc is not None and hasattr(acc, "memName"):
                return acc.memName[level] if level < len(acc.memName) else f"L{level}"
            return f"L{level}"
    # Non-MIREDO: derive from operand-O residency
    if hasattr(dataflow, "tm"):
        o_mems = [m.mem[2] for m in dataflow.tm if hasattr(m, "mem") and len(m.mem) > 2]
        if o_mems:
            return f"implicit (O at L{max(o_mems)})"
    return None


def available_profile_mask(profile, dataflow, framework_name: str) -> dict:
    """Per-framework availability of each metric family field.

    Marks N/A explicitly; never returns 0 or NaN as a substitute.
    Used in 5.3.2 setup paragraph as a fairness disclosure.
    """
    has_profile = profile is not None
    has_dataflow = dataflow is not None
    return {
        "F1_time_idle": {
            "compute_cycles": has_profile,
            "mode_switch_stall": has_profile,
            "mismatch_stall": has_profile,
            "writeback_stall": has_profile,
            "unattributed_cycles": has_profile,
        },
        "F2_utilization": {
            "spatial_util_pct": has_profile and getattr(profile, "count_mac", None) is not None,
            "temporal_util_pct": has_profile,
        },
        "F3_traffic": {
            "per_level_bytes": has_profile and bool(getattr(profile, "bytes_read", [])),
            "weight_reload_count": has_dataflow,
            "psum_traffic_bytes": has_profile,
            "traffic_per_mac_bytes": has_profile and getattr(profile, "count_mac", None) is not None,
        },
        "F4_mapping_decision": {
            "reload_trigger_boundary": has_dataflow,
            "psum_residency_level": (
                "explicit" if framework_name.lower() == "miredo" and has_dataflow
                else "implicit" if has_dataflow
                else None
            ),
            "double_buffer_pairs": has_dataflow,
            "operand_residency_path": has_dataflow,
        },
        "F5_macrowait_additive": {
            "compute": has_dataflow,
            "stall_input": has_dataflow,
            "stall_weight": has_dataflow,
            "stall_output": has_dataflow,
        },
    }


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

# 容量合法化 — baseline mapping → Simulax 之前的通用容量校验与修复。
#
# 任何外部 mapper（CoSA / CIMLoop / ZigZag …）产出的 ClassMapping 在进入
# convert_baseMapping_to_MIREDO 之前，可调用 legalize_capacity() 逐级校验
# tile × precision 是否超出 MIREDO 物理 memory 容量。仅在溢出时将 temporal
# factor 上推外层（保守方向），保证 Simulax 不触发 OverSize anomaly。

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from Architecture.HardwareSpec import HardwareSpec


# ---------------------------------------------------------------------------
# Tile computation helpers (match Simulax / WorkLoad.get_operand_size)
# ---------------------------------------------------------------------------

def _input_tile_extent(output_size: int, kernel_size: int,
                       stride: int, input_bound: int) -> int:
    """Matches WorkLoad._get_input_extent exactly."""
    unique_extent = kernel_size + (output_size - 1) * min(stride, kernel_size)
    return min(unique_extent, input_bound)


def _operand_tile_elements(acc_dims: Dict[str, int], op: str,
                           stride: int, H: int, W: int) -> int:
    """Compute operand tile elements from accumulated per-dim factors.

    Matches Simulax's ``get_operand_size`` (via ``WorkLoad.get_operand_size``).
    """
    if op == "I":
        p, q = acc_dims.get("P", 1), acc_dims.get("Q", 1)
        r, s = acc_dims.get("R", 1), acc_dims.get("S", 1)
        h = _input_tile_extent(p, r, stride, H)
        w = _input_tile_extent(q, s, stride, W)
        return h * w * acc_dims.get("C", 1) * acc_dims.get("G", 1)
    if op == "W":
        return (acc_dims.get("R", 1) * acc_dims.get("S", 1)
                * acc_dims.get("C", 1) * acc_dims.get("K", 1)
                * acc_dims.get("G", 1))
    if op == "O":
        return (acc_dims.get("P", 1) * acc_dims.get("Q", 1)
                * acc_dims.get("K", 1) * acc_dims.get("G", 1))
    return 0


# Dimensions that affect each operand's tile size.
_OP_RELEVANT_DIMS: Dict[str, frozenset] = {
    "I": frozenset({"R", "S", "P", "Q", "C", "G"}),
    "W": frozenset({"R", "S", "C", "K", "G"}),
    "O": frozenset({"P", "Q", "K", "G"}),
}

# Reduction dimensions for O — temporal loops on these dims at an O-keeping
# level mean that level (and all more-inner levels) hold partial sums, so
# Simulax charges precision_psum (16b) instead of precision_final (8b).
# Mirrors LoopNest.preprogress() psum_flag derivation.
_O_REDUCTION_DIMS = frozenset({"R", "S", "C"})


def _accumulated_dims(
    temporal_levels: List[List[Tuple[str, int]]],
    spatial_levels: List[List[Tuple[str, int]]],
    up_to_cm_level: int,
) -> Dict[str, int]:
    """Product of all factors from CM level 0 to *up_to_cm_level* (inclusive)."""
    acc: Dict[str, int] = {d: 1 for d in ("R", "S", "P", "Q", "C", "K", "G")}
    for j in range(up_to_cm_level + 1):
        if j < len(temporal_levels):
            for (dim, factor) in temporal_levels[j]:
                if dim in acc:
                    acc[dim] *= factor
        if j < len(spatial_levels):
            for (dim, factor) in spatial_levels[j]:
                if dim in acc:
                    acc[dim] *= factor
    return acc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def legalize_capacity(
    temporal_mapping: Dict[str, List[List[Tuple[str, int]]]],
    spatial_mapping: Dict[str, List[List[Tuple[str, int]]]],
    spec: HardwareSpec,
    loopdim: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Demote temporal factors outward when tile exceeds MIREDO memory capacity.

    Walks each MIREDO memory level (inner -> outer).  For every level where the
    total bit-occupancy of all kept operands exceeds the physical capacity, the
    outermost *relevant* temporal factor from the operand contributing the most
    bits is demoted to the next outer ClassMapping level for that operand.

    Only modifies the mapping when overflow actually occurs.  Returns a list of
    demotion records for metadata transparency.
    """
    stride = int(loopdim.get("Stride", 1))
    H = int(loopdim.get("H", 0))
    if H <= 0:
        P = int(loopdim.get("P", 1))
        R = int(loopdim.get("R", 1))
        H = (P - 1) * stride + R
    W = int(loopdim.get("W", 0))
    if W <= 0:
        Q = int(loopdim.get("Q", 1))
        S = int(loopdim.get("S", 1))
        W = (Q - 1) * stride + S

    # Per-operand kept-level names, inner -> outer.
    # spec.memory_hierarchy is outer -> inner; reversed gives inner -> outer.
    op_kept_names: Dict[str, List[str]] = {"I": [], "W": [], "O": []}
    for level in reversed(spec.memory_hierarchy):
        for op in ("I", "W", "O"):
            if op in level.operands:
                op_kept_names[op].append(level.name)

    mem_capacity = {lv.name: lv.size_bits for lv in spec.memory_hierarchy}

    I_prec = spec.macro.precision.I
    W_prec = spec.macro.precision.W
    psum_prec = spec.macro.precision.psum
    O_final_prec = spec.macro.precision.O_final

    def prec_bits(miredo_name: str, op: str) -> int:
        """Dynamic precision matching Simulax's LoopNest.get_precision.

        For O: if any temporal reduction dimension (R/S/C) with factor>1 exists
        at this O CM level or any more-outer O CM level, the level holds partial
        sums -> use psum precision.  Mirrors LoopNest.preprogress() psum_flag.
        Reads the *current* temporal_mapping["O"] so demotions that move
        reduction factors outward are reflected immediately.
        """
        if op == "I":
            return I_prec
        if op == "W":
            return W_prec
        # op == "O": dynamic psum check
        if miredo_name not in op_kept_names["O"]:
            return O_final_prec
        cm_idx = op_kept_names["O"].index(miredo_name)
        # Walk from outermost O CM level down to cm_idx.  If ANY of these
        # levels contains a reduction temporal factor, this level holds psums.
        for j in range(len(temporal_mapping["O"]) - 1, cm_idx - 1, -1):
            for (dim, factor) in temporal_mapping["O"][j]:
                if dim in _O_REDUCTION_DIMS and factor > 1:
                    return psum_prec
        return O_final_prec

    # Build check order: for each non-Dram level (inner -> outer), collect
    # the (operand, CM level index) pairs that map to it.
    check_order: List[Tuple[str, Dict[str, int]]] = []
    seen: set = set()
    for level in reversed(spec.memory_hierarchy):
        name = level.name
        if name in seen:
            continue
        seen.add(name)
        # Skip Dram -- effectively infinite.
        if any(name.lower().startswith(tag) for tag in ("dram",)):
            continue
        ops_at: Dict[str, int] = {}
        for op in ("I", "W", "O"):
            if name in op_kept_names[op]:
                ops_at[op] = op_kept_names[op].index(name)
        if ops_at:
            check_order.append((name, ops_at))

    demoted: List[Dict[str, Any]] = []

    for miredo_name, op_cm in check_order:
        cap = mem_capacity.get(miredo_name, 0)
        if cap <= 0:
            continue

        def _total_bits() -> int:
            total = 0
            for op, cm_idx in op_cm.items():
                ad = _accumulated_dims(
                    temporal_mapping[op], spatial_mapping[op], cm_idx)
                tile = _operand_tile_elements(ad, op, stride, H, W)
                total += tile * prec_bits(miredo_name, op)
            return total

        while _total_bits() > cap:
            # Identify the operand with the largest bit contribution.
            contribs: List[Tuple[int, str, int]] = []
            for op, cm_idx in op_cm.items():
                ad = _accumulated_dims(
                    temporal_mapping[op], spatial_mapping[op], cm_idx)
                tile = _operand_tile_elements(ad, op, stride, H, W)
                contribs.append((tile * prec_bits(miredo_name, op), op, cm_idx))
            contribs.sort(reverse=True)

            found = False
            for _, vop, vcm in contribs:
                relevant = _OP_RELEVANT_DIMS[vop]
                # Walk backwards (outermost first) for a relevant factor > 1.
                for fi in range(len(temporal_mapping[vop][vcm]) - 1, -1, -1):
                    dim, factor = temporal_mapping[vop][vcm][fi]
                    if dim in relevant and factor > 1:
                        temporal_mapping[vop][vcm].pop(fi)
                        next_cm = vcm + 1
                        if next_cm < len(temporal_mapping[vop]):
                            temporal_mapping[vop][next_cm].insert(0, (dim, factor))
                        demoted.append({
                            "miredo_level": miredo_name,
                            "operand": vop,
                            "cm_level": vcm,
                            "dim": dim,
                            "factor": factor,
                            "reason": "capacity_overflow",
                        })
                        found = True
                        break
                if found:
                    break

            if not found:
                break  # no relevant temporal factor left -- will be an anomaly

    return demoted

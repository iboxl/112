# CoSA map_16.yaml → ClassMapping (only naming / schema alignment, no invention).
#
# CoSA emits timeloop-compatible mapping YAML with 6 simba storage levels
# (Registers / AccumulationBuffer / WeightBuffer / InputBuffer / GlobalBuffer
# / DRAM) as per-target entries of type {temporal, spatial, bypass}. We
# translate simba levels to MIREDO storage names (see cosa_adapter.
# _SIMBA_TO_MIREDO), drop WeightBuffer (its keep/bypass pattern in our mapspace
# already funnels weights through GlobalBuffer), bubble any bypass-level loops
# outward per operand, and emit ClassMapping in the outer→inner convention
# ZigZag already consumes. MIREDO's IReg/OReg (no simba counterpart) get empty
# inner-level padding downstream via align_mapping_levels_to_acc.
#
# Restraint:
# - CoSA does not expose a double-buffer decision → double_buffer_flag = all False.
# - top_r_loop_size is a ZigZag artefact (no CoSA equivalent) → None.

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from utils.ClassMapping import ClassMapping
from utils.Workload import LoopNest
from Evaluation.Zigzag_imc.CompatibleZigzag import convert_baseMapping_to_MIREDO
from Evaluation.CoSA.cosa_adapter import CoSALayerOutput
from Evaluation.common.CapacityLegalization import legalize_capacity
from Architecture.HardwareSpec import HardwareSpec


_TIMELOOP_TO_MIREDO_DIM = {
    "R": "R", "S": "S", "P": "P", "Q": "Q",
    "C": "C", "K": "K",
}
_IGNORED_DIMS = {"N"}

_OPERANDS_MIREDO = ("I", "W", "O")
_TIMELOOP_DS_TO_OP = {"Inputs": "I", "Weights": "W", "Outputs": "O"}

# Simba level → MIREDO spatial-axis name. None = level has S=1 → no physical
# spatial slot; any spatial factor here is non-realizable and will be demoted.
# Derived from instances=[4096,128,128,8,1,1] → S=[1,32,1,16,8,1]:
#   AccumulationBuffer S=32 → dimX  (allowed_loops: R, S, C)
#   InputBuffer        S=16 → dimY  (allowed_loops: K)
#   GlobalBuffer       S=8  → cores (allowed_loops: P, Q, K, G)
_SIMBA_TO_AXIS = {
    "Registers":          None,
    "AccumulationBuffer": "dimX",
    "WeightBuffer":       None,
    "InputBuffer":        "dimY",
    "GlobalBuffer":       "cores",
    "DRAM":               None,
}


def _parse_cosa_factor_string(factors_str: str) -> Dict[str, int]:
    """CoSA emits `R=1 S=1 P=7 Q=1 C=1 K=1 N=1` (space-separated tokens with
    '='). Distinct from the timeloop v4 `C1 M1 R1 ...` form used by CIMLoop."""
    out: Dict[str, int] = {}
    for token in str(factors_str).split():
        m = re.match(r"^([A-Za-z]+)=(\d+)$", token)
        if not m:
            raise ValueError(f"cannot parse factor token {token!r} in {factors_str!r}")
        out[m.group(1)] = int(m.group(2))
    return out


def _factors_in_permutation_order(
    factor_map: Dict[str, int],
    permutation: str,
) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for ch in permutation:
        if ch in _IGNORED_DIMS:
            continue
        miredo_dim = _TIMELOOP_TO_MIREDO_DIM.get(ch)
        if miredo_dim is None:
            continue
        factor = int(factor_map.get(ch, 1))
        if factor > 1:
            result.append((miredo_dim, factor))
    return result


def _load_entries(map_yaml_path: Path) -> List[Dict[str, Any]]:
    with open(map_yaml_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "mapping" not in data:
        raise ValueError(f"{map_yaml_path}: expected top-level 'mapping' key")
    entries = data["mapping"]
    if not isinstance(entries, list):
        raise ValueError(f"{map_yaml_path}: 'mapping' must be a list")
    return entries


def _group_entries(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries:
        tgt = e.get("target")
        typ = e.get("type")
        if tgt is None or typ is None:
            raise ValueError(f"bad mapping entry (missing target/type): {e}")
        grouped.setdefault(tgt, {})[typ] = e
    return grouped


def _legalize_spatial_per_level(
    per_level_full: List[Dict[str, Any]],
    axis_allowed_loops: Dict[str, frozenset],
) -> List[Dict[str, Any]]:
    """Demote spatial factors whose dim is not allowed at the simba level's
    MIREDO axis. Moves (dim, factor) from lv['spatial'] → lv['temporal'] at
    the same simba level. Tile sizes at each Simulax memory level are
    preserved (tile = dim_tp × dim_sp at the same level regardless of
    temporal/spatial split), so no OverSize regression. Returns the list of
    demoted entries for metadata transparency."""
    demoted: List[Dict[str, Any]] = []
    for lv in per_level_full:
        axis = _SIMBA_TO_AXIS.get(lv["simba_name"])
        if axis is None:
            # S=1 slot — no physical spatial fanout; demote any stray entries.
            if lv["spatial"]:
                for (d, n) in lv["spatial"]:
                    demoted.append({
                        "simba_level": lv["simba_name"],
                        "axis": None,
                        "dim": d,
                        "factor": n,
                        "reason": "no_spatial_slot",
                    })
                lv["temporal"] = list(lv["spatial"]) + list(lv["temporal"])
                lv["spatial"] = []
            continue
        allowed = axis_allowed_loops[axis]
        legal, illegal = [], []
        for (d, n) in lv["spatial"]:
            (legal if d in allowed else illegal).append((d, n))
        if illegal:
            for (d, n) in illegal:
                demoted.append({
                    "simba_level": lv["simba_name"],
                    "axis": axis,
                    "dim": d,
                    "factor": n,
                    "reason": "dim_not_allowed_at_axis",
                })
            lv["temporal"] = list(lv["temporal"]) + illegal
            lv["spatial"] = legal
    return demoted


def baseMapping_from_cosa_output(
    out: CoSALayerOutput,
    spec: HardwareSpec,
):
    """Parse CoSA map_16.yaml into a MIREDO ClassMapping (outer→inner order).

    Returns ``(ClassMapping, legalization_metadata)`` where
    ``legalization_metadata`` contains ``"demoted"`` (list of dicts) and
    ``"demoted_count"`` (int) recording spatial factors that CoSA's MIP placed
    on incorrect MIREDO axes and were demoted to temporal at the same simba
    level.
    """
    entries = _load_entries(out.map_yaml_path)
    grouped = _group_entries(entries)

    # Walk simba levels inner → outer (ClassMapping convention, matching how
    # CompatibleZigzag's convert_ZZMP_to_loopMP pairs list[0]=innermost with
    # the innermost memory level in mappingArray). Drop simba_to_miredo=None
    # (WeightBuffer) — its factors are bubbled outward below. Record keep /
    # bypass + parsed temporal + spatial factors.
    per_level_full: List[Dict[str, Any]] = []
    for simba_name in out.simba_level_names:
        miredo_name = out.simba_to_miredo.get(simba_name)
        blk = grouped.get(simba_name, {})

        # Bypass / keep
        bp_entry = blk.get("bypass") or {}
        keep_ops = {_TIMELOOP_DS_TO_OP[d] for d in bp_entry.get("keep", []) if d in _TIMELOOP_DS_TO_OP}

        # Temporal
        t_entry = blk.get("temporal")
        if t_entry is not None:
            t_fmap = _parse_cosa_factor_string(t_entry.get("factors", ""))
            t_perm = str(t_entry.get("permutation", ""))
            temporal_parsed = _factors_in_permutation_order(t_fmap, t_perm)
        else:
            temporal_parsed = []

        # Spatial
        s_entry = blk.get("spatial")
        if s_entry is not None:
            s_fmap = _parse_cosa_factor_string(s_entry.get("factors", ""))
            s_perm = str(s_entry.get("permutation", ""))
            spatial_parsed = _factors_in_permutation_order(s_fmap, s_perm)
        else:
            spatial_parsed = []

        per_level_full.append({
            "simba_name": simba_name,
            "miredo_name": miredo_name,  # None for dropped levels (WeightBuffer)
            "keep": keep_ops,
            "temporal": temporal_parsed,
            "spatial": spatial_parsed,
        })

    # Legalize spatial factors against MIREDO per-axis allowed_loops. CoSA's MIP
    # treats each simba S[i] slot as a pure capacity budget, unaware that each
    # slot corresponds to a specific MIREDO axis with restricted dim routing
    # (dimX→RSC, dimY→K, cores→PQKG). Demoting illegal factors to temporal at
    # the same simba level preserves tile sizes (Simulax tile = dim_tp × dim_sp
    # regardless of temporal/spatial split) and memory access counts; only the
    # available parallelism changes to reflect physical hardware constraints.
    axis_allowed_loops = {
        ax.name: frozenset(ax.allowed_loops) for ax in spec.macro.spatial_axes
    }
    legalization_demoted = _legalize_spatial_per_level(per_level_full, axis_allowed_loops)

    # Drop simba levels that have no MIREDO counterpart (WeightBuffer).
    # Factors at dropped levels move outward to the next kept-and-non-dropped
    # simba level, preserving CoSA's iteration-count semantics at every MIREDO
    # level (dropping a level in the middle of the linear nest keeps inner /
    # outer iteration products unchanged). Loop-order within the destination
    # level's temporal list is best-effort — MIREDO's Simulax consumes
    # per-level factor lists without enforcing intra-level ordering.
    per_level: List[Dict[str, Any]] = [lv for lv in per_level_full if lv["miredo_name"] is not None]

    dropped_carryover_t: List[Tuple[str, int]] = []
    dropped_carryover_s: List[Tuple[str, int]] = []
    for full_idx, lv in enumerate(per_level_full):
        if lv["miredo_name"] is not None:
            if dropped_carryover_t:
                lv["temporal"] = list(dropped_carryover_t) + list(lv["temporal"])
                dropped_carryover_t = []
            if dropped_carryover_s:
                lv["spatial"] = list(dropped_carryover_s) + list(lv["spatial"])
                dropped_carryover_s = []
        else:
            dropped_carryover_t.extend(lv["temporal"])
            dropped_carryover_s.extend(lv["spatial"])
    if dropped_carryover_t or dropped_carryover_s:
        raise ValueError(
            "CoSA map: factors at a dropped simba level have no outer non-dropped "
            f"level to absorb them (temporal={dropped_carryover_t}, "
            f"spatial={dropped_carryover_s}); check _SIMBA_TO_MIREDO mapping."
        )

    # per_level is inner→outer over kept simba levels. Build per-operand lists
    # (ClassMapping convention: list[0] = innermost), containing only levels
    # where the operand is kept.
    temporal_mapping: Dict[str, List[List[Tuple[str, int]]]] = {op: [] for op in _OPERANDS_MIREDO}
    spatial_mapping: Dict[str, List[List[Tuple[str, int]]]] = {op: [] for op in _OPERANDS_MIREDO}
    op_kept_index: Dict[str, Dict[str, int]] = {op: {} for op in _OPERANDS_MIREDO}

    for op in _OPERANDS_MIREDO:
        for idx, lv in enumerate(per_level):
            if op in lv["keep"]:
                op_kept_index[op][idx] = len(temporal_mapping[op])
                temporal_mapping[op].append([])
                spatial_mapping[op].append([])

    # Attribute each simba level's factors to the correct per-operand slot.
    # For bypass levels (loop is at this level but op is not kept): bubble
    # outward to the next outer keep level. per_level is inner→outer, so
    # "outward" = later index.
    for idx, lv in enumerate(per_level):
        for op in _OPERANDS_MIREDO:
            if op in lv["keep"]:
                target_idx = idx
            else:
                # walk outward (toward higher idx)
                target_idx = None
                for j in range(idx + 1, len(per_level)):
                    if op in per_level[j]["keep"]:
                        target_idx = j
                        break
                if target_idx is None:
                    # op has no outer keep level — should not happen for a valid
                    # arch since DRAM keeps all ops.
                    if lv["temporal"] or lv["spatial"]:
                        raise ValueError(
                            f"CoSA map: operand '{op}' has factors at bypass level "
                            f"'{lv['simba_name']}' with no outer keep level"
                        )
                    continue
            if lv["temporal"]:
                temporal_mapping[op][op_kept_index[op][target_idx]].extend(lv["temporal"])
            if lv["spatial"]:
                spatial_mapping[op][op_kept_index[op][target_idx]].extend(lv["spatial"])

    # Pre-pad innermost empty levels so operand level counts match MIREDO's
    # 7-level spec directly. align_mapping_levels_to_acc would otherwise
    # append at the outer end of an inner→outer list, mis-pairing layers with
    # memory indices in convert_ZZMP_to_loopMP (inner→outer pairing via
    # reversed pos). Counts below track the canonical CIM_ACC_TEMPLATE
    # operand-kept hierarchy: I kept at IReg/Input_buffer/Global_buffer/Dram,
    # W at Macro/Global_buffer/Dram, O at OReg/Output_buffer/Global_buffer/Dram.
    _MIREDO_OP_KEPT_COUNT = {"I": 4, "W": 3, "O": 4}
    for op, target in _MIREDO_OP_KEPT_COUNT.items():
        cur = len(temporal_mapping[op])
        if cur < target:
            pad = target - cur
            temporal_mapping[op] = [[] for _ in range(pad)] + temporal_mapping[op]
            spatial_mapping[op] = [[] for _ in range(pad)] + spatial_mapping[op]

    # Capacity legalization: check per-level tile sizes (with correct Input
    # halo and per-operand precision) against MIREDO physical memory capacity.
    # Demotes temporal factors outward only when overflow actually occurs.
    capacity_demoted = legalize_capacity(
        temporal_mapping, spatial_mapping, spec, out.loopdim,
    )

    loop_dim = {
        "R": int(out.loopdim.get("R", 1)),
        "S": int(out.loopdim.get("S", 1)),
        "P": int(out.loopdim.get("P", 1)),
        "Q": int(out.loopdim.get("Q", 1)),
        "C": int(out.loopdim.get("C", 1)),
        "K": int(out.loopdim.get("K", 1)),
        "G": int(out.loopdim.get("G", 1)),
    }

    double_buffer_flag = {op: [False] * len(temporal_mapping[op]) for op in _OPERANDS_MIREDO}

    legalization_meta = {
        "demoted_count": len(legalization_demoted),
        "demoted": legalization_demoted,
        "capacity_demoted_count": len(capacity_demoted),
        "capacity_demoted": capacity_demoted,
    }
    return (
        ClassMapping(
            source="cosa",
            loop_dim=loop_dim,
            temporal_mapping=temporal_mapping,
            spatial_mapping=spatial_mapping,
            double_buffer_flag=double_buffer_flag,
            top_r_loop_size=None,
            raw=out,
        ),
        legalization_meta,
    )


def convert_CoSA_to_MIREDO(
    loops: LoopNest,
    out: CoSALayerOutput,
    spec: HardwareSpec,
):
    """Convert a CoSA mapping to a MIREDO LoopNest.

    Returns ``(LoopNest, legalization_metadata)`` — see
    ``baseMapping_from_cosa_output`` for metadata structure.
    """
    baseMapping, legalization_meta = baseMapping_from_cosa_output(out=out, spec=spec)
    loops = convert_baseMapping_to_MIREDO(loops=loops, baseMapping=baseMapping)
    return loops, legalization_meta

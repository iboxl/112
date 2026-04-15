# CIMLoop .map.yaml → ClassMapping 转换（只做命名/schema 对齐，不发明字段）
#
# Timeloop 的 mapping 输出是以 storage level 为主的条目列表（mapping.cpp:333-469）。
# 这里只负责解析成 MIREDO 的 ClassMapping 中介类型，随后走统一的
# convert_baseMapping_to_MIREDO → tranSimulator 路径。
#
# 克制原则：
# - M → K 命名翻译、X/Y/Z/N precision+batch 维度直接丢弃；
# - CIMLoop mapper 不暴露 double-buffer 决策 → double_buffer_flag = 全 False，
#   不发明 MIREDO-favorable 的默认值；macro-level double buffer 由 downstream
#   `loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro` 施加
#   （与 ZigZag 路径一致，属于硬件属性而非 mapping 决策）；
# - top_r_loop_size 是 ZigZag 特有的假设化浮窗，CIMLoop 没有对应语义，设为 None。

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from utils.ClassMapping import ClassMapping
from utils.Workload import LoopNest
from Evaluation.Zigzag_imc.CompatibleZigzag import convert_baseMapping_to_MIREDO
from Evaluation.CIMLoop.cimloop_adapter import CIMLoopLayerOutput


# Timeloop v4 flattened dim names present in our problem shape.
# R/S/P/Q/C/G are MIREDO dims; M maps to K; X/Y/Z are precision dims; N is batch.
_TIMELOOP_TO_MIREDO_DIM = {
    "R": "R", "S": "S", "P": "P", "Q": "Q",
    "C": "C", "M": "K", "G": "G",
}
_IGNORED_DIMS = {"X", "Y", "Z", "N"}

_OPERANDS_MIREDO = ("I", "W", "O")
_TIMELOOP_DS_TO_OP = {"Inputs": "I", "Weights": "W", "Outputs": "O"}


def _parse_factor_string(factors_str: str) -> Dict[str, int]:
    """Parse 'R1 S3 P7 Q7 C1 M64 N1 X1 Y1 Z1 G1' → {'R': 1, 'S': 3, ...}.
    Timeloop format (mapping.cpp:419-422): each token is <dim_char><factor_int>.
    """
    out: Dict[str, int] = {}
    for token in str(factors_str).split():
        m = re.match(r"^([A-Za-z]+)(\d+)$", token)
        if not m:
            raise ValueError(f"cannot parse factor token {token!r} in {factors_str!r}")
        out[m.group(1)] = int(m.group(2))
    return out


def _factors_in_permutation_order(
    factor_map: Dict[str, int],
    permutation: str,
    dim_filter: List[str] | None = None,
) -> List[Tuple[str, int]]:
    """Return (dim, factor) tuples in permutation order, keeping only factor > 1,
    translating timeloop dim → MIREDO dim, dropping ignored dims."""
    result: List[Tuple[str, int]] = []
    for ch in permutation:
        if dim_filter is not None and ch not in dim_filter:
            continue
        if ch in _IGNORED_DIMS:
            continue
        miredo_dim = _TIMELOOP_TO_MIREDO_DIM.get(ch)
        if miredo_dim is None:
            continue
        factor = int(factor_map.get(ch, 1))
        if factor > 1:
            result.append((miredo_dim, factor))
    return result


def _load_raw_entries(map_yaml_path: Path) -> List[Dict[str, Any]]:
    with open(map_yaml_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "mapping" not in data:
        raise ValueError(f"{map_yaml_path}: expected top-level 'mapping' key")
    entries = data["mapping"]
    if not isinstance(entries, list):
        raise ValueError(f"{map_yaml_path}: 'mapping' must be a list")
    return entries


def _group_entries_by_target(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Returns: {target_name: {type: entry_dict}} where type ∈ {datatype, spatial, temporal}."""
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries:
        tgt = e.get("target")
        typ = e.get("type")
        if tgt is None or typ is None:
            raise ValueError(f"bad mapping entry (missing target/type): {e}")
        grouped.setdefault(tgt, {})[typ] = e
    return grouped


def _next_outer_keep_level(per_level: List[Dict[str, Any]], bypass_idx: int, op: str) -> str | None:
    """Walk outward from `bypass_idx` (exclusive) in per_level (inner→outer order); return
    the first level name whose keep includes op. None if no such outer keep level exists.
    """
    for i in range(bypass_idx + 1, len(per_level)):
        if op in per_level[i]["keep"]:
            return per_level[i]["name"]
    return None


def _next_outer_keep_index(per_level: List[Dict[str, Any]], bypass_idx: int, op: str) -> int | None:
    for i in range(bypass_idx + 1, len(per_level)):
        if op in per_level[i]["keep"]:
            return i
    return None


def baseMapping_from_cimloop_output(out: CIMLoopLayerOutput) -> ClassMapping:
    """Parse timeloop-mapper .map.yaml into a MIREDO ClassMapping.

    Contract:
      - `out.storage_level_names` is inner→outer (level 0 = innermost).
      - ClassMapping per-operand lists are inner→outer, containing only levels
        where the operand is kept (not bypassed) at that storage.
      - Spatial factors are attributed per-operand according to
        `out.axis_sources[axis_idx][operand] = source_memory_name`.
    """
    entries = _load_raw_entries(out.map_yaml_path)
    grouped = _group_entries_by_target(entries)

    inner_to_outer = list(out.storage_level_names)  # inner → outer
    name_to_inner_idx = {name: i for i, name in enumerate(inner_to_outer)}

    # Step 1: for each storage level in inner→outer order, build keep-mask
    # and parsed temporal factors. Spatial factors come from fanout containers
    # (inter_<name>_spatial) in step 2, not from storage-level entries.
    per_level: List[Dict[str, Any]] = []
    for name in inner_to_outer:
        blk = grouped.get(name, {})
        dt = blk.get("datatype") or {}
        keep_ops = {_TIMELOOP_DS_TO_OP[d] for d in dt.get("keep", []) if d in _TIMELOOP_DS_TO_OP}

        temporal_entry = blk.get("temporal")
        if temporal_entry is not None:
            t_factors = _parse_factor_string(temporal_entry.get("factors", ""))
            t_perm = str(temporal_entry.get("permutation", ""))
            temporal_parsed = _factors_in_permutation_order(t_factors, t_perm)
        else:
            temporal_parsed = []

        per_level.append({
            "name": name,
            "keep": keep_ops,
            "temporal": temporal_parsed,
        })

    # Step 2: per spatial axis, extract the factors attributed to it.
    # Timeloop emits spatial entries with target=`inter_<container>_spatial`. Each fanout
    # `!Container` appears as its own target in .map.yaml. For an axis (container, side)
    # the relevant factors are:
    #   - side "X": spatial entry's SpaceX part (first `split` chars of permutation)
    #   - side "Y": spatial entry's SpaceY part (remaining chars)
    fanout_spatial: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
    for e in entries:
        if e.get("type") != "spatial":
            continue
        tgt = str(e.get("target", ""))
        if not tgt.startswith("inter_") or not tgt.endswith("_spatial"):
            continue
        container_name = tgt[len("inter_"):-len("_spatial")]
        factor_map = _parse_factor_string(e.get("factors", ""))
        perm = str(e.get("permutation", ""))
        split = int(e.get("split", len(perm)))
        space_x_chars = perm[:split]
        space_y_chars = perm[split:]
        fanout_spatial[container_name] = {
            "X": _factors_in_permutation_order(factor_map, space_x_chars),
            "Y": _factors_in_permutation_order(factor_map, space_y_chars),
        }

    axis_factors: List[List[Tuple[str, int]]] = []
    for container_name, side in out.axis_to_fanout:
        axis_factors.append(fanout_spatial.get(container_name, {}).get(side, []))

    # Step 3: build per-operand ClassMapping lists (inner → outer, only kept levels).
    # CIMLoop's linear arch may place loops at storage levels that are BYPASS for some
    # operand. From that operand's view, such loops are OUTER of its innermost keep
    # level but INNER of its next-outer keep level. MIREDO's ClassMapping convention
    # (matching ZigZag) only lists keep levels; bypass-level loops must be attributed
    # to the NEXT OUTER keep level for each operand (NOT fallback-to-innermost, which
    # would make tranSimulator think those loops' factors reside at the inner buffer
    # and inflate its tile size).
    temporal_mapping: Dict[str, List[List[Tuple[str, int]]]] = {op: [] for op in _OPERANDS_MIREDO}
    spatial_mapping: Dict[str, List[List[Tuple[str, int]]]] = {op: [] for op in _OPERANDS_MIREDO}
    op_level_index: Dict[str, Dict[str, int]] = {op: {} for op in _OPERANDS_MIREDO}
    # per-operand: list of (full_level_index_inner_to_outer → op_kept_level_index)
    # used to bubble bypass-level loops to the next outer keep level.
    op_keep_mask: Dict[str, List[bool]] = {op: [op in lv["keep"] for lv in per_level] for op in _OPERANDS_MIREDO}

    # Build empty lists sized to number of kept levels per operand.
    for op in _OPERANDS_MIREDO:
        kept_level_names: List[str] = []
        for lv in per_level:
            if op in lv["keep"]:
                op_level_index[op][lv["name"]] = len(temporal_mapping[op])
                temporal_mapping[op].append([])
                spatial_mapping[op].append([])
                kept_level_names.append(lv["name"])

    # Attribute each storage level's temporal factors to the correct per-operand slot:
    # - if level is keep for op: attribute to that level's slot in op's kept list
    # - if level is bypass for op: attribute to the next outer keep level for op
    #   (i.e., walk outward from this level until a keep level is found)
    for lv_idx, lv in enumerate(per_level):
        factors = lv["temporal"]
        if not factors:
            continue
        for op in _OPERANDS_MIREDO:
            if op in lv["keep"]:
                target_name = lv["name"]
            else:
                target_name = _next_outer_keep_level(per_level, lv_idx, op)
            if target_name is None:
                # Loop sits at an outer-bypass level with no further outer keep for op.
                # Silent drop here would be unsafe: convert_ZZMP_to_loopMP still emits
                # this loop (for other operands) and falls back to op's innermost keep,
                # recreating the tile-inflation that bubble-up was designed to avoid.
                # Surface explicitly — valid MIREDO hardware always has DRAM as an
                # outermost keep for I/W/O, so hitting this means a spec/arch mismatch.
                raise ValueError(
                    f"CIMLoop map.yaml: operand '{op}' has loop(s) {factors} at bypass "
                    f"level '{lv['name']}' with no outer keep level — cannot honour without "
                    f"triggering tranSimulator fallback-to-innermost (tile inflation). "
                    f"Check that HardwareSpec keeps {op} at DRAM."
                )
            temporal_mapping[op][op_level_index[op][target_name]].extend(factors)

    # Step 4: distribute spatial axis factors to each operand's kept-level.
    for axis_idx, factors in enumerate(axis_factors):
        if not factors:
            continue
        per_op_source = out.axis_sources[axis_idx]  # {"I": "Global_buffer", ...}
        for op in _OPERANDS_MIREDO:
            source_name = per_op_source.get(op)
            if source_name is None:
                continue
            # Find this operand's kept-level index whose name == source_name.
            idx = op_level_index[op].get(source_name)
            if idx is None:
                # Source memory not in operand's kept chain (shouldn't happen for valid spec);
                # skip silently to avoid synthesizing placement not present in CIMLoop's output.
                continue
            spatial_mapping[op][idx].extend(factors)

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

    return ClassMapping(
        source="cimloop",
        loop_dim=loop_dim,
        temporal_mapping=temporal_mapping,
        spatial_mapping=spatial_mapping,
        double_buffer_flag=double_buffer_flag,
        top_r_loop_size=None,
        raw=out,
    )


def convert_CIMLoop_to_MIREDO(loops: LoopNest, out: CIMLoopLayerOutput) -> LoopNest:
    baseMapping = baseMapping_from_cimloop_output(out)
    return convert_baseMapping_to_MIREDO(loops=loops, baseMapping=baseMapping)

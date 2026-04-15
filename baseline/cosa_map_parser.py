from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from baseline.types import BaselineLayer


OPERANDS = ("Weights", "Inputs", "Outputs")
MAP_TYPES = ("temporal", "spatial")
OPERAND_TO_BASELINE_KEY = {"Weights": "W", "Inputs": "I", "Outputs": "O"}
COARSE_DIMS = ("R", "S", "P", "Q", "C", "K", "N")


@dataclass
class CoSAConstraintEntry:
    target: str
    map_type: str
    factors: dict[str, int]
    permutation: str


@dataclass
class CoSATargetMapping:
    temporal: list[CoSAConstraintEntry] = field(default_factory=list)
    spatial: list[CoSAConstraintEntry] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)
    bypass: list[str] = field(default_factory=list)


@dataclass
class CoSAMapIR:
    source_path: str
    target_order_inner_to_outer: list[str]
    target_order_outer_to_inner: list[str]
    targets: dict[str, CoSATargetMapping]
    operand_paths_outer_to_inner: dict[str, list[str]]
    raw: dict[str, Any]
    source_loop_dim: dict[str, Any] | None = None


def _parse_factors(factors_expr: str) -> dict[str, int]:
    factors = {}
    for token in factors_expr.split():
        dim, val = token.split("=", 1)
        factors[dim] = int(val)
    return factors


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict) or "mapping" not in data:
        raise ValueError(f"Invalid CoSA map file: {path}")
    return data


def _build_target_order(mapping_items: list[dict[str, Any]]) -> list[str]:
    order = []
    seen = set()
    for item in mapping_items:
        target = item.get("target")
        if target is None or target in seen:
            continue
        seen.add(target)
        order.append(target)
    return order


def _build_targets(mapping_items: list[dict[str, Any]]) -> dict[str, CoSATargetMapping]:
    targets: dict[str, CoSATargetMapping] = {}
    for item in mapping_items:
        target = item["target"]
        mapping = targets.setdefault(target, CoSATargetMapping())
        map_type = item["type"]

        if map_type in MAP_TYPES:
            entry = CoSAConstraintEntry(
                target=target,
                map_type=map_type,
                factors=_parse_factors(item["factors"]),
                permutation=item["permutation"],
            )
            getattr(mapping, map_type).append(entry)
        elif map_type == "bypass":
            mapping.keep = list(item.get("keep", []))
            mapping.bypass = list(item.get("bypass", []))
        else:
            raise ValueError(f"Unsupported map type '{map_type}' in target '{target}'")
    return targets


def _derive_operand_paths(
    targets: dict[str, CoSATargetMapping],
    target_order_inner_to_outer: list[str],
) -> dict[str, list[str]]:
    paths = {}
    for operand in OPERANDS:
        path_inner_to_outer = []
        for target in target_order_inner_to_outer:
            keep = targets[target].keep
            if operand in keep:
                path_inner_to_outer.append(target)
        paths[operand] = list(reversed(path_inner_to_outer))
    return paths


def parse_cosa_map_16(path: str | Path) -> CoSAMapIR:
    map_path = Path(path).expanduser().resolve()
    data = _read_yaml(map_path)
    mapping_items = data["mapping"]
    if not isinstance(mapping_items, list):
        raise ValueError(f"'mapping' must be a list in {map_path}")

    target_order_inner_to_outer = _build_target_order(mapping_items)
    targets = _build_targets(mapping_items)
    operand_paths_outer_to_inner = _derive_operand_paths(targets, target_order_inner_to_outer)

    return CoSAMapIR(
        source_path=str(map_path),
        target_order_inner_to_outer=target_order_inner_to_outer,
        target_order_outer_to_inner=list(reversed(target_order_inner_to_outer)),
        targets=targets,
        operand_paths_outer_to_inner=operand_paths_outer_to_inner,
        raw=data,
    )


def _merge_entries(entries: list[CoSAConstraintEntry]) -> tuple[dict[str, int], str]:
    merged = {d: 1 for d in COARSE_DIMS}
    permutation = ""
    for entry in entries:
        if permutation == "":
            permutation = entry.permutation
        elif permutation != entry.permutation:
            raise ValueError(
                f"Permutation mismatch in target={entry.target}, type={entry.map_type}: "
                f"'{permutation}' vs '{entry.permutation}'"
            )
        for d, v in entry.factors.items():
            if d in merged:
                merged[d] *= v
    if permutation == "":
        permutation = "".join(COARSE_DIMS)
    return merged, permutation


def _factors_to_loop_tuples(factors: dict[str, int], permutation: str) -> list[tuple[str, int]]:
    tuples: list[tuple[str, int]] = []
    for dim in permutation:
        if dim not in factors:
            continue
        val = factors[dim]
        if val > 1:
            tuples.append((dim, val))
    return tuples


def _build_operand_mapping(ir: CoSAMapIR, map_type: str) -> dict[str, list[list[tuple[str, int]]]]:
    result: dict[str, list[list[tuple[str, int]]]] = {}
    for operand, baseline_key in OPERAND_TO_BASELINE_KEY.items():
        path = ir.operand_paths_outer_to_inner.get(operand, [])
        layers: list[list[tuple[str, int]]] = []
        for target in path:
            entries = getattr(ir.targets[target], map_type)
            merged, permutation = _merge_entries(entries)
            layers.append(_factors_to_loop_tuples(merged, permutation))
        result[baseline_key] = layers
    return result


def _infer_loop_dim(ir: CoSAMapIR) -> dict[str, int]:
    totals = {d: 1 for d in COARSE_DIMS}
    for target in ir.targets.values():
        for entry in target.temporal + target.spatial:
            for d in COARSE_DIMS:
                totals[d] *= entry.factors.get(d, 1)
    return {
        "R": totals["R"],
        "S": totals["S"],
        "P": totals["P"],
        "Q": totals["Q"],
        "C": totals["C"],
        "K": totals["K"],
        "B": totals["N"],
        # CoSA map_16 does not carry group metadata explicitly in this parser path.
        "G": 1,
        "StrideH": 1,
        "StrideW": 1,
        "DilationH": 1,
        "DilationW": 1,
        # Keep legacy aliases used by older code paths.
        "Stride": 1,
        "Dilation": 1,
    }


def ir_to_baseline_layer(ir: CoSAMapIR) -> BaselineLayer:
    temporal_mapping = _build_operand_mapping(ir, "temporal")
    spatial_mapping = _build_operand_mapping(ir, "spatial")

    double_buffer_flag = {
        "I": [False] + [False] * len(temporal_mapping["I"]),
        "W": [False] + [False] * len(temporal_mapping["W"]),
        "O": [False] + [False] * len(temporal_mapping["O"]),
    }

    # Prioritize source_loop_dim to preserve original G/Stride/Dilation semantics.
    # If unavailable, fall back to inferred dimensions from map_16.yaml.
    if ir.source_loop_dim is not None:
        loop_dim = dict(ir.source_loop_dim)
    else:
        loop_dim = _infer_loop_dim(ir)

    return BaselineLayer(
        source="cosa",
        loop_dim=loop_dim,
        temporal_mapping=temporal_mapping,
        spatial_mapping=spatial_mapping,
        double_buffer_flag=double_buffer_flag,
        top_r_loop_size={
            "I": [1] * (len(temporal_mapping["I"]) + 1),
            "W": [1] * (len(temporal_mapping["W"]) + 1),
            "O": [1] * (len(temporal_mapping["O"]) + 1),
        },
        raw=ir,
    )


def parse_cosa_map_16_to_baseline(path: str | Path) -> BaselineLayer:
    ir = parse_cosa_map_16(path)
    return ir_to_baseline_layer(ir)

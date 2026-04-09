from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from baseline.cosa_map_parser import parse_cosa_map_16_to_baseline
from baseline.types import BaselineLayer


_OPERAND_ALIAS = {
    "I": "I",
    "INPUT": "I",
    "INPUTS": "I",
    "W": "W",
    "WEIGHT": "W",
    "WEIGHTS": "W",
    "O": "O",
    "OUTPUT": "O",
    "OUTPUTS": "O",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}, expect mapping")
    return data


def _norm_operand_key(key: str) -> str:
    mapped = _OPERAND_ALIAS.get(str(key).strip().upper())
    if mapped is None:
        raise ValueError(f"Unsupported operand key '{key}'")
    return mapped


def _parse_loop_dim(data: dict[str, Any]) -> dict[str, int]:
    if "loop_dim" not in data or not isinstance(data["loop_dim"], dict):
        raise ValueError("Missing 'loop_dim' mapping in cimloop baseline yaml")

    raw = data["loop_dim"]
    required = ("R", "S", "P", "Q", "C", "K")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"loop_dim missing required keys: {missing}")

    loop_dim = {k: int(raw[k]) for k in required}
    loop_dim["G"] = int(raw.get("G", 1))

    for optional in ("B", "H", "W", "Stride", "Padding", "Dilation"):
        if optional in raw:
            loop_dim[optional] = int(raw[optional])
    return loop_dim


def _parse_factor_item(item: Any) -> tuple[str, int]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        dim, size = item[0], item[1]
        return str(dim), int(size)

    if isinstance(item, dict) and "dim" in item and "size" in item:
        return str(item["dim"]), int(item["size"])

    raise ValueError(f"Invalid factor item: {item}")


def _parse_operand_levels(node: Any, field_name: str) -> list[list[tuple[str, int]]]:
    if not isinstance(node, list):
        raise ValueError(f"{field_name} operand entry must be a list of levels")
    levels: list[list[tuple[str, int]]] = []
    for level in node:
        if not isinstance(level, list):
            raise ValueError(f"{field_name} level must be a list")
        levels.append([_parse_factor_item(x) for x in level])
    return levels


def _parse_mapping_block(data: dict[str, Any], field_name: str) -> dict[str, list[list[tuple[str, int]]]]:
    if field_name not in data or not isinstance(data[field_name], dict):
        raise ValueError(f"Missing '{field_name}' mapping in cimloop baseline yaml")

    raw = data[field_name]
    parsed: dict[str, list[list[tuple[str, int]]]] = {"I": [], "W": [], "O": []}
    for key, value in raw.items():
        op = _norm_operand_key(str(key))
        parsed[op] = _parse_operand_levels(value, field_name)
    return parsed


def _parse_double_buffer_flag(
    data: dict[str, Any],
    temporal_mapping: dict[str, list[list[tuple[str, int]]]],
) -> dict[str, list[bool]]:
    raw = data.get("double_buffer_flag", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("double_buffer_flag must be a mapping when provided")

    parsed: dict[str, list[bool]] = {}
    for op in ("I", "W", "O"):
        levels = len(temporal_mapping[op])
        op_raw = raw.get(op, raw.get(op.lower(), raw.get(op.upper(), None)))
        if op_raw is None:
            parsed[op] = [False] + [False] * levels
            continue
        if not isinstance(op_raw, list):
            raise ValueError(f"double_buffer_flag[{op}] must be list")

        values = [bool(x) for x in op_raw]
        if len(values) == levels:
            parsed[op] = [False] + values
        elif len(values) == levels + 1:
            parsed[op] = values
        else:
            raise ValueError(
                f"double_buffer_flag[{op}] length mismatch: got {len(values)}, expect {levels} or {levels + 1}"
            )
    return parsed


def _parse_top_r_loop_size(data: dict[str, Any], temporal_mapping: dict[str, list[list[tuple[str, int]]]]) -> dict[str, list[int]] | None:
    raw = data.get("top_r_loop_size")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("top_r_loop_size must be a mapping when provided")

    parsed: dict[str, list[int]] = {}
    for op in ("I", "W", "O"):
        levels = len(temporal_mapping[op])
        op_raw = raw.get(op, raw.get(op.lower(), raw.get(op.upper(), None)))
        if op_raw is None:
            parsed[op] = [1] * (levels + 1)
            continue
        if not isinstance(op_raw, list):
            raise ValueError(f"top_r_loop_size[{op}] must be list")
        values = [int(x) for x in op_raw]
        if len(values) == levels:
            parsed[op] = [1] + values
        elif len(values) == levels + 1:
            parsed[op] = values
        else:
            raise ValueError(
                f"top_r_loop_size[{op}] length mismatch: got {len(values)}, expect {levels} or {levels + 1}"
            )
    return parsed


def parse_cimloop_baseline_yaml(path: str | Path) -> BaselineLayer:
    yaml_path = Path(path).expanduser().resolve()
    data = _read_yaml(yaml_path)

    # Compatibility path: accept CoSA-style map yaml as an intermediate format.
    # Keep source="cosa" so conversion uses CoSA-specific ordering logic.
    if "mapping" in data:
        return parse_cosa_map_16_to_baseline(yaml_path)

    loop_dim = _parse_loop_dim(data)
    temporal_mapping = _parse_mapping_block(data, "temporal_mapping")
    spatial_mapping = _parse_mapping_block(data, "spatial_mapping")
    double_buffer_flag = _parse_double_buffer_flag(data, temporal_mapping)
    top_r_loop_size = _parse_top_r_loop_size(data, temporal_mapping)

    return BaselineLayer(
        source="cimloop",
        loop_dim=loop_dim,
        temporal_mapping=temporal_mapping,
        spatial_mapping=spatial_mapping,
        double_buffer_flag=double_buffer_flag,
        top_r_loop_size=top_r_loop_size,
        raw={"path": str(yaml_path), "payload": data},
    )

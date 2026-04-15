from collections import defaultdict, deque
import copy
from typing import Callable, Dict, List, Tuple

from baseline.types import BaselineLayer
from utils.GlobalUT import Logger
from utils.Workload import LoopNest, Mapping


def convert_cosa_mp_to_loopMP(
    mapping_dict: Dict[str, List[List[Tuple[str, int]]]],
    mappingArray: List[List[int]],
    mappingList: List[Mapping],
    *,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
    dim_code=None,
    layer_tags: Dict[str, List[str]] | None = None,
    tag_rank: Dict[str, int] | None = None,
) -> List[Mapping]:
    """
    CoSA-specific converter: preserve CoSA's outer->inner layer semantics and
    avoid forcing missing operand tuples into the innermost register level.
    """
    if dim_code is None:
        dim_code = {
            "FX": 1,
            "FY": 2,
            "OX": 3,
            "OY": 4,
            "R": 1,
            "S": 2,
            "P": 3,
            "Q": 4,
            "C": 5,
            "K": 6,
            "G": 7,
            "N": 7,
            "B": 7,
        }

    if len(mappingArray) != len(array_row_order):
        raise ValueError("mappingArray 行数必须等于 array_row_order 长度")

    # token = (layer_tag, (dim, size), occ_idx_in_layer_for_same_tuple)
    # layer_tag defaults to layer_idx, but CoSA path tags can be injected to avoid
    # merging tuples that come from different targets but happen to share layer_idx.
    level_map: Dict[str, Dict[Tuple[object, Tuple[str, int], int], deque[int]]] = {
        op: defaultdict(deque) for op in array_row_order
    }
    fallback_mem_level: Dict[str, int] = {}
    op_rank_mem_seq: Dict[str, List[Tuple[int, int]]] = {}

    for row_idx, op in enumerate(array_row_order):
        layers = mapping_dict[op]  # 外 -> 内
        pos = [col for col, v in enumerate(mappingArray[row_idx]) if v == 1]

        if len(pos) != len(layers):
            raise ValueError(
                f"{op}: 1 的数量 {len(pos)} ≠ 层数 {len(layers)}"
                "请检查 mappingArray / mapping_dict"
            )

        # Missing tuples should stay at a safe outer level unless explicitly moved inward.
        fallback_mem_level[op] = pos[0]
        op_rank_mem_seq[op] = []

        for layer_idx, layer in enumerate(layers):  # 外 -> 内
            mem_level = pos[layer_idx]
            layer_tag = (
                layer_tags[op][layer_idx]
                if layer_tags is not None and op in layer_tags and layer_idx < len(layer_tags[op])
                else layer_idx
            )
            layer_rank = tag_rank.get(layer_tag, layer_idx) if tag_rank is not None else layer_idx
            op_rank_mem_seq[op].append((int(layer_rank), int(mem_level)))
            occ_in_layer: Dict[Tuple[str, int], int] = defaultdict(int)
            for tup in layer:
                tup_key = tuple(tup)
                occ = occ_in_layer[tup_key]
                occ_in_layer[tup_key] += 1
                token = (layer_tag, tup_key, occ)
                level_map[op][token].append(mem_level)

    order: List[Tuple[object, Tuple[str, int], int]] = []
    seen_tokens = set()

    if tag_rank is None:
        max_levels = max(len(mapping_dict[op]) for op in array_row_order)
        for layer_idx in range(max_levels):  # 外 -> 内
            for op in array_row_order:
                layers = mapping_dict[op]
                if layer_idx >= len(layers):
                    continue
                layer_tag = (
                    layer_tags[op][layer_idx]
                    if layer_tags is not None and op in layer_tags and layer_idx < len(layer_tags[op])
                    else layer_idx
                )
                occ_in_layer: Dict[Tuple[str, int], int] = defaultdict(int)
                for tup in layers[layer_idx]:
                    tup_key = tuple(tup)
                    occ = occ_in_layer[tup_key]
                    occ_in_layer[tup_key] += 1
                    token = (layer_tag, tup_key, occ)
                    if token in seen_tokens:
                        continue
                    seen_tokens.add(token)
                    order.append(token)
    else:
        ranked_tokens: List[Tuple[int, int, int, Tuple[object, Tuple[str, int], int]]] = []
        for op_idx, op in enumerate(array_row_order):
            layers = mapping_dict[op]
            for layer_idx, layer in enumerate(layers):
                layer_tag = (
                    layer_tags[op][layer_idx]
                    if layer_tags is not None and op in layer_tags and layer_idx < len(layer_tags[op])
                    else layer_idx
                )
                layer_rank = int(tag_rank.get(layer_tag, layer_idx))
                occ_in_layer: Dict[Tuple[str, int], int] = defaultdict(int)
                for tup in layer:
                    tup_key = tuple(tup)
                    occ = occ_in_layer[tup_key]
                    occ_in_layer[tup_key] += 1
                    token = (layer_tag, tup_key, occ)
                    if token in seen_tokens:
                        continue
                    seen_tokens.add(token)
                    ranked_tokens.append((layer_rank, layer_idx, op_idx, token))

        ranked_tokens.sort(key=lambda x: (x[0], x[1], x[2]))
        order = [token for _, _, _, token in ranked_tokens]

    def _resolve_missing_level(op: str, token_layer_tag: object) -> int:
        if tag_rank is None:
            return int(last_level[op])

        token_rank = tag_rank.get(token_layer_tag)
        if token_rank is None:
            return int(last_level[op])

        candidates = [mem for rank, mem in op_rank_mem_seq.get(op, []) if rank <= int(token_rank)]
        if len(candidates) > 0:
            return int(candidates[-1])

        return int(fallback_mem_level[op])

    last_level = dict(fallback_mem_level)
    for token in order:
        token_layer_tag, tup, _ = token
        levels = []
        for op in array_row_order:
            dq = level_map[op].get(token)
            if dq:
                lv = dq.popleft()
            else:
                lv = _resolve_missing_level(op, token_layer_tag)
            last_level[op] = lv
            levels.append(lv)

        mappingList.append(
            Mapping(
                dim=dim_code.get(tup[0], tup[0]),
                dimSize=round(tup[1]),
                mem=levels,
            )
        )
    return mappingList


def _build_cosa_layer_tags(
    baseline: BaselineLayer,
    aligned_mapping: Dict[str, List[List[Tuple[str, int]]]],
    *,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
) -> Dict[str, List[str]] | None:
    if baseline.source != "cosa":
        return None
    if not hasattr(baseline, "raw"):
        return None

    ir = baseline.raw
    raw_paths = getattr(ir, "operand_paths_outer_to_inner", None)
    if raw_paths is None:
        return None

    op_name_map = {"I": "Inputs", "W": "Weights", "O": "Outputs"}
    layer_tags: Dict[str, List[str]] = {}
    for op in array_row_order:
        path = list(raw_paths.get(op_name_map[op], []))
        expected_len = len(aligned_mapping.get(op, []))
        if len(path) > expected_len:
            path = path[:expected_len]
        if len(path) < expected_len:
            pad = [f"__PAD_INNER_{idx}" for idx in range(expected_len - len(path))]
            path = path + pad
        layer_tags[op] = path
    return layer_tags


def _build_cosa_tag_rank(
    baseline: BaselineLayer,
    layer_tags: Dict[str, List[str]] | None,
) -> Dict[str, int] | None:
    if baseline.source != "cosa":
        return None
    if layer_tags is None:
        return None
    if not hasattr(baseline, "raw"):
        return None

    ir = baseline.raw
    target_order_outer_to_inner = getattr(ir, "target_order_outer_to_inner", None)
    if target_order_outer_to_inner is None:
        return None

    rank: Dict[str, int] = {
        str(tag): idx for idx, tag in enumerate(target_order_outer_to_inner)
    }
    next_rank = max(rank.values(), default=-1) + 1

    for tags in layer_tags.values():
        for tag in tags:
            key = str(tag)
            if key in rank:
                continue
            rank[key] = next_rank
            next_rank += 1

    return rank


def align_mapping_levels_to_acc(
    mapping_dict: Dict[str, List[List[Tuple[str, int]]]],
    mapping_array: List[List[int]],
    *,
    source_tag: str,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
    warn: Callable[[str], None] | None = None,
) -> Dict[str, List[List[Tuple[str, int]]]]:
    """
    Align mapping hierarchy depth to accelerator-visible levels.
    Missing levels are padded at the innermost side.
    """
    aligned = copy.deepcopy(mapping_dict)
    for row_idx, op in enumerate(array_row_order):
        if op not in aligned:
            raise ValueError(f"{source_tag}: missing operand mapping '{op}'")

        allowed_levels = sum(1 for v in mapping_array[row_idx] if v == 1)
        cur_levels = len(aligned[op])
        if cur_levels > allowed_levels:
            raise ValueError(
                f"{source_tag}: operand '{op}' has {cur_levels} levels, "
                f"but accelerator allows only {allowed_levels}"
            )
        if cur_levels < allowed_levels:
            gap = allowed_levels - cur_levels
            aligned[op] = aligned[op] + [[] for _ in range(gap)]
            if warn is not None:
                warn(
                    f"{source_tag}: padded operand '{op}' with "
                    f"{gap} empty inner level(s) ({cur_levels} -> {allowed_levels})"
                )
    return aligned


def align_double_buffer_flags_to_acc(
    double_buffer_flag: Dict[str, List[bool]],
    mapping_array: List[List[int]],
    *,
    source_tag: str,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
    warn: Callable[[str], None] | None = None,
) -> Dict[str, List[bool]]:
    """
    Align double-buffer flags to accelerator-visible levels.
    Output format matches ZigZag convention: [prefix] + per-level flags.
    """
    aligned: Dict[str, List[bool]] = {}
    for row_idx, op in enumerate(array_row_order):
        raw = list(double_buffer_flag.get(op, []))
        allowed_levels = sum(1 for v in mapping_array[row_idx] if v == 1)

        if len(raw) == 0:
            prefix = False
            level_flags: List[bool] = []
        elif len(raw) == allowed_levels:
            # Compatibility path for inputs without the ZigZag prefix bit.
            prefix = False
            level_flags = [bool(x) for x in raw]
        else:
            prefix = bool(raw[0])
            level_flags = [bool(x) for x in raw[1:]]

        if len(level_flags) > allowed_levels:
            raise ValueError(
                f"{source_tag}: operand '{op}' has {len(level_flags)} double-buffer flags, "
                f"but accelerator allows only {allowed_levels}"
            )
        if len(level_flags) < allowed_levels:
            gap = allowed_levels - len(level_flags)
            level_flags.extend([False] * gap)
            if warn is not None:
                warn(
                    f"{source_tag}: padded operand '{op}' double-buffer flags with "
                    f"{gap} false level(s)"
                )

        aligned[op] = [prefix] + level_flags

    return aligned


def fail_fast_if_capacity_mismatch(
    loops,
    *,
    source_tag: str,
    calc_explicit_mem_usage: Callable,
) -> None:
    """
    Fail fast before replay when mapped data footprint exceeds hardware capacity.
    """
    _, used_bits = calc_explicit_mem_usage(loops)
    op_names = ("I", "W", "O")
    violations = []

    for mem in range(1, loops.acc.Num_mem):
        cap = int(loops.acc.memSize[mem])
        used_total = int(sum(int(x) for x in used_bits[mem]))
        if used_total <= cap:
            continue

        per_op = []
        for op in range(3):
            if loops.acc.mappingArray[op][mem] == 1:
                per_op.append(f"{op_names[op]}={int(used_bits[mem][op])}")

        violations.append(
            f"{loops.acc.mem2dict(mem)}(mem={mem}): used={used_total}, cap={cap}, "
            f"overflow={used_total - cap}, detail[{', '.join(per_op)}]"
        )

    if violations:
        raise ValueError(
            f"{source_tag}: mapping/hardware mismatch before replay. "
            f"Capacity overflow detected -> {' | '.join(violations)}"
        )


def _assert_mapping_unrolling_complete(loops: LoopNest, source_tag: str) -> None:
    unrolling = [1 for _ in loops.ops.dim2Dict]
    for mapping in loops.tm + loops.sm:
        if mapping.dim < 0 or mapping.dim >= len(unrolling):
            raise ValueError(
                f"{source_tag}: illegal dimension id {mapping.dim} in mapping before replay"
            )
        unrolling[mapping.dim] *= int(round(mapping.dimSize))

    deficits = []
    for dim, dim_name in enumerate(loops.ops.dim2Dict):
        bound = int(loops.ops.dim2bound[dim])
        got = int(unrolling[dim])
        if got < bound:
            deficits.append(f"{dim_name}: got={got}, need={bound}")

    if deficits:
        raise ValueError(
            f"{source_tag}: incomplete unrolling before replay -> " + "; ".join(deficits)
        )


def _compute_unrolling(loops: LoopNest) -> list[int]:
    unrolling = [1 for _ in loops.ops.dim2Dict]
    for mapping in loops.tm + loops.sm:
        if mapping.dim < 0 or mapping.dim >= len(unrolling):
            continue
        unrolling[mapping.dim] *= int(round(mapping.dimSize))
    return unrolling


def _build_raw_dim_totals_from_cosa_map(baseline: BaselineLayer) -> Dict[str, int] | None:
    if baseline.source != "cosa" or not hasattr(baseline, "raw"):
        return None
    ir = baseline.raw
    targets = getattr(ir, "targets", None)
    if targets is None:
        return None

    totals = {"R": 1, "S": 1, "P": 1, "Q": 1, "C": 1, "K": 1, "N": 1}
    for target in targets.values():
        for entry in target.temporal + target.spatial:
            for dim in totals:
                totals[dim] *= int(entry.factors.get(dim, 1))
    return totals


def _patch_missing_unrolling_from_raw_map(
    loops: LoopNest,
    baseline: BaselineLayer,
    *,
    source_tag: str,
) -> None:
    raw_totals = _build_raw_dim_totals_from_cosa_map(baseline)
    if raw_totals is None:
        return

    dim_alias = {"B": "N", "G": "N"}
    current = _compute_unrolling(loops)
    appended: List[Tuple[str, int]] = []

    recover_mem = []
    for op in range(3):
        valid = [m for m in range(loops.acc.Num_mem) if int(loops.acc.mappingArray[op][m]) == 1]
        if not valid:
            recover_mem.append(int(loops.acc.Dram2mem))
            continue
        valid_sorted = sorted(valid)
        chosen = int(valid_sorted[-1])
        if chosen == int(loops.acc.Dram2mem) and len(valid_sorted) > 1:
            chosen = int(valid_sorted[-2])
        recover_mem.append(chosen)

    for dim, dim_name in enumerate(loops.ops.dim2Dict):
        need = int(loops.ops.dim2bound[dim])
        got = int(current[dim])
        if got >= need:
            continue

        raw_dim_name = dim_alias.get(dim_name, dim_name)
        raw_total = int(raw_totals.get(raw_dim_name, 1))
        if raw_total < need:
            continue

        if need % max(1, got) != 0:
            continue

        missing_factor = int(need // max(1, got))
        if missing_factor <= 1:
            continue

        mem_for_fix = list(recover_mem)
        # Missing factors often originate from bypass-only WeightBuffer levels.
        # Put recovered factors at DRAM in spatial mapping so they do not
        # inflate inner-memory footprints (IReg/Macro/Output_buffer).
        if dim_name in ("R", "S", "P", "Q", "C", "K"):
            mem_for_fix[0] = int(loops.acc.Dram2mem)
            mem_for_fix[1] = int(loops.acc.Dram2mem)
            mem_for_fix[2] = int(loops.acc.Dram2mem)

        loops.sm.append(
            Mapping(
                dim=dim,
                dimSize=missing_factor,
                mem=mem_for_fix,
            )
        )
        current[dim] *= missing_factor
        appended.append((dim_name, missing_factor))

    if appended:
        detail = ", ".join([f"{d}x{f}" for d, f in appended])
        Logger.warning(
            f"{source_tag}: recovered missing unrolling factors from CoSA raw map: {detail}"
        )


def convert_cosa_baseline_to_MIREDO(loops: LoopNest, baseline: BaselineLayer) -> LoopNest:
    """
    CoSA-only conversion path.
    This path is intentionally isolated from Evaluation/Zigzag_imc/CompatibleZigzag.py.
    """
    if baseline.source != "cosa":
        raise ValueError(f"CoSA converter received baseline source='{baseline.source}'")

    temporal_aligned = align_mapping_levels_to_acc(
        mapping_dict=baseline.temporal_mapping,
        mapping_array=loops.acc.mappingArray,
        source_tag="cosa:temporal",
        warn=Logger.warning,
    )
    spatial_aligned = align_mapping_levels_to_acc(
        mapping_dict=baseline.spatial_mapping,
        mapping_array=loops.acc.mappingArray,
        source_tag="cosa:spatial",
        warn=Logger.warning,
    )
    _ = align_double_buffer_flags_to_acc(
        double_buffer_flag=baseline.double_buffer_flag,
        mapping_array=loops.acc.mappingArray,
        source_tag="cosa:double_buffer",
        warn=Logger.warning,
    )

    temporal_layer_tags = _build_cosa_layer_tags(
        baseline=baseline,
        aligned_mapping=temporal_aligned,
    )
    temporal_tag_rank = _build_cosa_tag_rank(
        baseline=baseline,
        layer_tags=temporal_layer_tags,
    )
    loops.tm = convert_cosa_mp_to_loopMP(
        mapping_dict=temporal_aligned,
        mappingArray=loops.acc.mappingArray,
        mappingList=loops.tm,
        layer_tags=temporal_layer_tags,
        tag_rank=temporal_tag_rank,
    )

    spatial_layer_tags = _build_cosa_layer_tags(
        baseline=baseline,
        aligned_mapping=spatial_aligned,
    )
    spatial_tag_rank = _build_cosa_tag_rank(
        baseline=baseline,
        layer_tags=spatial_layer_tags,
    )
    loops.sm = convert_cosa_mp_to_loopMP(
        mapping_dict=spatial_aligned,
        mappingArray=loops.acc.mappingArray,
        mappingList=loops.sm,
        layer_tags=spatial_layer_tags,
        tag_rank=spatial_tag_rank,
    )

    if not loops.tm:
        loops.tm.append(
            Mapping(
                dim=0,
                dimSize=1,
                mem=[loops.acc.Dram2mem for _ in range(3)],
            )
        )

    _patch_missing_unrolling_from_raw_map(
        loops,
        baseline,
        source_tag="cosa:unrolling_recover",
    )
    _assert_mapping_unrolling_complete(loops, source_tag="cosa:unrolling_check")

    # CoSA parser currently exports all-false dflags; keep explicit and safe here.
    loops.usr_defined_double_flag = [
        [0 for _ in range(3)] for __ in range(loops.acc.Num_mem + 1)
    ]
    loops.usr_defined_double_flag[loops.acc.Macro2mem][1] = loops.acc.double_Macro

    return loops
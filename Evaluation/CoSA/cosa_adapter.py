# HardwareSpec → CoSA (simba-style) arch adapter.
#
# CoSA accepts 6-level simba-style storage (Registers / AccumulationBuffer /
# WeightBuffer / InputBuffer / GlobalBuffer / DRAM) and a cnn-layer problem
# (R, S, P, Q, C, K, N + stride/dilation). It runs a Gurobi MIP to pick the
# tile factors + spatial/temporal split + outer permutation, then emits a
# map_16.yaml mapping file.
#
# We fold MIREDO's 7-level hierarchy into simba:
#   Registers          ← Macro (weights); entries=1 blocks the MIP from
#                        placing inner-loop factors here. MIREDO IReg/OReg
#                        each hold 1 element and Simulax's OverSize check
#                        sums temporal+spatial factors across inner levels;
#                        keeping Macro empty is the only way to stay
#                        simulator-safe.
#   AccumulationBuffer ← Output_buffer share per dimX-group (OReg stays empty).
#   WeightBuffer       ← placeholder with entries=1, bypass pattern avoids it.
#   InputBuffer        ← Input_buffer per core (IReg stays empty).
#   GlobalBuffer       ← Global_buffer.
#   DRAM               ← Dram.
#
# Intra-core dimX/dimY parallelism is exposed via simba `instances` ratios so
# CoSA sees the full 4096 = cores × dimX × dimY MAC array. CoSA computes
# spatial fanout from `inner_instances / cur_instances` (see
# cosa_input_objs.Arch.gen_spatial_constraint), so the sequence
# [4096, 128, 128, 8, 1, 1] yields per-level S = [1, 32, 1, 16, 8, 1]
# which matches MIREDO's [cores=8 × dimX=32 × dimY=16] axes.
#
# CoSA's MIP does not respect MIREDO's per-axis allowed_loops (cores→PQKG,
# dimX→RSC, dimY→K). Mapping legality is therefore enforced downstream by
# Simulax's OverSize check: an illegal assignment (e.g. K spatial at dimX)
# inflates OReg/IReg tiles and raises Dataflow Over MemSize Error, which
# the baseline-comparison runner records as an anomaly.
#
# Only the MIP solver is invoked from the CoSA submodule; timeloop-model is
# not called (performance evaluation is done by MIREDO's Simulax).

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from Architecture.HardwareSpec import HardwareSpec


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_COSA_ADAPTER_DIR = _THIS_FILE.parent
_COSA_REPO = _COSA_ADAPTER_DIR / "cosa"
_COSA_SRC = _COSA_REPO / "src"


# ---------------------------------------------------------------------------
# CoSA submodule loader
# ---------------------------------------------------------------------------

_cosa_module = None


def _load_cosa_module():
    """Import the cosa.cosa module from the submodule (deferred to first call).

    Returns the module object; callers access cosa(), Prob, Arch, Mapspace,
    _A, _B, utils as module attributes.  Importing the module triggers
    Gurobi's licence check and check_timeloop_version() (warning only).
    """
    global _cosa_module
    if _cosa_module is not None:
        return _cosa_module
    cosa_src = str(_COSA_SRC)
    if cosa_src not in sys.path:
        sys.path.insert(0, cosa_src)
    os.environ.setdefault("COSA_DIR", str(_COSA_REPO))
    _cosa_module = importlib.import_module("cosa.cosa")
    return _cosa_module


# ---------------------------------------------------------------------------
# Public metadata for CompatibleCoSA parser
# ---------------------------------------------------------------------------

# Simba storage levels, inner → outer (matches CoSA Arch.mem_name indexing).
_SIMBA_INNER_TO_OUTER = [
    "Registers", "AccumulationBuffer", "WeightBuffer",
    "InputBuffer", "GlobalBuffer", "DRAM",
]

# Simba → MIREDO storage name (None means "drop the loop; no MIREDO level").
# Registers→Macro is weights; AccumulationBuffer→Output_buffer lets OReg stay
# empty so Simulax does not charge Macro-level factors against OReg's tile.
# Same logic for InputBuffer→Input_buffer (IReg stays empty). WeightBuffer is a
# bypass placeholder — no loops reach it under the mapspace below.
_SIMBA_TO_MIREDO: Dict[str, Optional[str]] = {
    "Registers": "Macro",
    "AccumulationBuffer": "Output_buffer",
    "WeightBuffer": None,
    "InputBuffer": "Input_buffer",
    "GlobalBuffer": "Global_buffer",
    "DRAM": "Dram",
}


@dataclass
class CoSALayerOutput:
    layer_fp: str
    loopdim: Dict[str, int]
    objective: str
    map_yaml_path: Path
    simba_level_names: List[str]
    simba_to_miredo: Dict[str, Optional[str]]
    runtime_s: float


# ---------------------------------------------------------------------------
# HardwareSpec → simba arch.yaml
# ---------------------------------------------------------------------------

def _storage_level_by_name(spec: HardwareSpec, name: str):
    for mem in spec.memory_hierarchy:
        if mem.name == name:
            return mem
    raise ValueError(f"storage level {name!r} not in spec.memory_hierarchy")


def _bits_to_entries(capacity_bits: int, word_bits: int) -> int:
    if word_bits <= 0:
        return 1
    return max(1, int(capacity_bits) // int(word_bits))


def _render_arch_yaml(spec: HardwareSpec) -> str:
    mac = spec.macro
    glb = _storage_level_by_name(spec, "Global_buffer")
    ibuf = _storage_level_by_name(spec, "Input_buffer")
    obuf = _storage_level_by_name(spec, "Output_buffer")

    input_bits = int(mac.precision.I)
    weight_bits = int(mac.precision.W)
    psum_bits = int(mac.precision.psum)
    n_cores = int(spec.cores)
    dim_x = int(mac.dimX)
    dim_y = int(mac.dimY)

    mac_total = n_cores * dim_x * dim_y
    # AccumulationBuffer holds Outputs shared across dimX accumulation within a
    # core — one instance per (core × dimY) lane, so mac_total / dimX.
    accum_instances = n_cores * dim_y
    # WeightBuffer is a pass-through placeholder; CoSA's _B forces Weights to
    # flow Registers / WeightBuffer / DRAM. Keeping it at accum_instances lets
    # the S[2]=128/128=1 boundary carry no spatial fanout.
    weight_buffer_instances = accum_instances
    # InputBuffer is per-core (MIREDO Input_buffer replication=per_core).
    input_instances = n_cores

    registers_entries = 1
    # Per-instance Output staging: per-core Output_buffer capacity divided
    # among dimY lanes, so MIP can't allocate more output psums than the
    # real buffer would hold.
    accumulation_entries = max(1, _bits_to_entries(obuf.size_bits, psum_bits) // dim_y)
    weight_buffer_entries = 1
    input_entries = _bits_to_entries(ibuf.size_bits, input_bits)
    global_entries = _bits_to_entries(glb.size_bits, input_bits)

    # cluster-size values are not used by the MIP but must be consistent with
    # the instances ratios.
    registers_cluster = dim_x * dim_y
    accum_cluster = dim_y

    return f"""arch:
  arithmetic:
    instances: {mac_total}
    word-bits: {input_bits}
  storage:
  - name: Registers
    entries: {registers_entries}
    instances: {mac_total}
    word-bits: {weight_bits}
    cluster-size: {registers_cluster}
    num-ports: 2
    num-banks: 8
  - name: AccumulationBuffer
    entries: {accumulation_entries}
    instances: {accum_instances}
    word-bits: {psum_bits}
    cluster-size: {accum_cluster}
    network-word-bits: {psum_bits}
    num-ports: 2
    num-banks: 2
  - name: WeightBuffer
    entries: {weight_buffer_entries}
    instances: {weight_buffer_instances}
    word-bits: {weight_bits}
    block-size: 1
    num-ports: 1
    num-banks: 1
  - name: InputBuffer
    entries: {input_entries}
    instances: {input_instances}
    word-bits: {input_bits}
    block-size: 8
    num-ports: 2
    num-banks: 1
  - name: GlobalBuffer
    entries: {global_entries}
    instances: 1
    word-bits: {input_bits}
    block-size: 8
    num-ports: 2
    num-banks: 256
  - name: DRAM
    technology: "DRAM"
    instances: 1
    word-bits: {input_bits}
    block-size: 64
    bandwidth: 20.0
"""


# Mapspace adapted from simba defaults, with one change: GlobalBuffer also
# keeps Weights. The simba default bypasses Weights at GlobalBuffer (W flows
# DRAM→WBuf→Registers), but MIREDO keeps W at Global_buffer. Matching MIREDO
# here gives a 3-level W chain (DRAM/GlobalBuffer/Registers after dropping
# WBuf), aligning with MIREDO's 3 W-levels (Dram/Global_buffer/Macro). WBuf
# stays on the path as a placeholder with entries=1 so no tile lands there.
_MAPSPACE_YAML = """mapspace:
  constraints:
  - target: Registers
    type: datatype
    keep:
    - Weights
    bypass:
    - Inputs
    - Outputs
  - target: AccumulationBuffer
    type: datatype
    keep:
    - Outputs
    bypass:
    - Weights
    - Inputs
  - target: WeightBuffer
    type: datatype
    keep:
    - Weights
    bypass:
    - Inputs
    - Outputs
  - target: InputBuffer
    type: datatype
    keep:
    - Inputs
    bypass:
    - Weights
    - Outputs
  - target: GlobalBuffer
    type: datatype
    keep:
    - Inputs
    - Weights
    - Outputs
    bypass: []
"""


def _render_prob_yaml(loopdim: Dict[str, int]) -> str:
    g = int(max(1, loopdim.get("G", 1)))
    if g != 1:
        raise NotImplementedError(
            f"CoSA adapter does not support grouped convolution (G={g}); "
            "CoSA's cnn-layer shape has no G dimension and a G-fold temporal "
            "unroll would break the MIP's factorization."
        )
    return f"""problem:
  C: {int(loopdim.get('C', 1))}
  K: {int(loopdim.get('K', 1))}
  R: {int(loopdim.get('R', 1))}
  S: {int(loopdim.get('S', 1))}
  P: {int(loopdim.get('P', 1))}
  Q: {int(loopdim.get('Q', 1))}
  N: 1
  Hstride: {int(loopdim.get('Stride', 1))}
  Wstride: {int(loopdim.get('Stride', 1))}
  Hdilation: {int(loopdim.get('Dilation', 1))}
  Wdilation: {int(loopdim.get('Dilation', 1))}
  shape: cnn-layer
"""


# ---------------------------------------------------------------------------
# Layer fingerprint (cache key)
# ---------------------------------------------------------------------------

def loopdim_fingerprint(loopdim: Dict[str, int]) -> str:
    canon_keys = (
        "R", "S", "P", "Q", "C", "K", "G",
        "H", "W", "Stride", "Padding", "Dilation", "B",
    )
    # ONNX dynamic-batch inputs carry B=None in loopdim; treat None as 1 but
    # keep numeric 0 intact so Padding=0 layers don't collide with Padding=1.
    def _coerce(v):
        return 1 if v is None else int(v)
    canon = {k: _coerce(loopdim.get(k)) for k in canon_keys}
    raw = json.dumps(canon, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Mapper invocation
# ---------------------------------------------------------------------------

def _run_cosa_mip_and_write_mapping(cm, prob_path, arch_path, mapspace_path, work_dir):
    """Run CoSA's MIP solver and write the resulting mapping YAML.

    Replicates the orchestration from cosa.cosa.run_timeloop() but skips the
    redundant timeloop-model subprocess call — performance evaluation is done
    downstream by MIREDO's Simulax.

    Returns the path to the written map_16.yaml.
    """
    prob = cm.Prob(prob_path)
    arch = cm.Arch(arch_path)
    mapspace = cm.Mapspace(mapspace_path)
    mapspace.init(prob, arch)

    part_ratios = [
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0.33, 0.33, 0.33],
    ]
    factor_config, spatial_config, outer_perm_config, _ = cm.cosa(
        prob, arch, cm._A, cm._B, part_ratios, global_buf_idx=4, Z=None,
    )

    # Encode spatial mapping (mirrors cosa.cosa.run_timeloop)
    spatial_to_factor_map = {}
    idx = arch.mem_levels
    for i, val in enumerate(arch.S):
        if val > 1:
            spatial_to_factor_map[i] = idx
            idx += 1
    for j, f_j in enumerate(prob.prob_factors):
        for n, f_jn in enumerate(f_j):
            if spatial_config[j][n] == 1:
                factor_config[j][n] = spatial_to_factor_map[factor_config[j][n]]

    perm_config = mapspace.get_default_perm()
    perm_config[4] = outer_perm_config

    # Generate and write mapping YAML directly (no timeloop-model)
    mapspace.reset_mapspace(None, [])
    mapspace.update_mapspace(perm_config, factor_config)
    mapping = mapspace.generate_mapping()

    map_path = work_dir / "map_16.yaml"
    cm.utils.store_yaml(str(map_path), mapping)
    return map_path


def run_cosa_mapper_for_layer(
    spec: HardwareSpec,
    loopdim: Dict[str, int],
    objective: str,
    work_dir: Path,
) -> CoSALayerOutput:
    """Generate arch/mapspace/prob yamls, run CoSA's MIP solver, and write the
    resulting map_16.yaml.  `objective` is recorded but not passed to CoSA —
    CoSA's MIP objective is fixed (weighted compute/util/traffic)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    arch_path = work_dir / "arch.yaml"
    mapspace_path = work_dir / "mapspace.yaml"
    prob_path = work_dir / "prob.yaml"
    arch_path.write_text(_render_arch_yaml(spec))
    mapspace_path.write_text(_MAPSPACE_YAML)
    prob_path.write_text(_render_prob_yaml(loopdim))

    layer_fp = loopdim_fingerprint(loopdim)

    cm = _load_cosa_module()
    t0 = time.time()
    map_yaml = _run_cosa_mip_and_write_mapping(
        cm, prob_path, arch_path, mapspace_path, work_dir,
    )
    runtime_s = time.time() - t0

    if not map_yaml.is_file():
        raise RuntimeError(
            f"CoSA did not produce a map_16.yaml under {work_dir}; the MIP may "
            "have returned infeasible."
        )

    return CoSALayerOutput(
        layer_fp=layer_fp,
        loopdim=dict(loopdim),
        objective=objective,
        map_yaml_path=map_yaml,
        simba_level_names=list(_SIMBA_INNER_TO_OUTER),
        simba_to_miredo=dict(_SIMBA_TO_MIREDO),
        runtime_s=runtime_s,
    )

from typing import Optional as _Optional


def supports_loopdim(loopdim: Dict[str, int]) -> _Optional[str]:
    if (int(loopdim.get('R', 1)) == 1 and
        int(loopdim.get('S', 1)) == 1 and
        int(loopdim.get('Q', 1)) == 1):
        return ("CoSA cnn-layer problem shape is conv-only; "
                "matmul/Gemm (R=S=1, Q=1) not supported")
    g = int(loopdim.get('G', 1) or 1)
    if g > 1:
        return f"CoSA cnn-layer has no G dimension (G={g} > 1)"
    return None


def run_for_layer(acc, ops, loopdim, model_name, architecture, objective):
    import copy as _copy
    from Simulator.Simulax import tranSimulator
    from utils.Workload import LoopNest
    from Evaluation.CoSA.CompatibleCoSA import convert_CoSA_to_MIREDO
    from Evaluation.common.BaselineProvider import (
        BaselineRunResult,
        _resolve_default_spec,
        _spec_fingerprint,
        _load_cosa_outputs,
        _save_cosa_index,
    )

    spec = getattr(acc, "source_spec", None)
    if spec is None:
        spec = _resolve_default_spec(architecture)

    outputs, cache_root, index_path = _load_cosa_outputs(
        model_name=model_name,
        architecture=architecture,
        objective=objective,
        spec=spec,
    )
    layer_fp = loopdim_fingerprint(loopdim)

    if layer_fp not in outputs:
        if spec is None:
            raise RuntimeError(
                f"cosa_adapter.run_for_layer: no HardwareSpec resolved for "
                f"architecture={architecture!r}; CoSA requires registered default_spec()"
            )
        work_dir = cache_root / "per_layer" / layer_fp
        outputs[layer_fp] = run_cosa_mapper_for_layer(
            spec=spec, loopdim=loopdim, objective=objective, work_dir=work_dir,
        )
        _save_cosa_index(index_path, outputs)

    out = outputs[layer_fp]
    loops = LoopNest(acc=acc, ops=ops)
    loops, legalization_meta = convert_CoSA_to_MIREDO(loops=loops, out=out, spec=spec)
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro
    simulator = tranSimulator(acc=_copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run_analytical()

    return BaselineRunResult(
        method="cosa",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": "cosa_mip",
            "model": model_name,
            "architecture": architecture,
            "spec_fingerprint": _spec_fingerprint(spec) if spec is not None else "legacy",
            "mapper_runtime_s": out.runtime_s,
            "legalization_demoted_count": legalization_meta["demoted_count"],
            "legalization_demoted": legalization_meta["demoted"],
            "capacity_demoted_count": legalization_meta.get("capacity_demoted_count", 0),
            "capacity_demoted": legalization_meta.get("capacity_demoted", []),
        },
    )


def run_cosa_constrained_mapper_for_layer(
    spec: HardwareSpec,
    loopdim: Dict[str, int],
    objective: str,
    work_dir: Path,
) -> CoSALayerOutput:
    """Like run_cosa_mapper_for_layer but uses the axis-constrained MIP fork.

    The forked MIP (cosa_constrained.run_constrained_timeloop) injects
    per-axis spatial-dimension constraints derived from spec before solving,
    so the resulting mapping is already axis-legal.  Objective is recorded in
    metadata but not passed to the MIP (same caveat as the unconstrained path).
    """
    from Evaluation.CoSA.cosa_constrained import run_constrained_timeloop

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    arch_path = work_dir / "arch.yaml"
    mapspace_path = work_dir / "mapspace.yaml"
    prob_path = work_dir / "prob.yaml"
    arch_path.write_text(_render_arch_yaml(spec))
    mapspace_path.write_text(_MAPSPACE_YAML)
    prob_path.write_text(_render_prob_yaml(loopdim))

    layer_fp = loopdim_fingerprint(loopdim)

    t0 = time.time()
    run_constrained_timeloop(
        prob_path=prob_path,
        arch_path=arch_path,
        mapspace_path=mapspace_path,
        output_path=str(work_dir),
        spec=spec,
    )
    runtime_s = time.time() - t0

    map_yaml = work_dir / "map_16.yaml"
    if not map_yaml.is_file():
        raise RuntimeError(
            f"CoSA constrained did not produce a map_16.yaml under {work_dir}; "
            "the MIP may have returned infeasible."
        )

    return CoSALayerOutput(
        layer_fp=layer_fp,
        loopdim=dict(loopdim),
        objective=objective,
        map_yaml_path=map_yaml,
        simba_level_names=list(_SIMBA_INNER_TO_OUTER),
        simba_to_miredo=dict(_SIMBA_TO_MIREDO),
        runtime_s=runtime_s,
    )

def supports_loopdim_constrained(loopdim: Dict[str, int]) -> _Optional[str]:
    return supports_loopdim(loopdim)


def run_for_layer_constrained(acc, ops, loopdim, model_name, architecture, objective):
    import copy as _copy
    from Simulator.Simulax import tranSimulator
    from utils.Workload import LoopNest
    from Evaluation.CoSA.CompatibleCoSA import convert_CoSA_to_MIREDO
    from Evaluation.common.BaselineProvider import (
        BaselineRunResult,
        _resolve_default_spec,
        _spec_fingerprint,
        _load_cosa_constrained_outputs,
        _save_cosa_constrained_index,
    )

    spec = getattr(acc, "source_spec", None)
    if spec is None:
        spec = _resolve_default_spec(architecture)

    outputs, cache_root, index_path = _load_cosa_constrained_outputs(
        model_name=model_name,
        architecture=architecture,
        objective=objective,
        spec=spec,
    )
    layer_fp = loopdim_fingerprint(loopdim)

    if layer_fp not in outputs:
        if spec is None:
            raise RuntimeError(
                f"cosa_adapter.run_for_layer_constrained: no HardwareSpec resolved for "
                f"architecture={architecture!r}; CoSA requires registered default_spec()"
            )
        work_dir = cache_root / "per_layer" / layer_fp
        outputs[layer_fp] = run_cosa_constrained_mapper_for_layer(
            spec=spec, loopdim=loopdim, objective=objective, work_dir=work_dir,
        )
        _save_cosa_constrained_index(index_path, outputs)

    out = outputs[layer_fp]
    loops = LoopNest(acc=acc, ops=ops)
    loops, legalization_meta = convert_CoSA_to_MIREDO(loops=loops, out=out, spec=spec)
    loops.usr_defined_double_flag[acc.Macro2mem][1] = acc.double_Macro
    simulator = tranSimulator(acc=_copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run_analytical()

    return BaselineRunResult(
        method="cosa-constrained",
        objective=objective,
        latency=latency,
        energy=energy,
        profile=simulator.PD,
        dataflow=loops,
        metadata={
            "policy": "cosa_constrained_mip",
            "model": model_name,
            "architecture": architecture,
            "spec_fingerprint": _spec_fingerprint(spec) if spec is not None else "legacy",
            "mapper_runtime_s": out.runtime_s,
            "legalization_demoted_count": legalization_meta["demoted_count"],
            "legalization_demoted": legalization_meta["demoted"],
            "capacity_demoted_count": legalization_meta.get("capacity_demoted_count", 0),
            "capacity_demoted": legalization_meta.get("capacity_demoted", []),
        },
    )

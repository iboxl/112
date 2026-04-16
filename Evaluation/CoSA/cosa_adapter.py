# HardwareSpec → CoSA (simba-style) arch adapter.
#
# CoSA accepts 6-level simba-style storage (Registers / AccumulationBuffer /
# WeightBuffer / InputBuffer / GlobalBuffer / DRAM) and a cnn-layer problem
# (R, S, P, Q, C, K, N + stride/dilation). It runs a Gurobi MIP to pick the
# tile factors + spatial/temporal split + outer permutation, then emits a
# timeloop-compatible map_16.yaml that is validated with timeloop-model.
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
# which matches MIREDO's [cores=8 × dimX=32 × dimY=16] axes. cluster-size
# values are only consumed by timeloop-model for topology validation; they
# don't affect the MIP but must be consistent with the instances ratios.
#
# CoSA's MIP does not respect MIREDO's per-axis allowed_loops (cores→PQKG,
# dimX→RSC, dimY→K). Mapping legality is therefore enforced downstream by
# Simulax's OverSize check: an illegal assignment (e.g. K spatial at dimX)
# inflates OReg/IReg tiles and raises Dataflow Over MemSize Error, which
# the baseline-comparison runner records as an anomaly.
#
# timeloop-model is invoked in-process by cosa.run_timeloop; we reuse the
# CIMLoop adapter's timeloop submodule via _timeloop_runtime_env.

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from contextlib import contextmanager
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
_CIMLOOP_ADAPTER_DIR = _COSA_ADAPTER_DIR.parent / "CIMLoop"
_TIMELOOP_SUBMODULE_ROOT = _CIMLOOP_ADAPTER_DIR / "timeloop-accelergy-infra" / "src" / "timeloop"
_TIMELOOP_SUBMODULE_BIN = _TIMELOOP_SUBMODULE_ROOT / "bin"
_TIMELOOP_SUBMODULE_LIB = _TIMELOOP_SUBMODULE_ROOT / "lib"

_CONDA_PREFIX = os.environ.get("CONDA_PREFIX") or sys.prefix
_CONDA_LIB = Path(_CONDA_PREFIX) / "lib"


# ---------------------------------------------------------------------------
# Runtime environment helpers
# ---------------------------------------------------------------------------

@contextmanager
def _timeloop_runtime_env():
    """Prepend the CIMLoop-submodule timeloop bin/lib to PATH/LD_LIBRARY_PATH
    so CoSA's `utils.run_timeloop` subprocess call picks up the pinned
    timeloop-model binary."""
    old_path = os.environ.get("PATH", "")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    try:
        os.environ["PATH"] = f"{_TIMELOOP_SUBMODULE_BIN}:{old_path}"
        parts = [str(_TIMELOOP_SUBMODULE_LIB)]
        if _CONDA_LIB.is_dir():
            parts.append(str(_CONDA_LIB))
        if old_ld:
            parts.append(old_ld)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        yield
    finally:
        os.environ["PATH"] = old_path
        if old_ld:
            os.environ["LD_LIBRARY_PATH"] = old_ld
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)


def _load_cosa_run_timeloop():
    """Import cosa.cosa.run_timeloop from the submodule. Ensures COSA_DIR env
    var is set so CoSA's module-level paths resolve relative to the submodule."""
    cosa_src = str(_COSA_SRC)
    if cosa_src not in sys.path:
        sys.path.insert(0, cosa_src)
    os.environ.setdefault("COSA_DIR", str(_COSA_REPO))
    cosa_module = importlib.import_module("cosa.cosa")
    return cosa_module.run_timeloop


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
    stats_xml_path: Optional[Path]
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

    # cluster-size: only consumed by timeloop-model topology, not by CoSA's
    # MIP. Pick values consistent with the fanout so timeloop accepts the
    # arch. Registers cluster spans dimX×dimY within a core. Accumulation
    # cluster spans dimY within a core.
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
    canon = {k: int(loopdim.get(k, 1)) for k in canon_keys}
    raw = json.dumps(canon, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Mapper invocation
# ---------------------------------------------------------------------------

def run_cosa_mapper_for_layer(
    spec: HardwareSpec,
    loopdim: Dict[str, int],
    objective: str,
    work_dir: Path,
) -> CoSALayerOutput:
    """Generate arch/mapspace/prob yamls, invoke cosa.run_timeloop, locate the
    resulting map_16.yaml + timeloop-model.map+stats.xml. `objective` is
    recorded but not passed to CoSA — CoSA's MIP objective is fixed (weighted
    compute/util/traffic); cross-objective differentiation is not available."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    arch_path = work_dir / "arch.yaml"
    mapspace_path = work_dir / "mapspace.yaml"
    prob_path = work_dir / "prob.yaml"
    arch_path.write_text(_render_arch_yaml(spec))
    mapspace_path.write_text(_MAPSPACE_YAML)
    prob_path.write_text(_render_prob_yaml(loopdim))

    layer_fp = loopdim_fingerprint(loopdim)

    run_timeloop = _load_cosa_run_timeloop()
    t0 = time.time()
    with _timeloop_runtime_env():
        run_timeloop(
            prob_path=prob_path,
            arch_path=arch_path,
            mapspace_path=mapspace_path,
            output_path=str(work_dir),
        )
    runtime_s = time.time() - t0

    map_candidates = sorted(work_dir.rglob("map_16.yaml"))
    if not map_candidates:
        raise RuntimeError(
            f"CoSA did not produce a map_16.yaml under {work_dir}; the MIP may "
            "have returned infeasible or timeloop-model rejected the mapping."
        )
    map_yaml = map_candidates[0]
    xml_candidates = sorted(work_dir.rglob("timeloop-model.map+stats.xml"))
    stats_xml = xml_candidates[0] if xml_candidates else None

    return CoSALayerOutput(
        layer_fp=layer_fp,
        loopdim=dict(loopdim),
        objective=objective,
        map_yaml_path=map_yaml,
        stats_xml_path=stats_xml,
        simba_level_names=list(_SIMBA_INNER_TO_OUTER),
        simba_to_miredo=dict(_SIMBA_TO_MIREDO),
        runtime_s=runtime_s,
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
    with _timeloop_runtime_env():
        run_constrained_timeloop(
            prob_path=prob_path,
            arch_path=arch_path,
            mapspace_path=mapspace_path,
            output_path=str(work_dir),
            spec=spec,
        )
    runtime_s = time.time() - t0

    map_candidates = sorted(work_dir.rglob("map_16.yaml"))
    if not map_candidates:
        raise RuntimeError(
            f"CoSA constrained did not produce a map_16.yaml under {work_dir}; "
            "the MIP may have returned infeasible or timeloop-model rejected the mapping."
        )
    map_yaml = map_candidates[0]
    xml_candidates = sorted(work_dir.rglob("timeloop-model.map+stats.xml"))
    stats_xml = xml_candidates[0] if xml_candidates else None

    return CoSALayerOutput(
        layer_fp=layer_fp,
        loopdim=dict(loopdim),
        objective=objective,
        map_yaml_path=map_yaml,
        stats_xml_path=stats_xml,
        simba_level_names=list(_SIMBA_INNER_TO_OUTER),
        simba_to_miredo=dict(_SIMBA_TO_MIREDO),
        runtime_s=runtime_s,
    )

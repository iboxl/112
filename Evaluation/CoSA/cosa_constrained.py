# Axis-constrained CoSA MIP fork.
#
# Forks two functions from cosa.cosa (mip_solver / cosa) and injects per-axis
# spatial-dimension constraints derived from a MIREDO HardwareSpec.  Everything
# else is imported directly from the CoSA submodule.
#
# Why fork rather than patch:
#   cosa.cosa has `from gurobipy import *` and check_timeloop_version() at
#   module-level, making it unsafe to import as a dependency.  The two
#   functions below are the only ones that touch Gurobi; all other CoSA objects
#   (Prob, Arch, Mapspace, utils, constants) have no Gurobi dependency and are
#   imported directly.
#
# Constraint semantics:
#   CoSA MIP allocates spatial factors freely across levels, subject only to
#   per-level total-fanout budgets (S[i]).  MIREDO's physical hardware
#   constrains which loop dimensions may be spatially parallelised at each axis:
#     dimX  (AccumulationBuffer, S=32) → allowed_loops: R, S, C
#     dimY  (InputBuffer,        S=16) → allowed_loops: K
#     cores (GlobalBuffer,       S=8 ) → allowed_loops: P, Q, K, G
#   This fork adds `x[(i,j,n,0)] == 0` constraints for every disallowed
#   (level, dim) pair so the MIP can optimise within the legal subspace.
#
# OverSize note:
#   CoSA's capacity model uses a single word-bits per level plus part_ratios,
#   while Simulax checks per-operand precision sums.  The constrained MIP may
#   produce tile distributions that pass CoSA's model but trigger Simulax
#   OverSize; callers must catch ValueError("Dataflow Over MemSize Error") and
#   record it as an anomaly — identical to the unconstrained flow.

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from Architecture.HardwareSpec import HardwareSpec

# ---------------------------------------------------------------------------
# Submodule path bootstrap (mirrors cosa_adapter._load_cosa_module)
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_COSA_REPO = _THIS_FILE.parent / "cosa"
_COSA_SRC = _COSA_REPO / "src"


def _ensure_cosa_importable() -> None:
    cosa_src = str(_COSA_SRC)
    if cosa_src not in sys.path:
        sys.path.insert(0, cosa_src)
    os.environ.setdefault("COSA_DIR", str(_COSA_REPO))


# Delay submodule imports until first call so that importing this module does
# not trigger Gurobi's licence check on import.
_cosa_imports_done = False


def _import_cosa_deps():
    global _cosa_imports_done
    if _cosa_imports_done:
        return
    _ensure_cosa_importable()
    global _A, _B, Prob, Arch, Mapspace, utils, GRB, Model, max_
    from cosa.cosa_constants import _A, _B  # noqa: F841
    from cosa.cosa_input_objs import Prob, Arch, Mapspace  # noqa: F841
    import cosa.utils as utils  # noqa: F841
    from gurobipy import GRB, Model, max_  # noqa: F841
    _cosa_imports_done = True


# ---------------------------------------------------------------------------
# Axis constraint derivation
# ---------------------------------------------------------------------------

# Maps MIREDO spatial axis name → CoSA simba level index.
# Derived from _render_arch_yaml instances=[4096,128,128,8,1,1]:
#   S=[1,32,1,16,8,1] → AccumBuf(1)=dimX, InputBuf(3)=dimY, GlobalBuf(4)=cores
_SIMBA_AXIS_TO_LEVEL: Dict[str, int] = {
    "dimX": 1,   # AccumulationBuffer, S=32
    "dimY": 3,   # InputBuffer,        S=16
    "cores": 4,  # GlobalBuffer,       S=8
}

# Maps MIREDO / timeloop loop-dimension name → CoSA prob dim index j.
# CoSA's Prob uses: R=0, S=1, P=2, Q=3, C=4, K=5, N=6 (N=batch, always 1).
# G has no CoSA equivalent — skipped silently.
_DIM_NAME_TO_J: Dict[str, int] = {
    "R": 0, "S": 1, "P": 2, "Q": 3, "C": 4, "K": 5, "N": 6,
}


def _derive_axis_constraints(spec: HardwareSpec) -> Dict[int, Set[int]]:
    """Return {simba_level_i: frozenset_of_allowed_j} from spec.macro.spatial_axes.

    For the default CIM_ACC_TEMPLATE this produces:
        {1: {0,1,4},   # AccumBuf → dimX: R,S,C
         3: {5},        # InputBuf → dimY: K
         4: {2,3,5}}    # GlobalBuf → cores: P,Q,K  (G skipped)
    """
    constraints: Dict[int, Set[int]] = {}
    for axis in spec.macro.spatial_axes:
        level_i = _SIMBA_AXIS_TO_LEVEL.get(axis.name)
        if level_i is None:
            continue
        allowed_j: Set[int] = set()
        for loop_name in axis.allowed_loops:
            j = _DIM_NAME_TO_J.get(loop_name)
            if j is not None:
                allowed_j.add(j)
        constraints[level_i] = allowed_j
    return constraints


# ---------------------------------------------------------------------------
# Forked MIP solver (adds axis_constraints parameter)
# ---------------------------------------------------------------------------

def constrained_mip_solver(
    f, strides, arch, part_ratios, global_buf_idx, A, Z,
    compute_factor=10, util_factor=-1, traffic_factor=1,
    axis_constraints: Optional[Dict[int, Set[int]]] = None,
    debug_lp_path: Optional[str] = None,
):
    """CoSA MIP formulation with optional per-axis spatial-dimension constraints.

    Identical to cosa.cosa.mip_solver except:
    - New axis_constraints parameter: {simba_level_i: set_of_allowed_j}.
      For each inner-GB level and for the GlobalBuffer pooled-perm region,
      adds x[(i,j,n,0)] == 0 for every disallowed (i, j) pair.
    - debug_lp_path: if provided, writes debug.lp there instead of cwd.
    """
    _import_cosa_deps()

    num_vars = len(A[0])
    num_mems = len(Z[0])

    m = Model("mip")

    M = []
    for i in range(num_mems - 1):
        mem_cap = arch.mem_entries[i]
        mem_cap_arr = []
        for j in range(num_vars):
            var_mem_cap = mem_cap * part_ratios[i][j]
            mem_cap_arr.append(var_mem_cap)
        M.append(mem_cap_arr)

    M_log = []
    for i, mem in enumerate(M):
        M_v = []
        for bound in mem:
            if bound == 0:
                bound = 1
            M_v.append(bound)
        M_log.append(M_v)

    S = arch.S

    perm_levels = 0
    for j, f_j in enumerate(f):
        perm_levels += len(f_j)
    gb_start_level = global_buf_idx

    total_levels = num_mems - 1 + perm_levels

    x = {}
    for i in range(total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    x[(i, j, n, k)] = m.addVar(vtype=GRB.BINARY, name=name)
                spatial_temp_sum = 0
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    spatial_temp_sum += x[(i, j, n, k)]
                m.addConstr(spatial_temp_sum <= 1, "spatial_temp_sum_{}_{}_{}".format(i, j, n))

    for i in range(gb_start_level, gb_start_level + perm_levels):
        row_sum = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    row_sum += x[(i, j, n, k)]
        m.addConstr(row_sum <= 1, "row_sum_{}".format(i))

    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            col_sum = 0
            for i in range(total_levels):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    col_sum += x[(i, j, n, k)]
            m.addConstr(col_sum == 1, "col_sum_{}_{}".format(j, n))

    s = {}
    y = {}
    prefix = 0
    for v in range(num_vars):
        for i in range(gb_start_level, gb_start_level + perm_levels):
            row_sum = 0
            y[(v, i)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name="y({},{})".format(v, i))
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    row_sum += x[(i, j, n, 1)] * A[j][v]
            if i > gb_start_level:
                m.addConstr(y[(v, i)] >= y[(v, i - 1)], "y_v_i_sv_{}_{}".format(v, i))
                m.addConstr(y[(v, i)] >= row_sum, "y_v_i_row_sum_{}_{}".format(v, i))
            else:
                m.addConstr(y[(v, i)] == row_sum, "y_v_i_row_sum_{}_{}".format(v, i))
            s[(v, i)] = row_sum

    zz = {}
    for var in [2, 3]:
        for mem_level in [3]:
            zz[(var, mem_level)] = m.addVar(lb=0, ub=1, vtype=GRB.INTEGER,
                                            name="zz({},{},{})".format(prefix, var, mem_level))
            x_sums = 0
            for n, prime_factor in enumerate(f[var]):
                for inner_mem_level_i in range(mem_level + 1):
                    for k in range(2):
                        filter_in = x[(inner_mem_level_i, var, n, k)]
                        m.addConstr(zz[(var, mem_level)] >= filter_in,
                                    "zz_x_sum_{}_{}_{}_{}_{}_{}".format(prefix, var, n, mem_level, inner_mem_level_i, k))
                        x_sums += filter_in
            m.addConstr(zz[(var, mem_level)] <= x_sums, "z_x_sum_{}_{}_{}".format(prefix, var, mem_level))

    l = {}
    for v in range(num_vars):
        for i in range(gb_start_level, gb_start_level + perm_levels):
            row_sum = 0
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    row_sum += np.log2(f[j][n]) * (x[(i, j, n, 1)])
            l[(v, i)] = row_sum

    # -----------------------------------------------------------------------
    # Spatial capacity constraints (unchanged from original)
    # -----------------------------------------------------------------------
    spatial_tile = 0
    for i in range(gb_start_level, gb_start_level + perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
    m.addConstr(spatial_tile <= np.log2(S[gb_start_level]), "spatial_tile_gb_{}".format(prefix))

    for i in range(gb_start_level):
        spatial_tile = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
        m.addConstr(spatial_tile <= np.log2(S[i]), f"spatial_tile_{prefix}_{i}")

    for i in range(gb_start_level + perm_levels, total_levels):
        spatial_tile = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                spatial_tile += np.log2(f[j][n]) * x[(i, j, n, 0)]
        m.addConstr(spatial_tile <= np.log2(S[i - perm_levels + 1]), f"spatial_tile_{i - perm_levels + 1}")

    # -----------------------------------------------------------------------
    # MIREDO axis constraints (new)
    # For each (simba level, dim) pair that is physically illegal on the
    # corresponding MIREDO spatial axis, force x[(i,j,n,0)] = 0.
    # -----------------------------------------------------------------------
    if axis_constraints:
        # inner-GB levels (i < gb_start_level): AccumBuf and InputBuf
        for level_i, allowed_j_set in axis_constraints.items():
            if level_i >= gb_start_level:
                continue
            for j, f_j in enumerate(f):
                if j in allowed_j_set:
                    continue
                for n in range(len(f_j)):
                    m.addConstr(x[(level_i, j, n, 0)] == 0,
                                f"axis_block_{level_i}_{j}_{n}")

        # GlobalBuffer pooled perm region: all sub-levels share the same
        # physical spatial axis (cores), so the dim restriction applies to
        # every i in [gb_start_level, gb_start_level + perm_levels).
        if gb_start_level in axis_constraints:
            gb_allowed = axis_constraints[gb_start_level]
            for i in range(gb_start_level, gb_start_level + perm_levels):
                for j, f_j in enumerate(f):
                    if j in gb_allowed:
                        continue
                    for n in range(len(f_j)):
                        m.addConstr(x[(i, j, n, 0)] == 0,
                                    f"axis_block_gb_{i}_{j}_{n}")

    # -----------------------------------------------------------------------
    # Buffer capacity constraints (unchanged)
    # -----------------------------------------------------------------------
    buf_util = {}
    for v in range(num_vars):
        for i in range(num_mems):
            buf_util[(i, v)] = 0

    for v in range(num_vars):
        for i_ in range(gb_start_level + perm_levels):
            for i in range(num_mems):
                for j, f_j in enumerate(f):
                    for n, f_jn in enumerate(f_j):
                        factor = 1
                        if v == 1 and j == 2:
                            factor = strides[0]
                        if v == 1 and j == 3:
                            factor = strides[1]

                        if i_ > gb_start_level and i_ < gb_start_level + perm_levels:
                            Z_const = Z[v][i][gb_start_level]
                        else:
                            Z_const = Z[v][i][i_]
                        buf_util[(i, v)] += np.log2(factor * f[j][n]) * (x[(i_, j, n, 0)] + x[i_, j, n, 1]) * A[j][v] * Z_const
                        if i < 3 and j in [0, 1] and v == 1:
                            buf_util[(i, v)] += (x[(i_, j, n, 0)] + x[(i_, j, n, 1)]) * (1 - zz[(j + 2, 3)]) * np.log2(f[j][n])
                            buf_util[(i, v)] += (x[(i_, j, n, 0)] + x[(i_, j, n, 1)]) * zz[(j + 2, 3)] * np.log2(2)

    for v in range(num_vars):
        for i in range(num_mems - 1):
            if M_log[i][v] > 0:
                m.addConstr(buf_util[(i, v)] <= np.log2(M_log[i][v]), f"buffer_size_{i}_{v}")

    # -----------------------------------------------------------------------
    # Objective (unchanged)
    # -----------------------------------------------------------------------
    inner_gb_cycles = 0
    for i in range(gb_start_level):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                inner_gb_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])

    gb_cycles = 0
    for i in range(gb_start_level, gb_start_level + perm_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                gb_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])

    dram_cycles = 0
    for i in range(gb_start_level + perm_levels, total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                dram_cycles += np.log2(f[j][n]) * (x[(i, j, n, 1)])
    total_compute = inner_gb_cycles + gb_cycles + dram_cycles

    spatial_cost = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level, gb_start_level + perm_levels):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    size += np.log2(f[j][n]) * (x[(i, j, n, 0)])
        spatial_cost[v] = size

    data_size = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level):
            for j, f_j in enumerate(f):
                for n, f_jn in enumerate(f_j):
                    factors = 0.8 + 0.04 * i
                    size += factors * np.log2(f[j][n]) * (x[(i, j, n, 0)] + x[i, j, n, 1]) * A[j][v]
        data_size[v] = size

    gb_traffic = {}
    for v in range(num_vars):
        size = 0
        for i in range(gb_start_level, gb_start_level + perm_levels):
            size += l[(v, i)] * y[(v, i)]
        gb_traffic[v] = size

    dram_traffic = {}
    for v in range(num_vars):
        i = gb_start_level + perm_levels
        size = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                size += np.log2(f[j][n]) * (x[(i, j, n, 1)])
        dram_traffic[v] = size

    total_util = 0
    for i in range(gb_start_level):
        for v in range(num_vars):
            factor = 1.01 if i == 2 else 1
            total_util += buf_util[(i, v)] * factor

    total_traffic = 0
    for v in range(num_vars):
        factor = 1.01 if v == 0 else 1
        total_traffic += 0.99 * data_size[v] + 0.99 * spatial_cost[v] + gb_traffic[v] + dram_traffic[v] * factor

    cosa_obj = total_util * util_factor + total_compute * compute_factor + total_traffic * traffic_factor

    max_it = m.addVar(vtype=GRB.CONTINUOUS, name="max_it")
    its = []
    its.append(m.addVar(vtype=GRB.CONTINUOUS, name="a"))
    m.addConstr(its[-1] == total_traffic, "total_traffic")
    its.append(m.addVar(vtype=GRB.CONTINUOUS, name="b"))
    m.addConstr(its[-1] == total_compute, "total_compute")
    m.addConstr(max_it == max_(its), name="max_it_constr")

    total_util_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_util_var")
    total_comp_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_comp_var")
    total_traf_var = m.addVar(vtype=GRB.CONTINUOUS, name="total_traf_var")
    m.addConstr(total_util_var == total_util, "total_util_constraint")
    m.addConstr(total_comp_var == total_compute, "total_comp_constraint")
    m.addConstr(total_traf_var == total_traffic, "total_traf_constraint")

    m.ModelSense = GRB.MINIMIZE
    m.setObjective(cosa_obj, GRB.MINIMIZE)

    begin_time = time.time()
    m.optimize()
    milp_runtime = time.time() - begin_time

    if debug_lp_path:
        m.write(str(debug_lp_path))

    result_dict = {}
    for variable in m.getVars():
        assert variable.varName not in result_dict
        result_dict[variable.varName] = variable.x

    # -----------------------------------------------------------------------
    # Extract factor_config, spatial_config, outer_perm_config (unchanged)
    # -----------------------------------------------------------------------
    all_x = np.zeros((total_levels, perm_levels, 2))
    for i in range(total_levels):
        level_idx = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    all_x[i, level_idx, k] = result_dict[name]
                level_idx += 1
    np.set_printoptions(precision=0, suppress=True)

    var_outer_perm_config = [-1] * perm_levels
    outer_perm_config = [-1] * perm_levels
    x_arr = np.zeros((perm_levels, perm_levels, 2))
    for i in range(gb_start_level, gb_start_level + perm_levels):
        level_idx = 0
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    x_arr[i - gb_start_level, level_idx, k] = result_dict[name]
                name = "X({},{},{},{})".format(i, j, n, 1)
                if result_dict[name] == 1:
                    var_outer_perm_config[i - gb_start_level] = j
                level_idx += 1

    merge_outer_perm_config = []
    for i, var in enumerate(var_outer_perm_config):
        if var != -1 and var not in merge_outer_perm_config:
            merge_outer_perm_config.append(var)
    for i in range(len(f)):
        if i not in merge_outer_perm_config:
            merge_outer_perm_config.append(i)

    outer_perm_config = [1] * len(f)
    for i, var in enumerate(merge_outer_perm_config):
        outer_perm_config[var] = i

    factor_config = []
    spatial_config = []
    dram_level = -1
    for j, f_j in enumerate(f):
        sub_factor_config = []
        sub_spatial_config = []
        for n, f_jn in enumerate(f_j):
            sub_factor_config.append(dram_level)
            sub_spatial_config.append(0)
        factor_config.append(sub_factor_config)
        spatial_config.append(sub_spatial_config)

    for i in range(gb_start_level):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                if f[j][n] == 1:
                    factor_config[j][n] = num_mems - 1
                    spatial_config[j][n] = 0
                    continue
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    if result_dict[name] >= 0.9:
                        factor_config[j][n] = i
                        if k == 0:
                            spatial_config[j][n] = 1

    for i in range(gb_start_level + perm_levels, total_levels):
        for j, f_j in enumerate(f):
            for n, f_jn in enumerate(f_j):
                if f[j][n] == 1:
                    factor_config[j][n] = num_mems - 1
                    spatial_config[j][n] = 0
                    continue
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    if result_dict[name] >= 0.9:
                        if k == 0:
                            raise ValueError('Invalid Mapping')
                        factor_config[j][n] = i - perm_levels + 1

    for j, f_j in enumerate(f):
        for n, f_jn in enumerate(f_j):
            for i in range(gb_start_level, gb_start_level + perm_levels):
                for k in range(2):
                    name = "X({},{},{},{})".format(i, j, n, k)
                    if result_dict[name] >= 0.9:
                        factor_config[j][n] = gb_start_level
                        if k == 0:
                            spatial_config[j][n] = 1

    return (factor_config, spatial_config, outer_perm_config, milp_runtime)


# ---------------------------------------------------------------------------
# Forked cosa() wrapper
# ---------------------------------------------------------------------------

def constrained_cosa(prob, arch, A, B, part_ratios, global_buf_idx,
                     Z=None, axis_constraints=None, debug_lp_path=None):
    """Thin wrapper around constrained_mip_solver (mirrors cosa.cosa.cosa)."""
    _import_cosa_deps()

    prime_factors = prob.prob_factors
    strides = [prob.prob['Wstride'], prob.prob['Hstride']]

    if Z is None:
        Z = []
        for var in _B:
            Z_var = []
            for i, val in enumerate(var):
                rank_arr = [0] * len(var)
                if val == 1:
                    for j in range(i + 1):
                        rank_arr[j] = 1
                Z_var.append(rank_arr)
            Z.append(Z_var)

    factor_config, spatial_config, outer_perm_config, run_time = constrained_mip_solver(
        prime_factors, strides, arch, part_ratios,
        global_buf_idx=global_buf_idx, A=A, Z=Z,
        compute_factor=10, util_factor=-0.1, traffic_factor=1,
        axis_constraints=axis_constraints,
        debug_lp_path=debug_lp_path,
    )
    return factor_config, spatial_config, outer_perm_config, run_time


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_constrained_timeloop(
    prob_path,
    arch_path,
    mapspace_path,
    output_path,
    spec: HardwareSpec,
) -> None:
    """Run the axis-constrained MIP and write the resulting mapping YAML.

    Derives per-axis spatial-dimension constraints from spec, runs the
    constrained MIP, and writes map_16.yaml directly (no timeloop-model
    subprocess — performance evaluation is done by MIREDO's Simulax).
    """
    _import_cosa_deps()

    prob_path = Path(prob_path)
    arch_path = Path(arch_path)
    mapspace_path = Path(mapspace_path)
    output_path = Path(output_path)

    prob = Prob(prob_path)
    arch = Arch(arch_path)
    mapspace = Mapspace(mapspace_path)
    mapspace.init(prob, arch)

    part_ratios = [
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0.33, 0.33, 0.33],
    ]

    axis_constraints = _derive_axis_constraints(spec)
    debug_lp = output_path / "debug_constrained.lp"

    factor_config, spatial_config, outer_perm_config, _ = constrained_cosa(
        prob, arch, _A, _B, part_ratios, global_buf_idx=4,
        Z=None, axis_constraints=axis_constraints, debug_lp_path=debug_lp,
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

    map_path = output_path / "map_16.yaml"
    utils.store_yaml(str(map_path), mapping)

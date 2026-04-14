# this file is prepared for project 026
# Created by iboxl

import os
import struct
import utils.UtilsFunction.ToolFunction as _tool_func
import math
import psutil
from multiprocessing.shared_memory import SharedMemory
from utils.GlobalUT import *


class SharedUB:
    """跨进程共享的metric上界，基于SharedMemory的lock-free double。

    Race policy: 并发写入可能短暂覆盖更优值，但parent进程在每个worker完成后
    重新同步，恢复权威最小值。不使用锁以避免worker崩溃导致的死锁。

    平台说明: x86-64上，自然对齐的8字节mmap写入是架构级原子操作。
    SharedMemory使用mmap返回页对齐缓冲区，offset 0处的double不会被撕裂。
    """
    __slots__ = ('_shm',)

    def __init__(self, shm: SharedMemory):
        self._shm = shm

    @property
    def value(self) -> float:
        return struct.unpack_from('d', self._shm.buf, 0)[0]

    @value.setter
    def value(self, v: float):
        struct.pack_into('d', self._shm.buf, 0, v)

    def update_min(self, v: float):
        """Best-effort set value = min(current, v)."""
        if v < struct.unpack_from('d', self._shm.buf, 0)[0]:
            struct.pack_into('d', self._shm.buf, 0, v)


conv_im2col_info = _tool_func.func_conv_info

def get_PowerOfTwo(x):
    return math.pow(2, math.ceil(math.log2(x)))

def debug_get_im2col_info(FLAG_DEBUG):
    if FLAG_DEBUG:
        for idx, (layer_name, info) in enumerate(conv_im2col_info.items()):
            print(f"{idx}:    Layer: {layer_name}")
            print(f"  ori_M: {info['ori_M']}")
            print(f"  ori_K: {info['ori_K']}")
            print(f"  ori_N: {info['ori_N']}")
            print(f"  num_mul: {info['num_mul']}")
            print(f"  module: {info['module']}")
            print(f"  input shape: {info['input_shape']}, weight shape: {info['weight_shape']}")

def append_scheme_summary(outputdir:str, message:str):
    summary_file = os.path.join(outputdir, "Scheme-Summary.log")
    with open(summary_file, "a", encoding="utf-8") as file:
        file.write(message.rstrip() + "\n")


def detect_parallel_config():
    logical_cores = psutil.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    memory_info = psutil.virtual_memory()
    available_mem_gb = memory_info.available / (1024 ** 3)

    try:
        load_avg = os.getloadavg()[0]
        busy_cores = min(physical_cores - 1, math.floor(load_avg))
    except (AttributeError, OSError):
        busy_cores = 0

    usable_cores = max(1, physical_cores - busy_cores)

    return {
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "usable_cores": usable_cores,
        "available_mem_gb": available_mem_gb,
    }


def auto_parallel_config(usable_cores, available_mem_gb, num_schemes):
    """Decide threads_per_worker and max_workers based on hardware and workload.

    Strategy:
    - Many schemes (>= 2×cores): maximize workers (2 threads each) for faster
      cross-scheme pruning via shared_ub.
    - Few schemes (< cores): give each solver more threads to finish faster.
    - Memory constraint: each Gurobi instance needs ~2 GB; cap workers accordingly.
    """
    if num_schemes <= 1:
        return usable_cores, 1

    if num_schemes <= usable_cores:
        # Few schemes: allocate cores evenly, ensure >= 2 threads for Gurobi.
        threads_per_worker = max(2, usable_cores // num_schemes)
        max_workers = max(1, usable_cores // threads_per_worker)
    else:
        # Many schemes: 99%+ are quickly pruned (dominance/LB/metric_ub),
        # but each still costs ~0.3-0.5s for model build + presolve.
        # Throughput of infeasible scheme processing dominates wall time,
        # so maximize workers. 2 threads is sufficient for Gurobi presolve
        # and basic B&B on the few feasible schemes.
        threads_per_worker = max(1, min(2, usable_cores))
        max_workers = max(1, usable_cores // threads_per_worker)

    # Memory cap: ~2 GB per solver instance as conservative estimate
    mem_per_worker = 2.0
    max_by_mem = max(1, int(available_mem_gb * 0.8 / mem_per_worker))
    if max_by_mem < max_workers:
        max_workers = max_by_mem
        # Redistribute remaining cores, but keep at least the original thread count
        threads_per_worker = max(threads_per_worker, usable_cores // max_workers)

    return threads_per_worker, max_workers

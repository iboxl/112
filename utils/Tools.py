# this file is prepared for project 026
# Created by iboxl

import os
import configparser
import utils.UtilsFunction.ToolFunction as _tool_func
import math
import psutil
from utils.GlobalUT import *


conv_im2col_info = _tool_func.func_conv_info

def get_PowerOfTwo(x):
    return math.pow(2, math.ceil(math.log2(x)))

def get_ConfigFile(cfgname):
    path_cfg = os.path.join(os.getcwd(),f'Config/{cfgname}')
    if os.path.exists(path_cfg):
        cfg = configparser.ConfigParser()
        cfg.read(path_cfg)
    else:
        Logger.error(path_cfg)
        raise Exception('No Configuration File ! ! !')
    return cfg

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
    threads_per_worker = max(1, int(math.sqrt(usable_cores)))
    max_workers = max(1, usable_cores // threads_per_worker)

    return {
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "usable_cores": usable_cores,
        "available_mem_gb": available_mem_gb,
        "threads_per_worker": threads_per_worker,
        "max_workers": max_workers,
    }

# this file is prepared for project 026
# Created by iboxl

import os
import configparser
import utils.UtilsFunction.ToolFunction as _tool_func
import math
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
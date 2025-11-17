# this file is prepared for project 026
# Created by iboxl

import os
import configparser
import torch.nn as nn
import sympy
import pathlib

func_conv_info = {}
def _getDim(input_shape, weight_shape, stride=1, pad=0):
    # input_shape: (N, C, H, W)                                                     #  N = batch
    # weight_shape: (out_channels, in_channels, filter_h, filter_w)
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Expected input_shape and weight_shape to have 4 dimensions")
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    # 计算输出特征图的形状
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    return N * out_h * out_w, C * filter_h * filter_w, out_channels

def _getIm2colShape(input_shape, weight_shape, stride=1, pad=0):
    # input_shape: (N, C, H, W)
    # weight_shape: (out_channels, in_channels, filter_h, filter_w)
    if len(input_shape) != 4 or len(weight_shape) != 4:
        raise ValueError("Expected input_shape and weight_shape to have 4 dimensions")
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    
    # 计算输出特征图的形状
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    # 计算经过 im2col 操作后的输入特征图的形状
    im2col_input_shape = (N * out_h * out_w, C * filter_h * filter_w)
    
    # 计算经过 im2col 操作后的权重的形状
    im2col_weight_shape = (C * filter_h * filter_w, out_channels)
    
    return im2col_input_shape, im2col_weight_shape

def _im2col_output_shape_groups(input_shape, weight_shape, stride=1, pad=0, groups=1):
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    # 计算 im2col 形状和矩阵乘法的个数
    if groups == 1:
        # 普通卷积
        im2col_input_shape = (N * out_h * out_w, C * filter_h * filter_w)
        im2col_weight_shape = (out_channels, C * filter_h * filter_w)
        num_matrix_multiplications = 1
    elif groups == C:
        # 深度卷积
        im2col_input_shape = (N * out_h * out_w, filter_h * filter_w)
        im2col_weight_shape = (out_channels // groups, filter_h * filter_w)
        num_matrix_multiplications = C
    else:
        # 分组卷积
        im2col_input_shape = (N * out_h * out_w, (C // groups) * filter_h * filter_w)
        im2col_weight_shape = (out_channels // groups, (C // groups) * filter_h * filter_w)
        num_matrix_multiplications = groups
    
    return im2col_input_shape, im2col_weight_shape, num_matrix_multiplications

def _getDimGroups(input_shape, weight_shape, stride=1, pad=0, groups=1):
    N, C, H, W = input_shape
    out_channels, in_channels, filter_h, filter_w = weight_shape
    
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    # 计算 im2col 形状和矩阵乘法的个数
    if groups == 1:
        # 普通卷积
        return N * out_h * out_w, C * filter_h * filter_w, out_channels, 1
    elif groups == C:
        # 深度卷积
        return N * out_h * out_w, filter_h * filter_w, out_channels // groups, C
    else:
        # 分组卷积
        return N * out_h * out_w, (C // groups) * filter_h * filter_w, out_channels // groups, groups
    
def _hook_fn(module, input, output):
    if isinstance(module, nn.Conv2d):
        print('all ready getdim')
        input_shape = input[0].shape # 注意这里使用 input[0].shape
        weight_shape = module.weight.shape
        stride = module.stride[0]
        pad = module.padding[0]
        # im2col_input_shape, im2col_weight_shape = getIm2colShape(input_shape, weight_shape, stride, pad)
        ori_M, ori_K, ori_N = _getDim(input_shape, weight_shape, stride, pad)
        layer_name = str(module)
        func_conv_info[layer_name] = {
            'ori_M': ori_M,
            'ori_N': ori_N,
            'ori_K': ori_K
        }
        raise "Error hook function used !!!"

def make_hook_im2col(layer_idx_name):
    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            input_shape = input[0].shape
            weight_shape = module.weight.shape
            stride = module.stride[0]
            pad = module.padding[0]
            groups = module.groups  # 获取 groups 参数
        
            ori_M, ori_K, ori_N, num_multiplications = _getDimGroups(input_shape, weight_shape, stride, pad, groups)

            layer_name = f"{module.__class__.__name__}_{id(module)}"
            func_conv_info[layer_name] = {
                'ori_M': ori_M,
                'ori_N': ori_N,
                'ori_K': ori_K,
                'num_mul': num_multiplications,
                'module':module,
                'input_shape':input_shape,
                'weight_shape':weight_shape
            }
            raise "Error hook function used !!!"
    return hook_fn

def make_hook_linear_im2col(layer_idx_name):
    def hook_fn_linear(module, input, output):
        if isinstance(module, nn.Linear):
            input_shape = input[0].shape  # 形状 [N, in_features]
            weight_shape = module.weight.shape  # 形状 [out_features, in_features]

            ori_M = input_shape[0] 
            ori_N = weight_shape[0]    
            ori_K = weight_shape[1]  

            layer_name = f"{module.__class__.__name__}_{id(module)}"
            func_conv_info[layer_name] = {
                'ori_M': ori_M,
                'ori_N': ori_N,
                'ori_K': ori_K,
                'num_mul': 1,
                'module':module,
                'input_shape':input_shape,
                'weight_shape':weight_shape
            }
            raise "Error hook function used !!!"
    return hook_fn_linear

def make_hook(layer_idx_name):
    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):

            batch_size, _in_channels, input_height, input_width = input[0].shape

            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups

            output_height = ((input_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
            output_width = ((input_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

            # RSCKPQG分别对应卷积层各个维度的参数
            S = kernel_size[0]          # 卷积核高度
            R = kernel_size[1]          # 卷积核宽度
            C = in_channels             # 输入通道数
            K = out_channels            # 输出通道数
            Q = output_height           # 输出高度
            P = output_width            # 输出宽度
            G = groups                  # 分组卷积的组数
            B = batch_size              # 批次大小
            H = input_height            # 输入高度
            W = input_width             # 输入宽度

            assert B==1     # inference

            layer_name = f"{module.__class__.__name__}_{id(module)}"
            func_conv_info[layer_name] = {
                'R': R,
                'S': S,
                'C': C,
                'K': K,
                'P': P,
                'Q': Q,
                'G': G,
                'B': B,
                'H': H,
                'W': W,
                'Stride': stride[0],
                'Padding': padding[0]
            }
    return hook_fn

def make_hook_linear(layer_idx_name):
    def hook_fn_linear(module, input, output):
        if isinstance(module, nn.Linear):
            batch_size = input[0].shape[0]
            in_features = module.in_features
            out_features = module.out_features

            # 将全连接层视为特殊的1x1卷积层
            S = 1                   # 卷积核高度
            R = 1                   # 卷积核宽度
            C = in_features         # 输入特征数，对应卷积层的输入通道数
            K = out_features        # 输出特征数，对应卷积层的输出通道数
            Q = 1                   # 输出高度，对应卷积层的输出高度
            P = 1                   # 输出宽度，对应卷积层的输出宽度
            G = 1                   # 分组数，对于全连接层，假设为1
            B = batch_size          # 批次大小
            H = 1                   # 输入高度
            W = 1                   # 输入宽度

            layer_name = f"{module.__class__.__name__}_{id(module)}"
            func_conv_info[layer_name] = {
                'R': R,
                'S': S,
                'C': C,
                'K': K,
                'P': P,
                'Q': Q,
                'G': G,
                'B': B,
                'H': H,
                'W': W,
                'Stride': 1,
                'Padding': 0
            }
    return hook_fn_linear

def _prime_factors(n):
    factors = []
    for prime in sympy.primerange(2, n+1):
        while n % prime == 0:
            factors.append(prime)
            n //= prime
    return factors

def combine_factors(factors, min_factor, max_factor):
    small_factors = [f for f in factors if f < min_factor]
    medium_factors = [f for f in factors if min_factor <= f <= max_factor]
    large_factors = [f for f in factors if f > max_factor]
    combined_factors = []

    # 合并小于 min_factor 的因子
    small_factors.sort()
    temp_product = 1
    for f in small_factors:
        temp_product *= f
        if min_factor <= temp_product <= max_factor:
            combined_factors.append(temp_product)
            temp_product = 1
        elif temp_product > max_factor:
            # 无法再合并，分解 temp_product
            while temp_product > max_factor:
                temp_product //= f
                combined_factors.append(f)
            if min_factor <= temp_product <= max_factor:
                combined_factors.append(temp_product)
                temp_product = 1
            else:
                temp_product = f
    if temp_product != 1:
        combined_factors.append(temp_product)

    # 添加中间的因子
    combined_factors.extend(medium_factors)

    # 处理大于 max_factor 的因子
    for f in large_factors:
        # 尝试将大因子分解为较小的因子
        if sympy.isprime(f):
            # 如果是素数，无法分解，直接添加
            combined_factors.append(f)
        else:
            # 分解为因子
            sub_factors = _prime_factors(f)
            # 递归合并这些因子
            sub_combined = combine_factors(sub_factors, min_factor, max_factor)
            combined_factors.extend(sub_combined)

    # 最终结果中可能有因子不在范围内，需要再次检查
    final_factors = []
    for f in combined_factors:
        if f > max_factor:
            # 尝试进一步分解
            if sympy.isprime(f):
                final_factors.append(f)
            else:
                sub_factors = _prime_factors(f)
                sub_combined = combine_factors(sub_factors, min_factor, max_factor)
                final_factors.extend(sub_combined)
        else:
            final_factors.append(f)

    return final_factors

def prime_factorization_list(numbers, min_factor=2, max_factor=100):
    result = []
    for number in numbers:
        # 检查是否为素数，且大于 8
        if sympy.isprime(number) and number > max_factor:
            number += 1
        factors = _prime_factors(number)
        combined = combine_factors(factors, min_factor, max_factor)
        result.append(combined)
    return result

def getDivisors(N: int) -> list[int]:
    if N <= 0:
        return []
    small_divs = []
    large_divs = []
    i = 1
    # 只需遍历到 sqrt(N)
    while i * i <= N:
        if N % i == 0:
            small_divs.append(i)
            if i != N // i:
                large_divs.append(N // i)
        i += 1
    return small_divs + large_divs[::-1]

def getPrimeFactors(list_factors):
    """
    从嵌套列表 nested_list 中提取所有整数，去重并升序排序后返回。

    参数:
        nested_list (list of list of int): 形如 [[3], [2,4], [2,7,8], [4,7]] 的输入
    返回:
        list of int: 升序排列的唯一元素列表，例如 [2,3,4,7,8]
    """
    # 使用一个 set 结构用于去重
    unique_factors = set()

    # 遍历每个子列表，将其元素加入集合
    for sublist in list_factors:
        # 这里假设 sublist 中的所有元素均为 int 类型
        for element in sublist:
            unique_factors.add(element)

    # 将去重后的集合转换为列表并排序
    sorted_list = sorted(unique_factors)
    return sorted_list

def prepare_save_dir(save_dir: str) -> pathlib.Path:
    """
    检查 save_dir 路径是否可用：
        1. 若路径已经存在且为文件夹，原样使用；
        2. 若路径不存在，则递归创建文件夹；
        3. 若路径存在但为普通文件，抛出 ValueError。
    返回值
    ------
    pathlib.Path
        规范化后的绝对路径对象
    """
    p = pathlib.Path(save_dir).expanduser().resolve()  # 支持 ~、相对路径
    if p.exists():
        if p.is_dir():
            pass  # 目录已存在，可直接使用
        else:
            raise ValueError(f"目标路径 {p} 已存在且是文件，无法作为日志目录！")
    else:
        p.mkdir(parents=True, exist_ok=False)  # 递归创建
    return p


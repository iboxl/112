# this file is prepared for project 026
# Created by iboxl

import os
import configparser
import torch.nn as nn
import utils.UtilsFunction.ToolFunction as _tool_func
from model.ResNet18 import ResNet18
from model.MobileNet_v2_imagenet import MobileNetV2_img as mobileNet_v2
from model.VGG19 import VGG19, VGG19BN
from torchvision import models as torchmodel
from model.SingleLayer import testSingleNet
from model.AlexNet import AlexNet
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

def get_Model(model_name, num_classes:int=1000):
    if model_name in ['ResNet18', 'resnet18', 'resnet', 'res']:
        return ResNet18(num_classes=num_classes)
    elif model_name in ['mobilenetv2', 'mob', 'mobilenet']:
        return mobileNet_v2(num_classes=num_classes)
    elif model_name in ['vggNet19', 'vgg', 'vgg19']:
        return VGG19(num_classes=num_classes)
    elif model_name in ['vggNet19BN', 'vggbn', 'vgg19bn']:
        return VGG19BN(num_classes=num_classes)
    elif model_name in ['alexNet', 'alex']:
        return AlexNet(num_classes=num_classes)
    elif model_name in ['r50']:
        return torchmodel.resnet50()
    elif model_name in ['r50x']:
        return torchmodel.resnext50_32x4d()
    elif model_name in ['sque']:
        return torchmodel.squeezenet1_0()
    elif model_name in ['test']:
        return testSingleNet()
    else:   
        raise Exception(f" ~ ~ ~ MODEL {model_name} NOT FOUND ~ ~ ~ ")

def set_hook(model):
    model.eval()
    for idx, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.Conv2d):
            layer_idx_name = f"{idx}_{name}"  # 使用索引和名字来唯一标识每个层
            hook_fn = _tool_func.make_hook(layer_idx_name)
            layer.register_forward_hook(hook_fn)
        if isinstance(layer, nn.Linear):
            layer_idx_name = f"{idx}_{name}"  # 使用索引和名字来唯一标识每个层
            hook_fn = _tool_func.make_hook_linear(layer_idx_name)
            layer.register_forward_hook(hook_fn)

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
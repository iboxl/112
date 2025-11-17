import torch
import torch.nn as nn
import torch.onnx
import torchvision.models as models
from collections import OrderedDict

# 创建一个函数，用于单独提取模型的卷积层，并导出为ONNX文件
def export_conv_layers_to_onnx(model, input_shape, output_dir):
    # 确保模型在评估模式下
    model.eval()

    # 遍历模型的每一层，找到Conv2d层
    layer_index = 0
    hooks = []
    layer_inputs_outputs = {}
    
    # 定义一个前向钩子来捕获输入和输出的形状
    def hook_fn(module, input, output, name):
        layer_inputs_outputs[name] = (input[0].shape, output.shape)

    # 注册前向钩子以捕获输入和输出的大小
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))
    
    # 生成一次前向传播以捕获输入和输出形状
    dummy_input = torch.randn(*input_shape)
    model(dummy_input)
    
    # 移除所有钩子
    for hook in hooks:
        hook.remove()

    # 遍历模型的每一层，再次找到Conv2d层并导出ONNX
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            # 将模块名称中的点替换为下划线，以避免KeyError
            sanitized_name = name.replace('.', '_')

            # 获取卷积层的权重维度信息和输出特征图大小
            R, S = layer.kernel_size
            P, Q = layer_inputs_outputs[name][1][2], layer_inputs_outputs[name][1][3]  # 输出的高度和宽度
            C = layer.in_channels
            K = layer.out_channels

            # 生成唯一的导出文件名，包含权重维度信息和输出特征图大小
            layer_name = f"conv_{layer_index}_{sanitized_name}_{R}_{S}_{P}_{Q}_{C}_{K}.onnx"
            layer_index += 1

            # 创建一个新模型，只包含该卷积层
            single_layer_model = nn.Sequential(OrderedDict([(sanitized_name, layer)]))

            # 生成一个输入张量，其形状与原始模型中对应卷积层的输入相同
            input_tensor = torch.randn(layer_inputs_outputs[name][0])

            # 构造ONNX模型文件路径
            output_path = f"{output_dir}/{layer_name}"

            # 导出卷积层到ONNX
            torch.onnx.export(
                single_layer_model,                      # 被导出的模型
                input_tensor,                            # 输入张量
                output_path,                             # 导出的ONNX文件路径
                export_params=True,                      # 是否导出模型参数
                opset_version=11,                        # ONNX的opset版本
                do_constant_folding=True,                # 是否执行常量折叠优化
                input_names=['input'],                   # 输入张量的名称
                output_names=['output'],                 # 输出张量的名称
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴
            )
            print(f"Layer '{name}' exported to ONNX at '{output_path}'")

# 示例：导出ResNet18模型的卷积层
if __name__ == "__main__":
    # 使用torchvision加载预定义的resnet18模型
    # model = models.resnet18(pretrained=False)
    # model = models.alexnet()
    model = models.vgg19_bn()
    
    # 指定输入张量形状（例如，batch_size=1, channels=3, height=224, width=224）
    input_shape = (1, 3, 224, 224)
    
    # 指定ONNX模型保存的目录
    # output_directory = "model/Resnet18"
    # output_directory = "model/alexNet"
    output_directory = "model/vgg"

    # 执行导出
    export_conv_layers_to_onnx(model, input_shape, output_directory)
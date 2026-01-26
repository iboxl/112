# this file is prepared for project 112
# Created by iboxl

import os
import torch
import onnx

from torchvision.models import mobilenet_v2 as model

if __name__ == "__main__":
    
    modelName = "mobilenetV2"
    
    m = model().eval()
    dummy = torch.randn(1, 3, 224, 224)     # imageNet [3,224,224]
    modelNameSave = modelName + ".onnx"
    torch.onnx.export(m, dummy, modelName,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input":{0:"N"}, "output":{0:"N"}})
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(modelName)),modelNameSave )
    os.remove(modelName)

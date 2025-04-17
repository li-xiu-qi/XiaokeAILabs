#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第1部分：引言与基础准备
这个脚本用于检查环境并演示一个简单的PyTorch模型
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_environment():
    """检查环境配置"""
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    
    # 检查ONNX是否已安装
    try:
        import onnx
        print("ONNX版本:", onnx.__version__)
        onnx_available = True
    except ImportError:
        print("ONNX未安装，请使用pip install onnx安装")
        onnx_available = False
    
    # 检查ONNX Runtime是否已安装  
    try:
        import onnxruntime
        print("ONNX Runtime版本:", onnxruntime.__version__)
        onnxruntime_available = True
    except ImportError:
        print("ONNX Runtime未安装，请使用pip install onnxruntime安装")
        onnxruntime_available = False
    
    # 检查CUDA是否可用
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("当前CUDA设备:", torch.cuda.current_device())
        print("CUDA设备名称:", torch.cuda.get_device_name(0))
    
    return onnx_available and onnxruntime_available

class SimpleModel(nn.Module):
    """一个简单的全连接神经网络"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_simple_model():
    """测试简单模型的前向传播"""
    model = SimpleModel()
    model.eval()  # 设置为评估模式
    
    # 创建一个随机输入
    dummy_input = torch.randn(1, 10)
    print("\n输入形状:", dummy_input.shape)
    
    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)
    
    print("输出形状:", output.shape)
    print("模型输出:", output)
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)
    
    return model, dummy_input

def main():
    """主函数"""
    print("=" * 50)
    print("ONNX教程 - 第1部分: 引言与基础准备")
    print("=" * 50)
    
    # 检查环境
    print("\n[1] 环境检查")
    env_ready = check_environment()
    
    # 测试简单模型
    print("\n[2] 简单PyTorch模型测试")
    model, dummy_input = test_simple_model()
    
    print("\n环境检查完成。如果上述库均已正确安装，您可以继续下一部分的学习！")

if __name__ == "__main__":
    main()
    
    
"""
输出：

==================================================
ONNX教程 - 第1部分: 引言与基础准备
==================================================

[1] 环境检查
Python版本: 3.10.16 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:19:12) [MSC v.1929 64 bit (AMD64)]
PyTorch版本: 2.6.0+cpu
ONNX版本: 1.17.0
ONNX Runtime版本: 1.21.0
CUDA是否可用: False

[2] 简单PyTorch模型测试

输入形状: torch.Size([1, 10])
输出形状: torch.Size([1, 5])
模型输出: tensor([[ 0.0704,  0.1555, -0.0874, -0.2082,  0.2882]])

模型结构:
SimpleModel(
  (fc1): Linear(in_features=10, out_features=20, bias=True)
  (fc2): Linear(in_features=20, out_features=5, bias=True)
)

环境检查完成。如果上述库均已正确安装，您可以继续下一部分的学习！

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第3部分：从PyTorch导出ONNX模型
这个脚本展示如何将PyTorch模型导出为ONNX格式
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 重新定义模型结构（与训练时保持一致）
class MNISTModel(nn.Module):
    """MNIST手写数字识别模型"""
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_pytorch_model(model_path):
    """加载已保存的PyTorch模型"""
    # 创建模型实例
    model = MNISTModel()
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"PyTorch模型成功从{model_path}加载")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None
    
    # 设置为评估模式
    model.eval()
    return model

def export_to_onnx(model, onnx_path, input_shape=(1, 1, 28, 28)):
    """将PyTorch模型导出为ONNX格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 创建随机输入tensor（符合MNIST图像尺寸）
    dummy_input = torch.randn(input_shape, requires_grad=True)
    
    # 设置输出名称（可选）
    output_names = ['output']
    
    # 设置输入名称（可选）
    input_names = ['input']
    
    # 设置动态轴（可选，允许批量大小可变）
    dynamic_axes = {'input': {0: 'batch_size'},
                   'output': {0: 'batch_size'}}
    
    # 导出模型到ONNX
    try:
        torch.onnx.export(
            model,                     # 要导出的模型
            dummy_input,               # 模型输入
            onnx_path,                 # 输出ONNX文件路径
            export_params=True,        # 存储训练后的参数权重
            opset_version=12,          # ONNX算子集版本
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=input_names,   # 输入名称
            output_names=output_names, # 输出名称
            dynamic_axes=dynamic_axes, # 动态轴
            verbose=False              # 详细信息打印
        )
        print(f"ONNX模型已成功导出到: {os.path.abspath(onnx_path)}")
        return True
    except Exception as e:
        print(f"导出ONNX模型时出错: {e}")
        return False

def verify_onnx_model(onnx_path):
    """验证导出的ONNX模型是否有效"""
    try:
        import onnx
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 检查模型是否格式良好
        onnx.checker.check_model(onnx_model)
        
        print("ONNX模型检查通过！")
        
        # 打印一些基础模型信息
        print("\nONNX模型信息:")
        print(f"IR版本: {onnx_model.ir_version}")
        print(f"操作符集版本: {onnx_model.opset_import[0].version}")
        print(f"生产者名称: {onnx_model.producer_name}")
        
        # 打印输入输出信息
        print("\n输入信息:")
        for input in onnx_model.graph.input:
            print(f"  - 名称: {input.name}, 类型: {input.type.tensor_type.elem_type}, "
                  f"形状: {[d.dim_value if d.dim_value else 'dynamic' for d in input.type.tensor_type.shape.dim]}")
            
        print("\n输出信息:")
        for output in onnx_model.graph.output:
            print(f"  - 名称: {output.name}, 类型: {output.type.tensor_type.elem_type}, "
                  f"形状: {[d.dim_value if d.dim_value else 'dynamic' for d in output.type.tensor_type.shape.dim]}")
        
        # 计算模型大小
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # 转换为MB
        print(f"\nONNX模型大小: {model_size:.2f} MB")
        
        return True
    except ImportError:
        print("警告: 未安装onnx库，无法验证模型。请使用pip install onnx安装。")
        return False
    except Exception as e:
        print(f"验证ONNX模型时出错: {e}")
        return False

def compare_outputs(pytorch_model, onnx_path):
    """比较PyTorch模型与ONNX模型的输出是否一致"""
    try:
        import onnxruntime
        
        # 创建一个随机测试输入
        test_input = torch.randn(1, 1, 28, 28)
        
        # PyTorch模型推理
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX Runtime推理
        ort_session = onnxruntime.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # 比较两个输出
        is_close = np.allclose(pytorch_output, ort_output, rtol=1e-03, atol=1e-05)
        
        if is_close:
            print("✓ PyTorch和ONNX输出一致！")
        else:
            print("✗ PyTorch和ONNX输出不一致！")
            # 显示差异
            print(f"最大绝对误差: {np.max(np.abs(pytorch_output - ort_output))}")
        
        return is_close
    except ImportError:
        print("警告: 未安装onnxruntime库，无法比较输出。请使用pip install onnxruntime安装。")
        return False
    except Exception as e:
        print(f"比较模型输出时出错: {e}")
        return False

def explain_export_parameters():
    """解释torch.onnx.export的主要参数"""
    print("\n" + "=" * 50)
    print("torch.onnx.export 主要参数说明")
    print("=" * 50)
    
    parameters = [
        ("model", "要导出的PyTorch模型"),
        ("args", "模型的输入参数(通常是一个样例输入张量)"),
        ("f", "输出文件路径或类文件对象"),
        ("export_params", "如果为True，将导出模型参数；如果为False，则只导出模型结构"),
        ("opset_version", "导出模型使用的ONNX版本，默认为9"),
        ("do_constant_folding", "如果为True，在导出期间执行常量折叠优化"),
        ("input_names", "模型输入的名称列表"),
        ("output_names", "模型输出的名称列表"),
        ("dynamic_axes", "指定动态轴的字典，例如批处理维度"),
        ("verbose", "如果为True，打印导出过程的详细信息")
    ]
    
    for param, desc in parameters:
        print(f"{param.ljust(20)}: {desc}")

def main():
    """主函数：导出并验证ONNX模型"""
    print("=" * 50)
    print("ONNX教程 - 第3部分: 从PyTorch导出ONNX模型")
    print("=" * 50)
    
    # 设置文件路径
    pytorch_model_path = './models/mnist_cnn.pth'
    onnx_model_path = './models/mnist_cnn.onnx'
    
    # 1. 加载PyTorch模型
    print("\n[1] 加载PyTorch模型")
    pytorch_model = load_pytorch_model(pytorch_model_path)
    if pytorch_model is None:
        print(f"错误: 无法加载PyTorch模型。请先运行第2部分教程以训练和保存模型。")
        return
    
    # 2. 解释导出参数
    print("\n[2] 导出参数解释")
    explain_export_parameters()
    
    # 3. 导出为ONNX格式
    print("\n[3] 导出为ONNX格式")
    export_success = export_to_onnx(pytorch_model, onnx_model_path)
    if not export_success:
        print("错误: ONNX导出失败。")
        return
    
    # 4. 验证ONNX模型
    print("\n[4] 验证ONNX模型")
    verify_success = verify_onnx_model(onnx_model_path)
    if not verify_success:
        print("警告: ONNX模型验证步骤未完成。")
    
    # 5. 比较PyTorch和ONNX输出
    print("\n[5] 比较PyTorch和ONNX模型输出")
    compare_outputs(pytorch_model, onnx_model_path)
    
    print("\n导出过程完成！")

if __name__ == "__main__":
    main()
    
    
"""
输出：

==================================================
ONNX教程 - 第3部分: 从PyTorch导出ONNX模型
==================================================

[1] 加载PyTorch模型
PyTorch模型成功从./models/mnist_cnn.pth加载

[2] 导出参数解释

==================================================
torch.onnx.export 主要参数说明
==================================================
model               : 要导出的PyTorch模型
args                : 模型的输入参数(通常是一个样例输入张量)
f                   : 输出文件路径或类文件对象
export_params       : 如果为True，将导出模型参数；如果为False，则只导出模型结构
opset_version       : 导出模型使用的ONNX版本，默认为9
do_constant_folding : 如果为True，在导出期间执行常量折叠优化
input_names         : 模型输入的名称列表
output_names        : 模型输出的名称列表
dynamic_axes        : 指定动态轴的字典，例如批处理维度
verbose             : 如果为True，打印导出过程的详细信息

[3] 导出为ONNX格式
ONNX模型已成功导出到: C:\Users\k\Documents\project\programming_project\python_project\importance\XiaokeAILabs\datas\test_onnx\code\models\mnist_cnn.onnx

[4] 验证ONNX模型
ONNX模型检查通过！

ONNX模型信息:
IR版本: 7
操作符集版本: 12
生产者名称: pytorch

输入信息:
  - 名称: input, 类型: 1, 形状: ['dynamic', 1, 28, 28]

输出信息:
  - 名称: output, 类型: 1, 形状: ['dynamic', 10]

ONNX模型大小: 1.61 MB

[5] 比较PyTorch和ONNX模型输出
✓ PyTorch和ONNX输出一致！

导出过程完成！

"""
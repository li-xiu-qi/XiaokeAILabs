#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第8部分：ONNX与其他框架的互操作性
这个脚本展示如何在不同深度学习框架之间转换模型，以及使用可视化工具分析ONNX模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import onnx

# 抑制不必要的警告
warnings.filterwarnings('ignore')


def check_dependencies():
    """检查必要的依赖项"""
    dependencies = {
        "netron": "模型可视化工具",
        "tf2onnx": "TensorFlow到ONNX转换",
        "onnx2keras": "ONNX到Keras转换",
        "onnx_tf": "ONNX到TensorFlow转换",
        "skl2onnx": "Scikit-learn到ONNX转换",
        "keras2onnx": "Keras到ONNX转换"
    }
    
    print("检查依赖项安装状态:")
    for package, desc in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package} - {desc} [已安装]")
        except ImportError:
            print(f"✗ {package} - {desc} [未安装] - 可通过 'pip install {package}' 安装")
    
    print("\n注意: 并非所有依赖都需要，只需根据您的转换需求安装相应的包")


def create_pytorch_model():
    """创建一个简单的PyTorch模型"""
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super(SimpleClassifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 100)
            self.fc2 = nn.Linear(100, 10)
            self.dropout = nn.Dropout(0.25)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    model = SimpleClassifier()
    model.eval()
    
    # 保存PyTorch模型
    os.makedirs("./models", exist_ok=True)
    torch_path = "./models/simple_classifier.pth"
    torch.save(model.state_dict(), torch_path)
    print(f"PyTorch模型已保存到: {torch_path}")
    
    return model


def pytorch_to_onnx(model, input_shape=(1, 3, 32, 32)):
    """将PyTorch模型转换为ONNX"""
    # 创建随机输入
    dummy_input = torch.randn(input_shape)
    
    # 设置输出路径
    onnx_path = "./models/simple_classifier.onnx"
    
    # 导出为ONNX格式
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    
    print(f"ONNX模型已保存到: {onnx_path}")
    
    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过!")
    
    return onnx_path


def tensorflow_to_onnx():
    """从TensorFlow模型转换为ONNX（演示）"""
    try:
        import tensorflow as tf
        print("\n=== TensorFlow到ONNX转换演示 ===")
        
        # 创建一个简单的TensorFlow/Keras模型
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        
        model.compile(optimizer='adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        # 保存Keras模型
        model_path = "./models/tf_model"
        model.save(model_path)
        print(f"TensorFlow模型已保存到: {model_path}")
        
        # 转换为ONNX
        try:
            import tf2onnx
            
            # 创建具体形状的模型输入
            concrete_func = tf.function(lambda x: model(x))
            concrete_func = concrete_func.get_concrete_function(
                tf.TensorSpec([None, 784], tf.float32))
            
            # 转换成ONNX模型
            onnx_path = "./models/tf_model.onnx"
            tf2onnx.convert.from_concrete_function(
                concrete_func, 
                [tf2onnx.TensorProto.FLOAT],
                onnx_path
            )
            
            print(f"TensorFlow模型已转换为ONNX，保存路径: {onnx_path}")
            
            # 验证ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("从TensorFlow转换的ONNX模型验证通过!")
            
            return onnx_path
            
        except ImportError:
            print("无法进行转换：缺少tf2onnx包。请使用'pip install tf2onnx'安装。")
        
    except ImportError:
        print("无法创建TensorFlow模型：缺少TensorFlow包。请使用'pip install tensorflow'安装。")
    
    return None


def scikit_learn_to_onnx():
    """从scikit-learn模型转换为ONNX（演示）"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        
        print("\n=== Scikit-learn到ONNX转换演示 ===")
        
        # 加载示例数据集
        X, y = load_iris(return_X_y=True)
        
        # 训练一个简单的sklearn模型
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # 转换为ONNX
            initial_type = [('float_input', FloatTensorType([None, 4]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # 保存ONNX模型
            onnx_path = "./models/sklearn_model.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"scikit-learn模型已转换为ONNX，保存路径: {onnx_path}")
            return onnx_path
            
        except ImportError:
            print("无法进行转换：缺少skl2onnx包。请使用'pip install skl2onnx'安装。")
        
    except ImportError:
        print("无法创建scikit-learn模型：缺少scikit-learn包。请使用'pip install scikit-learn'安装。")
    
    return None


def onnx_to_tensorflow(onnx_path):
    """从ONNX转换为TensorFlow（演示）"""
    print("\n=== ONNX到TensorFlow转换演示 ===")
    
    try:
        import onnx_tf
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        
        # 转换为TensorFlow
        tf_path = "./models/from_onnx_tf_model"
        try:
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            tf_rep.export_graph(tf_path)
            print(f"ONNX模型已转换为TensorFlow，保存路径: {tf_path}")
            return tf_path
        except Exception as e:
            print(f"ONNX到TensorFlow转换失败: {str(e)}")
    
    except ImportError:
        print("无法进行转换：缺少onnx-tf包。请使用'pip install onnx-tf'安装。")
    
    return None


def visualize_with_netron(model_path):
    """使用Netron可视化模型"""
    print("\n=== 使用Netron可视化模型 ===")
    
    try:
        import netron
        print(f"正在启动Netron服务器以可视化模型: {model_path}")
        print("请在浏览器中查看模型。完成后在此控制台按Ctrl+C来停止服务器。")
        
        # 启动Netron服务器
        server = netron.start(model_path)
        
        return server
    except ImportError:
        print("无法可视化模型：缺少netron包。请使用'pip install netron'安装。")
    
    return None


def explain_model_exchange_formats():
    """解释不同的模型交换格式"""
    print("\n" + "=" * 50)
    print("模型交换格式比较")
    print("=" * 50)
    
    formats = [
        ("ONNX", 
         "Open Neural Network Exchange", 
         "通用的开放神经网络交换格式，支持多种框架", 
         "PyTorch, TensorFlow, MXNet, scikit-learn等"),
        
        ("PMML", 
         "Predictive Model Markup Language", 
         "基于XML的预测模型表示格式，主要用于传统机器学习", 
         "R, scikit-learn, SAS等"),
        
        ("TensorFlow SavedModel", 
         "TensorFlow's native format", 
         "TensorFlow的原生格式，适合TF生态系统", 
         "TensorFlow, Keras"),
        
        ("TorchScript", 
         "PyTorch's serialization format", 
         "序列化PyTorch模型的格式，支持即时编译", 
         "PyTorch"),
        
        ("Core ML", 
         "Apple's ML model format", 
         "专为Apple设备优化的模型格式", 
         "iOS, macOS应用"),
        
        ("TF Lite", 
         "TensorFlow Lite format", 
         "针对移动和边缘设备优化的格式", 
         "移动和嵌入式设备"),
    ]
    
    print(f"{"格式":<15}{"全称":<30}{"描述":<40}{"主要支持平台"}")
    print("-" * 100)
    
    for name, full_name, desc, platforms in formats:
        print(f"{name:<15}{full_name:<30}{desc:<40}{platforms}")


def compare_onnx_converters():
    """比较不同的ONNX转换器工具"""
    print("\n" + "=" * 50)
    print("ONNX转换工具比较")
    print("=" * 50)
    
    converters = [
        ("PyTorch → ONNX", "torch.onnx", "PyTorch官方支持，成熟稳定"),
        ("TensorFlow → ONNX", "tf2onnx", "由ONNX社区维护，支持TF 1.x和2.x"),
        ("Keras → ONNX", "keras2onnx/tf2onnx", "支持原生Keras和TF-Keras"),
        ("scikit-learn → ONNX", "skl2onnx", "转换常见ML模型"),
        ("ONNX → TensorFlow", "onnx-tf", "从ONNX转回TF模型"),
        ("ONNX → Keras", "onnx2keras", "转换为Keras格式，有算子兼容性限制"),
        ("ONNX → TFLite", "onnx2tflite", "转换为TensorFlow Lite格式")
    ]
    
    print(f"{"转换路径":<20}{"工具":<15}{"说明"}")
    print("-" * 65)
    
    for path, tool, desc in converters:
        print(f"{path:<20}{tool:<15}{desc}")


def explain_interoperability_challenges():
    """解释互操作性面临的挑战"""
    print("\n" + "=" * 50)
    print("互操作性挑战与解决方案")
    print("=" * 50)
    
    challenges = [
        ("算子支持差异", 
         "不同框架支持的操作符集不同，ONNX可能无法表示某些特定框架的独特操作。", 
         "使用自定义操作符扩展，尽量使用常见算子，或在导出前重写模型结构。"),
        
        ("精度差异", 
         "框架之间的数值计算实现可能略有不同，导致结果有微小差异。", 
         "设置合理的容错阈值，确保关键操作在不同框架中有一致的实现。"),
        
        ("动态特性支持", 
         "某些框架的动态特性（如动态图）在ONNX中表示困难。", 
         "使用dynamic_axes参数指定动态维度，考虑使用TorchScript等中间表示。"),
        
        ("复杂模型结构", 
         "具有条件分支、循环等复杂结构的模型在转换时可能遇到问题。", 
         "简化模型结构，将复杂模型分解为多个简单组件，或使用更高版本的opset。"),
        
        ("自定义层", 
         "自定义层或非标准操作在转换时可能丢失或错误转换。", 
         "实现自定义操作符映射，或用标准操作组合代替自定义操作。"),
        
        ("大模型支持", 
         "超大模型在转换过程中可能因内存限制而失败。", 
         "使用模型分片，增加内存，优化转换过程的内存使用。")
    ]
    
    for challenge, problem, solution in challenges:
        print(f"\n### {challenge} ###")
        print(f"问题: {problem}")
        print(f"解决方案: {solution}")


def main():
    """主函数"""
    print("=" * 50)
    print("ONNX教程 - 第8部分: ONNX与其他框架的互操作性")
    print("=" * 50)
    
    # 检查依赖项安装状态
    check_dependencies()
    
    # 1. 创建和导出PyTorch模型到ONNX
    print("\n[1] 创建PyTorch模型并导出为ONNX")
    pytorch_model = create_pytorch_model()
    onnx_path = pytorch_to_onnx(pytorch_model)
    
    # 2. TensorFlow到ONNX转换示例
    print("\n[2] TensorFlow到ONNX转换示例")
    tf_onnx_path = tensorflow_to_onnx()
    
    # 3. scikit-learn到ONNX转换示例
    print("\n[3] scikit-learn到ONNX转换示例")
    sklearn_onnx_path = scikit_learn_to_onnx()
    
    # 4. ONNX到TensorFlow转换示例
    print("\n[4] ONNX到TensorFlow转换示例")
    if onnx_path:
        tf_model_path = onnx_to_tensorflow(onnx_path)
    
    # 5. 模型交换格式比较
    print("\n[5] 模型交换格式比较")
    explain_model_exchange_formats()
    
    # 6. ONNX转换工具比较
    print("\n[6] ONNX转换工具比较")
    compare_onnx_converters()
    
    # 7. 互操作性挑战与解决方案
    print("\n[7] 互操作性挑战与解决方案")
    explain_interoperability_challenges()
    
    # 8. 使用Netron可视化模型
    print("\n[8] 使用Netron可视化模型")
    print("提示: 此步骤将启动Web服务器。如果要跳过，请按Ctrl+C")
    print("准备启动Netron可视化...")
    
    try:
        input("按Enter键继续...")
        if onnx_path:
            server = visualize_with_netron(onnx_path)
            # 此处会阻塞，直到用户按Ctrl+C
            if server:
                print("Netron服务器已关闭")
    except KeyboardInterrupt:
        print("\nNetron可视化步骤已跳过")
    
    print("\nONNX框架互操作性教程完成！")


if __name__ == "__main__":
    main()
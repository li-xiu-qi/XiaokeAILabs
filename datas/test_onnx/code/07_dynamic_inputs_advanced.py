#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第7部分：高级主题：动态输入与复杂模型
这个脚本展示如何处理动态输入、多输入输出以及复杂模型的ONNX导出和推理
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime
from PIL import Image
from torchvision import transforms, models


class DynamicReshapeModel(nn.Module):
    """具有动态批量大小的模型，使用reshape操作"""
    def __init__(self):
        super(DynamicReshapeModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # 输入x的形状：[batch_size, 3, 32, 32]
        batch_size = x.shape[0]  # 动态获取批量大小
        
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 32, 8, 8]
        
        # 动态reshape - 这是ONNX导出的关键点
        x = x.reshape(batch_size, 32 * 8 * 8)  # [batch_size, 2048]
        
        x = F.relu(self.fc1(x))  # [batch_size, 128]
        x = self.fc2(x)  # [batch_size, 10]
        return x


class MultiInputModel(nn.Module):
    """具有多个输入的模型示例"""
    def __init__(self):
        super(MultiInputModel, self).__init__()
        # 处理图像的卷积层
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.img_fc = nn.Linear(32 * 8 * 8, 128)
        
        # 处理辅助特征的层
        self.aux_fc1 = nn.Linear(5, 20)
        
        # 组合层
        self.combined_fc = nn.Linear(128 + 20, 10)
    
    def forward(self, image, features):
        # 处理图像输入
        img = self.pool(F.relu(self.conv1(image)))
        img = self.pool(F.relu(self.conv2(img)))
        img = img.reshape(img.shape[0], -1)
        img = F.relu(self.img_fc(img))
        
        # 处理辅助特征
        aux = F.relu(self.aux_fc1(features))
        
        # 组合两个特征
        combined = torch.cat((img, aux), dim=1)
        output = self.combined_fc(combined)
        
        return output


class MultiOutputModel(nn.Module):
    """具有多个输出的模型示例"""
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        
        # 修改最后一层，移除全连接分类器
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 多个任务头
        self.classifier = nn.Linear(512, 10)  # 分类任务
        self.regressor = nn.Linear(512, 2)    # 回归任务
        self.feature_extractor = nn.Linear(512, 64)  # 特征提取
    
    def forward(self, x):
        # 共享主干网络 [batch, 3, 224, 224] -> [batch, 512, 7, 7]
        features = self.backbone(x)
        
        # 全局平均池化 [batch, 512, 7, 7] -> [batch, 512]
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).reshape(x.shape[0], -1)
        
        # 应用多个任务头
        classification = self.classifier(pooled)  # [batch, 10]
        regression = self.regressor(pooled)       # [batch, 2]
        embedding = self.feature_extractor(pooled)  # [batch, 64]
        
        return classification, regression, embedding


def export_dynamic_model():
    """导出具有动态批量大小的模型"""
    print("\n" + "=" * 50)
    print("导出具有动态批量大小的模型")
    print("=" * 50)
    
    # 创建模型
    model = DynamicReshapeModel()
    model.eval()
    
    # 创建一个样本输入
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # 设置输出路径
    output_dir = "./models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "dynamic_model.onnx")
    
    # 定义动态轴，允许批量大小变化
    dynamic_axes = {
        'input': {0: 'batch_size'},  # 第一个维度是动态的
        'output': {0: 'batch_size'}  # 输出的第一个维度也是动态的
    }
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True
    )
    
    print(f"动态模型已导出到: {model_path}")
    
    # 验证模型
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("模型格式验证通过!")
    
    return model_path


def test_dynamic_batch_size(model_path):
    """测试动态批量大小模型的不同批量大小"""
    print("\n" + "=" * 50)
    print("测试动态批量大小模型")
    print("=" * 50)
    
    # 创建ONNX Runtime会话
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # 测试不同的批量大小
    batch_sizes = [1, 4, 8, 16]
    
    for batch in batch_sizes:
        print(f"\n测试批量大小 = {batch}")
        # 创建随机输入
        input_data = np.random.randn(batch, 3, 32, 32).astype(np.float32)
        
        # 运行推理
        outputs = session.run(None, {input_name: input_data})
        output = outputs[0]
        
        # 检查输出形状
        print(f"输入形状: {input_data.shape}")
        print(f"输出形状: {output.shape}")
        
        # 确认批量大小正确传递
        assert output.shape[0] == batch, f"输出批量大小({output.shape[0]})与输入({batch})不匹配!"
    
    print("\n所有批量大小测试通过!")


def export_multi_input_model():
    """导出具有多个输入的模型"""
    print("\n" + "=" * 50)
    print("导出具有多个输入的模型")
    print("=" * 50)
    
    # 创建模型
    model = MultiInputModel()
    model.eval()
    
    # 创建样本输入
    dummy_img = torch.randn(1, 3, 32, 32)  # 图像
    dummy_features = torch.randn(1, 5)     # 辅助特征
    
    # 设置输出路径
    output_dir = "./models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "multi_input_model.onnx")
    
    # 定义动态轴
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'features': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # 导出模型
    torch.onnx.export(
        model, 
        (dummy_img, dummy_features),  # 多个输入作为元组传递
        model_path,
        export_params=True,
        opset_version=11,
        input_names=['image', 'features'],  # 指定输入名称
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True
    )
    
    print(f"多输入模型已导出到: {model_path}")
    
    # 验证模型
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("模型格式验证通过!")
    
    return model_path


def test_multi_input_model(model_path):
    """测试多输入模型"""
    print("\n" + "=" * 50)
    print("测试多输入模型")
    print("=" * 50)
    
    # 创建ONNX Runtime会话
    session = onnxruntime.InferenceSession(model_path)
    
    # 获取输入名称
    input_names = [input.name for input in session.get_inputs()]
    print(f"模型输入名称: {input_names}")
    
    # 创建批量输入数据
    batch_size = 2
    img_data = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
    features_data = np.random.randn(batch_size, 5).astype(np.float32)
    
    # 准备输入字典
    inputs = {
        'image': img_data,
        'features': features_data
    }
    
    # 运行推理
    outputs = session.run(None, inputs)
    output = outputs[0]
    
    # 检查输出形状
    print(f"输入图像形状: {img_data.shape}")
    print(f"输入特征形状: {features_data.shape}")
    print(f"输出形状: {output.shape}")


def export_multi_output_model():
    """导出具有多个输出的模型"""
    print("\n" + "=" * 50)
    print("导出具有多个输出的模型")
    print("=" * 50)
    
    # 创建模型
    model = MultiOutputModel()
    model.eval()
    
    # 创建样本输入 (ResNet需要224x224图像)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 设置输出路径
    output_dir = "./models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "multi_output_model.onnx")
    
    # 定义动态轴
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'classification': {0: 'batch_size'},
        'regression': {0: 'batch_size'},
        'embedding': {0: 'batch_size'}
    }
    
    # 导出模型
    torch.onnx.export(
        model, 
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['classification', 'regression', 'embedding'],  # 多输出
        dynamic_axes=dynamic_axes,
        do_constant_folding=True
    )
    
    print(f"多输出模型已导出到: {model_path}")
    
    # 验证模型
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("模型格式验证通过!")
    
    return model_path


def test_multi_output_model(model_path):
    """测试多输出模型"""
    print("\n" + "=" * 50)
    print("测试多输出模型")
    print("=" * 50)
    
    # 创建ONNX Runtime会话
    session = onnxruntime.InferenceSession(model_path)
    
    # 获取输入和输出名称
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"模型输入名称: {input_name}")
    print(f"模型输出名称: {output_names}")
    
    # 创建批量输入数据
    batch_size = 2
    input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    # 运行推理
    outputs = session.run(output_names, {input_name: input_data})
    
    # 检查所有输出
    for i, output_name in enumerate(output_names):
        print(f"输出 '{output_name}' 形状: {outputs[i].shape}")


def explain_common_issues():
    """解释使用ONNX时的常见问题和解决方法"""
    print("\n" + "=" * 50)
    print("使用ONNX时的常见问题和解决方法")
    print("=" * 50)
    
    issues = [
        ("动态形状处理", 
         "问题：ONNX模型导出时固定形状，无法处理可变输入。\n"
         "解决：使用dynamic_axes参数指定哪些维度是动态的。"),
        
        ("不支持的操作或自定义操作", 
         "问题：某些PyTorch操作在ONNX中没有等效项。\n"
         "解决：使用symbolic_fn注册自定义操作，或重写模型以使用支持的操作。"),
        
        ("Reshape和视图操作", 
         "问题：PyTorch中的view和reshape操作在ONNX中有不同行为。\n"
         "解决：优先使用reshape而不是view，并明确指定目标形状。"),
        
        ("跟踪与脚本化", 
         "问题：torch.onnx.export使用跟踪，可能无法捕获所有控制流。\n"
         "解决：对于复杂控制流，先使用TorchScript脚本化模型，再导出。"),
        
        ("大型模型导出", 
         "问题：导出大型模型可能内存不足。\n"
         "解决：使用更大内存的机器，或尝试分部分导出模型。"),
        
        ("推理性能差异", 
         "问题：ONNX模型性能可能低于原始框架。\n"
         "解决：应用ONNX优化、使用特定硬件加速器、调整图优化级别。"),
        
        ("精度问题", 
         "问题：ONNX模型与原始模型输出有微小差异。\n"
         "解决：检查数值精度设置、使用custom_op_import等高级选项。")
    ]
    
    for topic, description in issues:
        print(f"\n#### {topic} ####")
        print(description)


def check_operator_support():
    """检查ONNX支持的操作符版本"""
    print("\n" + "=" * 50)
    print("ONNX操作符版本支持")
    print("=" * 50)
    
    # 打印可用的ONNX操作符集版本
    print("当前ONNX支持的操作符集版本:")
    for i in range(7, 14):  # 常见的ONNX操作符版本范围
        try:
            # 尝试使用不同版本的操作符集导出小模型
            model = nn.Linear(10, 5)
            dummy_input = torch.randn(1, 10)
            torch.onnx.export(model, dummy_input, f"./tmp_opset{i}.onnx", opset_version=i)
            print(f"  - Opset {i}: 支持 ✓")
            # 删除临时文件
            if os.path.exists(f"./tmp_opset{i}.onnx"):
                os.remove(f"./tmp_opset{i}.onnx")
        except Exception as e:
            print(f"  - Opset {i}: 不支持 ✗ ({str(e)[:50]}...)")
    
    print("\n推荐使用Opset 11或更高版本以获得最佳兼容性。")


def main():
    """主函数"""
    print("=" * 50)
    print("ONNX教程 - 第7部分: 高级主题：动态输入与复杂模型")
    print("=" * 50)
    
    print("\n本教程将演示处理高级ONNX场景:")
    print("1. 动态批量大小")
    print("2. 多输入模型")
    print("3. 多输出模型")
    print("4. 常见问题解决方案")
    print("5. 操作符支持检查")
    
    # 1. 导出和测试动态批量大小模型
    dynamic_model_path = export_dynamic_model()
    test_dynamic_batch_size(dynamic_model_path)
    
    # 2. 导出和测试多输入模型
    multi_input_path = export_multi_input_model()
    test_multi_input_model(multi_input_path)
    
    # 3. 导出和测试多输出模型
    try:
        multi_output_path = export_multi_output_model()
        test_multi_output_model(multi_output_path)
    except Exception as e:
        print(f"导出多输出模型时出错: {str(e)}")
        print("这可能是由于没有预训练的ResNet模型造成的，跳过此部分演示。")
    
    # 4. 解释使用ONNX时的常见问题
    explain_common_issues()
    
    # 5. 检查操作符支持
    check_operator_support()
    
    print("\n高级主题教程完成！")


if __name__ == "__main__":
    main()
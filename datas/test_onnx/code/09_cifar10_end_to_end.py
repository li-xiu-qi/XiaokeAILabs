#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第9部分：实战项目：CIFAR-10分类系统
这个脚本实现了一个完整的CIFAR-10图像分类项目，从数据准备、模型训练到ONNX导出和部署
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import onnx
import onnxruntime


# 定义参数解析器，让脚本支持不同的运行模式
def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 分类系统 - ONNX教程')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'export', 'infer', 'benchmark', 'all'],
                        help='运行模式: train(训练), export(导出ONNX), infer(推理), benchmark(性能测试), all(全部)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数 (默认: 5)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批量大小 (默认: 64)')
    parser.add_argument('--model-arch', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenet', 'custom'],
                        help='模型架构 (默认: resnet18)')
    parser.add_argument('--image', type=str, default=None,
                        help='用于推理的图像路径')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA')
    
    return parser.parse_args()


# 设置数据加载和预处理
def setup_data_loaders(batch_size):
    # 定义数据变换（数据增强和标准化）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 下载并加载CIFAR-10数据集
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


# 定义类别名称
def get_cifar10_classes():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# 定义自定义CNN模型
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        # 全连接层
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 创建模型
def create_model(model_arch, device):
    print(f"创建模型架构: {model_arch}")
    
    if model_arch == 'resnet18':
        # 使用预训练的ResNet18，修改最后一层以匹配CIFAR-10的10个类别
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_arch == 'mobilenet':
        # 使用MobileNetV2
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    elif model_arch == 'custom':
        # 使用自定义CNN
        model = CustomCNN()
    else:
        raise ValueError(f"不支持的模型架构: {model_arch}")
    
    return model.to(device)


# 训练模型
def train_model(model, device, train_loader, test_loader, epochs):
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    
    # 创建保存模型的目录
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 记录损失和准确率
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印训练进度
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%')
        
        # 学习率调度
        scheduler.step()
        
        # 计算平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 评估阶段
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                output = model(data)
                
                # 计算损失
                loss = criterion(output, target)
                test_loss += loss.item()
                
                # 计算准确率
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # 计算平均测试损失和准确率
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        
        # 保存训练历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 打印评估结果
        print(f'Epoch: {epoch}/{epochs} | 训练损失: {train_loss:.3f} | 训练准确率: {train_acc:.2f}% | '
              f'测试损失: {test_loss:.3f} | 测试准确率: {test_acc:.2f}%')
        
        # 保存最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'cifar10_best.pth'))
            print(f'测试准确率提高了！已保存模型。')
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'cifar10_final.pth'))
    print("训练完成！")
    
    # 保存训练历史
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # 绘制训练历史
    plot_training_history(history)
    
    return history


# 绘制训练历史
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['test_loss'], label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和测试损失')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['test_acc'], label='测试准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.title('训练和测试准确率')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/cifar10_training_history.png')
    print("训练历史图已保存到 './results/cifar10_training_history.png'")
    
    # 显示图像（如果在非交互环境中可能无法显示）
    try:
        plt.show()
    except Exception:
        pass


# 导出模型为ONNX格式
def export_to_onnx(model, model_arch, device):
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    
    # ONNX模型输出路径
    onnx_model_dir = './models'
    os.makedirs(onnx_model_dir, exist_ok=True)
    onnx_model_path = os.path.join(onnx_model_dir, f'cifar10_{model_arch}.onnx')
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    
    print(f"ONNX模型已保存到: {onnx_model_path}")
    
    # 验证导出的ONNX模型
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过！")
    
    # 保存类别信息供推理时使用
    classes = get_cifar10_classes()
    class_info_path = os.path.join(onnx_model_dir, 'cifar10_classes.json')
    with open(class_info_path, 'w') as f:
        json.dump(classes, f)
    
    print(f"类别信息已保存到: {class_info_path}")
    
    return onnx_model_path


# 使用ONNX模型进行推理
def inference_with_onnx(onnx_model_path, image_path=None):
    print("\n开始ONNX推理...")
    
    # 创建ONNX运行时会话
    session = onnxruntime.InferenceSession(onnx_model_path)
    
    # 获取模型输入名称
    input_name = session.get_inputs()[0].name
    
    # 加载类别信息
    model_dir = os.path.dirname(onnx_model_path)
    class_info_path = os.path.join(model_dir, 'cifar10_classes.json')
    
    try:
        with open(class_info_path, 'r') as f:
            classes = json.load(f)
    except FileNotFoundError:
        classes = get_cifar10_classes()
    
    # 图像预处理函数
    def preprocess_image(image_path):
        # 加载和调整图像大小
        image = Image.open(image_path).convert('RGB')
        image = image.resize((32, 32))
        
        # 转换为numpy数组，并进行通道调整和标准化
        img_array = np.array(image).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        img_array = img_array / 255.0
        
        # 标准化
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(-1, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(-1, 1, 1)
        img_array = (img_array - mean) / std
        
        # 添加批量维度
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, image
    
    # 如果没有提供图像路径，使用测试集中的样本
    if image_path is None or not os.path.exists(image_path):
        # 加载一些测试样本
        _, test_loader = setup_data_loaders(batch_size=5)
        data_iterator = iter(test_loader)
        images, labels = next(data_iterator)
        
        # 选择前5个样本进行推理和可视化
        results = []
        plt.figure(figsize=(15, 3))
        for i in range(min(5, len(images))):
            # 获取单个图像，转换为numpy数组
            img = images[i].numpy()
            
            # 运行推理
            outputs = session.run(None, {input_name: img.reshape(1, *img.shape)})
            output = outputs[0]
            
            # 获取预测类别
            predicted_class = np.argmax(output)
            predicted_label = classes[predicted_class]
            true_label = classes[labels[i].item()]
            
            # 存储结果
            results.append({
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': output.max()
            })
            
            # 反标准化图像用于显示
            img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # 绘制图像和结果
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            color = 'green' if predicted_label == true_label else 'red'
            plt.title(f"P: {predicted_label}\nT: {true_label}", color=color)
            plt.axis('off')
        
        plt.tight_layout()
        
        # 保存结果图像
        result_dir = './results'
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(os.path.join(result_dir, 'cifar10_inference_results.png'))
        try:
            plt.show()
        except Exception:
            pass
        
        print("推理结果:")
        for i, res in enumerate(results):
            print(f"  样本 {i+1}: 真实类别={res['true_label']}, "
                  f"预测类别={res['predicted_label']}, 置信度={res['confidence']:.4f}")
    else:
        # 处理单个图像
        img_array, original_image = preprocess_image(image_path)
        
        # 运行推理
        start_time = time.time()
        outputs = session.run(None, {input_name: img_array})
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        
        output = outputs[0]
        
        # 获取预测结果
        probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities)
        predicted_label = classes[predicted_class]
        confidence = probabilities[0][predicted_class]
        
        # 获取前3个预测结果
        top3_indices = np.argsort(-probabilities[0])[:3]
        top3_labels = [classes[idx] for idx in top3_indices]
        top3_probs = [probabilities[0][idx] for idx in top3_indices]
        
        # 打印结果
        print(f"推理时间: {inference_time:.2f}毫秒")
        print(f"预测类别: {predicted_label}")
        print(f"置信度: {confidence:.4f}")
        print("前3个预测:")
        for label, prob in zip(top3_labels, top3_probs):
            print(f"  {label}: {prob:.4f}")
        
        # 可视化结果
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image)
        plt.title(f"预测: {predicted_label} (置信度: {confidence:.4f})")
        plt.axis('off')
        
        # 添加置信度条
        plt.figure(figsize=(10, 3))
        bars = plt.barh(range(len(top3_labels)), top3_probs)
        plt.yticks(range(len(top3_labels)), top3_labels)
        plt.xlabel('置信度')
        plt.title('前3个预测类别')
        
        # 为每个条形添加值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{top3_probs[i]:.4f}", va='center')
        
        plt.tight_layout()
        
        # 保存结果
        result_dir = './results'
        os.makedirs(result_dir, exist_ok=True)
        plt.savefig(os.path.join(result_dir, 'single_image_prediction.png'))
        try:
            plt.show()
        except Exception:
            pass
    
    print("推理完成！")


# 性能基准测试
def benchmark_model(onnx_model_path, device):
    print("\n开始性能基准测试...")
    
    # 创建ONNX运行时会话
    session_options = onnxruntime.SessionOptions()
    # 设置图优化级别
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 根据设备选择提供程序
    if device.type == 'cuda' and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("使用CUDA执行提供程序")
    else:
        providers = ['CPUExecutionProvider']
        print("使用CPU执行提供程序")
    
    session = onnxruntime.InferenceSession(onnx_model_path, session_options, providers=providers)
    
    # 获取模型输入名称
    input_name = session.get_inputs()[0].name
    
    # 加载测试数据
    _, test_loader = setup_data_loaders(batch_size=1)
    
    # 预热运行
    print("预热中...")
    warmup_samples = min(50, len(test_loader.dataset))
    for i, (data, _) in enumerate(test_loader):
        if i >= warmup_samples:
            break
        _ = session.run(None, {input_name: data.numpy()})
    
    # 批量大小测试
    batch_sizes = [1, 8, 16, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n测试批量大小: {batch_size}")
        
        # 创建批量输入
        dummy_input = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
        
        # 测量推理时间
        total_time = 0
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.time()
            _ = session.run(None, {input_name: dummy_input})
            total_time += (time.time() - start_time)
        
        avg_time = total_time / iterations * 1000  # 毫秒
        throughput = batch_size * iterations / total_time  # 样本/秒
        
        results[batch_size] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'time_per_sample': avg_time / batch_size
        }
        
        print(f"  平均推理时间: {avg_time:.2f}毫秒")
        print(f"  每样本时间: {avg_time / batch_size:.2f}毫秒")
        print(f"  吞吐量: {throughput:.2f}样本/秒")
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # 绘制每样本推理时间
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, [results[bs]['time_per_sample'] for bs in batch_sizes], marker='o')
    plt.xlabel('批量大小')
    plt.ylabel('每样本推理时间 (毫秒)')
    plt.title('批量大小对每样本推理时间的影响')
    plt.grid()
    
    # 绘制吞吐量
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, [results[bs]['throughput'] for bs in batch_sizes], marker='o', color='green')
    plt.xlabel('批量大小')
    plt.ylabel('吞吐量 (样本/秒)')
    plt.title('批量大小对吞吐量的影响')
    plt.grid()
    
    plt.tight_layout()
    
    # 保存结果
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'benchmark_results.png'))
    try:
        plt.show()
    except Exception:
        pass
    
    print("\n性能基准测试完成！")
    
    # 保存基准测试结果
    with open(os.path.join(result_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f)


# 端到端流程
def end_to_end_pipeline(args):
    # 检查CUDA是否可用
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(args.model_arch, device)
    
    if args.mode == 'train' or args.mode == 'all':
        # 设置数据加载器
        train_loader, test_loader = setup_data_loaders(args.batch_size)
        # 训练模型
        train_model(model, device, train_loader, test_loader, args.epochs)
    
    # 加载训练好的模型权重
    model_path = os.path.join('./models', 'cifar10_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"从{model_path}加载模型权重")
    
    if args.mode == 'export' or args.mode == 'all':
        # 导出为ONNX模型
        onnx_model_path = export_to_onnx(model, args.model_arch, device)
    else:
        # 查找现有的ONNX模型
        onnx_model_path = os.path.join('./models', f'cifar10_{args.model_arch}.onnx')
        if not os.path.exists(onnx_model_path):
            print(f"ONNX模型{onnx_model_path}不存在，现在创建...")
            onnx_model_path = export_to_onnx(model, args.model_arch, device)
    
    if args.mode == 'infer' or args.mode == 'all':
        # 使用ONNX模型进行推理
        inference_with_onnx(onnx_model_path, args.image)
    
    if args.mode == 'benchmark' or args.mode == 'all':
        # 进行性能基准测试
        benchmark_model(onnx_model_path, device)


# 主函数
def main():
    print("=" * 50)
    print("ONNX教程 - 第9部分: 实战项目：CIFAR-10分类系统")
    print("=" * 50)
    
    # 解析命令行参数
    args = parse_args()
    
    # 运行端到端流程
    end_to_end_pipeline(args)
    
    print("\n实战项目完成！")


if __name__ == "__main__":
    main()
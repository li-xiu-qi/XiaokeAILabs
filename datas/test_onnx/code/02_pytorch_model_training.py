#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第2部分：PyTorch模型创建与训练
这个脚本展示如何创建、训练并保存一个简单的PyTorch模型（MNIST手写数字识别）
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 检查是否可以使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class MNISTModel(nn.Module):
    """MNIST手写数字识别模型"""
    def __init__(self):
        super(MNISTModel, self).__init__()
        # 第一个卷积层：1通道输入，32通道输出，3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层：32通道输入，64通道输出，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 全连接层1：将7x7x64维张量展平为1维，然后映射到128维
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        # 全连接层2：将128维映射到10维（对应10个数字类别）
        self.fc2 = nn.Linear(128, 10)
        # Dropout层，用于减少过拟合
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 第一个卷积层+ReLU激活+最大池化
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸: [batch, 32, 14, 14]
        # 第二个卷积层+ReLU激活+最大池化
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸: [batch, 64, 7, 7]
        # 展平张量
        x = x.view(-1, 7 * 7 * 64)  # 输出尺寸: [batch, 7*7*64]
        # Dropout
        x = self.dropout(x)
        # 全连接层1+ReLU激活
        x = F.relu(self.fc1(x))  # 输出尺寸: [batch, 128]
        # Dropout
        x = self.dropout(x)
        # 全连接层2
        x = self.fc2(x)  # 输出尺寸: [batch, 10]
        # 输出层使用log_softmax
        return F.log_softmax(x, dim=1)

def load_data():
    """加载MNIST数据集"""
    # 数据预处理和增强
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST数据集的均值和标准差）
    ])
    
    # 下载并加载训练集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 下载并加载测试集
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, epoch):
    """训练模型的一个epoch"""
    model.train()  # 设置为训练模式
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移至GPU（如果可用）
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 打印训练进度
        if (batch_idx + 1) % 100 == 0:
            print(f'训练: Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')
    
    # 计算平均损失和训练时间
    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    print(f'Epoch {epoch} 训练完成, 平均损失: {avg_loss:.6f}, 用时: {elapsed:.2f} 秒')
    
    return avg_loss

def test(model, device, test_loader):
    """评估模型性能"""
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # 在评估时不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 累加批次损失
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # 获取最大对数概率的索引
            pred = output.argmax(dim=1, keepdim=True)
            
            # 计算正确预测的数量
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # 计算平均损失
    test_loss /= len(test_loader.dataset)
    
    # 打印测试结果
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

def save_model(model, path='model.pth'):
    """保存PyTorch模型"""
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), path)
    print(f'模型已保存到 {os.path.abspath(path)}')

def main():
    """主函数：训练和评估MNIST模型"""
    print("=" * 50)
    print("ONNX教程 - 第2部分: PyTorch模型创建与训练")
    print("=" * 50)
    
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    
    # 加载数据
    print("\n[1] 加载MNIST数据集")
    train_loader, test_loader = load_data()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n[2] 创建MNIST识别模型")
    model = MNISTModel().to(device)
    print(model)
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("\n[3] 开始训练模型")
    n_epochs = 5
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, n_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
    
    # 保存训练好的模型
    print("\n[4] 保存模型")
    save_model(model, './models/mnist_cnn.pth')
    
    # 打印训练结果
    print("\n[5] 训练结束")
    print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()
    
    
    
    
"""
输出：

使用设备: cpu
==================================================
ONNX教程 - 第2部分: PyTorch模型创建与训练
==================================================

[1] 加载MNIST数据集
100.0%
100.0%
100.0%
100.0%
训练集大小: 60000
测试集大小: 10000

[2] 创建MNIST识别模型
MNISTModel(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
)

[3] 开始训练模型
训练: Epoch 1 [6336/60000 (11%)]        损失: 0.313702
训练: Epoch 1 [12736/60000 (21%)]       损失: 0.210255
训练: Epoch 1 [19136/60000 (32%)]       损失: 0.100655
训练: Epoch 1 [25536/60000 (43%)]       损失: 0.090547
训练: Epoch 1 [31936/60000 (53%)]       损失: 0.024115
训练: Epoch 1 [38336/60000 (64%)]       损失: 0.047463
训练: Epoch 1 [44736/60000 (75%)]       损失: 0.029490
训练: Epoch 1 [51136/60000 (85%)]       损失: 0.149914
训练: Epoch 1 [57536/60000 (96%)]       损失: 0.037385
Epoch 1 训练完成, 平均损失: 0.172681, 用时: 32.16 秒
测试集: 平均损失: 0.0530, 准确率: 9834/10000 (98.34%)
训练: Epoch 2 [6336/60000 (11%)]        损失: 0.035880
训练: Epoch 2 [12736/60000 (21%)]       损失: 0.126968
训练: Epoch 2 [19136/60000 (32%)]       损失: 0.027440
训练: Epoch 2 [25536/60000 (43%)]       损失: 0.068839
训练: Epoch 2 [31936/60000 (53%)]       损失: 0.026306
训练: Epoch 2 [38336/60000 (64%)]       损失: 0.055877
训练: Epoch 2 [44736/60000 (75%)]       损失: 0.053969
训练: Epoch 2 [51136/60000 (85%)]       损失: 0.094864
训练: Epoch 2 [57536/60000 (96%)]       损失: 0.066359
Epoch 2 训练完成, 平均损失: 0.062655, 用时: 33.67 秒
测试集: 平均损失: 0.0306, 准确率: 9900/10000 (99.00%)
训练: Epoch 3 [6336/60000 (11%)]        损失: 0.024332
训练: Epoch 3 [12736/60000 (21%)]       损失: 0.011601
训练: Epoch 3 [19136/60000 (32%)]       损失: 0.009040
训练: Epoch 3 [25536/60000 (43%)]       损失: 0.010512
训练: Epoch 3 [31936/60000 (53%)]       损失: 0.005006
训练: Epoch 3 [38336/60000 (64%)]       损失: 0.043408
训练: Epoch 3 [44736/60000 (75%)]       损失: 0.048316
训练: Epoch 3 [51136/60000 (85%)]       损失: 0.007906
训练: Epoch 3 [57536/60000 (96%)]       损失: 0.032745
Epoch 3 训练完成, 平均损失: 0.046090, 用时: 40.56 秒
测试集: 平均损失: 0.0321, 准确率: 9892/10000 (98.92%)
训练: Epoch 4 [6336/60000 (11%)]        损失: 0.029591
训练: Epoch 4 [12736/60000 (21%)]       损失: 0.061966
训练: Epoch 4 [19136/60000 (32%)]       损失: 0.015733
训练: Epoch 4 [25536/60000 (43%)]       损失: 0.032805
训练: Epoch 4 [31936/60000 (53%)]       损失: 0.014238
训练: Epoch 4 [38336/60000 (64%)]       损失: 0.125204
训练: Epoch 4 [44736/60000 (75%)]       损失: 0.019043
训练: Epoch 4 [51136/60000 (85%)]       损失: 0.026008
训练: Epoch 4 [57536/60000 (96%)]       损失: 0.081384
Epoch 4 训练完成, 平均损失: 0.037881, 用时: 35.26 秒
测试集: 平均损失: 0.0322, 准确率: 9894/10000 (98.94%)
训练: Epoch 5 [6336/60000 (11%)]        损失: 0.030131
训练: Epoch 5 [12736/60000 (21%)]       损失: 0.022695
训练: Epoch 5 [19136/60000 (32%)]       损失: 0.025452
训练: Epoch 5 [25536/60000 (43%)]       损失: 0.054379
训练: Epoch 5 [31936/60000 (53%)]       损失: 0.013315
训练: Epoch 5 [38336/60000 (64%)]       损失: 0.036957
训练: Epoch 5 [44736/60000 (75%)]       损失: 0.008009
训练: Epoch 5 [51136/60000 (85%)]       损失: 0.001014
训练: Epoch 5 [57536/60000 (96%)]       损失: 0.005562
Epoch 5 训练完成, 平均损失: 0.032956, 用时: 28.24 秒
测试集: 平均损失: 0.0267, 准确率: 9915/10000 (99.15%)

[4] 保存模型
模型已保存到 C:\Users\k\Documents\project\programming_project\python_project\importance\XiaokeAILabs\datas\test_onnx\code\models\mnist_cnn.pth

[5] 训练结束
最终测试准确率: 99.15%
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第5部分：使用ONNX Runtime进行推理
这个脚本展示如何使用ONNX Runtime加载和运行ONNX模型进行推理
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms

# 检查ONNX Runtime是否安装
try:
    import onnxruntime
    print(f"ONNX Runtime已成功安装，版本：{onnxruntime.__version__}")
except ImportError:
    print("错误：未安装ONNX Runtime。请使用 'pip install onnxruntime' 安装。")
    exit(1)

def get_available_providers():
    """获取ONNX Runtime可用的执行提供程序"""
    providers = onnxruntime.get_available_providers()
    print("\n可用的执行提供程序：")
    for i, provider in enumerate(providers):
        print(f"  [{i}] {provider}")
    
    # 检查CUDA是否可用
    if 'CUDAExecutionProvider' in providers:
        print("\nCUDA可用于加速！模型将默认使用GPU执行")
    else:
        print("\n注意：未检测到CUDA支持。模型将在CPU上运行。")
        print("如需GPU加速，请安装支持CUDA的ONNX Runtime版本。")
    
    return providers

def create_inference_session(model_path, providers=None):
    """创建ONNX Runtime推理会话"""
    try:
        # 如果未指定提供程序，使用默认设置
        if providers is None:
            session = onnxruntime.InferenceSession(model_path)
        else:
            # 创建会话选项
            options = onnxruntime.SessionOptions()
            # 启用优化
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            # 启用内存模式
            options.enable_mem_pattern = True
            # 启用并行执行
            options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            # 创建推理会话
            session = onnxruntime.InferenceSession(
                model_path, 
                options, 
                providers=providers
            )
        
        # 打印模型输入信息
        print("\n模型输入：")
        for i, input_node in enumerate(session.get_inputs()):
            print(f"  [{i}] 名称: {input_node.name}")
            print(f"      形状: {input_node.shape}")
            print(f"      类型: {input_node.type}")
        
        # 打印模型输出信息
        print("\n模型输出：")
        for i, output_node in enumerate(session.get_outputs()):
            print(f"  [{i}] 名称: {output_node.name}")
            print(f"      形状: {output_node.shape}")
            print(f"      类型: {output_node.type}")
        
        return session
    except Exception as e:
        print(f"创建推理会话时出错：{str(e)}")
        return None

def load_mnist_test_data(num_samples=10, data_dir='./data'):
    """加载MNIST测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    try:
        # 下载并加载MNIST测试集
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        
        # 选择指定数量的样本
        data_loader = []
        for i in range(num_samples):
            image, label = test_dataset[i]
            data_loader.append((image.numpy(), label))
        
        print(f"成功加载{len(data_loader)}个MNIST测试样本")
        return data_loader
    except Exception as e:
        print(f"加载MNIST数据时出错：{str(e)}")
        return None

def run_inference(session, input_data, input_name):
    """使用ONNX Runtime运行推理"""
    # 准备输入，确保是4维的 [batch_size, channels, height, width]
    if input_data.ndim == 3:  # 如果输入是3维的，添加批次维度
        input_data = np.expand_dims(input_data, axis=0)
    
    # 准备输入
    inputs = {input_name: input_data}
    
    # 运行推理
    start_time = time.time()
    outputs = session.run(None, inputs)
    inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    return outputs, inference_time

def evaluate_performance(session, test_data, input_name, num_runs=5):
    """评估模型性能（速度和准确性）"""
    inference_times = []
    correct_predictions = 0
    
    for image, label in test_data:
        # 多次运行以获得平均推理时间
        times = []
        predictions = []
        
        for _ in range(num_runs):
            # 运行单次推理
            outputs, inf_time = run_inference(session, image, input_name)
            times.append(inf_time)
            
            # 获取预测结果（对数概率的索引）
            prediction = np.argmax(outputs[0], axis=1)[0]
            predictions.append(prediction)
        
        # 计算平均推理时间
        avg_time = sum(times) / len(times)
        inference_times.append(avg_time)
        
        # 检查预测是否正确（使用最常见的预测结果）
        most_common_prediction = max(set(predictions), key=predictions.count)
        if most_common_prediction == label:
            correct_predictions += 1
    
    # 计算整体性能指标
    accuracy = correct_predictions / len(test_data) * 100
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    print("\n性能评估结果：")
    print(f"准确率: {accuracy:.2f}% ({correct_predictions}/{len(test_data)})")
    print(f"平均推理时间: {avg_inference_time:.2f}毫秒")
    print(f"最短推理时间: {min(inference_times):.2f}毫秒")
    print(f"最长推理时间: {max(inference_times):.2f}毫秒")
    
    return accuracy, avg_inference_time

def visualize_predictions(test_data, predictions, num_samples=5):
    """可视化模型预测结果"""
    # 选择要显示的样本数量
    num_samples = min(num_samples, len(test_data))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        image, label = test_data[i]
        # 从输出数组中提取单个预测值作为标量
        prediction = int(np.argmax(predictions[i][0], axis=1)[0])
        
        # 将图像从NCHW格式转换为显示格式
        display_image = image.reshape(28, 28)
        
        # 添加子图
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(display_image, cmap='gray')
        
        # 设置标题（绿色表示预测正确，红色表示预测错误）
        if prediction == label:
            plt.title(f"预测: {prediction}\n实际: {label}", color='green')
        else:
            plt.title(f"预测: {prediction}\n实际: {label}", color='red')
        
        plt.axis('off')  # 隐藏坐标轴
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/mnist_predictions.png')
    print(f"\n预测可视化结果已保存到 ./results/mnist_predictions.png")
    
    # 显示图形
    try:
        plt.show()
    except Exception:
        print("无法显示图形。图像已保存到文件。")

def run_batch_inference(session, batch_size=16):
    """批量推理演示"""
    print("\n批量推理演示:")
    
    # 准备批量输入
    batch_input = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # 批量推理
    start_time = time.time()
    outputs = session.run(None, {input_name: batch_input})
    batch_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    print(f"批量大小: {batch_size}")
    print(f"批量推理时间: {batch_time:.2f}毫秒")
    print(f"每样本平均时间: {batch_time / batch_size:.2f}毫秒")
    
    return batch_time

def compare_execution_providers(model_path, input_data):
    """比较不同执行提供程序的性能"""
    providers = onnxruntime.get_available_providers()
    results = []
    
    if len(providers) <= 1:
        print("\n只有一个执行提供程序可用，无法进行比较。")
        return None
    
    print("\n比较不同执行提供程序的性能：")
    
    # 准备单个输入样本
    sample_input = input_data[0][0]  # 第一个样本的图像
    
    for provider in providers:
        try:
            # 创建使用特定提供程序的会话
            print(f"\n使用 {provider}:")
            session = onnxruntime.InferenceSession(
                model_path, 
                providers=[provider]
            )
            
            input_name = session.get_inputs()[0].name
            
            # 预热运行
            _ = session.run(None, {input_name: sample_input})
            
            # 测量性能（多次运行）
            times = []
            for _ in range(10):
                _, inf_time = run_inference(session, sample_input, input_name)
                times.append(inf_time)
            
            avg_time = sum(times) / len(times)
            results.append((provider, avg_time))
            
            print(f"平均推理时间: {avg_time:.2f}毫秒")
            
        except Exception as e:
            print(f"{provider} 运行失败: {str(e)}")
    
    # 打印性能比较结果
    if len(results) > 1:
        print("\n执行提供程序性能比较：")
        # 按推理时间排序
        results.sort(key=lambda x: x[1])
        
        # 使用最快的提供程序作为基准
        baseline_provider, baseline_time = results[0]
        
        for provider, avg_time in results:
            speedup = baseline_time / avg_time if avg_time > 0 else float('inf')
            print(f"{provider.ljust(30)}: {avg_time:.2f}毫秒 (速度比: {speedup:.2f}x)")
    
    return results

def explain_session_options():
    """解释ONNX Runtime会话选项"""
    print("\nONNX Runtime会话选项说明：")
    options = [
        ("graph_optimization_level", "图优化级别，可设置为：DISABLE_ALL, ENABLE_BASIC, ENABLE_EXTENDED, ENABLE_ALL"),
        ("intra_op_num_threads", "算子内并行的线程数，设置为处理器核心数通常效果最佳"),
        ("inter_op_num_threads", "算子间并行的线程数"),
        ("execution_mode", "执行模式：ORT_SEQUENTIAL（顺序）或ORT_PARALLEL（并行）"),
        ("enable_profiling", "是否启用性能分析"),
        ("enable_mem_pattern", "是否启用内存模式优化"),
        ("enable_cpu_mem_arena", "是否启用CPU内存竞技场优化"),
        ("session_log_severity_level", "日志级别：0(Verbose), 1(Info), 2(Warning), 3(Error), 4(Fatal)")
    ]
    
    for option, description in options:
        print(f"  - {option.ljust(25)}: {description}")

def visualize_batch_performance(session):
    """可视化不同批次大小下的性能比较"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 设置要测试的批次大小
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    total_times = []      # 总推理时间
    per_sample_times = [] # 每样本平均时间
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    
    # 测量每个批次大小的性能
    print("\n测量不同批次大小的性能...")
    for batch_size in batch_sizes:
        # 准备批量输入
        batch_input = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
        
        # 预热运行
        _ = session.run(None, {input_name: batch_input})
        
        # 多次运行以获得平均时间
        times = []
        for _ in range(5):  # 运行5次取平均值
            start_time = time.time()
            _ = session.run(None, {input_name: batch_input})
            inference_time = (time.time() - start_time) * 1000  # 毫秒
            times.append(inference_time)
        
        # 计算平均时间
        avg_time = sum(times) / len(times)
        avg_per_sample = avg_time / batch_size
        
        total_times.append(avg_time)
        per_sample_times.append(avg_per_sample)
        
        print(f"批次大小: {batch_size}, 总时间: {avg_time:.2f}毫秒, 每样本: {avg_per_sample:.2f}毫秒")
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 第一个子图：总推理时间
    plt.subplot(2, 1, 1)
    plt.plot(batch_sizes, total_times, 'o-', color='blue', linewidth=2, markersize=8)
    plt.title('不同批次大小的总推理时间', fontsize=14)
    plt.xlabel('批次大小', fontsize=12)
    plt.ylabel('推理时间 (毫秒)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(batch_sizes)
    
    # 第二个子图：每样本平均时间
    plt.subplot(2, 1, 2)
    plt.plot(batch_sizes, per_sample_times, 'o-', color='green', linewidth=2, markersize=8)
    plt.title('不同批次大小的每样本平均推理时间', fontsize=14)
    plt.xlabel('批次大小', fontsize=12)
    plt.ylabel('每样本推理时间 (毫秒)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(batch_sizes)
    
    # 使显示更美观
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/batch_performance_comparison.png')
    print(f"\n批次性能对比图已保存到 ./results/batch_performance_comparison.png")
    
    # 显示图表
    try:
        plt.show()
    except Exception:
        print("无法显示图形。图像已保存到文件。")
    
    # 返回结果数据，方便后续使用
    return {
        "batch_sizes": batch_sizes,
        "total_times": total_times,
        "per_sample_times": per_sample_times
    }

def main():
    """主函数"""
    print("=" * 50)
    print("ONNX教程 - 第5部分：使用ONNX Runtime进行推理")
    print("=" * 50)
    
    # 检查可用的执行提供程序
    providers = get_available_providers()
    
    # 设置ONNX模型路径（使用之前导出的MNIST模型）
    model_path = './models/mnist_cnn.onnx'
    
    # 1. 创建推理会话
    print("\n[1] 创建ONNX Runtime推理会话")
    session = create_inference_session(model_path)
    if session is None:
        print(f"错误：无法创建推理会话。请确保模型文件'{model_path}'存在。")
        print(f"请先运行第3部分教程以导出ONNX模型。")
        return
    
    # 2. 加载MNIST测试数据
    print("\n[2] 加载MNIST测试数据")
    test_data = load_mnist_test_data(num_samples=10)
    if test_data is None:
        return
    
    # 3. 运行单个推理
    print("\n[3] 运行单个示例推理")
    # 获取第一个样本
    sample_image, sample_label = test_data[0]
    print(f"样本实际标签: {sample_label}")
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    
    # 运行推理
    outputs, inference_time = run_inference(session, sample_image, input_name)
    
    # 打印结果
    prediction = np.argmax(outputs[0], axis=1)[0]
    print(f"模型预测: {prediction}")
    print(f"推理时间: {inference_time:.2f}毫秒")
    
    # 4. 评估性能（准确率和速度）
    print("\n[4] 模型性能评估")
    accuracy, avg_time = evaluate_performance(session, test_data, input_name)
    
    # 5. 批量推理演示
    print("\n[5] 批量推理性能测试")
    run_batch_inference(session, batch_size=16)
    run_batch_inference(session, batch_size=32)
    run_batch_inference(session, batch_size=64)
    
    # 6. 比较不同执行提供程序的性能（如果有多个提供程序）
    print("\n[6] 比较执行提供程序")
    provider_results = compare_execution_providers(model_path, test_data)
    
    # 7. 可视化一些预测结果
    print("\n[7] 可视化预测结果")
    # 收集多个样本的预测结果
    predictions = []
    for image, _ in test_data[:5]:  # 只使用前5个样本
        output, _ = run_inference(session, image, input_name)
        predictions.append(output)
    
    # 可视化结果
    visualize_predictions(test_data, predictions, num_samples=5)
    
    # 8. 解释会话选项
    print("\n[8] 会话选项说明")
    explain_session_options()
    
    # 9. 可视化批次性能对比
    print("\n[9] 可视化批次性能对比")
    visualize_batch_performance(session)
    
    print("\nONNX Runtime推理教程完成！")

if __name__ == "__main__":
    main()
    
    
"""
ONNX Runtime已成功安装，版本：1.21.0
==================================================
ONNX教程 - 第5部分：使用ONNX Runtime进行推理
==================================================

可用的执行提供程序：
  [0] AzureExecutionProvider
  [1] CPUExecutionProvider

注意：未检测到CUDA支持。模型将在CPU上运行。
如需GPU加速，请安装支持CUDA的ONNX Runtime版本。

[1] 创建ONNX Runtime推理会话

模型输入：
  [0] 名称: input
      形状: ['batch_size', 1, 28, 28]
      类型: tensor(float)

模型输出：
  [0] 名称: output
      形状: ['batch_size', 10]
      类型: tensor(float)

[2] 加载MNIST测试数据
成功加载10个MNIST测试样本

[3] 运行单个示例推理
样本实际标签: 7
模型预测: 7
推理时间: 1.00毫秒

[4] 模型性能评估

性能评估结果：
准确率: 100.00% (10/10)
平均推理时间: 0.12毫秒
最短推理时间: 0.00毫秒
最长推理时间: 0.20毫秒

[5] 批量推理性能测试

批量推理演示:
批量大小: 16
批量推理时间: 1.02毫秒
每样本平均时间: 0.06毫秒

批量推理演示:
批量大小: 32
批量推理时间: 1.00毫秒
每样本平均时间: 0.03毫秒

批量推理演示:
批量大小: 64
批量推理时间: 13.06毫秒
每样本平均时间: 0.20毫秒

[6] 比较执行提供程序

比较不同执行提供程序的性能：

使用 AzureExecutionProvider:
AzureExecutionProvider 运行失败: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid rank for input: input Got: 3 Expected: 4 Please fix either the inputs/outputs or the model.

使用 CPUExecutionProvider:
CPUExecutionProvider 运行失败: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid rank for input: input Got: 3 Expected: 4 Please fix either the inputs/outputs or the model.

[7] 可视化预测结果

预测可视化结果已保存到 ./results/mnist_predictions.png

[8] 会话选项说明

ONNX Runtime会话选项说明：
  - graph_optimization_level : 图优化级别，可设置为：DISABLE_ALL, ENABLE_BASIC, ENABLE_EXTENDED, ENABLE_ALL
  - intra_op_num_threads     : 算子内并行的线程数，设置为处理器核心数通常效果最佳
  - inter_op_num_threads     : 算子间并行的线程数
  - execution_mode           : 执行模式：ORT_SEQUENTIAL（顺序）或ORT_PARALLEL（并行）
  - enable_profiling         : 是否启用性能分析
  - enable_mem_pattern       : 是否启用内存模式优化
  - enable_cpu_mem_arena     : 是否启用CPU内存竞技场优化
  - session_log_severity_level: 日志级别：0(Verbose), 1(Info), 2(Warning), 3(Error), 4(Fatal)

[9] 可视化批次性能对比

测量不同批次大小的性能...
批次大小: 1, 总时间: 0.00毫秒, 每样本: 0.00毫秒
批次大小: 2, 总时间: 0.20毫秒, 每样本: 0.10毫秒
批次大小: 4, 总时间: 0.40毫秒, 每样本: 0.10毫秒
批次大小: 8, 总时间: 0.20毫秒, 每样本: 0.02毫秒
批次大小: 16, 总时间: 1.26毫秒, 每样本: 0.08毫秒
批次大小: 32, 总时间: 6.52毫秒, 每样本: 0.20毫秒
批次大小: 64, 总时间: 7.87毫秒, 每样本: 0.12毫秒
批次大小: 128, 总时间: 13.17毫秒, 每样本: 0.10毫秒

批次性能对比图已保存到 ./results/batch_performance_comparison.png

ONNX Runtime推理教程完成！
"""
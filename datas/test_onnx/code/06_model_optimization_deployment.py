#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第6部分：ONNX模型优化与部署
这个脚本展示如何对ONNX模型进行优化，并介绍各种部署选项
"""

import os
import time
import numpy as np
import onnx
from onnx import shape_inference

# 检查依赖库
try:
    import onnxruntime
    print(f"ONNX Runtime已成功安装，版本：{onnxruntime.__version__}")
except ImportError:
    print("错误：未安装ONNX Runtime。请使用 'pip install onnxruntime' 安装。")
    exit(1)

# 尝试导入onnxruntime-tools（用于优化）
try:
    from onnxruntime.tools.onnxruntime_tools import optimizer
    ort_tools_available = True
    print("ONNX Runtime Tools已成功导入")
except ImportError:
    ort_tools_available = False
    print("提示：未安装'onnxruntime-tools'，一些优化功能将不可用")

# 尝试导入onnx-simplifier
try:
    import onnxsim
    onnxsim_available = True
    print("ONNX Simplifier已成功导入")
except ImportError:
    onnxsim_available = False
    print("提示：未安装'onnx-simplifier'，模型简化功能将不可用。使用 'pip install onnx-simplifier' 安装")

def check_model_size(model_path):
    """检查ONNX模型文件大小"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"模型大小: {size_mb:.2f} MB ({size_bytes:,} 字节)")
    return size_bytes

def count_ops_params(model_path):
    """统计模型中的操作数和参数数量"""
    model = onnx.load(model_path)
    graph = model.graph
    
    # 计算操作数
    op_count = len(graph.node)
    op_types = {}
    for node in graph.node:
        if node.op_type in op_types:
            op_types[node.op_type] += 1
        else:
            op_types[node.op_type] = 1
    
    # 计算参数数量
    param_count = 0
    for initializer in graph.initializer:
        shape = initializer.dims
        count = 1
        for dim in shape:
            count *= dim
        param_count += count
    
    print(f"操作数: {op_count}")
    print(f"参数数量: {param_count:,}")
    print("操作类型分布:")
    for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {op_type}: {count}")
    
    return op_count, param_count, op_types

def optimize_with_onnxsim(model_path, output_path):
    """使用ONNX-Simplifier优化模型"""
    if not onnxsim_available:
        print("错误：ONNX Simplifier未安装")
        return None
    
    try:
        print("使用ONNX Simplifier优化模型...")
        model = onnx.load(model_path)
        model_opt, check = onnxsim.simplify(
            model,
            skip_fuse_bn=False,
            skip_reshape=False,
            skip_shape_inference=False
        )
        
        if not check:
            print("警告：简化后的模型可能与原始模型不等效")
        
        # 保存简化后的模型
        onnx.save(model_opt, output_path)
        print(f"简化后的模型已保存到: {output_path}")
        
        return output_path
    except Exception as e:
        print(f"使用ONNX Simplifier优化模型时出错：{str(e)}")
        return None

def optimize_with_ort(model_path, output_path, optimization_level=99):
    """使用ONNX Runtime优化模型（使用配置选项而非工具包）"""
    try:
        print("使用ONNX Runtime会话选项优化模型...")
        
        # 创建会话选项
        options = onnxruntime.SessionOptions()
        # 设置优化级别
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 启用内存优化
        options.enable_mem_pattern = True
        # 启用CPU内存arena
        options.enable_cpu_mem_arena = True
        # 设置执行模式为并行
        options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        # 可选：设置线程数
        options.intra_op_num_threads = 4
        
        # 创建会话（这会加载并优化模型）
        session = onnxruntime.InferenceSession(model_path, options)
        
        # 模型已经被加载并优化，但ONNX Runtime没有直接提供导出优化后模型的功能
        print("注意：ONNX Runtime已应用优化，但优化后的模型无法直接导出")
        print("      优化已应用于内存中的模型，未来推理会使用这些优化")
        
        return model_path  # 返回原始路径，因为我们没有导出优化模型
    except Exception as e:
        print(f"使用ONNX Runtime优化模型时出错：{str(e)}")
        return None

def convert_to_int8(model_path, output_path):
    """模拟将模型量化为INT8（演示用途）
    注意：真正的量化需要校准数据集和更复杂的过程
    """
    print("注意：此示例仅用于演示目的。在实际应用中，需要使用完整的量化流程和校准数据集。")
    
    try:
        # 我们只是在这里简单地描述量化步骤，而不是真正执行它
        print("\nINT8量化过程包括以下步骤：")
        print("1. 收集校准数据集（代表性样本）")
        print("2. 计算每个tensor的动态范围")
        print("3. 将浮点值缩放到INT8范围")
        print("4. 量化权重和激活值")
        
        # 返回原始模型路径
        print("\n这个演示不会真正执行量化。在实际应用中，请使用：")
        print("- ONNX Runtime的量化工具")
        print("- TensorRT的量化API")
        print("- OpenVINO的量化工具")
        
        return model_path
    except Exception as e:
        print(f"演示量化流程时出错：{str(e)}")
        return None

def benchmark_model(model_path, num_iterations=100):
    """对模型进行基准测试"""
    try:
        print(f"\n对模型 {os.path.basename(model_path)} 进行基准测试...")
        
        # 创建推理会话
        session = onnxruntime.InferenceSession(model_path)
        
        # 获取输入名称和形状
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # 创建随机输入数据
        # 注意：对于MNIST模型，输入形状应该是[batch, 1, 28, 28]
        batch_size = 1
        if 'batch_size' in input_shape or input_shape[0] == -1:
            # 处理动态批量大小
            if len(input_shape) == 4:  # 假设是NCHW格式
                input_data = np.random.randn(batch_size, input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
            else:
                # 回退到默认MNIST输入
                input_data = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
        else:
            # 使用固定形状
            input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # 预热运行
        print("预热推理...")
        for _ in range(10):
            session.run(None, {input_name: input_data})
        
        # 基准测试
        print(f"运行 {num_iterations} 次迭代的基准测试...")
        start_time = time.time()
        for _ in range(num_iterations):
            session.run(None, {input_name: input_data})
        end_time = time.time()
        
        # 计算性能指标
        total_time = end_time - start_time
        avg_time = total_time / num_iterations * 1000  # 毫秒
        fps = num_iterations / total_time
        
        print(f"总时间: {total_time:.2f} 秒")
        print(f"平均推理时间: {avg_time:.2f} 毫秒/张")
        print(f"吞吐量: {fps:.2f} FPS (每秒推理次数)")
        
        return avg_time, fps
    except Exception as e:
        print(f"运行基准测试时出错：{str(e)}")
        return None, None

def explain_deployment_options():
    """解释ONNX模型的不同部署选项"""
    print("\n" + "=" * 50)
    print("ONNX模型部署选项")
    print("=" * 50)
    
    # 服务器部署选项
    print("\n服务器部署:")
    server_options = [
        ("ONNX Runtime Server", "性能优秀，支持多种硬件加速，适合大多数服务器场景"),
        ("TensorRT", "英伟达GPU上性能最佳，适合高吞吐量要求场景"),
        ("OpenVINO", "英特尔硬件上性能最佳，CPU和VPU优化"),
        ("TorchServe", "提供模型服务，支持多种格式包括ONNX"),
        ("Triton Inference Server", "适合生产环境的模型服务器，支持多种格式")
    ]
    
    for option, desc in server_options:
        print(f"  - {option.ljust(25)}: {desc}")
    
    # 边缘设备部署选项
    print("\n边缘设备部署:")
    edge_options = [
        ("ONNX Runtime Mobile", "适合移动设备，占用空间小"),
        ("TensorFlow Lite", "可以转换ONNX模型，适合移动设备"),
        ("OpenVINO", "适合Intel边缘设备（NCS2等）"),
        ("TensorRT", "适合Jetson等英伟达边缘设备"),
        ("CoreML", "适合苹果设备，可从ONNX转换")
    ]
    
    for option, desc in edge_options:
        print(f"  - {option.ljust(25)}: {desc}")
    
    # 浏览器部署选项
    print("\n浏览器部署:")
    web_options = [
        ("ONNX.js", "直接在浏览器中运行ONNX模型"),
        ("TensorFlow.js", "可以转换ONNX模型，在浏览器中运行"),
        ("WebNN", "利用Web Neural Network API，硬件加速")
    ]
    
    for option, desc in web_options:
        print(f"  - {option.ljust(25)}: {desc}")

def explain_optimization_techniques():
    """解释ONNX模型优化技术"""
    print("\n" + "=" * 50)
    print("ONNX模型优化技术")
    print("=" * 50)
    
    techniques = [
        ("图优化", "合并操作、删除冗余节点、常量折叠等"),
        ("算子融合", "将多个小算子融合为一个更高效的算子"),
        ("量化", "将浮点运算降为低精度整数运算（FP32→INT8/FP16）"),
        ("剪枝", "移除对输出影响小的权重和连接"),
        ("知识蒸馏", "使用大模型训练小模型"),
        ("动态形状优化", "针对不同输入大小优化执行路径"),
        ("内存优化", "减少峰值内存使用，重用缓冲区")
    ]
    
    for technique, desc in techniques:
        print(f"  - {technique.ljust(25)}: {desc}")

def main():
    """主函数"""
    print("=" * 50)
    print("ONNX教程 - 第6部分：ONNX模型优化与部署")
    print("=" * 50)
    
    # 设置ONNX模型路径（使用之前导出的MNIST模型）
    model_path = './models/mnist_cnn.onnx'
    
    # 1. 检查原始模型大小和复杂度
    print("\n[1] 检查原始模型")
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行第3部分教程以导出ONNX模型。")
        return
    
    original_size = check_model_size(model_path)
    op_count, param_count, _ = count_ops_params(model_path)
    
    # 2. 使用ONNX Simplifier优化模型
    print("\n[2] 使用ONNX Simplifier优化模型")
    simplified_model_path = './models/mnist_cnn_simplified.onnx'
    if onnxsim_available:
        simplified_path = optimize_with_onnxsim(model_path, simplified_model_path)
        if simplified_path:
            simplified_size = check_model_size(simplified_path)
            print(f"模型大小减少: {(original_size - simplified_size) / original_size * 100:.2f}%")
            count_ops_params(simplified_path)
    else:
        print("跳过ONNX Simplifier优化，因为库未安装。")
        simplified_path = model_path
    
    # 3. 使用ONNX Runtime优化模型
    print("\n[3] 使用ONNX Runtime优化模型")
    ort_model_path = './models/mnist_cnn_ort_optimized.onnx'
    # 注意：这里我们实际上无法导出优化后的模型，但会应用内存中的优化
    optimize_with_ort(simplified_path, ort_model_path)
    
    # 4. 模拟INT8量化（仅演示）
    print("\n[4] 模拟INT8量化")
    quantized_model_path = './models/mnist_cnn_int8.onnx'
    convert_to_int8(simplified_path, quantized_model_path)
    
    # 5. 基准测试比较不同模型
    print("\n[5] 基准测试比较")
    print("\na) 原始模型性能：")
    orig_time, orig_fps = benchmark_model(model_path, num_iterations=50)
    
    if onnxsim_available and simplified_path != model_path:
        print("\nb) 简化后模型性能：")
        simp_time, simp_fps = benchmark_model(simplified_path, num_iterations=50)
        
        if orig_time and simp_time:
            speedup = orig_time / simp_time
            print(f"\n简化后的模型速度提升: {speedup:.2f}x")
    
    # 6. 解释模型优化技术
    print("\n[6] 模型优化技术")
    explain_optimization_techniques()
    
    # 7. 解释部署选项
    print("\n[7] 部署选项")
    explain_deployment_options()
    
    print("\nONNX模型优化与部署教程完成！")
    print("恭喜你完成了整个ONNX教程系列！👏")

if __name__ == "__main__":
    main()
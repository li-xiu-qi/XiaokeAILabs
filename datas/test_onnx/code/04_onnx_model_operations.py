#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX教程 - 第4部分：ONNX模型的基本操作
这个脚本展示如何读取、分析、可视化和修改ONNX模型
"""

import os
import sys
import numpy as np
import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto

def check_onnx_installation():
    """检查ONNX库是否正确安装"""
    try:
        import onnx
        print(f"ONNX库已成功安装，版本：{onnx.__version__}")
        
        # 检查ONNX Runtime
        try:
            import onnxruntime
            print(f"ONNX Runtime已成功安装，版本：{onnxruntime.__version__}")
        except ImportError:
            print("警告：未安装ONNX Runtime，将无法执行推理。可使用 'pip install onnxruntime' 安装。")
        
        # 检查可视化工具
        try:
            import netron
            print(f"Netron已成功安装，可用于可视化ONNX模型。")
        except ImportError:
            print("提示：未安装Netron，无法进行图形化可视化。可使用 'pip install netron' 安装。")
        
        return True
    except ImportError:
        print("错误：未安装ONNX库。请使用 'pip install onnx' 安装。")
        return False

def load_and_validate_model(model_path):
    """加载并验证ONNX模型"""
    try:
        # 加载ONNX模型
        print(f"正在加载ONNX模型：{model_path}")
        model = onnx.load(model_path)
        
        # 检查模型是否格式正确
        print("验证ONNX模型格式...")
        onnx.checker.check_model(model)
        print("✓ 模型格式验证通过！")
        
        # 运行形状推断，确保所有中间张量的形状都已知
        print("正在进行形状推断...")
        inferred_model = shape_inference.infer_shapes(model)
        print("✓ 形状推断完成！")
        
        return model
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        return None
    except Exception as e:
        print(f"错误：加载或验证模型时出错：{str(e)}")
        return None

def explore_model_metadata(model):
    """探索ONNX模型的元数据"""
    print("\n" + "=" * 50)
    print("ONNX模型元数据")
    print("=" * 50)
    
    # 显示IR版本
    print(f"IR版本：{model.ir_version}")
    print(f"Opset版本：{model.opset_import[0].version}")
    
    # 显示生产者信息
    print(f"生产者名称：{model.producer_name}")
    print(f"生产者版本：{model.producer_version}")
    
    # 显示模型版本
    print(f"模型版本：{model.model_version}")
    
    # 显示文档字符串
    if model.doc_string:
        print(f"\n模型文档：\n{model.doc_string}")
    
    # 检查并显示自定义元数据
    if len(model.metadata_props) > 0:
        print("\n自定义元数据：")
        for prop in model.metadata_props:
            print(f"  - {prop.key}: {prop.value}")

def analyze_model_graph(model):
    """分析ONNX模型的计算图结构"""
    graph = model.graph
    
    print("\n" + "=" * 50)
    print("ONNX模型图结构分析")
    print("=" * 50)
    
    # 分析输入
    print("\n输入节点：")
    for i, input_node in enumerate(graph.input):
        print(f"  [{i}] 名称：{input_node.name}")
        # 获取输入形状
        shape_info = []
        if hasattr(input_node.type.tensor_type, 'shape'):
            for dim in input_node.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape_info.append(dim.dim_param)
                else:
                    shape_info.append(dim.dim_value)
        print(f"      形状：{shape_info}")
        print(f"      数据类型：{TensorProto.DataType.Name(input_node.type.tensor_type.elem_type)}")
    
    # 分析输出
    print("\n输出节点：")
    for i, output_node in enumerate(graph.output):
        print(f"  [{i}] 名称：{output_node.name}")
        # 获取输出形状
        shape_info = []
        if hasattr(output_node.type.tensor_type, 'shape'):
            for dim in output_node.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape_info.append(dim.dim_param)
                else:
                    shape_info.append(dim.dim_value)
        print(f"      形状：{shape_info}")
        print(f"      数据类型：{TensorProto.DataType.Name(output_node.type.tensor_type.elem_type)}")
    
    # 分析节点
    print(f"\n计算节点总数：{len(graph.node)}")
    op_type_counts = {}
    for node in graph.node:
        if node.op_type in op_type_counts:
            op_type_counts[node.op_type] += 1
        else:
            op_type_counts[node.op_type] = 1
    
    print("\n操作类型统计：")
    for op_type, count in sorted(op_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {op_type}: {count}")
    
    # 分析权重（初始化器）
    print(f"\n权重（初始化器）总数：{len(graph.initializer)}")
    total_params = 0
    for initializer in graph.initializer:
        size = 1
        for dim in initializer.dims:
            size *= dim
        total_params += size
    
    print(f"总参数数量：{total_params:,}")
    
    # 内存占用估计（粗略计算）
    memory_bytes = total_params * 4  # 假设使用float32（4字节）
    print(f"估计内存占用：{memory_bytes / (1024*1024):.2f} MB")

def visualize_model(model_path):
    """使用Netron可视化ONNX模型"""
    try:
        import netron
        print("\n" + "=" * 50)
        print("ONNX模型可视化")
        print("=" * 50)
        print("\n正在启动Netron服务器来可视化模型...")
        print("请在网页浏览器中查看模型（通常会自动打开）")
        print("完成查看后，请在此控制台按Ctrl+C终止服务器")
        
        # 启动Netron服务器
        netron.start(model_path)
    except ImportError:
        print("\n无法可视化模型：未安装Netron。")
        print("请使用 'pip install netron' 安装Netron后再尝试。")
    except Exception as e:
        print(f"\n可视化模型时出错：{str(e)}")

def extract_node_by_name(model, node_name):
    """通过名称查找并提取特定节点的信息"""
    graph = model.graph
    found = False
    
    # 在所有节点中查找
    for node in graph.node:
        if node.name == node_name or node_name in node.output:
            found = True
            print(f"\n找到节点：{node.name}")
            print(f"操作类型：{node.op_type}")
            print(f"输入：{node.input}")
            print(f"输出：{node.output}")
            
            # 显示属性
            if len(node.attribute) > 0:
                print("属性：")
                for attr in node.attribute:
                    print(f"  - {attr.name}: {get_attribute_value(attr)}")
    
    if not found:
        print(f"\n未找到名称为 '{node_name}' 的节点")

def get_attribute_value(attr):
    """获取节点属性值的辅助函数"""
    if attr.type == AttributeProto.FLOAT:
        return attr.f
    elif attr.type == AttributeProto.INT:
        return attr.i
    elif attr.type == AttributeProto.STRING:
        return attr.s
    elif attr.type == AttributeProto.TENSOR:
        return "<tensor>"
    elif attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == AttributeProto.STRINGS:
        return list(attr.strings)
    else:
        return f"<未知类型：{attr.type}>"

def modify_model_metadata(model, output_path):
    """修改ONNX模型的元数据"""
    # 添加描述
    model.doc_string = "这是在ONNX教程中修改过的MNIST模型"
    
    # 添加自定义元数据属性
    metadata_props = {"修改时间": "2025-04-05", "修改者": "XiaokeAILabs", "用途": "教学演示"}
    for key, value in metadata_props.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value
    
    # 保存修改后的模型
    onnx.save(model, output_path)
    print(f"\n已修改模型元数据并保存到：{output_path}")
    
    return output_path

def convert_model_to_text(model_path, output_path=None):
    """将ONNX模型转换为可读文本格式"""
    model = onnx.load(model_path)
    
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + ".txt"
    
    with open(output_path, 'w') as f:
        f.write(str(model))
    
    print(f"\n已将模型转换为文本格式并保存到：{output_path}")
    print(f"文本文件大小：{os.path.getsize(output_path) / 1024:.2f} KB")
    
    return output_path

def main():
    """主函数"""
    print("=" * 50)
    print("ONNX教程 - 第4部分：ONNX模型的基本操作")
    print("=" * 50)
    
    # 检查ONNX安装
    if not check_onnx_installation():
        return
    
    # 设置ONNX模型路径
    model_path = './models/mnist_cnn.onnx'
    
    # 创建models目录（如果不存在）
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 加载与验证模型
    print("\n[1] 加载与验证模型")
    model = load_and_validate_model(model_path)
    if model is None:
        print(f"请先运行第3部分教程以导出ONNX模型。")
        return
    
    # 探索模型元数据
    print("\n[2] 探索模型元数据")
    explore_model_metadata(model)
    
    # 分析模型图结构
    print("\n[3] 分析模型图结构")
    analyze_model_graph(model)
    
    # 查找特定节点 (一个卷积节点示例)
    print("\n[4] 提取特定节点信息")
    # 注意：由于我们不确定具体节点名称，这里使用一个输出节点的名称
    extract_node_by_name(model, "output")
    
    # 修改模型元数据
    print("\n[5] 修改模型元数据")
    modified_model_path = './models/mnist_cnn_modified.onnx'
    modify_model_metadata(model, modified_model_path)
    
    # 转换为文本格式
    print("\n[6] 转换为文本格式")
    text_file_path = convert_model_to_text(modified_model_path)
    
    # 可视化模型（放在最后，因为这是一个阻塞操作）
    print("\n[7] 可视化模型")
    print("提示：此步骤将启动Web服务器。如果要跳过，请按Ctrl+C")
    print("准备启动模型可视化...")
    
    try:
        input("按Enter键继续...")
        visualize_model(modified_model_path)
    except KeyboardInterrupt:
        print("\n可视化步骤已跳过。")
    
    print("\nONNX模型基本操作教程完成！")

if __name__ == "__main__":
    main()
    
    
"""
==================================================
ONNX教程 - 第4部分：ONNX模型的基本操作
==================================================
ONNX库已成功安装，版本：1.17.0
ONNX Runtime已成功安装，版本：1.21.0
Netron已成功安装，可用于可视化ONNX模型。

[1] 加载与验证模型
正在加载ONNX模型：./models/mnist_cnn.onnx
验证ONNX模型格式...
✓ 模型格式验证通过！
正在进行形状推断...
✓ 形状推断完成！

[2] 探索模型元数据

==================================================
ONNX模型元数据
==================================================
IR版本：7
Opset版本：12
生产者名称：pytorch
生产者版本：2.6.0
模型版本：0

[3] 分析模型图结构

==================================================
ONNX模型图结构分析
==================================================

输入节点：
  [0] 名称：input
      形状：['batch_size', 1, 28, 28]
      数据类型：FLOAT

输出节点：
  [0] 名称：output
      形状：['batch_size', 10]
      数据类型：FLOAT

计算节点总数：12

操作类型统计：
  - Relu: 3
  - Conv: 2
  - MaxPool: 2
  - Gemm: 2
  - Constant: 1
  - Reshape: 1
  - LogSoftmax: 1

权重（初始化器）总数：8
总参数数量：421,642
估计内存占用：1.61 MB

[4] 提取特定节点信息

找到节点：/LogSoftmax
操作类型：LogSoftmax
输入：['/fc2/Gemm_output_0']
输出：['output']
属性：
  - axis: 1

[5] 修改模型元数据

已修改模型元数据并保存到：./models/mnist_cnn_modified.onnx

[6] 转换为文本格式

已将模型转换为文本格式并保存到：./models/mnist_cnn_modified.txt
文本文件大小：4670.54 KB

[7] 可视化模型
提示：此步骤将启动Web服务器。如果要跳过，请按Ctrl+C
准备启动模型可视化...
按Enter键继续...

==================================================
ONNX模型可视化
==================================================

正在启动Netron服务器来可视化模型...
请在网页浏览器中查看模型（通常会自动打开）
完成查看后，请在此控制台按Ctrl+C终止服务器
Serving './models/mnist_cnn_modified.onnx' at http://localhost:8080

ONNX模型基本操作教程完成！
"""
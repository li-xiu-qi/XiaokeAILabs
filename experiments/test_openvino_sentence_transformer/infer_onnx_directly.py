import openvino # 直接导入 openvino 替代 openvino.runtime
from pathlib import Path
import numpy as np # 用于创建示例输入数据
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import requests # 用于从 URL 加载图片示例
from scipy.spatial.distance import cosine # 用于计算余弦相似度

# --- 配置 ---
# 输入：你的 ONNX 模型文件路径
# 从上次的 dir 命令结果中，我们选择 model.onnx 作为示例
# 你可以更改为列表中的其他 .onnx 文件，例如 model_fp16.onnx
onnx_model_path = r"C:\\Users\\k\\Desktop\\BaiduSyncdisk\\baidu_sync_documents\\hf_models\\jina-clip-v2\\onnx\\model.onnx"

# 推理设备，例如 "CPU", "GPU", "NPU"
# 如果你有 Intel NPU 并且驱动和 OpenVINO 环境配置正确，可以尝试 "NPU"
device_name = "CPU"

# 预处理配置路径
tokenizer_directory_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\jina-clip-v2"
processor_directory = r"./preprocessor"

# 示例图片URL和文本
sample_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
sample_texts = [
    "小猫在草地上面运动", 
    "小狗在广场上跑步", 
    "猫咪正在绿色的草坪上玩耍",
    "一只狗狗在公园里奔跑",
    "学生们在教室里认真学习",
    "孩子们在操场上打篮球"
]

# 序列最大长度
max_sequence_length = 77

def infer_onnx_model():
    """
    直接加载 ONNX 模型并使用 OpenVINO Runtime 进行推理。
    """
    if not Path(onnx_model_path).exists():
        print(f"错误：输入的 ONNX 模型文件 '{onnx_model_path}' 不存在。请检查路径。")
        return    print(f"正在初始化 OpenVINO Runtime Core...")
    core = openvino.Core()

    print(f"可用的设备: {core.available_devices}")

    try:
        print(f"正在从 '{onnx_model_path}' 读取 ONNX 模型...")
        # 直接读取 .onnx 模型
        model = core.read_model(model=onnx_model_path)
        print("模型读取成功。")

        # 打印模型的输入和输出信息，这对于准备输入和解析输出至关重要
        print("\\n--- 模型输入信息 ---")
        for input_tensor in model.inputs:
            print(f"  输入名称: {input_tensor.any_name}")
            print(f"  形状: {input_tensor.partial_shape}")
            print(f"  数据类型: {input_tensor.element_type}")

        print("\\n--- 模型输出信息 ---")
        for output_tensor in model.outputs:
            print(f"  输出名称: {output_tensor.any_name}")
            print(f"  形状: {output_tensor.partial_shape}")
            print(f"  数据类型: {output_tensor.element_type}")

        print(f"\\n正在将模型编译到设备 '{device_name}'...")
        compiled_model = core.compile_model(model=model, device_name=device_name)
        print("模型编译成功。")

        # 创建推理请求
        infer_request = compiled_model.create_infer_request()
        print("推理请求创建成功。")        # --- 准备输入数据 ---
        # 基于 notebook (explore_onnx_model.ipynb) 的发现，我们知道模型需要以下输入:
        # 1. "input_ids": 形状 [batch_size, sequence_length], 类型 int64
        #    示例中 batch_size=1, sequence_length=77 (CLIP 文本模型的常见值)
        # 2. "pixel_values": 形状 [batch_size, num_channels, height, width], 类型 float32
        #    示例中 batch_size=1, num_channels=3, height=512, width=512

        print("\n--- 准备实际输入数据 ---")

        # 实现文本预处理逻辑
        try:
            print("正在加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory_path)
            print(f"成功从 '{tokenizer_directory_path}' 加载分词器: {type(tokenizer)}")
            
            print("正在处理文本...")
            inputs_text = tokenizer(
                sample_texts,
                padding='max_length',
                truncation=True,
                max_length=max_sequence_length,
                return_tensors='np',
                add_special_tokens=True
            )
            
            actual_input_ids = inputs_text['input_ids'].astype(np.int64)
            print(f"为 'input_ids' 准备了形状为 {actual_input_ids.shape}，类型为 {actual_input_ids.dtype} 的实际数据。")
        except Exception as e:
            print(f"文本预处理过程中发生错误：{e}")
            print("继续使用占位符数据...")
            batch_size = 1
            actual_input_ids = np.zeros((batch_size, max_sequence_length), dtype=np.int64)
            print(f"使用占位符 'actual_input_ids'，形状: {actual_input_ids.shape}")
        
        # 实现图像预处理逻辑
        try:
            print("正在加载图像预处理器...")
            image_processor = AutoImageProcessor.from_pretrained(processor_directory, trust_remote_code=True)
            print("图像预处理器加载成功！")
            
            print("正在加载图片...")
            image = Image.open(requests.get(sample_image_url, stream=True).raw)
            if image.mode != "RGB":
                image = image.convert("RGB")
            print(f"原始图片尺寸: {image.size}, 模式: {image.mode}")
            
            print("正在预处理图片...")
            inputs_image = image_processor(images=image, return_tensors="np")
            actual_pixel_values = inputs_image.pixel_values
            print(f"为 'pixel_values' 准备了形状为 {actual_pixel_values.shape}，类型为 {actual_pixel_values.dtype} 的实际数据。")
        except Exception as e:
            print(f"图像预处理过程中发生错误：{e}")
            print("继续使用占位符数据...")
            batch_size = 1
            num_channels = 3
            height = 512
            width = 512
            actual_pixel_values = np.zeros((batch_size, num_channels, height, width), dtype=np.float32)
            print(f"使用占位符 'actual_pixel_values'，形状: {actual_pixel_values.shape}")

        # 确保输入字典使用新的变量名
        inputs = {
            "input_ids": actual_input_ids,
            "pixel_values": actual_pixel_values
        }

        # 验证模型是否确实有这些输入名称
        model_input_names = {inp.any_name for inp in model.inputs}
        if "input_ids" not in model_input_names:
            print("警告: 模型输入中未找到 'input_ids'。请检查模型结构。")
        if "pixel_values" not in model_input_names:
            print("警告: 模型输入中未找到 'pixel_values'。请检查模型结构。")

        print(f"\\n正在使用准备好的输入进行推理...")
        results = infer_request.infer(inputs=inputs)
        print("推理完成。")

        # --- 处理输出 ---
        # 基于 explore_onnx_model.ipynb 的发现处理输出
        print("\\n--- 推理结果 ---")
        
        # 从 notebook 我们知道有以下输出:
        # 1. text_embeddings: [?,1024], float32
        # 2. image_embeddings: [?,1024], float32
        # 3. l2norm_text_embeddings: [?,1024], float32
        # 4. l2norm_image_embeddings: [?,1024], float32

        # compiled_model.outputs 中的顺序可能与 ONNX 模型的原始顺序一致
        # results 字典的键是输出张量的名称

        # 首先，我们打印 compiled_model.outputs 以了解 OpenVINO 如何命名它们
        # print("\\n--- Compiled Model Output Node Info ---")
        # for i, output_node in enumerate(compiled_model.outputs):
        #     print(f"  Output Node #{i}:")
        #     print(f"    Any Name (original): {output_node.get_any_name()}") 
        #     print(f"    Tensor Names (used in results dict): {output_node.get_names()}")
            # 通常 results 字典的键是 output_node.get_tensor().name 或 output_node.get_any_name()
            # 如果 get_names() 返回多个，其中一个应该是 results 字典的键

        # 假设 ONNX 模型中定义的输出名称是 OpenVINO IR 中以及 results 字典中使用的键
        # 如果不是，你可能需要检查 compiled_model.outputs[i].get_tensor().name
        expected_output_names = ["text_embeddings", "image_embeddings", "l2norm_text_embeddings", "l2norm_image_embeddings"]

        for name in expected_output_names:
            if name in results:
                result_data = results[name]
                print(f"  输出名称: {name}")
                print(f"    形状: {result_data.shape}")
                print(f"    数据类型: {result_data.dtype}")
                print(f"    部分数据: {result_data.flatten()[:10]}...")
            else:
                # 尝试查找 compiled_model.outputs 中是否有匹配的 any_name
                found = False
                for output_node in compiled_model.outputs:
                    # results 字典的键通常是 output_node.get_tensor().name
                    # 我们也检查 output_node.get_any_name()
                    # 以及 output_node.get_names() 中的任何一个是否匹配
                    
                    # 获取此输出节点在 results 字典中实际使用的键
                    # compiled_model.output(i) or compiled_model.output("tensor_name")
                    # The name used in `results` dictionary is usually one of the names in `output_node.get_names()`
                    # or `output_node.get_tensor().get_name()` if it's unique.
                    
                    # Let's iterate through the results dictionary and see if we can match it back to an expected name
                    # This is a bit more robust if the exact key name in 'results' is slightly different
                    # but corresponds to one of the known output concepts.
                    
                    # A simpler way: the 'results' dictionary keys are usually the tensor names.
                    # We can iterate through compiled_model.outputs and get their tensor names.
                    actual_tensor_name_in_results = ""
                    for key_in_results in results.keys():
                        # Heuristic: if an expected name is a substring of a key in results, assume it's a match.
                        # Or if a key in results is a substring of an expected name.
                        # This is not perfect but can help if names are slightly mangled.
                        # A more robust way is to use compiled_model.output(index).get_any_name()
                        # and then find that in the results dictionary.
                        # The `results` dictionary keys are the output tensor names as defined in the graph after compilation.
                        
                        # Let's try to find the result data by iterating through compiled_model.outputs
                        # and using its properties to look up in the `results` dictionary.
                        if output_node.get_any_name() == name: # Check original name
                            # The key in results might be different, often one of output_node.get_names()
                            for potential_key in output_node.get_names():
                                if potential_key in results:
                                    actual_tensor_name_in_results = potential_key
                                    break
                            if not actual_tensor_name_in_results and hasattr(output_node.get_tensor(), 'name'): # Fallback to tensor name
                                if output_node.get_tensor().name in results:
                                     actual_tensor_name_in_results = output_node.get_tensor().name

                            if actual_tensor_name_in_results:
                                result_data = results[actual_tensor_name_in_results]
                                print(f"  输出名称 (原始名: {name}, 实际键: {actual_tensor_name_in_results}):")
                                print(f"    形状: {result_data.shape}")
                                print(f"    数据类型: {result_data.dtype}")
                                print(f"    部分数据: {result_data.flatten()[:10]}...")
                                found = True
                                break # Found this expected output
                    if found:
                        continue # Move to next expected_output_name
                
                if not found:
                     print(f"警告: 在推理结果中未直接找到期望的输出 '{name}'。")
                     print(f"  可用的输出键: {list(results.keys())}")
        
        # Fallback: Print all outputs found in the results dictionary if specific ones weren't matched
        if not all(name in results for name in expected_output_names):
            print("\\n  --- 所有实际输出 (回退) ---")
            for res_name, res_data_val in results.items():
                is_expected = res_name in expected_output_names
                print(f"  输出名称 (实际键): {res_name} {'(已作为期望输出打印)' if is_expected else ''}")
                if not is_expected: # Only print if not already printed as an expected output
                    print(f"    形状: {res_data_val.shape}")
                    print(f"    数据类型: {res_data_val.dtype}")
                    print(f"    部分数据: {res_data_val.flatten()[:10]}...")

        # Old generic loop (can be removed or kept as a final fallback)
        # for output_tensor in model.outputs:
        #     output_name = output_tensor.any_name
        #     if output_name in results:
        #         result_data = results[output_name]
        #         print(f"  输出名称: {output_name}")
        #         print(f"  形状: {result_data.shape}")
        #         print(f"  部分数据: {result_data.flatten()[:10]}...") # 打印前10个扁平化数据
        #     else:
        #         # 有时 compiled_model.outputs 和 infer_request.results 的键可能略有不同
        #         # 我们可以尝试通过索引访问
        #         try:
        #             # compiled_model.outputs 是一个 OutputVector, results 是一个字典
        #             # 我们需要找到匹配的键或按顺序获取
        #             # 这是一个更健壮的方式来获取所有输出
        #             for res_name, res_data_val in results.items():
        #                  if output_tensor.get_any_name() in res_name or \
        #                     any(out_name in res_name for out_name in output_tensor.get_names()):
        #                     print(f"  输出名称 (匹配): {res_name}")
        #                     print(f"  形状: {res_data_val.shape}")
        #                     print(f"  部分数据: {res_data_val.flatten()[:10]}...")
        #                     break
        #             else:
        #                  print(f"警告: 在推理结果中未直接找到输出 '{output_name}'。")
        #         except Exception as e_out:
        #             print(f"警告: 处理输出 '{output_name}' 时出错: {e_out}")

        # 打印所有实际获得的输出的名称，以供调试
        print("\\n--- 实际获得的输出键 ---")
        if results:
            for res_key in results.keys():
                print(f"  {res_key}")
        else:
            print("  没有获得任何输出结果。")
            
        # 计算并展示文本嵌入之间的相似度
        print("\n--- 文本相似度分析 ---")
        try:
            # 尝试获取文本嵌入向量，优先使用标准化后的嵌入（效果更好）
            text_embeddings = None
            if "l2norm_text_embeddings" in results:
                text_embeddings = results["l2norm_text_embeddings"]
                print("使用 L2 标准化后的文本嵌入向量计算相似度")
            elif "text_embeddings" in results:
                text_embeddings = results["text_embeddings"]
                print("使用原始文本嵌入向量计算相似度")
            
            if text_embeddings is not None:
                # 计算余弦相似度矩阵
                num_texts = len(sample_texts)
                similarity_matrix = np.zeros((num_texts, num_texts))
                
                for i in range(num_texts):
                    for j in range(num_texts):
                        # 余弦相似度 = 1 - 余弦距离
                        similarity = 1 - cosine(text_embeddings[i], text_embeddings[j])
                        similarity_matrix[i, j] = similarity
                
                # 打印相似度矩阵
                print("\n相似度矩阵:")
                print("    " + "    ".join([f"句子{i+1}" for i in range(num_texts)]))
                
                for i in range(num_texts):
                    row = [f"句子{i+1}"]
                    for j in range(num_texts):
                        # 格式化相似度值，保留4位小数
                        row.append(f"{similarity_matrix[i, j]:.4f}")
                    print("  ".join(row))
                
                # 打印句子对应关系
                print("\n句子对应关系:")
                for i in range(num_texts):
                    print(f"句子{i+1}: {sample_texts[i]}")
                
                # 找出每个句子最相似的其他句子（排除自身）
                print("\n每个句子的最相似句子:")
                for i in range(num_texts):
                    # 创建相似度数组的副本，并将自身与自身的相似度设为-1（排除）
                    sim_scores = similarity_matrix[i].copy()
                    sim_scores[i] = -1
                    
                    # 找出最相似的句子索引
                    most_similar_idx = np.argmax(sim_scores)
                    similarity = similarity_matrix[i, most_similar_idx]
                    
                    print(f"'{sample_texts[i]}' 最相似的句子是: '{sample_texts[most_similar_idx]}' (相似度: {similarity:.4f})")
            else:
                print("未找到文本嵌入向量，无法计算相似度。")
        
        except Exception as e:
            print(f"计算相似度时发生错误: {e}")
            print("请确保模型输出中包含文本嵌入向量。")

    except Exception as e:
        print(f"\\n推理过程中发生错误: {e}")
        print("请检查以下几点：")
        print(f"1. ONNX 模型路径 '{onnx_model_path}' 是否正确。")
        print(f"2. OpenVINO Runtime 是否正确安装并配置在你的环境中。")
        print(f"3. 设备 '{device_name}' 是否可用且受支持。")
        print(f"4. ONNX 模型是否与当前 OpenVINO 版本兼容。")
        print(f"5. 输入数据准备部分是否与模型期望的输入完全一致（名称、形状、数据类型）。")

if __name__ == "__main__":
    infer_onnx_model()

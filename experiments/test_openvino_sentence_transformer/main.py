import os
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 尝试导入 OpenVINO 和 Transformers
try:
    from openvino.runtime import Core
    from transformers import AutoTokenizer
except ImportError:
    print("错误：运行此脚本需要 openvino-dev 和 transformers 包。")
    print("请确保已在你的环境中安装它们：")
    print("pip install openvino-dev transformers scikit-learn")
    exit()

# --- 配置 ---
# IR 模型所在的目录 (由转换脚本创建)
MODEL_DIR = Path(r"c:\Users\k\Documents\project\programming_project\python_project\unimportance\test_openvino_sentence_transformer\jina-clip-v2-openvino-ir")
# 指定推理设备 ("NPU", "CPU", "GPU" 等)
DEVICE = "NPU"

# --- 全局变量，用于缓存模型和 tokenizer ---
core = None
compiled_model = None
tokenizer = None
input_layer_name_ids = None
input_layer_name_mask = None

def initialize_model_and_tokenizer():
    """
    加载 OpenVINO IR 模型和相关的 Tokenizer。
    """
    global core, compiled_model, tokenizer, input_layer_name_ids, input_layer_name_mask

    if compiled_model is not None and tokenizer is not None:
        print("模型和 Tokenizer 已加载。")
        return True

    if not MODEL_DIR.exists():
        print(f"错误：模型目录 '{MODEL_DIR}' 不存在。请先运行转换脚本。")
        return False

    model_xml_path = MODEL_DIR / "openvino_model.xml" # optimum 通常保存为 openvino_model.xml
    if not model_xml_path.exists():
        model_xml_path = MODEL_DIR / "model.xml" # 备用名称
        if not model_xml_path.exists():
            print(f"错误：在 '{MODEL_DIR}' 中未找到 openvino_model.xml 或 model.xml。")
            return False

    print("正在初始化 OpenVINO Core...")
    core = Core()
    
    # 检查 NPU 是否可用
    available_devices = core.available_devices
    print(f"可用设备: {available_devices}")
    if DEVICE not in available_devices and DEVICE != "CPU": # CPU 总是可用的后备
        print(f"警告: 设备 '{DEVICE}' 未在可用设备列表中找到。将尝试使用 CPU。")
        # current_device = "CPU"
    # else:
    #     current_device = DEVICE
    # 即使NPU不在列表中，也尝试编译到NPU，如果失败OpenVINO会报错

    print(f"正在从 '{model_xml_path}' 加载模型...")
    model = core.read_model(model=model_xml_path)

    # 识别输入层的名称 (通常是 'input_ids' 和 'attention_mask')
    # 对于 CLIP text encoder, 通常是这两个输入
    # 我们假设模型有两个输入，按顺序是 input_ids 和 attention_mask
    # 或者我们可以从模型中获取它们的名称
    inputs_info = model.inputs
    if len(inputs_info) < 2:
        print(f"错误：模型期望至少2个输入，但只找到 {len(inputs_info)} 个。")
        # 尝试打印找到的输入以帮助调试
        for inp in inputs_info:
            print(f"  找到的输入: 名称: {inp.any_name}, 类型: {inp.element_type}, 形状: {inp.partial_shape}")
        return False

    # 尝试基于常见的名称或顺序来确定
    # Jina CLIP text encoder 通常有 'input_ids' 和 'attention_mask'
    found_ids = False
    found_mask = False
    for inp in inputs_info:
        if "input_ids" in inp.any_name:
            input_layer_name_ids = inp.any_name
            found_ids = True
        elif "attention_mask" in inp.any_name:
            input_layer_name_mask = inp.any_name
            found_mask = True
    
    if not (found_ids and found_mask):
        print("警告：未能通过名称自动识别 'input_ids' 和 'attention_mask'。")
        print("将假设模型的前两个输入分别是 input_ids 和 attention_mask。")
        # 如果名称不匹配，这可能导致运行时错误或不正确的结果。
        # 确保转换后的模型输入名称是已知的。
        # 对于 optimum 导出的模型，输入名称通常是原始模型的名称。
        input_layer_name_ids = model.input(0).any_name # 第一个输入
        input_layer_name_mask = model.input(1).any_name # 第二个输入
        print(f"  假设 input_ids 为: {input_layer_name_ids}")
        print(f"  假设 attention_mask 为: {input_layer_name_mask}")


    print(f"正在将模型编译到设备: {DEVICE}...")
    try:
        compiled_model = core.compile_model(model=model, device_name=DEVICE)
    except Exception as e:
        print(f"错误：编译模型到设备 '{DEVICE}' 失败: {e}")
        if DEVICE == "NPU":
            print("请确保 NPU 驱动和 OpenVINO NPU 插件已正确安装和配置。")
            print("如果 NPU 不可用，可以尝试将 DEVICE 设置为 'CPU'。")
        return False

    print(f"正在从 '{MODEL_DIR}' 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception as e:
        print(f"错误：加载 Tokenizer 失败: {e}")
        return False
    
    print("模型和 Tokenizer 初始化成功。")
    return True

def get_sentence_embeddings(sentences: list[str]) -> np.ndarray | None:
    """
    使用加载的 OpenVINO 模型获取句子的嵌入。

    Args:
        sentences: 一个包含待处理句子的列表。

    Returns:
        一个 NumPy 数组，其中每行是对应句子的嵌入向量。
        如果出错则返回 None。
    """
    global compiled_model, tokenizer, input_layer_name_ids, input_layer_name_mask

    if compiled_model is None or tokenizer is None:
        print("错误：模型或 Tokenizer 未初始化。请先调用 initialize_model_and_tokenizer()。")
        if not initialize_model_and_tokenizer(): # 尝试初始化
             return None


    if not sentences:
        return np.array([])

    # 1. 预处理 (Tokenization)
    # Jina-CLIP 的 tokenizer 通常需要 padding 和 truncation
    # max_length 可以从 tokenizer.model_max_length 获取，或者设一个合理的值
    max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 77 # CLIP 通常是 77
    
    print(f"正在对输入文本进行 Tokenize (max_length={max_length})...")
    inputs = tokenizer(
        sentences,
        padding='max_length', # 或者 True，让 tokenizer 自动处理
        truncation=True,
        max_length=max_length,
        return_tensors="np" # 返回 NumPy 数组
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # 对于某些 CLIP 实现，可能不需要 token_type_ids，如果模型需要，需要添加
    # token_type_ids = inputs.get('token_type_ids') 

    # 2. 执行推理
    print("正在执行推理...")
    infer_request = compiled_model.create_infer_request()
    
    # 准备输入字典
    # 确保这里的键名与模型 .xml 文件中定义的输入名称一致
    input_feed = {
        input_layer_name_ids: input_ids,
        input_layer_name_mask: attention_mask
    }
    # if token_type_ids is not None and compiled_model.input("token_type_ids"): # 如果模型有这个输入
    #    input_feed[compiled_model.input("token_type_ids").any_name] = token_type_ids

    results = infer_request.infer(inputs=input_feed)
    
    # 3. 获取输出
    # OVModelForFeatureExtraction 的输出通常是 pooled_output 或 last_hidden_state
    # 对于 CLIP text encoder, 输出通常是句子的嵌入向量
    # 假设模型只有一个输出，即句子嵌入
    output_node = compiled_model.output(0) # 获取第一个（通常是唯一的）输出节点
    embeddings = results[output_node]
    
    # (可选) L2 归一化，CLIP 嵌入通常会进行归一化以用于余弦相似度
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings

def calculate_similarity_demo():
    """
    演示如何计算两个句子之间的余弦相似度。
    """
    if not initialize_model_and_tokenizer():
        print("无法执行相似度计算演示，因为模型初始化失败。")
        return

    sentence1 = "这里是测试使用文本"
    sentence2 = "这里还是测试文本"
    
    print(f"\n计算句子相似度示例:")
    print(f"句子1: \"{sentence1}\"")
    print(f"句子2: \"{sentence2}\"")

    embeddings = get_sentence_embeddings([sentence1, sentence2])

    if embeddings is not None and len(embeddings) == 2:
        embedding1 = embeddings[0:1] # 保持二维形状 (1, embedding_dim)
        embedding2 = embeddings[1:2] # 保持二维形状 (1, embedding_dim)
        
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
        print(f"\n句子之间的余弦相似度: {similarity_score:.4f}")
    else:
        print("\n获取句子嵌入失败，无法计算相似度。")

if __name__ == "__main__":
    print("开始 OpenVINO 推理和相似度计算演示...")
    
    # 运行演示
    calculate_similarity_demo()

    # 你也可以在这里测试其他句子
    # print("\n--- 测试其他句子 ---")
    # if initialize_model_and_tokenizer(): # 确保已初始化
    #     test_sentences = [
    #         "A photo of a cat.",
    #         "A drawing of a dog.",
    #         "The quick brown fox jumps over the lazy dog."
    #     ]
    #     test_embeddings = get_sentence_embeddings(test_sentences)
    #     if test_embeddings is not None:
    #         for i, sentence in enumerate(test_sentences):
    #             print(f"\n句子: {sentence}")
    #             print(f"嵌入 (前5个维度): {test_embeddings[i, :5]}")
    #             print(f"嵌入形状: {test_embeddings[i].shape}")
    # else:
    #     print("模型未初始化，跳过额外测试。")
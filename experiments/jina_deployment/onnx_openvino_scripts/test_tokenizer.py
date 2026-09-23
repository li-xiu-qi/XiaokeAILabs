from transformers import AutoTokenizer

# 直接使用您提供的目录路径
tokenizer_directory_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\jina-clip-v2"

try:
    # AutoTokenizer 会自动从该目录加载配置和必要文件
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory_path)
    print(f"成功从 '{tokenizer_directory_path}' 加载分词器: {type(tokenizer)}")

    # --- 后续的分词步骤与之前相同 ---
    texts = ["你好，世界！", "这是一个使用 Jina CLIP v2 的例子。"]


    actual_max_length_for_onnx = 77 

    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=actual_max_length_for_onnx,
        return_tensors='np', # 返回 NumPy 数组，方便 ONNX Runtime 使用
        add_special_tokens=True
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print("Input IDs:\n", input_ids)
    print("Attention Mask:\n", attention_mask)

except Exception as e:
    print(f"从目录 '{tokenizer_directory_path}' 加载分词器失败: {e}")
    print("请检查：")
    print("1. 路径是否正确。")
    print("2. 目录中是否包含 'tokenizer_config.json' 以及 'tokenizer.json' 或其他必要的分词器文件（如 sentencepiece.model, vocab.json, merges.txt 等，取决于分词器类型）。")
    print("3. Transformers 库是否已正确安装。")
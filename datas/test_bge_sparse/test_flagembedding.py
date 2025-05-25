from FlagEmbedding import BGEM3FlagModel

sentences = [
    "这是一个测试句子。",
    "BGE-M3 模型用于处理自然语言。",
    "我们正在演示如何使用 BGE-M3 的词汇权重。"
]

model = BGEM3FlagModel(r'C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3', use_fp16=True) # 使用本地模型路径
output = model.encode(sentences, return_dense=False, return_sparse=True, return_colbert_vecs=False)
sparse_lexical_weights = output["lexical_weights"]

# 获取tokenizer用于ID到文本的映射
tokenizer = model.tokenizer

print("原始稀疏权重（token ID格式）:")
print(sparse_lexical_weights)
print("\n" + "="*50 + "\n")

print("稀疏权重（token文本映射）:")
for i, weights in enumerate(sparse_lexical_weights):
    print(f"句子 {i+1}: \"{sentences[i]}\"")
    print("Token权重映射:")
    for token_id, weight in weights.items():
        token_text = tokenizer.decode([int(token_id)])
        print(f"  '{token_text}' (ID: {token_id}) -> 权重: {weight:.6f}")
    print()

print("完整输出:")
print(output)

# 创建映射后的稀疏向量（token ID -> token文本）
mapped_sparse_weights = []
for weights in sparse_lexical_weights:
    mapped_weights = {}
    for token_id, weight in weights.items():
        token_text = tokenizer.decode([int(token_id)])
        mapped_weights[token_text] = weight
    mapped_sparse_weights.append(mapped_weights)

print("\n" + "="*50 + "\n")
print("映射后的稀疏向量（token文本格式）:")
for i, weights in enumerate(mapped_sparse_weights):
    print(f"句子 {i+1}: {weights}")

# 返回映射后的稀疏向量
print(f"\n返回的映射稀疏向量: {mapped_sparse_weights}")
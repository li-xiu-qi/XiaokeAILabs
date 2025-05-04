import sentencepiece as spm

# SentencePiece是Google开发的文本分词工具，支持多种语言的分词处理

# 训练新的分词器
spm.SentencePieceTrainer.train(
    input="corpus.txt",           # 训练语料：包含文本样本的文件路径
    model_prefix="my_sp_model",   # 模型名称前缀：将生成my_sp_model.model和my_sp_model.vocab
    vocab_size=2000,              # 词汇表大小：最终词表将包含约2000个token
    character_coverage=0.9995,    # 字符覆盖率：确保模型覆盖语料中99.95%的字符
    model_type="bpe"              # 模型类型：使用BPE(Byte Pair Encoding)算法，alternative是unigram
)

# 加载训练好的模型
sp = spm.SentencePieceProcessor()
sp.load("my_sp_model.model")      # 加载保存的模型文件

# 分词示例
text = "中文分词是自然语言处理的基础任务，深度学习技术在这个领域取得了突破性的进展。"   # 待分词的中文文本
tokens = sp.encode_as_pieces(text)  # 将文本转换为子词tokens
ids = sp.encode_as_ids(text)        # 将文本转换为token IDs

# 打印分词结果
print("Tokens:", tokens)            # 打印分词后的子词列表
print("Token IDs:", ids)            # 打印每个子词对应的ID

# 验证：将tokens重新组合为原文本
decoded_text = sp.decode_pieces(tokens)
print("Decoded text:", decoded_text)  # 打印重建的文本，应与原文本一致

"""
Tokens: ['▁中文分词', '是', '自然语言处理', '的', '基础任务', ',', '深度', '学习', '技术', '在', '这', '个', '领域', '取得了突破', '性的进展', '。']
Token IDs: [956, 1488, 130, 1478, 854, 1481, 192, 6, 8, 1514, 1978, 1520, 69, 1007, 882, 1480]
Decoded text: 中文分词是自然语言处理的基础任务,深度学习技术在这个领域取得了突破性的进展。

"""

# ================ 更多高级功能示例 ================

print("\n===== 高级功能示例 =====")

# 1. 批量处理文本
text_batch = [
    "自然语言处理是人工智能的重要分支。",
    "SentencePiece支持多种语言的分词。",
    "词元化是预处理文本的重要步骤。"
]
tokens_batch = sp.encode_as_pieces(text_batch)  # 批量分词
ids_batch = sp.encode_as_ids(text_batch)        # 批量转换为ID

print("\n批量处理示例:")
for i, (text, tokens, ids) in enumerate(zip(text_batch, tokens_batch, ids_batch)):
    print(f"文本{i+1}: {text}")
    print(f"分词结果: {tokens}")
    print(f"ID序列: {ids}")

# 2. 特殊标记的使用
print("\n特殊标记示例:")
# 获取特殊标记ID
unk_id = sp.piece_to_id("<unk>")   # 未知词标记
bos_id = sp.piece_to_id("<s>")     # 句子开始标记
eos_id = sp.piece_to_id("</s>")    # 句子结束标记
pad_id = sp.piece_to_id("<pad>")   # 填充标记

print(f"未知词标记ID: {unk_id}, 对应的token: {sp.id_to_piece(unk_id) if unk_id != sp.unk_id() else '<unk>'}")
print(f"句子开始标记ID: {bos_id}, 对应的token: {sp.id_to_piece(bos_id) if bos_id != sp.unk_id() else '<不存在>'}")
print(f"句子结束标记ID: {eos_id}, 对应的token: {sp.id_to_piece(eos_id) if eos_id != sp.unk_id() else '<不存在>'}")
print(f"填充标记ID: {pad_id}, 对应的token: {sp.id_to_piece(pad_id) if pad_id != sp.unk_id() else '<不存在>'}")

# 添加带特殊标记的编码
text = "这是一个测试句子。"
ids_with_special = [bos_id] + sp.encode_as_ids(text) + [eos_id]
print(f"带特殊标记的编码: {ids_with_special}")
print(f"解码结果: {sp.decode_ids(ids_with_special)}")

# 3. 子词正则化 (基于概率采样的分词)
if hasattr(sp, 'SampleEncodeAsIds'):  # 检查是否支持此功能
    print("\n子词正则化示例:")
    # alpha控制采样温度，较高的值增加多样性
    for alpha in [0.1, 0.5, 1.0]:
        sampled_ids = sp.SampleEncodeAsIds(text, alpha=alpha, nbest_size=-1)
        print(f"alpha={alpha}, 采样结果: {sampled_ids}")
        print(f"对应的tokens: {[sp.id_to_piece(id) for id in sampled_ids]}")

# 4. 词汇表操作
print("\n词汇表操作示例:")
# 获取词汇表大小
vocab_size = sp.get_piece_size()
print(f"词汇表大小: {vocab_size}")

# 查看词汇表中的一些词条
print("词汇表示例:")
for i in range(10):  # 展示前10个token
    piece = sp.id_to_piece(i)
    print(f"ID {i}: {piece}")

# 查询特定词的ID
common_words = ["自然", "语言", "处理", "学习"]
for word in common_words:
    word_id = sp.piece_to_id(word)
    found = "找到" if word_id != sp.unk_id() else "未找到"
    print(f"词 '{word}' 在词汇表中{found}, ID: {word_id if word_id != sp.unk_id() else '无'}")

# 5. 多语言处理示例
print("\n多语言处理示例:")
multilingual_texts = [
    "这是中文文本。",                           # 中文
    "This is English text.",                    # 英文
    "これは日本語のテキストです。",              # 日文
    "这是混合了English和中文的双语文本。",       # 中英混合
]

for text in multilingual_texts:
    tokens = sp.encode_as_pieces(text)
    print(f"原文: {text}")
    print(f"分词: {tokens}")
    print(f"ID序列: {sp.encode_as_ids(text)}")
    print()

# 6. 控制未知词的处理
print("\n未知词处理示例:")
rare_text = "这个词汇表中应该没有的稀有词语：XYZ123和αβγ"
tokens = sp.encode_as_pieces(rare_text)
print(f"原文: {rare_text}")
print(f"分词: {tokens}")
# 查找未知词的标记
for token in tokens:
    if sp.piece_to_id(token) == sp.unk_id():
        print(f"未知词标记: {token}")

# 7. 保存和加载分词结果
import json
print("\n保存和加载分词结果示例:")
# 保存分词结果
tokenization_result = {
    "text": text,
    "tokens": tokens,
    "ids": sp.encode_as_ids(text)
}
with open("tokenization_example.json", "w", encoding="utf-8") as f:
    json.dump(tokenization_result, f, ensure_ascii=False, indent=2)
print("分词结果已保存到tokenization_example.json")

print("\n完成所有示例演示")
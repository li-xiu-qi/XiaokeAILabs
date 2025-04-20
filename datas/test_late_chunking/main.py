# 导入必要的库
# pip install sentence-transformers torch transformers regex # 可能需要安装这些库
from sentence_transformers import SentenceTransformer
import torch
import re # 用于简单的句子分割 (更精确的可以用pysbd等库)
import numpy as np

# --- 参数设置 ---

# 选择一个支持中文并且性能较好的 Sentence Transformer 模型
# 'paraphrase-multilingual-mpnet-base-v2' 是一个强大的多语言模型
# 'shibing624/text2vec-base-chinese' 是一个专注于中文的模型
# 'nghuyong/ernie-3.0-base-zh' 基于ERNIE的模型
model_name = 'paraphrase-multilingual-mpnet-base-v2'

# 准备一个长中文文档用于演示
long_document = "这是一个用于演示延迟分块的长文档。人工智能（AI）正在飞速发展，深刻影响着医疗、金融、交通等各个行业。自然语言处理（NLP）是AI的一个重要分支，致力于让计算机理解和生成人类语言。对于长文本，我们需要将其有效分割成语义连贯的块，并为这些块生成高质量的向量表示，这对于信息检索、问答系统至关重要。传统的朴素分块方法（先分割文本再独立编码）常常因为忽略了块与块之间的联系而丢失重要的上下文信息。延迟分块（Late Chunking）策略试图克服这一缺点。它首先利用强大的预训练模型对整个文档进行编码，捕捉全局上下文，获得词元级别的向量表示。然后，根据预先定义的边界线索（如句子结束符），在词元向量序列上进行分割。最后，通过对每个分割出的向量片段进行池化操作（如平均池化），生成最终的块向量。"

# 定义分块的边界：这里我们用简单的标点符号来分割句子
# 注意：实际应用中可能需要更健壮的句子分割库，如pysbd或基于NLP模型的分割器
sentence_delimiters = r'[。？！]' # 匹配中文句号、问号、感叹号

# 定义池化策略：'mean' (平均池化), 'cls' (如果模型输出CLS向量且适用), 'max' (最大池化)
pooling_strategy = 'mean'

# --- 加载模型 ---
print(f"正在加载 Sentence Transformer 模型: {model_name}")
# 自动选择使用 GPU (如果可用) 或 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载模型实例
model = SentenceTransformer(model_name, device=device)
print(f"模型已成功加载到: {device}")

# --- 步骤 1: 对整个文档进行编码，获取词元级(Token-level)向量 ---
# 为了获取词元向量，我们需要访问模型底层的 Transformer 输出
# SentenceTransformer 默认的 encode() 会直接进行池化得到单个句子/文档向量
# 我们需要获取 'last_hidden_state'

print("对整个文档进行编码以获取词元向量...")
# 使用模型的 tokenizer 对文档进行分词
# return_tensors='pt' 返回 PyTorch tensors
# truncation=True 如果文档过长则截断 (需要注意!)
# padding=True 填充到最大长度或指定长度 (这里获取整个文档，一般不需填充到固定长，但要处理截断)
# max_length 可以根据模型限制设置
max_length = model.tokenizer.model_max_length
encoded_input = model.tokenizer(long_document, return_tensors='pt', truncation=True, max_length=max_length, padding=False) # 先不padding，看实际长度
encoded_input = encoded_input.to(device) # 将输入数据移动到模型所在的设备

# 获取 input_ids (词元的数字表示)
input_ids = encoded_input['input_ids'].squeeze() # 移除 batch 维度

# 通过底层 Transformer 模型获取输出
# model[0] 通常是底层的 Transformer 模块
transformer_module = model[0].auto_model
with torch.no_grad(): # 关闭梯度计算，节省内存
    # 传入 input_ids 和 attention_mask
    outputs = transformer_module(**encoded_input)
    # 获取最后一层的隐藏状态，这就是我们需要的词元向量
    # 形状: [batch_size, sequence_length, hidden_size]
    token_embeddings = outputs.last_hidden_state.squeeze(0) # 移除 batch 维度
    # 形状变为: [sequence_length, hidden_size]

print(f"成功获取了 {token_embeddings.shape[0]} 个词元的向量，每个向量维度为 {token_embeddings.shape[1]}")

# --- 步骤 2 & 3: 定义边界并映射到词元索引 ---
print("正在分割文档为句子，并尝试映射到词元边界...")

# 使用正则表达式分割句子，并保留分隔符以便映射
sentences_with_delimiters = re.split(f'({sentence_delimiters})', long_document)
sentences = []
if sentences_with_delimiters:
    # 将分隔符附加到前面的句子，并去除空字符串
    current_sentence = sentences_with_delimiters[0].strip()
    for i in range(1, len(sentences_with_delimiters) - 1, 2):
        part = sentences_with_delimiters[i]
        delimiter = sentences_with_delimiters[i+1]
        if part and part.strip():
           current_sentence += part.strip() # 合并可能被错误分割的部分
        if delimiter and delimiter.strip():
           current_sentence += delimiter.strip() # 添加分隔符
        if current_sentence:
           sentences.append(current_sentence) # 添加完整句子
        current_sentence = "" # 重置
    # 处理最后一个片段（如果没有以分隔符结尾）
    last_part = sentences_with_delimiters[-1].strip()
    if last_part:
        if sentences: # 如果已有句子，加到最后一个后面
            sentences[-1] += " " + last_part # 或者直接作为一个新句子
        else:
            sentences.append(last_part)

print(f"文档被分割成 {len(sentences)} 个句子块。")

# 映射句子边界到词元索引 (这是 Late Chunking 的难点)
# 我们需要知道每个句子对应原始 `input_ids` 中的哪些 token
chunk_token_indices = [] # 存储每个块对应的词元索引范围 (start, end)
current_token_pos = 0 # 追踪在完整 token_embeddings 中的当前位置

# 考虑特殊标记 [CLS] 和 [SEP] 对索引的影响
# 不同模型的 tokenizer 可能行为不同
cls_token_id = model.tokenizer.cls_token_id
sep_token_id = model.tokenizer.sep_token_id
adds_special_tokens = cls_token_id is not None and sep_token_id is not None

token_offset = 0
if adds_special_tokens and input_ids[0] == cls_token_id:
    token_offset = 1 # 跳过开头的 [CLS] token

print("正在映射...")
for i, sentence in enumerate(sentences):
    # 对当前句子进行分词 (不添加特殊标记，因为它们已在整体编码中)
    sentence_encoded = model.tokenizer(sentence, add_special_tokens=False, return_tensors=None)
    sentence_token_ids = sentence_encoded['input_ids']
    num_sentence_tokens = len(sentence_token_ids)

    # 查找当前句子的 token 序列在完整 token 序列中的位置
    start_index = -1
    # 在 token_embeddings (对应 input_ids) 中查找子序列
    # 注意：这里的查找基于 ID，需要确保一致性
    # 这是一个简化的查找，实际可能因为规范化等原因不完全匹配，需要更鲁棒的方法
    search_start_pos = current_token_pos
    found = False
    max_search_limit = token_embeddings.shape[0] - num_sentence_tokens + 1

    # 从上一个结束位置开始查找
    for k in range(search_start_pos, max_search_limit):
        # 比较 input_ids 中的序列段
        if input_ids[k : k + num_sentence_tokens].tolist() == sentence_token_ids:
            # 找到了匹配！
            start_index = k
            end_index = k + num_sentence_tokens
            # 记录的是 token_embeddings 中的索引 (需要加上 offset)
            # 但池化时用的就是原始 token_embeddings 的索引，offset 已在查找时考虑
            chunk_token_indices.append((start_index, end_index))
            current_token_pos = end_index # 更新下一个句子的起始搜索位置
            print(f"  句子 {i+1} 映射到词元索引范围: [{start_index}, {end_index})")
            found = True
            break # 找到就停止当前句子的搜索

    if not found:
        print(f"  **警告**: 句子 {i+1} 未能在词元序列中精确定位。可能是句子分割或分词不一致导致。跳过此块。")
        # 尝试向前移动一点，避免卡死 (非常粗糙)
        # current_token_pos += 1

# --- 步骤 4 & 5: 分割向量序列并进行池化 ---
print("\n正在对映射到的词元向量块进行池化...")
late_chunk_embeddings = []

for i, (start_idx, end_idx) in enumerate(chunk_token_indices):
    # 从完整的词元向量序列中提取对应块的向量
    segment_token_embeddings = token_embeddings[start_idx:end_idx]

    if segment_token_embeddings.shape[0] > 0:
        # 应用池化策略
        if pooling_strategy == 'mean':
            # 计算平均值 (在 hidden_size 维度上)
            pooled_embedding = torch.mean(segment_embeddings, dim=0)
        elif pooling_strategy == 'max':
             # 计算最大值 (在 hidden_size 维度上)
             pooled_embedding = torch.max(segment_embeddings, dim=0).values
        # 可以添加其他池化策略, e.g., CLS token (如果适用且被包含在块内)
        # elif pooling_strategy == 'cls':
        #     pooled_embedding = segment_token_embeddings[0] # 假设CLS是第一个token
        else:
             # 默认使用平均池化
             pooled_embedding = torch.mean(segment_embeddings, dim=0)

        late_chunk_embeddings.append(pooled_embedding)
        print(f"  生成块 {i+1} 的向量 (维度: {pooled_embedding.shape[0]})")
    else:
        print(f"  块 {i+1} 没有有效的词元向量，跳过。")

# --- 输出结果 ---
print(f"\n成功生成了 {len(late_chunk_embeddings)} 个延迟分块向量。")

# 可以将结果转换为 NumPy 数组，方便后续使用
if late_chunk_embeddings:
    late_chunk_embeddings_np = torch.stack(late_chunk_embeddings).cpu().numpy()
    print(f"最终输出的 NumPy 数组形状: {late_chunk_embeddings_np.shape}") # (块数量, 向量维度)
else:
    print("没有生成任何有效的块向量。")

# 现在 late_chunk_embeddings_np (或 late_chunk_embeddings 列表)
# 包含了每个句子的、考虑了全局上下文的向量表示。
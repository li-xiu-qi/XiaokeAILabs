import numpy as np
from FlagEmbedding import FlagModel # 导入 FlagModel

def cosine_similarity(vec1, vec2):
  """计算两个 NumPy 向量的余弦相似度"""
  vec1 = np.asarray(vec1)
  vec2 = np.asarray(vec2)
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  if norm_vec1 == 0 or norm_vec2 == 0:
    return 0.0
  similarity = dot_product / (norm_vec1 * norm_vec2)
  # 根据需要，可以考虑相似度范围，例如映射到 [0, 1]
  # return max(0.0, similarity) # 如果需要非负相似度
  return similarity # 使用原始余弦相似度 [-1, 1]

def mmr_selection(query_embedding, item_embeddings, item_ids, lambda_param, num_results):
  """
  使用 MMR 算法执行项目选择

  Args:
    query_embedding (np.array): 查询的向量表示。
    item_embeddings (dict): 候选项目向量表示的字典 {item_id: np.array}。
    item_ids (list): 初始候选项目 ID 列表 (通常是字符串ID)。
    lambda_param (float): MMR 的权衡参数 lambda (0 <= lambda <= 1)。
    num_results (int): 需要选择的结果数量 N。

  Returns:
    list: 最终选出的项目 ID 列表 (字符串ID)。
  """

  if not item_ids or not item_embeddings or num_results <= 0:
      return []

  # 筛选出有效的候选ID（存在于embeddings字典中）
  valid_candidate_ids = [id for id in item_ids if id in item_embeddings]
  if not valid_candidate_ids:
      return []

  candidate_pool = set(valid_candidate_ids)
  selected_item_ids = []

  # 预计算所有有效候选项目与查询的相关性 (Sim_1)
  candidate_relevance = {
      id: cosine_similarity(query_embedding, item_embeddings[id])
      for id in valid_candidate_ids
  }

  # 确保 N 不超过有效候选者数量
  num_results = min(num_results, len(valid_candidate_ids))

  # 第一步：选择最相关的项目
  if valid_candidate_ids:
      first_selection_id = max(candidate_relevance, key=candidate_relevance.get)
      selected_item_ids.append(first_selection_id)
      candidate_pool.remove(first_selection_id)

  # 后续迭代选择
  while len(selected_item_ids) < num_results and candidate_pool:
    mmr_scores = {}
    selected_embeddings_list = [item_embeddings[id] for id in selected_item_ids] # 获取已选项目的向量

    for candidate_id in candidate_pool:
        candidate_emb = item_embeddings[candidate_id]

        # Sim_1: 获取预计算的相关性
        relevance_score = candidate_relevance.get(candidate_id, -1.0) # 使用预计算的相关性, -1.0作为默认值

        # Sim_2: 计算与已选项目的最大相似度
        max_similarity_with_selected = -1.0 # 初始化为可能的最低余弦相似度
        if selected_item_ids: # 仅当 S 非空时计算
             similarities_to_selected = [cosine_similarity(candidate_emb, sel_emb) for sel_emb in selected_embeddings_list]
             if similarities_to_selected:
                 max_similarity_with_selected = max(similarities_to_selected)

        # 计算 MMR 分数
        # MMR Score = λ * Sim1(Di, Q) - (1 - λ) * max(Sim2(Di, Dj)) for Dj in S
        # 注意：如果 Sim1 和 Sim2 可能为负，需要确保公式逻辑正确
        mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity_with_selected
        mmr_scores[candidate_id] = mmr_score

    if not mmr_scores: # 如果没有更多可计算分数的候选者
        break

    # 选择当前迭代中 MMR 分数最高的项目
    best_next_id = max(mmr_scores, key=mmr_scores.get)
    selected_item_ids.append(best_next_id)
    candidate_pool.remove(best_next_id) # 从候选池中移除

  return selected_item_ids

# --- 使用 FlagEmbedding 获取向量并运行 MMR ---

# 1. 加载模型 (请确保模型路径正确)
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-large-zh-v1.5"
try:
    model = FlagModel(model_path, use_fp16=True) # 尝试使用 FP16 加速
    print("模型加载成功。")
except Exception as e:
    print(f"模型加载失败: {e}")
    # 在此可以添加退出或使用备用逻辑
    exit() # 或者 return, raise e 等

# 2. 定义查询和候选句子
query_sentence = "大型语言模型有哪些应用？"

candidate_sentences = [
    # 与查询直接相关 - 应用类
    "大语言模型可用于文本生成，例如写诗歌或代码。", # id=s1
    "机器翻译是大语言模型的常见应用场景之一。",     # id=s2
    "聊天机器人和智能客服常常基于大型语言模型构建。",# id=s3
    "大型模型能够进行文本摘要和信息抽取。",         # id=s4

    # 与查询相关 - 原理/定义类 (与应用类有差异)
    "大型语言模型通常指参数量巨大的深度学习模型。", # id=s5
    "Transformer架构是现代大语言模型的基础。",       # id=s6
    "训练大型语言模型需要海量的文本数据和计算资源。",# id=s7

    # 不太相关或离题
    "今天天气真不错。",                           # id=s8
    "人工智能的研究历史悠久。",                   # id=s9
]
# 为句子分配 ID
candidate_ids = [f"s{i+1}" for i in range(len(candidate_sentences))]

# 创建ID到句子的映射字典
id_to_sentence = {candidate_ids[i]: candidate_sentences[i] for i in range(len(candidate_sentences))}

# 3. 获取所有句子的嵌入向量
all_sentences = [query_sentence] + candidate_sentences
print("开始计算嵌入向量...")
embeddings = model.encode(all_sentences)
print(f"嵌入向量计算完成，形状: {embeddings.shape}") # 应为 (1 + len(candidate_sentences), 1024)

query_embedding = embeddings[0]
item_embeddings_dict = {candidate_ids[i]: embeddings[i+1] for i in range(len(candidate_sentences))}

# 4. 设定参数并运行 MMR
# 假设初始列表基于某种粗排得到（这里简化为原始顺序）
initial_ranked_ids = candidate_ids
num_select = 5 # 期望选出5个结果

# 场景1: 更注重相关性
lambda_high = 0.7
selected_high_lambda = mmr_selection(query_embedding, item_embeddings_dict, initial_ranked_ids, lambda_high, num_select)
print(f"\n--- MMR 选择结果 (lambda={lambda_high}, N={num_select}) ---")
print("选定句子ID:", selected_high_lambda)
print("选定句子内容:")
for i, item_id in enumerate(selected_high_lambda):
    print(f"{i+1}. ID={item_id}: {id_to_sentence[item_id]}")

# 场景2: 更注重多样性
lambda_low = 0.3
selected_low_lambda = mmr_selection(query_embedding, item_embeddings_dict, initial_ranked_ids, lambda_low, num_select)
print(f"\n--- MMR 选择结果 (lambda={lambda_low}, N={num_select}) ---")
print("选定句子ID:", selected_low_lambda)
print("选定句子内容:")
for i, item_id in enumerate(selected_low_lambda):
    print(f"{i+1}. ID={item_id}: {id_to_sentence[item_id]}")
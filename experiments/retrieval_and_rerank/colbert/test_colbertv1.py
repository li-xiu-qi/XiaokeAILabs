# 作者: 筱可
# 日期: 2025 年 3 月 30 日
# 版权所有 (c) 2025 筱可 & 筱可AI研习社. 保留所有权利.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from FlagEmbedding import BGEM3FlagModel
import numpy as np

model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"

# 初始化模型
model = BGEM3FlagModel(model_path, use_fp16=True)

# 输入数据
sentences = ["What is BGE M3?", "Defination of BM25"]
output = model.encode(sentences, return_colbert_vecs=True)

# 提取向量
dense_vecs = output['dense_vecs'] 
print(f"Dense vectors shape: {dense_vecs.shape}") # 形状: (2, 1024)

dense_vecs1 = output['dense_vecs'][0]  # 第一个句子的向量
print(f"Dense vectors 1: {dense_vecs1.shape}")

dense_vecs2 = output['dense_vecs'][1]  # 修正为第二个句子的向量
print(f"Dense vectors 2: {dense_vecs2.shape}")

query_vecs = output['colbert_vecs'][0]  # 形状: (8, 1024)
doc_vecs = output['colbert_vecs'][1]    # 形状: (7, 1024)
print(f"Query vectors: {query_vecs.shape}")
print(f"Document vectors: {doc_vecs.shape}")

# 原始计算
sim_matrix = np.dot(query_vecs, doc_vecs.T)  # 形状: (8, 7)
score_raw = sim_matrix.max(axis=1).sum()
print(f"Similarity score (raw): {score_raw}")

# 归一化

# 归一化
query_vecs_norm = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
doc_vecs_norm = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
sim_matrix_correct = np.dot(query_vecs_norm, doc_vecs_norm.T)
score_correct = sim_matrix_correct.max(axis=1).sum() / len(query_vecs)
print(f"Similarity score (normalized vectors): {score_correct}")

# 官方方法
score_colbert = model.colbert_score(query_vecs, doc_vecs)
print(f"Similarity score (colbert_score): {score_colbert}")

# 计算dense_vecs1和dense_vecs2之间的余弦相似度
cosine_similarity = np.dot(dense_vecs1, dense_vecs2) / (np.linalg.norm(dense_vecs1) * np.linalg.norm(dense_vecs2))

print(f"Cosine similarity between dense_vecs1 and dense_vecs2: {cosine_similarity}")

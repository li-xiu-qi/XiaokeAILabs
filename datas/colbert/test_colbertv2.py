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
from sklearn.cluster import KMeans
import pickle
import os

# 初始化BGE-M3模型
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
model = BGEM3FlagModel(model_path, use_fp16=True)

# 加载示例文档集
documents = [
    "BGE-M3是一个多语言嵌入模型，支持多种语言的文本检索。",
    "ColBERT是一种延迟交互检索模型，通过最大相似度匹配计算相关性。",
    "BM25是一种经典的词袋模型，基于词频和逆文档频率计算相关性。",
    "向量检索在大规模信息检索任务中越来越受欢迎，替代了传统的倒排索引方法。",
    "残差压缩是一种有效的向量压缩技术，可以大幅减少存储需求。"
]

# 生成ColBERT向量
def generate_colbert_vectors(docs):
    all_vectors = []
    for doc in docs:
        output = model.encode([doc], return_colbert_vecs=True)
        doc_vectors = output['colbert_vecs'][0]  # 单个文档的token向量集合
        all_vectors.append(doc_vectors)
    return all_vectors

# 残差压缩实现
class ResidualCompressor:
    def __init__(self, n_centroids=256, quantization_bits=8):
        self.n_centroids = n_centroids
        self.bits = quantization_bits
        self.kmeans = None
        self.r_min = None
        self.r_max = None
        
    def fit(self, vectors_list):
        # 将所有文档的所有token向量扁平化为一个大矩阵
        all_vectors = np.vstack([v for doc_vecs in vectors_list for v in doc_vecs])
        print(f"训练聚类模型，向量总数: {all_vectors.shape[0]}")
        
        # 动态调整聚类中心数量
        actual_n_centroids = min(self.n_centroids, all_vectors.shape[0])
        if actual_n_centroids < self.n_centroids:
            print(f"警告: 样本数({all_vectors.shape[0]})小于指定的聚类中心数({self.n_centroids})。")
            print(f"已将聚类中心数调整为: {actual_n_centroids}")
            self.n_centroids = actual_n_centroids
        
        # K-means聚类
        self.kmeans = KMeans(n_clusters=self.n_centroids, random_state=42)
        self.kmeans.fit(all_vectors)
        return self
    
    def compress(self, vectors_list):
        compressed_docs = []
        total_original_size = 0
        total_compressed_size = 0
        
        for doc_vectors in vectors_list:
            # 计算每个向量最近的质心
            centroids_idx = self.kmeans.predict(doc_vectors)
            
            # 计算残差
            residuals = doc_vectors - self.kmeans.cluster_centers_[centroids_idx]
            
            # 确定残差的范围
            if self.r_min is None or self.r_max is None:
                self.r_min = np.min(residuals)
                self.r_max = np.max(residuals)
            
            # 量化残差
            quantized = np.round(
                (residuals - self.r_min) / (self.r_max - self.r_min) * (2**self.bits - 1)
            ).astype(np.uint8)
            
            # 计算原始大小和压缩后大小
            original_size = doc_vectors.nbytes
            compressed_size = centroids_idx.nbytes + quantized.nbytes
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            
            compressed_docs.append({
                'centroids_idx': centroids_idx,
                'quantized': quantized
            })
            
        compression_ratio = total_original_size / total_compressed_size
        print(f"原始向量大小: {total_original_size / 1024:.2f} KB")
        print(f"压缩后大小: {total_compressed_size / 1024:.2f} KB")
        print(f"压缩比: {compression_ratio:.2f}x")
        
        return compressed_docs
    
    def decompress(self, compressed_docs):
        decompressed_docs = []
        
        for doc in compressed_docs:
            centroids_idx = doc['centroids_idx']
            quantized = doc['quantized']
            
            # 反量化残差
            residuals = self.r_min + (quantized / (2**self.bits - 1)) * (self.r_max - self.r_min)
            
            # 重建向量
            reconstructed = self.kmeans.cluster_centers_[centroids_idx] + residuals
            decompressed_docs.append(reconstructed)
            
        return decompressed_docs
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'r_min': self.r_min,
                'r_max': self.r_max,
                'n_centroids': self.n_centroids,
                'bits': self.bits
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.r_min = data['r_min']
            self.r_max = data['r_max']
            self.n_centroids = data['n_centroids']
            self.bits = data['bits']
        return self

# 执行ColBERT向量生成和残差压缩
colbert_vectors = generate_colbert_vectors(documents)

# 计算原始向量的总体大小
original_size = sum([doc_vecs.nbytes for doc_vecs in colbert_vectors])
print(f"原始未压缩向量大小: {original_size / 1024:.2f} KB")

# 应用残差压缩
compressor = ResidualCompressor(n_centroids=256, quantization_bits=8)
compressor.fit(colbert_vectors)
compressed_vectors = compressor.compress(colbert_vectors)

# 重建向量并评估质量
decompressed_vectors = compressor.decompress(compressed_vectors)

# 计算重建误差
mse_list = []
for orig, recon in zip(colbert_vectors, decompressed_vectors):
    mse = np.mean((orig - recon) ** 2)
    mse_list.append(mse)
avg_mse = np.mean(mse_list)
print(f"平均重建均方误差: {avg_mse:.6f}")

# 评估重建前后的相似度计算结果差异
def compute_similarity(query_vec, doc_vec):
    """计算ColBERT相似度得分"""
    query_vec_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    doc_vec_norm = doc_vec / np.linalg.norm(doc_vec, axis=1, keepdims=True)
    sim_matrix = np.dot(query_vec_norm, doc_vec_norm.T)
    return sim_matrix.max(axis=1).sum() / len(query_vec)

# 假设使用第一个文档作为查询
query_vec = colbert_vectors[0]

# 计算原始向量的相似度
original_scores = [compute_similarity(query_vec, doc_vec) for doc_vec in colbert_vectors]

# 计算重建向量的相似度
reconstructed_scores = [compute_similarity(query_vec, doc_vec) for doc_vec in decompressed_vectors]

# 比较相似度差异
for i, (orig_score, recon_score) in enumerate(zip(original_scores, reconstructed_scores)):
    rel_diff = abs(orig_score - recon_score) / orig_score * 100
    print(f"文档 {i+1} - 原始得分: {orig_score:.4f}, 重建得分: {recon_score:.4f}, 相对差异: {rel_diff:.2f}%")

# 保存压缩模型以供生产环境使用
compressor.save('colbert_compressor.pkl')

print("压缩模型已保存，可用于生产环境")
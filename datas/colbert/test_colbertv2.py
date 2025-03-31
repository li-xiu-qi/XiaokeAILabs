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
# 请确保模型路径正确，如果模型未下载，FlagEmbedding库会自动下载
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3" # 使用你的本地路径
# model_path = 'BAAI/bge-m3' # 或者直接使用Hugging Face模型ID
try:
    model = BGEM3FlagModel(model_path, use_fp16=True) # 尝试使用FP16以节省内存和加速
except Exception as e:
    print(f"无法加载模型 {model_path}，请检查路径或网络连接: {e}")
    # 在无法加载模型时退出或采取备用措施
    exit()

# 示例文档集
documents = [
    "BGE-M3是一个多语言嵌入模型，支持多种语言的文本检索。",
    "ColBERT是一种延迟交互检索模型，通过最大相似度匹配计算相关性。",
    "BM25是一种经典的词袋模型，基于词频和逆文档频率计算相关性。",
    "向量检索在大规模信息检索任务中越来越受欢迎，替代了传统的倒排索引方法。",
    "残差压缩是一种有效的向量压缩技术，可以大幅减少存储需求。",
    "机器学习模型需要大量数据进行训练，以达到良好的泛化能力。",
    "自然语言处理是人工智能的一个重要分支，专注于计算机与人类语言的交互。",
    "深度学习利用深度神经网络学习数据的复杂模式。"
] # 增加了文档数量以更好地模拟聚类

# 生成ColBERT向量
def generate_colbert_vectors(docs):
    print("正在生成ColBERT向量...")
    all_vectors = []
    for doc in docs:
        # BGE-M3的encode方法可以直接处理列表，但这里为了清晰，逐个处理
        try:
            output = model.encode([doc], return_colbert_vecs=True, batch_size=1) # 指定batch_size=1
            # 注意：BGE-M3 返回的 colbert_vecs 可能已经是 list of list of floats，需要转np.array
            doc_vectors_list = output['colbert_vecs'][0] # 提取token级别向量表示 (可能是list)
            doc_vectors_np = np.array(doc_vectors_list, dtype=np.float32) # 转换为numpy数组
            if doc_vectors_np.ndim == 2 and doc_vectors_np.shape[0] > 0: # 确保不是空或维度错误
                 all_vectors.append(doc_vectors_np)
            else:
                 print(f"警告: 文档 '{doc[:30]}...' 生成的向量为空或格式不正确，已跳过。Shape: {doc_vectors_np.shape}")
        except Exception as e:
            print(f"处理文档 '{doc[:30]}...' 时出错: {e}")
            continue # 跳过出错的文档
    print(f"向量生成完毕，共处理 {len(all_vectors)} 个有效文档。")
    return all_vectors

# 残差压缩实现类
class ResidualCompressor:
    def __init__(self, n_centroids=256, quantization_bits=8):
        self.n_centroids = n_centroids
        self.bits = quantization_bits
        self.kmeans = None
        self.r_min = None
        self.r_max = None
        # BGE-M3的维度是1024
        self.dimension = 1024 # 显式指定维度或从数据推断

    def fit(self, vectors_list):
        # 将所有文档的token向量合并为训练集
        # 过滤掉空向量列表
        non_empty_vectors_list = [v for v in vectors_list if v.shape[0] > 0]
        if not non_empty_vectors_list:
             print("错误：没有有效的向量用于训练KMeans。")
             return self

        try:
            all_vectors = np.vstack(non_empty_vectors_list)
            if all_vectors.shape[0] == 0:
                print("错误：合并后的向量集为空。")
                return self
            self.dimension = all_vectors.shape[1] # 更新维度信息
            print(f"训练聚类模型，向量总数: {all_vectors.shape[0]}, 维度: {self.dimension}")

            # 自适应调整聚类中心数量，避免过拟合
            actual_n_centroids = min(self.n_centroids, all_vectors.shape[0])
            if actual_n_centroids < self.n_centroids:
                print(f"警告: 样本数({all_vectors.shape[0]})小于指定的聚类中心数({self.n_centroids})。")
                print(f"已将聚类中心数调整为: {actual_n_centroids}")
                self.n_centroids = actual_n_centroids
            elif actual_n_centroids == 0:
                print("错误: 没有可用的聚类中心。")
                return self


            # 执行K-means聚类，学习向量空间的质心分布
            # 增加 n_init 和 max_iter 以提高聚类质量
            self.kmeans = KMeans(n_clusters=self.n_centroids, random_state=42, n_init=10, max_iter=300)
            self.kmeans.fit(all_vectors)

            # 计算所有残差以确定全局范围
            print("计算残差范围...")
            all_residuals = []
            for doc_vectors in non_empty_vectors_list:
                if doc_vectors.shape[0] > 0:
                    centroids_idx = self.kmeans.predict(doc_vectors)
                    residuals = doc_vectors - self.kmeans.cluster_centers_[centroids_idx]
                    all_residuals.append(residuals)

            if not all_residuals:
                 print("错误：无法计算残差。")
                 return self

            all_residuals_np = np.vstack(all_residuals)
            self.r_min = np.min(all_residuals_np)
            self.r_max = np.max(all_residuals_np)
            print(f"残差范围确定: min={self.r_min:.4f}, max={self.r_max:.4f}")

        except ValueError as ve:
             print(f"KMeans训练或残差计算中发生数值错误: {ve}")
             # 可能需要检查输入数据是否有NaN或无穷大值
        except Exception as e:
             print(f"拟合过程中发生未知错误: {e}")
        return self

    def compress(self, vectors_list):
        if self.kmeans is None or self.r_min is None or self.r_max is None:
            print("错误: 压缩器未训练或训练不完整。请先调用 fit 方法。")
            return None

        compressed_docs = []
        total_original_size = 0
        total_compressed_size = 0

        # 确定索引和量化值的numpy类型以精确计算大小
        index_dtype = np.uint16 if self.n_centroids > 255 else np.uint8
        quantized_dtype = np.uint8 # 假设 b=8

        for doc_vectors in vectors_list:
             if doc_vectors.shape[0] == 0: # 跳过空向量
                  compressed_docs.append(None) # 用None占位或跳过
                  continue

             # 步骤1: 为每个向量分配最近的质心
             centroids_idx = self.kmeans.predict(doc_vectors).astype(index_dtype)

             # 步骤2: 计算向量与质心之间的残差
             residuals = doc_vectors - self.kmeans.cluster_centers_[centroids_idx]

             # 步骤3: (范围已在fit中确定)

             # 步骤4: 将残差量化为指定位数的整数
             # 防止除零错误
             if self.r_max == self.r_min:
                 print("警告: 残差范围为零，无法进行量化。")
                 # 可以选择将所有量化值设为0或中间值
                 quantized = np.zeros_like(residuals, dtype=quantized_dtype)
             else:
                 quantized_float = (residuals - self.r_min) / (self.r_max - self.r_min) * (2**self.bits - 1)
                 # 将值裁剪到[0, 2^b-1]范围内，然后取整
                 quantized = np.round(np.clip(quantized_float, 0, 2**self.bits - 1)).astype(quantized_dtype)

             # 计算压缩效果
             original_size = doc_vectors.nbytes
             # 压缩大小 = 索引大小 + 量化残差大小
             compressed_size = centroids_idx.nbytes + quantized.nbytes

             total_original_size += original_size
             total_compressed_size += compressed_size

             # 存储压缩结果：质心索引和量化残差
             compressed_docs.append({
                 'centroids_idx': centroids_idx,
                 'quantized': quantized
             })

        # 输出压缩统计信息
        if total_compressed_size > 0:
             compression_ratio = total_original_size / total_compressed_size
             print(f"原始向量总大小: {total_original_size / 1024:.2f} KB")
             print(f"压缩后总大小: {total_compressed_size / 1024:.2f} KB")
             print(f"压缩比: {compression_ratio:.2f}x")
        else:
             print("没有可压缩的数据。")

        return compressed_docs

    def decompress(self, compressed_docs):
        if self.kmeans is None or self.r_min is None or self.r_max is None:
            print("错误: 压缩器未训练或训练不完整。")
            return None

        decompressed_docs = []
        for doc_data in compressed_docs:
            if doc_data is None: # 处理之前跳过的空文档
                # 可以返回空数组或特定标记
                decompressed_docs.append(np.empty((0, self.dimension), dtype=np.float32))
                continue

            centroids_idx = doc_data['centroids_idx']
            quantized = doc_data['quantized']

            # 步骤1: 反量化残差
            if self.r_max == self.r_min:
                 residuals = np.zeros_like(quantized, dtype=np.float32) # 如果范围为0，则残差为0
            else:
                 residuals = self.r_min + (quantized.astype(np.float32) / (2**self.bits - 1)) * (self.r_max - self.r_min)

            # 步骤2: 重建原始向量 = 质心 + 残差
            # 确保使用正确的质心和类型
            reconstructed = self.kmeans.cluster_centers_[centroids_idx].astype(np.float32) + residuals.astype(np.float32)
            decompressed_docs.append(reconstructed)

        return decompressed_docs

    def save(self, path):
        if self.kmeans is None:
            print("错误: 模型未训练，无法保存。")
            return
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans_centers': self.kmeans.cluster_centers_,
                'r_min': self.r_min,
                'r_max': self.r_max,
                'n_centroids': self.n_centroids,
                'bits': self.bits,
                'dimension': self.dimension # 保存维度信息
            }, f)
        print(f"压缩器状态已保存到 {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # 重新构建一个KMeans对象，只设置centers，因为我们不需要再用它来predict或fit
            self.kmeans = KMeans(n_clusters=data['n_centroids'])
            self.kmeans.cluster_centers_ = data['kmeans_centers']
            # 初始化一个虚拟的 _n_threads 属性 (部分sklearn版本需要)
            # setattr(self.kmeans, '_n_threads', 1) # 或者根据需要设置
            self.r_min = data['r_min']
            self.r_max = data['r_max']
            self.n_centroids = data['n_centroids']
            self.bits = data['bits']
            self.dimension = data.get('dimension', 1024) # 向后兼容，如果旧模型没存维度，则默认1024
        print(f"压缩器状态已从 {path} 加载")
        return self

# --- 主流程 ---
colbert_vectors = generate_colbert_vectors(documents)

if not colbert_vectors:
    print("未能生成任何有效的ColBERT向量，程序终止。")
    exit()

# 计算原始向量占用空间
original_size = sum([doc_vecs.nbytes for doc_vecs in colbert_vectors if doc_vecs.shape[0]>0])
print(f"原始未压缩向量总大小: {original_size / 1024:.2f} KB")

# 应用残差压缩 (使用更多质心以获得更好的效果，但受限于样本数量)
# n_centroids 设为 64 或 128 可能是个更实际的起点
compressor = ResidualCompressor(n_centroids=64, quantization_bits=8)
compressor.fit(colbert_vectors)

# 检查压缩器是否成功训练
if compressor.kmeans is not None:
    compressed_vectors_data = compressor.compress(colbert_vectors)

    if compressed_vectors_data:
        # 重建向量并评估压缩质量
        decompressed_vectors = compressor.decompress(compressed_vectors_data)

        # 计算均方误差评估重建质量 (只对非空向量计算)
        mse_list = []
        valid_indices = [i for i, v in enumerate(colbert_vectors) if v.shape[0] > 0]
        for i in valid_indices:
            orig = colbert_vectors[i]
            recon = decompressed_vectors[i]
            if orig.shape == recon.shape and orig.shape[0] > 0: # 确保形状匹配且非空
                mse = np.mean((orig - recon) ** 2)
                mse_list.append(mse)
            else:
                 print(f"警告: 文档 {i} 的原始和重建向量形状不匹配或为空，跳过MSE计算。")

        if mse_list:
            avg_mse = np.mean(mse_list)
            print(f"平均重建均方误差 (MSE): {avg_mse:.6f}")
        else:
            print("无法计算平均重建均方误差。")

        # 评估重建前后的相似度排序一致性 (只对有效向量操作)
        def compute_similarity(query_vec, doc_vec):
            """计算标准化后的ColBERT相似度得分 (MaxSim)"""
            if query_vec.shape[0] == 0 or doc_vec.shape[0] == 0: return 0.0 # 处理空向量
            # 归一化每个token向量
            query_vec_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8) # 加epsilon防除零
            doc_vec_norm = doc_vec / (np.linalg.norm(doc_vec, axis=1, keepdims=True) + 1e-8)
            # 计算每个查询token与所有文档token的最大相似度
            sim_matrix = np.dot(query_vec_norm, doc_vec_norm.T)
            max_sim_per_query_token = sim_matrix.max(axis=1)
            # 求和并平均 (ColBERT原始得分)
            return max_sim_per_query_token.sum()

        # 以第一个有效文档作为查询示例
        query_index = valid_indices[0] if valid_indices else -1
        if query_index != -1:
             query_vec_orig = colbert_vectors[query_index]
             query_vec_recon = decompressed_vectors[query_index] # 重建后的查询向量

             print("\n比较查询与各文档的相似度得分 (使用原始查询向量):")
             original_scores = []
             reconstructed_scores = []
             for i in valid_indices: # 只和有效文档比较
                 orig_score = compute_similarity(query_vec_orig, colbert_vectors[i])
                 recon_score = compute_similarity(query_vec_orig, decompressed_vectors[i]) # 查询是原始的，文档是重建的
                 original_scores.append(orig_score)
                 reconstructed_scores.append(recon_score)

                 rel_diff = abs(orig_score - recon_score) / (abs(orig_score) + 1e-9) * 100 # 加epsilon防除零
                 print(f"文档 {i+1} - 原始得分: {orig_score:.4f}, 重建文档得分: {recon_score:.4f}, 相对差异: {rel_diff:.2f}%")

             # 还可以比较使用重建查询向量的情况，但这通常不是典型用法
             # recon_query_scores = [compute_similarity(query_vec_recon, dv) for dv in decompressed_vectors]

             # 序列化压缩模型以便在推理阶段使用
             compressor.save('colbert_compressor_bge_m3.pkl')
             print("\n压缩模型已保存，可用于生产环境")
        else:
            print("没有有效的查询向量用于相似度评估。")
    else:
        print("压缩过程未能生成有效数据。")
else:
    print("压缩器训练失败，后续步骤已跳过。")

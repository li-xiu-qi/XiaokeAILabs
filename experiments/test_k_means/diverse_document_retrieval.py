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
from sklearn.metrics.pairwise import cosine_similarity

# 加载模型
# 使用BGE-M3模型进行文本向量化，该模型支持多语言文本编码
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
model = BGEM3FlagModel(model_path, use_fp16=True)  # 使用FP16加速推理

def diversity_enhanced_retrieval(query, doc_vectors, doc_texts, top_k=5, diversity_clusters=3):
    """
    返回多样化的检索结果，覆盖不同语义簇。
    该方法通过K-means聚类确保结果多样性，同时保持相关性。
    
    Args:
        query (str): 查询文本
        doc_vectors (np.ndarray): 文档向量集合
        doc_texts (list): 文档文本列表
        top_k (int): 返回的文档数量
        diversity_clusters (int): 聚类数量，用于增加结果多样性
        
    Returns:
        list: 选择的文档索引列表
    """
    # 编码查询并计算相似度
    # 将查询文本转换为向量表示
    query_vec = model.encode([query])['dense_vecs'][0]
    # 计算查询向量与所有文档向量的余弦相似度
    similarities = cosine_similarity([query_vec], doc_vectors)[0]
    
    # 获取候选文档
    # 选择相似度最高的top_n个文档作为候选集
    top_n = min(top_k * 3, len(doc_vectors))  # 取top_k的3倍或全部文档
    candidate_indices = np.argsort(similarities)[-top_n:][::-1]  # 按相似度降序排列
    candidate_vectors = doc_vectors[candidate_indices]  # 获取候选文档向量
    
    # 执行K-means聚类
    # 对候选文档进行聚类，以确保语义多样性
    kmeans = KMeans(n_clusters=min(diversity_clusters, len(candidate_vectors)), 
                   random_state=42, n_init=10)  # 设置聚类参数
    clusters = kmeans.fit_predict(candidate_vectors)  # 执行聚类并预测簇标签
    
    # 从每个簇中选择最相似文档
    selected_indices = []
    cluster_dict = {}
    
    # 按簇分组并记录相似度
    # 将每个文档按簇ID分组，并保存其原始索引和相似度
    for idx, cluster_id in enumerate(clusters):
        cluster_dict.setdefault(cluster_id, []).append((candidate_indices[idx], similarities[candidate_indices[idx]]))
    
    # 从每个簇中选最佳文档
    # 对每个簇，选择相似度最高的文档
    for cluster_id in range(min(diversity_clusters, len(cluster_dict))):
        if cluster_dict.get(cluster_id):
            best_doc = max(cluster_dict[cluster_id], key=lambda x: x[1])[0]
            selected_indices.append(best_doc)
    
    # 补充不足的文档
    # 如果从聚类中选出的文档数量不足top_k，从剩余候选文档中补充
    remaining = [i for i in candidate_indices if i not in selected_indices]
    if len(selected_indices) < top_k and remaining:
        remaining_similarities = [similarities[i] for i in remaining]
        extra_indices = [remaining[i] for i in np.argsort(remaining_similarities)[-top_k + len(selected_indices):]]
        selected_indices.extend(extra_indices)
    
    return selected_indices[:top_k]

def generate_sample_news_data(n_docs=20):
    """
    生成模拟新闻标题及其向量表示
    
    Args:
        n_docs (int): 需要生成的文档数量
        
    Returns:
        tuple: (文档向量数组, 文档文本列表)
    """
    # 中文新闻标题示例
    news_titles = [
        "人工智能在医学研究中取得重大突破",
        "新兴科技初创公司融资达数十亿",
        "气候变化影响全球农业发展",
        "2025年量子计算技术取得新进展",
        "人工智能模型预测股市趋势",
        "可再生能源超过传统化石燃料",
        "癌症治疗新突破得益于AI技术",
        "科技巨头面临新的隐私法规",
        "全球数据泄露事件创历史新高",
        "AI助手变得更加人性化",
        "自动驾驶汽车革新交通运输",
        "气候技术获得巨额投资支持",
        "新AI算法解决复杂问题",
        "数字化时代网络安全威胁上升",
        "医疗AI减少诊断错误",
        "技术创新推动经济增长",
        "AI驱动的机器人进入职场",
        "可持续技术解决方案受到关注",
        "数据科学改变商业策略",
        "量子AI研究开启新领域"
    ]
    
    # 如果需要更多文档，重复标题列表
    news_titles = news_titles[:n_docs] if len(news_titles) >= n_docs else news_titles * (n_docs // len(news_titles) + 1)
    news_titles = news_titles[:n_docs]
    
    # 生成向量
    # 使用模型将文本转换为向量表示
    news_vectors = model.encode(news_titles)['dense_vecs']
    return news_vectors, news_titles

# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    doc_vectors, doc_texts = generate_sample_news_data(n_docs=20)
    
    # 测试查询
    query = "人工智能研究进展"
    result_indices = diversity_enhanced_retrieval(query, doc_vectors, doc_texts, top_k=5, diversity_clusters=3)
    
    # 打印结果
    print("查询:", query)
    print("\n检索结果:")
    # 计算查询与所有文档的相似度
    similarities = cosine_similarity([model.encode([query])['dense_vecs'][0]], doc_vectors)[0]
    
    # 按相似度降序排序结果
    sorted_results = sorted([(idx, similarities[idx]) for idx in result_indices], 
                           key=lambda x: x[1], reverse=True)
    
    # 打印排序后的结果
    for idx, sim in sorted_results:
        print(f"文档 {idx}: {doc_texts[idx]} (相似度: {sim:.4f})")
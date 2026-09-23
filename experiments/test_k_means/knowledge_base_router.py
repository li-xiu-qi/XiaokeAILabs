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

# 初始化 BGE-M3 模型
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
model = BGEM3FlagModel(model_path, use_fp16=True)

class ClusterAwareRouter:
    """基于聚类的查询路由系统，将查询导向最相关的专业知识库"""
    
    def __init__(self, knowledge_bases, n_clusters=5):
        """
        初始化路由系统
        
        参数:
            knowledge_bases: 字典 {知识库名称: {"vectors": 文档向量, "documents": 文档}}
            n_clusters: 每个知识库的簇数，默认值为5
        """
        self.knowledge_bases = knowledge_bases
        self.kb_centers = {}
        self.n_clusters = n_clusters
        
        # 为每个知识库创建聚类模型
        for kb_name, kb_data in knowledge_bases.items():
            if len(kb_data["vectors"]) < n_clusters:
                raise ValueError(f"知识库 {kb_name} 的向量数量少于指定簇数 {n_clusters}")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(kb_data["vectors"])
            self.kb_centers[kb_name] = kmeans.cluster_centers_
    
    def route_query(self, query, top_k=1):
        """
        将查询路由到最相关的知识库
        
        参数:
            query: 字符串，用户输入的查询
            top_k: 返回前k个最相关知识库，默认为1
            
        返回:
            如果 top_k=1，返回最佳知识库名称和相似度分数 (str, float)
            如果 top_k>1，返回按相似度排序的知识库名称和相似度分数对列表 [(str, float), ...]
        """
        # 将查询编码为向量
        query_vec = model.encode([query], max_length=512)['dense_vecs'][0]
        
        # 计算查询与各知识库簇中心的最大相似度
        similarities = {}
        for kb_name, centers in self.kb_centers.items():
            sim_scores = cosine_similarity([query_vec], centers)[0]
            similarities[kb_name] = np.max(sim_scores)
        
        # 按相似度排序并选择top_k结果
        sorted_kbs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        if top_k == 1:
            return sorted_kbs[0][0], sorted_kbs[0][1]  # 返回最佳知识库名称和相似度分数
        return [(kb[0], kb[1]) for kb in sorted_kbs[:top_k]]  # 返回知识库名称和相似度分数对
    
    def retrieve_documents(self, query, kb_name, top_k=3):
        """
        在指定知识库中检索与查询最相关的文档
        
        参数:
            query: 用户查询
            kb_name: 知识库名称
            top_k: 返回前k个最相关文档，默认为3
            
        返回:
            包含文档和相似度分数的列表，格式为 [(document, similarity_score), ...]
        """
        # 检查知识库是否存在
        if kb_name not in self.knowledge_bases:
            return []
            
        # 将查询编码为向量
        query_vec = model.encode([query], max_length=512)['dense_vecs'][0]
        
        # 获取知识库中的文档向量和文档内容
        kb_vectors = self.knowledge_bases[kb_name]["vectors"]
        kb_documents = self.knowledge_bases[kb_name]["documents"]
        
        # 计算查询与知识库中所有文档的相似度
        similarities = cosine_similarity([query_vec], kb_vectors)[0]
        
        # 获取相似度排名最高的文档索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 返回相关文档及其相似度分数
        return [(kb_documents[i], similarities[i]) for i in top_indices]

# 示例使用
if __name__ == "__main__":
    # 创建示例知识库数据
    sample_docs = {
        "医学": {
            "documents": [
                "糖尿病的治疗方法包括胰岛素治疗和饮食控制。",
                "流感的常见症状包括发热、咳嗽和疲劳。",
                "心脏病预防需要定期锻炼和健康饮食。"
            ],
            "vectors": None
        },
        "技术": {
            "documents": [
                "Python 被广泛用于机器学习和数据分析。",
                "云计算提供了可扩展的基础设施解决方案。",
                "AI模型需要大量的计算资源。"
            ],
            "vectors": None
        }
    }
    
    # 生成向量表示
    for kb_name in sample_docs:
        texts = sample_docs[kb_name]["documents"]
        vectors = model.encode(texts, batch_size=3)['dense_vecs']
        sample_docs[kb_name]["vectors"] = vectors
    
    # 初始化路由器
    router = ClusterAwareRouter(sample_docs, n_clusters=2)
    
    # 测试查询
    test_queries = [
        "感冒的症状有哪些？",
        "AI开发最好的编程语言是什么？"
    ]
    
    # 执行路由
    for query in test_queries:
        best_kb, similarity = router.route_query(query)  # 获取最佳知识库及其相似度
        print(f"查询: {query}")
        print(f"路由到的知识库: {best_kb} (相似度: {similarity:.4f})")
        
        # 检索相关文档并显示
        relevant_docs = router.retrieve_documents(query, best_kb)
        print("检索结果:")
        for i, (doc, score) in enumerate(relevant_docs, 1):
            print(f"{i}. 文档: {doc} (相似度: {score:.4f})")
        print("\n" + "-"*50 + "\n")
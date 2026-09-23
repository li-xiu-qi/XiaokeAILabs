from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba  # 添加jieba导入

# 1. 准备语料库
corpus = [
    "市盈率是衡量股票价格相对于每股收益的指标，计算公式为股票价格除以每股收益。",
    "市净率是股价与每股净资产的比率，常用于评估银行等资产密集型公司价值。",
    "股息收益率是公司年度总派息额与股票现价之比，衡量投资回报的指标。",
    "市销率是股票价格与每股销售收入的比值，适用于评估尚未盈利的成长型公司。",
    "企业价值倍数是企业价值与EBITDA的比率，考虑了公司债务水平的估值指标。",
    "现金流折现模型通过预测未来现金流并折现至今来评估公司内在价值。",
    "技术分析主要关注股票价格和交易量的历史数据，预测未来趋势。",
    "基本面分析关注公司财务状况、管理层质量和市场地位等因素。",
    "投资组合理论主张通过资产多样化来分散风险，优化风险回报比。",
    "被动投资策略通过购买指数基金或ETF来追踪特定市场指数表现。"
]

# 查询和已知的正例
query = "什么是市盈率？如何使用它评估股票价值？"
true_positive = corpus[0]

# 2. 使用jieba分词进行BM25检索
tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]  # 对corpus进行分词
tokenized_query = list(jieba.cut(query))  # 对query进行分词

bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(tokenized_query)

# 获取BM25排序后的文档索引（按相关性从高到低排序）
sorted_indices = np.argsort(bm25_scores)[::-1]  # 降序排序
print("BM25检索结果排序:")
for idx in sorted_indices[:5]:  # 取前5名
    print(f"文档{idx} (得分: {bm25_scores[idx]:.4f}): {corpus[idx][:50]}...")

# 3. 使用Embedding模型进行重排序（稠密检索阶段）
# 加载预训练的Embedding模型
model = SentenceTransformer(r'C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3',trust_remote_code=True)  # 示例模型

# 计算查询和所有文档的嵌入向量
query_embedding = model.encode([query])[0]
corpus_embeddings = model.encode(corpus)

# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]

# 获取嵌入模型排序后的文档索引
sorted_indices_emb = np.argsort(similarities)[::-1]  # 降序排序
print("\n嵌入模型重排序结果:")
for idx in sorted_indices_emb[:5]:  # 取前5名
    print(f"文档{idx} (相似度: {similarities[idx]:.4f}): {corpus[idx][:50]}...")

# 4. 识别难负样本（高相似度但实际不相关的文档）
# 去除真正的正例
hard_negatives_candidates = [idx for idx in sorted_indices_emb if corpus[idx] != true_positive]

# 从候选中选择前N个作为难负样本
hard_negatives = [corpus[idx] for idx in hard_negatives_candidates[:2]]  # 取前2个作为难负样本

# 5. 最终的训练样本结构
training_sample = {
    "query": query,
    "pos": [true_positive],
    "neg": hard_negatives
}

print("\n最终构建的包含难负样本的训练数据:")
print(f"查询: {training_sample['query']}")
print(f"正例: {training_sample['pos'][0]}")
print("难负样本:")
for i, neg in enumerate(training_sample['neg']):
    print(f"  {i+1}. {neg}")
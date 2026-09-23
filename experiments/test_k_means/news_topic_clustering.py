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
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import pandas as pd

# 加载模型
model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
model = BGEM3FlagModel(model_path, use_fp16=True)

# 样本新闻标题集合（涵盖科技、体育、政治、娱乐和健康五个主题）
news_titles = [
    # 科技主题
    "苹果发布最新iPhone 15系列，搭载A17芯片",
    "谷歌推出新一代人工智能助手，支持自然语言理解",
    "特斯拉自动驾驶技术获突破，事故率降低30%",
    "微软宣布收购AI初创公司，强化云服务能力",
    
    # 体育主题
    "梅西在巴黎首秀进球，球迷欢呼雀跃",
    "东京奥运会闭幕，中国队金牌榜位列第二",
    "NBA季后赛：湖人击败热火，夺得总冠军",
    "国足世预赛不敌日本队，出线形势严峻",
    
    # 政治主题
    "中美元首通话，就双边关系交换意见",
    "欧盟通过新气候法案，承诺2050年实现碳中和",
    "联合国大会召开，各国领导人讨论全球治理",
    "英国宣布脱欧后新贸易政策，加强与亚洲合作",
    
    # 娱乐主题
    "新电影《沙丘》全球热映，票房突破4亿美元",
    "流行歌手泰勒·斯威夫特发布新专辑，粉丝热情高涨",
    "网飞热门剧集《鱿鱼游戏》创收视纪录",
    "奥斯卡颁奖典礼举行，《无依之地》获最佳影片",
    
    # 健康主题
    "新研究发现常规锻炼可降低阿尔茨海默病风险",
    "全球新冠疫苗接种突破30亿剂，发展中国家覆盖率仍低",
    "医学专家建议减少超加工食品摄入，降低慢性病风险",
    "心理健康问题在年轻人中上升，专家呼吁加强关注"
]

# 生成文本向量
news_vectors = model.encode(news_titles)['dense_vecs']

# 对向量进行归一化，准备基于余弦相似度的聚类
normalized_vectors = normalize(news_vectors, norm='l2')

# 使用归一化向量进行K-means聚类(等效于基于余弦相似度)
kmeans = KMeans(
    n_clusters=5, 
    init='k-means++',
    n_init=10,
    random_state=42
)
clusters = kmeans.fit_predict(normalized_vectors)

# 创建结果DataFrame
results_df = pd.DataFrame({
    'title': news_titles,
    'cluster': clusters
})

# 打印每个簇的新闻标题
for cluster_id in range(5):
    print(f"\n=== 簇 {cluster_id} ===")
    cluster_titles = results_df[results_df['cluster'] == cluster_id]['title']
    for title in cluster_titles:
        print(f"- {title}")

# 计算聚类评估指标
silhouette_avg = silhouette_score(normalized_vectors, clusters)
print(f"\n聚类轮廓系数: {silhouette_avg:.4f}")
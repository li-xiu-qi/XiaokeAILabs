import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



# 首先尝试加载本地模型
model_path = r'C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3'
model = SentenceTransformer(model_path)
print(f"成功加载本地模型: {model_path}")


# 获取模型的输出维度
embedding_dim = model.get_sentence_embedding_dimension()
print(f"模型输出维度: {embedding_dim}")


# 创建一个SQLite数据库连接
db_path = "vec.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


# 尝试加载sqlite-vec扩展
try:
    conn.enable_load_extension(True)
    # 根据操作系统加载不同的扩展文件
    conn.execute("SELECT load_extension('./vec0.dll')")
    # 测试扩展是否成功加载
    cursor.execute("SELECT vec_version()")
    version = cursor.fetchone()[0]
    print(f"成功加载sqlite-vec扩展，版本: {version}")
except Exception as e:
    print(f"加载sqlite-vec扩展失败: {e}")
    print("将使用纯Python实现向量操作")
    print("如需使用sqlite-vec扩展，请从 https://github.com/asg017/sqlite-vec/releases 下载对应版本")
    
    
    
# 创建表格
cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                    document_id INTEGER PRIMARY KEY,
                    content_embedding FLOAT[{embedding_dim}] DISTANCE_METRIC=cosine,
                    category TEXT,
                    +original_content TEXT
                );
                """)


def insert_documents(cursor, documents):
    # 清空表
    cursor.execute("DELETE FROM vec_documents")
    for doc in documents:
        # 虚拟表插入（如果可用）
        embedding_json = json.dumps(doc["embedding"].tolist())
        try:
            cursor.execute(
                "INSERT INTO vec_documents(document_id, content_embedding, category, original_content) VALUES (?, vec_f32(?), ?, ?)",
                (doc["id"], embedding_json, doc["category"], doc["content"])
            )
        except Exception as e:
            print(f"插入虚拟表失败: {e}")
            
def fetch_vectors_from_db(cursor, doc_ids=None):
    """从数据库中获取向量数据
    
    Args:
        cursor: 数据库游标
        doc_ids: 指定要获取的文档ID列表，默认获取所有文档
        
    Returns:
        包含文档ID、内容、类别和向量的字典列表
    """
    try:
        if doc_ids:
            # 获取特定ID的文档
            placeholders = ','.join(['?'] * len(doc_ids))
            query = f"SELECT document_id, original_content, category, content_embedding FROM vec_documents WHERE document_id IN ({placeholders})"
            cursor.execute(query, doc_ids)
        else:
            # 获取所有文档
            cursor.execute("SELECT document_id, original_content, category, content_embedding FROM vec_documents")
        
        results = []
        for row in cursor.fetchall():
            doc_id, content, category, embedding_blob = row
            # 解析向量数据
            try:
                # 尝试从BLOB中提取向量数据
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                print(f"成功解析向量数据: {embedding}")
                print("embedding的数据类型：",type(embedding))
                results.append({
                    "id": doc_id,
                    "content": content,
                    "category": category,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"解析向量数据失败: {e}")
        
        return results
    except Exception as e:
        print(f"从数据库获取向量失败: {e}")
        return []

# 准备一些示例文本
documents = [
    {"id": 1, "content": "机器学习是人工智能的一个子领域", "category": "技术"},
    {"id": 2, "content": "深度学习是机器学习的一种方法", "category": "技术"},
    {"id": 3, "content": "向量数据库可以高效存储和检索向量数据", "category": "数据库"},
    {"id": 4, "content": "SQLite是一个轻量级的关系型数据库", "category": "数据库"},
    {"id": 5, "content": "Python是一种流行的编程语言", "category": "编程"},
    {"id": 6, "content": "自然语言处理是处理人类语言的技术", "category": "技术"},
    {"id": 7, "content": "向量相似度搜索在推荐系统中很常用", "category": "技术"},
    {"id": 8, "content": "大数据分析需要高效的数据存储和处理", "category": "数据"}
]

# 为每个文档生成embedding向量
for doc in documents:
    embedding = model.encode(doc["content"])
    doc["embedding"] = embedding

# 展示部分数据
for doc in documents[:2]:
    print(f"ID: {doc['id']}, 内容: {doc['content']}")
    print(f"向量维度: {len(doc['embedding'])}, 向量前几个元素: {doc['embedding'][:5]}...\n")
    
    
# 这里是使用向量浮点数插入的
insert_documents(cursor,documents)

# 从数据库获取向量数据并展示
print("\n从数据库获取存储的向量数据:")
db_vectors = fetch_vectors_from_db(cursor)
if db_vectors:
    print(f"成功从数据库获取了 {len(db_vectors)} 条向量数据")
    # 展示前2条数据的向量
    for vec_data in db_vectors[:2]:
        print(f"ID: {vec_data['id']}, 类别: {vec_data['category']}, 内容: {vec_data['content']}")
        if isinstance(vec_data['embedding'], np.ndarray):
            print(f"向量维度: {len(vec_data['embedding'])}, 向量前几个元素: {vec_data['embedding'][:5]}...\n")
        else:
            print(f"向量类型: {type(vec_data['embedding'])}, 无法显示向量内容\n")
else:
    print("未能从数据库获取向量数据")

def knn_search(cursor, query_embedding, k=3, category=None):
    """执行KNN搜索"""
        # 使用vec0虚拟表

    query_json = json.dumps(query_embedding.tolist())
    if category:
        cursor.execute("""
        SELECT document_id, original_content, category, distance
        FROM vec_documents 
        WHERE content_embedding MATCH ? AND k = ? AND category = ?
        """, (query_json, k, category))
    else:
        cursor.execute("""
        SELECT document_id, original_content, category, distance
        FROM vec_documents 
        WHERE content_embedding MATCH ? AND k = ?
        """, (query_json, k))
    
    return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]


# 生成查询向量
query_text = "数据库技术与应用"
query_embedding = model.encode(query_text)


# 执行KNN查询
k = 3
# 试试knn_with_cosine_distance
results = knn_search(cursor, query_embedding, k)
print(f"查询：'{query_text}'的最近{k}个结果（使用vec_distance_cosine函数）:")
for doc_id, content, category, distance in results:
    print(f"ID: {doc_id}, 距离: {distance:.4f}, 类别: {category}, 内容: {content}")
    

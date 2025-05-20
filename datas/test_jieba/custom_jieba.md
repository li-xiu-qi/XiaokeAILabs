### 注意事项与优化

1. **分词优化**：可以根据需要调整jieba的分词模式，如精确模式、全模式或搜索引擎模式
2. **停用词处理**：可以添加中文停用词列表，滤除常见但无意义的词语
3. **自定义词典**：对于特定领域，可以通过jieba的自定义词典功能增强分词效果
4. **索引刷新**：当原始数据更新时，需要重新处理文本并更新索引
5. **性能考虑**：对于大量文本，建议批量处理和索引，以提高性能

### jieba自定义词典的使用

jieba分词器支持自定义词典，可以显著提高特定领域文本的分词准确度。

#### 自定义词典格式

jieba的自定义词典是一个文本文件，每行一个词条，格式为：

词语 词频(可省略) 词性(可省略)

例如：

机器学习 100 n
深度学习 80 n
自然语言处理 60 n
神经网络模型 50 n
DuckDB全文检索 40 n

- 词频越高，该词语被分出来的可能性越高
- 词性是可选的，如n(名词)、v(动词)、adj(形容词)等

#### 加载自定义词典

以下是正确加载和使用自定义词典的方式：

```python
import jieba

# 在任何分词操作之前加载自定义词典
def load_custom_dict():
    # 方式1：从文件加载
    jieba.load_userdict("custom_dict.txt")
    
    # 方式2：动态添加词条
    jieba.add_word("DuckDB全文检索", freq=100, tag='n')
    jieba.add_word("中文分词系统", freq=80)
    
    print("已加载自定义词典")

# 确保在任何分词操作前调用此函数
load_custom_dict()

# 自定义词典加载后的分词效果示例
text = "DuckDB全文检索系统支持中文分词"
seg_list = jieba.cut(text)
print("分词结果: " + " / ".join(seg_list))
```

#### 完整示例：结合自定义词典的中文全文检索

```python
import duckdb
import jieba
import re
import os

# 创建一个简单的自定义词典文件
def create_custom_dict(dict_path):
    """创建一个临时的自定义词典文件"""
    # 确保目录存在
    dir_path = os.path.dirname(os.path.abspath(dict_path))
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write("数据库系统 100 n\n")
        f.write("全文检索 120 n\n")
        f.write("机器学习模型 90 n\n")
        f.write("自然语言处理技术 85 n\n")
        f.write("深度神经网络 95 n\n")
    print(f"自定义词典已创建在 {dict_path}")

def preprocess_chinese_text(text):
    """对中文文本进行分词处理"""
    if not text or not isinstance(text, str):
        return ""
    # 使用精确模式进行分词
    words = jieba.cut(text, cut_all=False)
    # 过滤掉空白和标点符号
    filtered_words = [word for word in words 
                     if word.strip() and not re.match(r'[^\w\u4e00-\u9fff]+', word)]
    return " ".join(filtered_words)

# 创建持久性自定义词典文件
def create_persistent_custom_dict(dict_path):
    """创建一个持久性的自定义词典文件用于jieba分词"""
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(dict_path)), exist_ok=True)
    
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write("数据库系统 100 n\n")
        f.write("全文检索 120 n\n")
        f.write("机器学习模型 90 n\n")
        f.write("自然语言处理技术 85 n\n")
        f.write("深度神经网络 95 n\n")
        f.write("DuckDB全文检索 110 n\n")
        f.write("中文分词系统 80 n\n")
        # 添加更多特定领域词汇
        f.write("数据挖掘算法 75 n\n")
        f.write("语义理解框架 70 n\n")
        f.write("知识图谱构建 85 n\n")
    print(f"持久性自定义词典已创建在 {dict_path}")
    return dict_path

def run_chinese_fts_demo_with_custom_dict():
    # 创建临时词典文件
    dict_path = "temp_custom_dict.txt"
    create_custom_dict(dict_path)
    
    # 在任何分词操作前加载自定义词典
    jieba.load_userdict(dict_path)
    print("已加载jieba自定义词典")
    
    # 测试分词效果
    test_text = "使用DuckDB进行全文检索和自然语言处理技术分析"
    seg_list = jieba.cut(test_text)
    print("使用自定义词典的分词效果: " + " / ".join(seg_list))
    
    # 创建内存数据库连接并执行FTS操作
    conn = duckdb.connect(':memory:')
    try:
        # 安装和加载FTS扩展
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        
        # 创建文档表并插入数据
        conn.execute("CREATE TABLE chinese_docs (id VARCHAR, content VARCHAR)")
        conn.execute("""
        INSERT INTO chinese_docs VALUES
            ('1', '使用DuckDB进行全文检索分析'),
            ('2', '自然语言处理技术应用于搜索引擎'),
            ('3', '深度神经网络在机器学习模型中的应用')
        """)
        
        # 创建预处理表
        conn.execute("CREATE TABLE processed_docs (id VARCHAR, content_processed VARCHAR)")
        
        # 预处理并填充数据
        docs = conn.execute("SELECT * FROM chinese_docs").fetchall()
        for doc in docs:
            doc_id, content = doc
            processed = preprocess_chinese_text(content)
            conn.execute("INSERT INTO processed_docs VALUES (?, ?)", 
                        (doc_id, processed))
        
        # 创建FTS索引
        conn.execute("""
        PRAGMA create_fts_index('processed_docs', 'id', 'content_processed')
        """)
        
        # 搜索示例
        query = "全文检索"
        processed_query = preprocess_chinese_text(query)
        print(f"\n原始查询: '{query}'")
        print(f"处理后查询: '{processed_query}'")
        
        results = conn.execute("""
        SELECT pd.id, cd.content, score
        FROM (
            SELECT *, fts_main_processed_docs.match_bm25(id, ?) AS score
            FROM processed_docs
        ) pd
        JOIN chinese_docs cd ON pd.id = cd.id
        WHERE score IS NOT NULL
        ORDER BY score DESC
        """, (processed_query,)).fetchall()
        
        if results:
            print("\n搜索结果:")
            for res in results:
                print(f"ID: {res[0]}, 内容: {res[1]}, 得分: {res[2]:.6f}")
        else:
            print("\n未找到匹配结果")
            
    finally:
        # 清理资源
        conn.close()
        # 删除临时词典文件
        if os.path.exists(dict_path):
            os.remove(dict_path)
            print(f"\n已删除临时词典文件 {dict_path}")

if __name__ == "__main__":
    run_chinese_fts_demo_with_custom_dict()
```

# 使用持久性自定义词典的完整示例

以下代码展示了如何创建和使用持久性的自定义词典，这对于长期项目更为实用：

```python
import os
import jieba
import duckdb
import re

def create_and_use_persistent_dict():
    # 定义自定义词典路径（使用相对或绝对路径）
    dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_jieba_dict.txt")
    
    # 创建持久性词典
    create_persistent_custom_dict(dict_path)
    
    # 加载词典
    jieba.load_userdict(dict_path)
    print(f"已加载持久性词典: {dict_path}")
    
    # 测试分词效果
    test_texts = [
        "DuckDB全文检索系统支持中文分词和语义理解框架",
        "知识图谱构建需要自然语言处理技术支持",
        "数据挖掘算法在机器学习模型中的应用"
    ]
    
    print("\n分词测试结果:")
    for text in test_texts:
        seg_list = jieba.cut(text)
        print(f"原文: {text}")
        print(f"分词: {' / '.join(seg_list)}")
        print("-"*50)
    
    return dict_path

if __name__ == "__main__":
    # 创建并使用持久性词典
    dict_path = create_and_use_persistent_dict()
    
    print("\n此词典文件可以在后续项目中重复使用")
    print(f"词典位置: {os.path.abspath(dict_path)}")
```

### 自定义词典管理的最佳实践

1. **词典位置**：将自定义词典放在项目易于访问的位置，并使用绝对路径或相对于项目根目录的路径
2. **定期维护**：根据领域需求定期更新词典内容，添加新词条或调整词频
3. **分词质量检测**：定期检查分词结果，确保自定义词典正常工作
4. **版本控制**：将词典文件纳入版本控制系统，跟踪词典的变更历史
5. **备份策略**：对重要的自定义词典建立备份机制

通过以上改进，您可以更有效地管理jieba分词器的自定义词典，提高中文文本处理的精准度。

# 红楼梦知识图谱构建器

这是一个基于NLP技术的红楼梦知识图谱构建工具，能够从红楼梦文本中提取人物、地点、关系等信息，并构建可视化的知识图谱。

## 功能特点

- **实体提取**: 自动识别人物、地点、物品等实体
- **关系抽取**: 基于规则和模式匹配提取实体间关系
- **LLM增强**: 支持使用大语言模型提升提取质量（可选）
- **图谱构建**: 构建NetworkX格式的知识图谱
- **可视化**: 生成美观的图谱可视化结果
- **分析功能**: 提供中心性分析、类型分布等统计信息

## 文件结构

```
test_hong_lou_meng_kg/
├── 红楼梦.txt                    # 红楼梦原文（必需）
├── build_hongloumeng_kg.py       # 主要的知识图谱构建器
├── demo_hongloumeng_kg.py        # 演示脚本
├── simple_semantic_splitter.py   # 简化版文本分割器
├── embedding_models.py           # 嵌入模型封装
├── fallback_openai_client.py     # LLM客户端
├── requirements_kg.txt           # 依赖包列表
├── .env                          # 环境变量配置
└── README.md                     # 本文件
```

## 安装依赖

```bash
# 安装Python依赖
pip install -r requirements_kg.txt

# 安装jieba分词
pip install jieba

# 如果需要使用spacy（可选）
pip install spacy
```

## 快速开始

### 1. 基础使用（不需要LLM）

```bash
# 运行基础演示
python demo_hongloumeng_kg.py basic
```

这将：
- 使用基础NLP方法提取实体和关系
- 构建知识图谱
- 生成可视化结果
- 保存分析结果到 `demo_kg_output` 目录

### 2. 高级使用（需要LLM API）

首先配置环境变量（在`.env`文件中）：

```env
# 智谱AI配置（主要）
ZHIPU_API_KEY=your_zhipu_api_key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
ZHIPU_MODEL=glm-4-flash

# 硅基流动配置（备用）
GUIJI_API_KEY=your_guiji_api_key
GUIJI_BASE_URL=https://api.siliconflow.cn/v1
GUIJI_MODEL=THUDM/GLM-4-9B-0414
```

然后运行高级演示：

```bash
python demo_hongloumeng_kg.py advanced
```

### 3. 编程使用

```python
import asyncio
from build_hongloumeng_kg import HongLouMengKGBuilder

async def build_kg():
    # 创建构建器
    kg_builder = HongLouMengKGBuilder("红楼梦.txt")
    
    # 构建完整知识图谱
    await kg_builder.build_complete_kg(
        use_llm=True,  # 是否使用LLM
        output_dir="my_kg_output"
    )

# 运行
asyncio.run(build_kg())
```

## 输出结果

构建完成后，会在输出目录生成以下文件：

- `entities.json`: 提取的实体信息
- `relations.json`: 提取的关系信息
- `knowledge_graph.gexf`: 图谱文件（可用Gephi等工具打开）
- `analysis.json`: 图谱分析结果
- `hongloumeng_kg.png`: 图谱可视化图片

## 核心类说明

### HongLouMengKGBuilder

主要的知识图谱构建器类。

#### 主要方法：

- `extract_entities_basic()`: 基础实体提取
- `extract_entities_with_llm()`: LLM增强实体提取
- `extract_relations_basic()`: 基础关系提取
- `build_graph()`: 构建图谱
- `analyze_graph()`: 分析图谱
- `visualize_graph()`: 可视化图谱
- `save_results()`: 保存结果
- `build_complete_kg()`: 完整构建流程

#### 预定义实体：

- **人物**: 贾宝玉、林黛玉、薛宝钗、王熙凤等主要人物
- **地点**: 大观园、怡红院、潇湘馆、荣国府等重要场所

#### 关系类型：

- **家庭关系**: 父子、母子、夫妻、兄弟姐妹等
- **社会关系**: 主仆、朋友、师生等
- **情感关系**: 恋人、知己、情敌等
- **位置关系**: 居住、拜访、经过等

## 分析功能

- **度中心性**: 识别图谱中的核心人物
- **介数中心性**: 找出起桥梁作用的关键人物
- **实体类型分布**: 统计各类实体的数量
- **关系类型分布**: 统计各类关系的频率

## 注意事项

1. **文本文件**: 确保`红楼梦.txt`文件在运行目录下
2. **编码**: 文本文件使用UTF-8编码
3. **内存**: 处理大文本时可能需要较多内存
4. **API配置**: 使用LLM功能需要正确配置API密钥
5. **可视化**: 生成的图片可能需要中文字体支持

## 扩展功能

### 自定义实体

可以在`HongLouMengKGBuilder`类中添加自定义的实体列表：

```python
# 添加自定义人物
kg_builder.predefined_characters.update({"新人物1", "新人物2"})

# 添加自定义地点
kg_builder.predefined_places.update({"新地点1", "新地点2"})
```

### 自定义关系模式

可以添加新的关系提取模式：

```python
# 在extract_relations_basic方法中添加新模式
custom_patterns = [
    (r'(\w+)喜欢(\w+)', "喜欢"),
    (r'(\w+)讨厌(\w+)', "讨厌"),
]
```

## 问题排查

### 常见问题：

1. **找不到红楼梦.txt**: 检查文件是否在正确目录
2. **中文显示问题**: 安装中文字体或设置matplotlib字体
3. **LLM连接失败**: 检查API密钥和网络连接
4. **内存不足**: 减少处理的文本块数量

### 日志信息：

程序会输出详细的进度信息，包括：
- ✅ 成功操作
- ⚠️ 警告信息
- ❌ 错误信息
- 📊 统计数据

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目遵循MIT许可证。

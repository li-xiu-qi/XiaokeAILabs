# 问答对生成器

这个工具用于从文档中生成高质量的问答对，可用于训练和评估检索增强生成（RAG）系统。生成的问答对会被评估三个维度的质量：

1. **扎根性（Groundedness）**：问题是否可以从给定的上下文中明确回答？
2. **相关性（Relevance）**：问题对用户是否相关？
3. **独立性（Stand-alone）**：问题在没有任何上下文的情况下是否可以理解？

## 安装

```bash
pip install openai sentence-transformers tqdm pandas datasets python-dotenv
```

## 使用方法

### 基本用法

```bash
python qa_generator.py
```

这将使用默认设置（Hugging Face数据集、中文提示、10个问答对）生成问答对。

### 自定义选项

```bash
python qa_generator.py \
  --model "THUDM/GLM-4-32B-0414" \
  --embedding_model "BAAI/bge-m3" \
  --dataset_name "m-ric/huggingface_doc" \
  --n_generations 20 \
  --output_file "my_qa_pairs.json"
```

### 使用本地数据集

```bash
python qa_generator.py \
  --dataset_path "path/to/your/documents.txt" \
  --n_generations 5
```

### 所有可用选项

```
--model               OpenAI模型名称
--embedding_model     用于嵌入的模型名称
--api_key             OpenAI API密钥（也可通过环境变量设置）
--base_url            OpenAI API基础URL（也可通过环境变量设置）
--dataset_path        本地数据集路径
--dataset_name        Hugging Face数据集名称
--n_generations       要生成的问答对数量
--output_file         输出文件路径
--use_english         使用英文提示（默认为中文）
--chunk_size          文本块大小
--chunk_overlap       文本块重叠大小
--skip_evaluation     跳过评估步骤
```

## 环境变量

可以通过环境变量或`.env`文件设置API密钥：

```
API_KEY=your_api_key
BASE_URL=https://api.example.com/v1
```

## 输出格式

生成的JSON文件包含问答对及其评估结果：

```json
[
  {
    "context": "文档内容...",
    "question": "生成的问题?",
    "answer": "答案",
    "source_doc": "文档来源",
    "groundedness_score": 5,
    "groundedness_eval": "评价理由...",
    "relevance_score": 4,
    "relevance_eval": "评价理由...",
    "standalone_score": 5,
    "standalone_eval": "评价理由..."
  },
  ...
]
```

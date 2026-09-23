# spaCy 中文分句模型微调

用 spaCy 训练一个中文句子边界检测（senter）模型。中文分句没有西文那种可靠的句号空格规则，标点歧义多（小数点、引号、省略号都可能不是句尾），标准模型效果不够时值得自己训一个。

## 文件

| 文件 | 作用 |
|---|---|
| `data/gen_jieba_data.py` | 用 jieba 生成分句训练数据 |
| `data/jsonl_to_spacy.py` | 训练数据从 jsonl 转 spaCy 训练格式 |
| `data/senter_train.jsonl`、`data/senter_train.spacy` | 训练语料（两种格式） |
| `data/show_spacy_content.py` | 检查训练数据内容 |
| `spacy_object_examples.py` | spaCy 对象与 pipeline 概念演示 |
| `check_cuda_version.py` | GPU 可用性检查（训练用） |
| `load_trained_model.py` | 加载训练好的模型，对比标准模型与 transformer 结构模型的分句差异 |
| `eval_spacy_model.py` | 模型效果评估 |

## 流程

准备数据（jieba 切分标注）转格式、用 `spacy train` 训练 senter 组件、`load_trained_model.py` 看效果、`eval_spacy_model.py` 量化评估。依赖见 `requirements.txt`，GPU 训练前先跑 `check_cuda_version.py`。

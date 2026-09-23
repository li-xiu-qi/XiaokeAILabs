import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training import Example

# 加载训练好的 spaCy 分句模型
nlp = spacy.load("output_model/model-best")
# 加载标注好的分句数据（.spacy 格式）
doc_bin = DocBin().from_disk("data/senter_train.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

scorer = Scorer()  # 初始化评测器
examples = []  # 存放 Example 对象
for doc in docs:
    pred_doc = nlp(doc.text)  # 用模型对原文重新分句
    example = Example(pred_doc, doc)  # 构造评测用的 Example 对象
    examples.append(example)
# 评测所有 Example，返回各项指标
results = scorer.score(examples)
print("分句评测结果：")
print(results)
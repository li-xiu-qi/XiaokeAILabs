import spacy
from spacy.tokens import DocBin

# 读取 .spacy 文件并打印每个 doc 的句子分割结果
nlp = spacy.blank("zh")
doc_bin = DocBin().from_disk("senter_train.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

for i, doc in enumerate(docs):
    print(f"Doc {i+1}:")
    for sent in doc.sents:
        print(f"  句子: {sent.text}")
    print("-"*20)

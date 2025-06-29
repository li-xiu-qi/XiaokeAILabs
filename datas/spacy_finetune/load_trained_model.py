import spacy

# 加载标准模型
test_text = "今天天气很好。我想去公园。"
nlp = spacy.load("output_model/model-best")
doc = nlp(test_text)  # 直接用pipeline处理原始文本
print("标准模型分句结果（spaCy自带分词+完整pipeline）：")
print("Pipeline 组件：", nlp.pipe_names)
for token in doc:
    print(token.text, token.is_sent_start)
print("\n分句结果：")
for sent in doc.sents:
    print(sent.text)

# 加载 transformer 结构训练的模型
nlp_trf = spacy.load("output_model_trf/model-best")
doc_trf = nlp_trf(test_text)  # 直接用pipeline处理原始文本
print("\nTransformer 模型分句结果（spaCy自带分词+完整pipeline）：")
print("Pipeline 组件：", nlp_trf.pipe_names)
for token in doc_trf:
    print(token.text, token.is_sent_start)
print("\n分句结果：")
for sent in doc_trf.sents:
    print(sent.text)

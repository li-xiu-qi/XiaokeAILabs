import spacy

# 加载模型
nlp = spacy.load("output_model_trf/model-best")

# nlp 对象：可以直接处理原始文本
text = "小明喜欢看书，也喜欢运动。你呢？"
doc = nlp(text)

# pipe_names：查看 pipeline 组件
print("Pipeline 组件：", nlp.pipe_names)  # 例如 ['transformer', 'senter']

# doc 对象：结构化文本，支持遍历 token
print("所有 token：")
for token in doc:
    print(f"{token.text}\t是否句子起始: {token.is_sent_start}")

# token 对象：每个 token 的属性
first_token = doc[0]
print(f"第一个 token: {first_token.text}, 是否句子起始: {first_token.is_sent_start}")

# sents：遍历分句结果
print("\n分句结果：")
for sent in doc.sents:
    print(sent.text)

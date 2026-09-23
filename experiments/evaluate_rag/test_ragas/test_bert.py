from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 定义两个句子
sentence1 = "我喜欢看电影。"
sentence2 = "我很享受看电影。"

# 对句子进行编码
inputs1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True, max_length=128)
inputs2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True, max_length=128)

# 获取 BERT 输出
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# 提取 [CLS] token 的表示（也可以用 mean pooling）
cls_embedding1 = outputs1.last_hidden_state[:, 0, :].squeeze().numpy()
cls_embedding2 = outputs2.last_hidden_state[:, 0, :].squeeze().numpy()

# 计算余弦相似度
cos_sim = np.dot(cls_embedding1, cls_embedding2) / (np.linalg.norm(cls_embedding1) * np.linalg.norm(cls_embedding2))
print(f"两个句子的余弦相似度: {cos_sim:.4f}")
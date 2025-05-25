import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

"""
SimCSE (Simple Contrastive Sentence Embedding) 是一种通过对比学习改进句子嵌入的方法。
核心思想：
1. 无监督学习：利用同一句子通过不同dropout mask生成的两个表示作为正样本对
2. 同一批次中的其他句子作为负样本
3. 训练目标是使正样本对的表示相似，而与负样本的表示不相似
"""

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 准备输入句子 - 有意设计了语义相似和不相似的句子对
sentences = [
    "市盈率是衡量股票价格相对于每股收益的指标。",  # 金融指标相关
    "P/E比率用于评估股票估值的合理性。",         # 与第一句语义相似
    "通货膨胀是物价持续上涨的经济现象。",        # 经济现象，与前两句相关但不同概念
    "每股收益是公司净利润除以流通股数。"         # 与第一句相关，都涉及每股收益
]

def get_sentence_embeddings(model, tokenizer, sentences, use_simcse=False):
    """获取句子嵌入，可选是否使用SimCSE方法"""
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    if use_simcse:
        # SimCSE方法：对每个句子使用不同dropout生成两个表示，然后取平均
        model.train()  # 激活dropout
        # 运行两次获取不同的表示
        outputs1 = model(**inputs, output_hidden_states=True)
        outputs2 = model(**inputs, output_hidden_states=True)
        # 取CLS token
        embeddings1 = outputs1.last_hidden_state[:, 0]
        embeddings2 = outputs2.last_hidden_state[:, 0]
        # 取平均作为最终表示
        embeddings = (embeddings1 + embeddings2) / 2
    else:
        # 传统方法：直接获取句子表示
        model.eval()  # 关闭dropout
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state[:, 0]
    
    return embeddings

# 演示SimCSE训练过程
def demonstrate_simcse_training():
    print("=== SimCSE训练过程演示 ===")
    # 将句子转换为模型输入，同一批次输入两次（以使用不同的dropout mask）
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    inputs_repeated = {k: torch.cat([v, v]) for k, v in inputs.items()}
    
    # 前向传播，获取CLS表示
    model.train()  # 确保dropout被激活
    outputs = model(**inputs_repeated, output_hidden_states=True)
    last_hidden = outputs.last_hidden_state
    cls_embeds = last_hidden[:, 0]  # 取CLS token表示
    
    # 分开原始样本和重复样本的表示
    batch_size = len(sentences)
    z1, z2 = torch.split(cls_embeds, batch_size)
    
    # 计算余弦相似度
    cosine_sim = torch.nn.functional.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2)
    
    # 计算对比学习损失（InfoNCE/NT-Xent）
    # 对角线上的元素代表相同句子的两个不同表示之间的相似度（正样本）
    # 非对角线元素代表不同句子之间的相似度（负样本）
    # 训练目标是最大化对角线元素的值
    labels = torch.arange(batch_size).to(cosine_sim.device)
    temperature = 0.05  # 温度参数，控制分布的平滑程度
    loss = torch.nn.CrossEntropyLoss()(cosine_sim / temperature, labels)
    
    print(f"SimCSE 对比损失：{loss.item():.4f}")
    print("余弦相似度矩阵（训练中）：")
    print(cosine_sim.detach().numpy())
    print("对角线元素（正样本对）平均相似度：", torch.mean(torch.diag(cosine_sim)).item())
    print("非对角线元素（负样本对）平均相似度：", 
          (torch.sum(cosine_sim) - torch.sum(torch.diag(cosine_sim))) / (batch_size * batch_size - batch_size))

# 比较使用SimCSE和不使用SimCSE的句子嵌入效果
def compare_embeddings():
    print("\n=== 比较传统嵌入与SimCSE嵌入效果 ===")
    
    # 获取传统句子嵌入
    traditional_embeddings = get_sentence_embeddings(model, tokenizer, sentences, use_simcse=False)
    traditional_embeddings = traditional_embeddings.detach().numpy()
    
    # 获取SimCSE增强的句子嵌入
    simcse_embeddings = get_sentence_embeddings(model, tokenizer, sentences, use_simcse=True)
    simcse_embeddings = simcse_embeddings.detach().numpy()
    
    # 计算相似度矩阵
    traditional_sim = cosine_similarity(traditional_embeddings)
    simcse_sim = cosine_similarity(simcse_embeddings)
    
    # 显示结果
    print("传统方法的相似度矩阵：")
    print(np.round(traditional_sim, 3))
    
    print("\nSimCSE方法的相似度矩阵：")
    print(np.round(simcse_sim, 3))
    
    print("\n句子对的语义关系:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            print(f"句子{i+1}与句子{j+1}:")
            print(f"  - 传统相似度: {traditional_sim[i,j]:.3f}")
            print(f"  - SimCSE相似度: {simcse_sim[i,j]:.3f}")
            print(f"  - 句子{i+1}: {sentences[i]}")
            print(f"  - 句子{j+1}: {sentences[j]}")
            print()

# 运行演示
if __name__ == "__main__":
    demonstrate_simcse_training()
    compare_embeddings()
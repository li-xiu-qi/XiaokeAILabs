import os
import sentencepiece as spm
import random
from pathlib import Path

def create_sample_corpus(file_path, num_samples=1000):
    """
    创建一个示例语料库文件用于训练
    
    Args:
        file_path: 语料库文件路径
        num_samples: 样本数量
    """
    print(f"创建示例语料库文件: {file_path}")
    
    # 准备一些中英文混合的示例文本
    samples_zh = [
        "自然语言处理是人工智能的重要分支。",
        "深度学习模型在机器翻译任务中表现出色。",
        "词向量能够捕捉词语之间的语义关系。",
        "注意力机制是Transformer模型的核心组件。",
        "预训练语言模型大大提高了下游任务的性能。",
        "中文分词是处理中文文本的第一步。",
        "命名实体识别可以从文本中提取人名、地名和组织名等信息。",
        "语义理解是自然语言处理中的关键挑战。",
        "情感分析可以判断文本表达的情感倾向。",
        "机器阅读理解测试模型对文本的理解能力。",
        # 增加更多多样性的中文样本
        "大模型时代，计算资源成为关键瓶颈。",
        "知识图谱结合神经网络可以增强推理能力。",
        "多模态学习融合文本、图像和语音信息。",
        "生成式AI正在改变内容创作的方式。",
        "对抗训练可以提高模型的鲁棒性。",
        "迁移学习减少了对大规模标注数据的需求。",
        "强化学习通过奖励信号指导模型行为。",
        "小样本学习让模型能够从少量数据中学习。",
        "可解释性AI帮助理解模型的决策过程。",
        "联邦学习保护用户隐私的同时实现模型训练。"
    ]
    
    samples_en = [
        "Natural Language Processing is a subfield of AI.",
        "Deep learning models perform well on machine translation tasks.",
        "Word embeddings capture semantic relationships between words.",
        "Attention mechanism is a core component of Transformer models.",
        "Pre-trained language models significantly improve downstream tasks.",
        "Chinese word segmentation is the first step in processing Chinese text.",
        "Named Entity Recognition extracts information like names, places and organizations.",
        "Semantic understanding is a key challenge in NLP.",
        "Sentiment analysis determines the emotional tone of a text.",
        "Machine reading comprehension tests a model's ability to understand text.",
        # 增加更多多样性的英文样本
        "Large language models have revolutionized the field of AI.",
        "Reinforcement learning from human feedback improves alignment.",
        "Prompt engineering is becoming an essential skill for AI practitioners.",
        "Fine-tuning adapts pre-trained models to specific domains.",
        "Knowledge distillation transfers knowledge from larger to smaller models.",
        "Retrieval-augmented generation improves factuality in LLMs.",
        "Model quantization reduces computational requirements without significant performance loss.",
        "Multimodal models can understand and generate both text and images.",
        "Few-shot learning enables models to learn from a small number of examples.",
        "Self-supervised learning leverages unlabeled data for pre-training."
    ]
    
    # 生成更多样本并增加随机变化
    all_samples = []
    for _ in range(num_samples):
        # 随机选择中文或英文样本或混合样本
        if random.random() < 0.4:
            sample = random.choice(samples_zh)
            # 随机替换一些标点符号增加多样性
            if random.random() < 0.3:
                sample = sample.replace("。", "!")
            if random.random() < 0.3:
                sample = sample.replace("，", ";")
        elif random.random() < 0.8:
            sample = random.choice(samples_en)
            # 随机替换一些标点符号增加多样性
            if random.random() < 0.3:
                sample = sample.replace(".", "?")
            if random.random() < 0.3:
                sample = sample.replace(",", ":")
        else:
            # 创建中英混合文本，增加混合模式的多样性
            zh_part = random.choice(samples_zh).split("。")[0]
            en_part = random.choice(samples_en).split(".")[0]
            
            mix_pattern = random.randint(0, 3)
            if mix_pattern == 0:
                sample = f"{zh_part}，{en_part}。"
            elif mix_pattern == 1:
                sample = f"{en_part}. {zh_part}。"
            elif mix_pattern == 2:
                sample = f"{zh_part}（{en_part}）。"
            else:
                sample = f"{en_part} - {zh_part}。"
        
        # 增加数字和特殊字符
        if random.random() < 0.2:
            sample += f" V{random.randint(1, 10)}.{random.randint(0, 9)}"
        if random.random() < 0.1:
            sample += f" #{random.randint(100, 999)}"
            
        all_samples.append(sample)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(sample + '\n')
    
    print(f"已写入 {len(all_samples)} 个样本到文件")
    return file_path

def train_bpe_model(corpus_file, model_prefix, vocab_size=1000, model_type="bpe", character_coverage=0.9995):
    """
    训练BPE模型
    
    Args:
        corpus_file: 语料库文件路径
        model_prefix: 模型名称前缀
        vocab_size: 词汇表大小
        model_type: 模型类型，这里使用"bpe"
        character_coverage: 字符覆盖率
    """
    print(f"开始训练BPE模型: {model_prefix}")
    
    # 设置训练参数
    train_args = {
        'input': corpus_file,                # 训练语料路径
        'model_prefix': model_prefix,        # 模型前缀
        'vocab_size': vocab_size,            # 词汇表大小 (降低为1000)
        'character_coverage': character_coverage,  # 字符覆盖率
        'model_type': model_type,            # 模型类型：BPE
        'input_sentence_size': 100000,       # 训练中使用的句子数上限
        'shuffle_input_sentence': True,      # 打乱输入句子
        'seed_sentencepiece_size': 1000,     # 用于训练sentencepiece模型的句子数量
        'pad_id': 0,                         # PAD的ID
        'unk_id': 1,                         # UNK的ID
        'bos_id': 2,                         # BOS (beginning of sentence)的ID
        'eos_id': 3,                         # EOS (end of sentence)的ID
        'user_defined_symbols': ['<mask>'],  # 自定义符号，用于掩码语言模型
        # 添加更多训练参数增加模型稳定性
        'max_sentence_length': 4192,         # 最大句子长度
        'normalization_rule_name': 'nmt_nfkc', # 标准化规则
        'num_threads': 4,                    # 使用更多线程加速训练
        'allow_whitespace_only_pieces': True, # 允许只包含空白的片段
    }
    
    # 尝试训练模型，如果失败则降低词汇量再试
    try:
        spm.SentencePieceTrainer.train(**train_args)
    except RuntimeError as e:
        if "Vocabulary size too high" in str(e):
            # 从错误消息中提取建议的最大词汇表大小
            import re
            match = re.search(r"Please set it to a value <= (\d+)", str(e))
            if match:
                new_vocab_size = int(match.group(1))
                print(f"词汇表大小过大，自动调整为 {new_vocab_size}")
                train_args['vocab_size'] = new_vocab_size
                spm.SentencePieceTrainer.train(**train_args)
            else:
                # 如果无法提取，降低为原来的一半再试
                train_args['vocab_size'] = vocab_size // 2
                print(f"词汇表大小过大，降低为 {train_args['vocab_size']}")
                spm.SentencePieceTrainer.train(**train_args)
        else:
            raise e
    
    print(f"模型训练完成，模型文件保存在: {model_prefix}.model")
    print(f"词汇表文件保存在: {model_prefix}.vocab")
    
    return f"{model_prefix}.model"

def test_bpe_model(model_path):
    """
    测试训练好的BPE模型
    
    Args:
        model_path: 模型文件路径
    """
    print(f"加载模型: {model_path}")
    
    # 加载模型
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # 准备测试文本
    test_texts = [
        "自然语言处理是人工智能的重要分支。",
        "This is a mixed English and 中文 text for testing.",
        "BPE算法在多语言处理中非常有效。",
        "Transformers模型在NLP任务中表现出色。",
        "这个词汇表中应该没有的稀有词语：XYZ123和αβγ"
    ]
    
    print("开始测试分词效果:")
    
    for text in test_texts:
        # 分词
        tokens = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        
        # 打印结果
        print("\n原文:", text)
        print("分词:", tokens)
        print("ID序列:", ids)
        
        # 重建文本
        decoded = sp.decode_pieces(tokens)
        print("重建文本:", decoded)
        print("重建是否与原文一致:", decoded == text)
        
    # 打印词汇表信息
    print(f"\n词汇表大小: {sp.get_piece_size()}")
    print("特殊标记:")
    special_tokens = ["<unk>", "<s>", "</s>", "<mask>"]
    for token in special_tokens:
        token_id = sp.piece_to_id(token)
        print(f"  {token}: ID={token_id}")
    
    # 打印一些词汇表中的词条
    print("\n词汇表示例 (前20个):")
    for i in range(min(20, sp.get_piece_size())):
        piece = sp.id_to_piece(i)
        print(f"  ID {i}: {piece}")

def main():
    # 设置路径
    current_dir = Path(__file__).parent
    data_dir = current_dir / "bpe_model"  # 简化路径结构
    os.makedirs(data_dir, exist_ok=True)
    
    # 文件路径
    corpus_file = data_dir / "bpe_training_corpus.txt"
    model_prefix = data_dir / "bpe_model"
    
    # 创建语料库 - 增加样本数量提高多样性
    create_sample_corpus(corpus_file, num_samples=5000)
    
    # 训练模型 - 降低词汇表大小到1000
    model_path = train_bpe_model(
        corpus_file=str(corpus_file),
        model_prefix=str(model_prefix),
        vocab_size=1000,  # 降低词汇表大小
        model_type="bpe",
        character_coverage=0.9995
    )
    
    # 测试模型
    test_bpe_model(model_path)

if __name__ == "__main__":
    main()

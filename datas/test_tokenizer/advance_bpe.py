"""
BPE分词器的实现模块

本模块实现了基于字节对编码(Byte Pair Encoding, BPE)的分词器，
提供了训练、加载和测试BPE分词器的功能。
BPE是一种常用的次词级(subword)分词算法，能够有效处理未登录词问题。
"""

# 导入必要的库
from tokenizers import Tokenizer  # 基础分词器类，提供分词器的核心功能
from tokenizers.models import BPE  # BPE模型实现
from tokenizers.trainers import BpeTrainer  # BPE训练器，用于训练BPE模型
from tokenizers.pre_tokenizers import Whitespace  # 预分词器，基于空白符划分文本
from tokenizers.processors import TemplateProcessing  # 后处理器，用于处理分词结果
import os  # 操作系统接口，用于文件路径操作


def train_bpe_tokenizer(corpus_file, vocab_size=2000, min_frequency=2, save_path="test-my-bpe-tokenizer.json",
                        show_progress=True, continuing_subword_prefix="##", end_of_word_suffix="",
                        max_token_length=None, initial_alphabet=None, limit_alphabet=None,
                        dropout=None, fuse_unk=False):
    """
    训练一个BPE分词器
    
    该函数从语料文件中训练BPE分词器，设置必要的特殊tokens，配置预处理和后处理器，
    最后保存训练好的分词器到指定路径。
    
    参数:
        corpus_file: 训练语料文件的路径，文件应包含用于训练的文本语料
        vocab_size: 词汇表大小，默认为2000，决定了最终词汇表中的token数量
        min_frequency: 最小词频，默认为2，低于此频率的token不会被加入词汇表
        save_path: 保存模型的路径，默认为"test-my-bpe-tokenizer.json"
        show_progress: 是否显示训练进度条，默认为True
        continuing_subword_prefix: 子词前缀，用于标记单词中间的子词，默认为"##"
        end_of_word_suffix: 单词结尾的后缀，默认为空字符串""
        max_token_length: token的最大长度限制，默认为None
        initial_alphabet: 初始字母表，默认为None，使用系统默认
        limit_alphabet: 限制字母表大小，默认为None，不限制
        dropout: 合并操作的随机丢弃率，用于子词正则化，默认为None
        fuse_unk: 是否将未知token融合为单个token，默认为False
    
    返回:
        训练好的tokenizer对象
    """
    # 初始化BPE分词器，设置未知token为[UNK]
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 配置预分词器 - 使用空白符将文本分割为初始tokens
    # 对中文，可以考虑使用字符级分词，但此处使用Whitespace作为简单示例
    tokenizer.pre_tokenizer = Whitespace()
    
    # 准备BpeTrainer参数，只包含非None值
    trainer_kwargs = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        "show_progress": show_progress,
        "continuing_subword_prefix": continuing_subword_prefix,
        "fuse_unk": fuse_unk
    }
    
    # 为end_of_word_suffix提供默认的空字符串而不是None
    if end_of_word_suffix is not None:
        trainer_kwargs["end_of_word_suffix"] = end_of_word_suffix
    
    # 只有在提供了非None值时，才添加这些可选参数
    if max_token_length is not None:
        trainer_kwargs["max_token_length"] = max_token_length
    
    if initial_alphabet is not None:
        trainer_kwargs["initial_alphabet"] = initial_alphabet
        
    if limit_alphabet is not None:
        trainer_kwargs["limit_alphabet"] = limit_alphabet
        
    if dropout is not None:
        trainer_kwargs["dropout"] = dropout
    
    # 使用筛选后的参数创建BpeTrainer
    trainer = BpeTrainer(**trainer_kwargs)
    
    # 对语料库进行训练，学习BPE合并规则
    tokenizer.train(files=[corpus_file], trainer=trainer)
    
    # 添加后处理步骤，配置BERT风格的输入格式
    # 单句输入格式：[CLS] 句子内容 [SEP]
    # 双句输入格式：[CLS] 句子1 [SEP] 句子2 [SEP]
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",  # 单句模板
        pair="[CLS] $A [SEP] $B [SEP]",  # 双句模板
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),  # CLS标记对应的ID
            ("[SEP]", tokenizer.token_to_id("[SEP]")),  # SEP标记对应的ID
        ],
    )
    
    # 将训练好的分词器保存到指定路径
    tokenizer.save(save_path)
    print(f"分词器已保存至: {save_path}")
    
    return tokenizer


def load_tokenizer(path="test-my-bpe-tokenizer.json"):
    """
    加载已保存的分词器
    
    从指定路径加载先前训练并保存的分词器模型。
    
    参数:
        path: 分词器文件的路径，默认为"test-my-bpe-tokenizer.json"
    
    返回:
        加载的tokenizer对象
    """
    # 从文件加载分词器模型
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


def test_tokenizer(tokenizer, text):
    """
    测试分词器
    
    使用给定的分词器对输入文本进行编码和解码，展示分词结果。
    
    参数:
        tokenizer: 分词器实例，用于执行分词操作
        text: 要分词的文本，作为测试输入
    
    返回:
        tuple: (tokens, ids) - 分词后的token列表和对应的ID列表
    """
    # 对输入文本进行编码，将文本转换为tokens和ids
    encoded = tokenizer.encode(text)
    tokens = encoded.tokens  # 获取分词后的token列表
    ids = encoded.ids  # 获取token对应的ID列表
    
    # 输出分词结果
    print(f"输入文本: {text}")
    print(f"分词结果: {tokens}")
    print(f"Token IDs: {ids}")
    
    # 将token ID解码回文本，验证解码功能
    decoded = tokenizer.decode(ids)
    print(f"解码结果: {decoded}")
    
    return tokens, ids


# 主程序入口
if __name__ == "__main__":
    # 设置语料文件路径
    corpus_path = "test_corpus.txt"
    
    # 检查语料文件是否存在
    if os.path.exists(corpus_path):
        # 语料文件存在，训练新的分词器
        tokenizer = train_bpe_tokenizer(corpus_path)
        
        # 使用示例中文句子测试分词器效果
        test_text = "低温会使硅基芯片性能降低。高温会使碳基芯片性能提高。"
        test_tokenizer(tokenizer, test_text)
    else:
        # 语料文件不存在，输出错误信息
        print(f"语料文件不存在: {corpus_path}")
        

'''
分词器已保存至: test-my-bpe-tokenizer.json
输入文本: 低温会使硅基芯片性能降低。高温会使碳基芯片性能提高。
分词结果: ['[CLS]', '低', '##温会使', '##硅', '##基芯片性能', '##降', '##低', '。', '高', '##温会使', '##碳', '##基芯片性能', '##提', '##高', '。', '[SEP]']
Token IDs: [1, 7, 37, 31, 39, 32, 33, 5, 19, 37, 23, 39, 29, 30, 5, 2]
解码结果: 低 ##温会使 ##硅 ##基芯片性能 ##降 ##低 。 高 ##温会使 ##碳 ##基芯片性能 ##提 ##高 。
'''
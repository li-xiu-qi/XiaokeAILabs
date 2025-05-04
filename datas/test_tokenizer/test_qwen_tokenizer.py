# filepath: c:\Users\k\Documents\project\programming_project\md_project\文章\原始版本\test_part2_bpe_advanced.md
# -*- coding: utf-8 -*-
"""
分词器测试模块

本模块用于测试不同的分词器实现，包括：
1. 自定义训练的BPE分词器 (见第一部分)
2. 清华大学开发的Qwen(通义千问)分词器

通过本模块可以比较不同分词器对中文和英文文本的处理效果。
"""

# 确保已安装所需库:
# pip install transformers sentencepiece tiktoken -i https://pypi.tuna.tsinghua.edu.cn/simple

from transformers import AutoTokenizer  # 用于加载预训练模型的分词器



class QwenTokenizerTool:
    """
    一个基于 Qwen Tokenizer 的简单分词工具类。

    Qwen(通义千问)是清华大学开发的大语言模型，其分词器针对中文进行了优化。
    该工具类封装了Qwen的分词功能，可以将文本切分为Qwen Tokenizer定义的子词单元(Subword Tokens)。
    """
    def __init__(self, model_name_or_path="Qwen/Qwen1.5-0.5B-Chat"):
        """
        初始化并加载 Qwen Tokenizer。

        参数:
            model_name_or_path: Hugging Face 上的模型名称或本地路径。
                               例如: "Qwen/Qwen1.5-0.5B-Chat" (较小模型)
                                     "Qwen/Qwen1.5-7B-Chat" (较大模型)
                               可以根据需要选择不同的 Qwen 模型。
        """
        self.model_name = model_name_or_path
        self.tokenizer = None
        try:
            # 加载预训练的Qwen分词器
            # trust_remote_code=True 对加载某些模型 (包括 Qwen) 是必要的，允许执行模型仓库中的自定义代码
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print(f"成功加载 Tokenizer: '{self.model_name}'")
        except ImportError as e:
             # 导入错误通常是缺少依赖包导致的
             print(f"加载 Tokenizer '{self.model_name}' 失败: {e}")
             print("请确保已安装 'transformers', 'tiktoken', 'sentencepiece'。")
             print("运行: pip install transformers sentencepiece tiktoken")
        except Exception as e:
            # 捕获其他可能的错误，如网络问题或模型名称错误
            print(f"加载 Tokenizer '{self.model_name}' 时发生错误: {e}")
            print("请检查模型名称是否正确以及网络连接。")

    def tokenize(self, text: str) -> list[str]:
        """
        使用加载的 Qwen Tokenizer 对文本进行分词 (切分为子词)。

        参数:
            text: 需要进行分词的输入字符串。

        返回:
            一个包含解码后的子词 (Subword) 字符串的列表。
            如果 Tokenizer 未加载成功，则返回空列表。
        """
        if not self.tokenizer:
            print("错误: Tokenizer 未成功加载，无法进行分词。")
            return []
        if not isinstance(text, str):
            print("错误: 输入必须是字符串。")
            return []

        # 1. 使用 tokenizer 获取 token IDs
        #    add_special_tokens=False 避免在结果中加入特殊标记如[CLS], [SEP]等
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # 2. 将token IDs转换为可读的token字符串
        tokens = []
        for token_id in token_ids:
            # 使用tokenizer将每个ID解码为对应的token字符串
            token = self.tokenizer.decode([token_id])
            tokens.append(token)

        return tokens



if __name__ == "__main__":
    # 实例化分词器工具
    qwen_tokenizer = QwenTokenizerTool()

    # 测试不同类型的中文和英文文本
    test_texts = [
        "你好，世界！",
        "清华大学是中国顶尖的学府之一。",
        "自然语言处理（NLP）是人工智能的重要分支。",
        "Qwen(通义千问)是一个由阿里云开发的大型语言模型。",
        "这个分词器能够处理中英文混合的文本，数字123，以及标点符号！？。",
        "深度学习技术在近年来取得了突飞猛进的发展，特别是在自然语言处理领域。",
        "我来自清华大学，正在学习自然语言处理相关的课程。",
        "Understanding BPE tokenization is crucial for working with large language models." # 英文测试句
    ]

    # 对每个测试文本进行分词并打印结果
    if qwen_tokenizer.tokenizer: # 确保tokenizer已加载
        for i, text in enumerate(test_texts):
            print(f"\n测试 {i+1}:")
            print(f"原文: {text}")
            tokens = qwen_tokenizer.tokenize(text)
            print(f"分词结果: {tokens}")
            print(f"Token数量: {len(tokens)}")
    else:
        print("Qwen Tokenizer 未能加载，跳过测试。")
        
"""
成功加载 Tokenizer: 'Qwen/Qwen1.5-0.5B-Chat'

测试 1:
原文: 你好，世界！
分词结果: ['你好', '，', '世界', '！']
Token数量: 4

测试 2:
原文: 清华大学是中国顶尖的学府之一。
分词结果: ['清华大学', '是中国', '顶尖', '的', '学', '府', '之一', '。']
Token数量: 8

测试 3:
原文: 自然语言处理（NLP）是人工智能的重要分支。
分词结果: ['自然', '语言', '处理', '（', 'N', 'LP', '）', '是', '人工智能', '的重要', '分支', '。']
Token数量: 12

测试 4:
原文: Qwen(通义千问)是一个由阿里云开发的大型语言模型。
分词结果: ['Q', 'wen', '(', '通', '义', '千', '问', ')', '是一个', '由', '阿里', '云', '开发', '的', '大型', '语言', '模型', '。']
Token数量: 18

测试 5:
原文: 这个分词器能够处理中英文混合的文本，数字123，以及标点符号！？。
分词结果: ['这个', '分', '词', '器', '能够', '处理', '中', '英文', '混合', '的', '文本', '，', '数字', '1', '2', '3', '，', '以及', '标', '点', '符号', '！', '？', '。']
Token数量: 24

测试 6:
原文: 深度学习技术在近年来取得了突飞猛进的发展，特别是在自然语言处理领域。
分词结果: ['深度', '学习', '技术', '在', '近年来', '取得了', '突', '飞', '猛', '进', '的发展', '，', '特别是在', '自然', '语言', '处理', '领域', '。']
Token数量: 18

测试 7:
原文: 我来自清华大学，正在学习自然语言处理相关的课程。
分词结果: ['我', '来自', '清华大学', '，', '正在', '学习', '自然', '语言', '处理', '相关的', '课程', '。']
Token数量: 12

测试 8:
原文: Understanding BPE tokenization is crucial for working with large language models.
分词结果: ['Understanding', ' B', 'PE', ' token', 'ization', ' is', ' crucial', ' for', ' working', ' with', ' large', ' language', ' models', '.']
Token数量: 14

"""
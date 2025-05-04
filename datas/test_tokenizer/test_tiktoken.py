import tiktoken
import time

def main():
    print("===== TikToken 功能展示 =====\n")
    
    # 1. 基本编码和解码
    basic_encoding_decoding()
    
    # 2. 不同编码器比较
    different_encoders()
    
    # 3. 模型特定编码器
    model_specific_encoders()
    
    # 4. 查看token内容 - 展示每个token对应的具体字节和文本内容
    inspect_tokens()
    
    # 5. 多语言比较 - 比较不同语言文本的分词效率和token数量
    multilingual_comparison()
    
    # 6. 批处理演示 - 展示批量处理文本的性能优势
    batch_processing()
    
    # 7. 可视化分词效果 - 直观展示文本如何被拆分成tokens
    visualize_tokenization()
    

def basic_encoding_decoding():
    """
    演示tiktoken的基本编码和解码功能
    包括基本的文本编码、解码以及encode和encode_ordinary方法的比较
    """
    print("1. 基本编码和解码功能:")
    # 获取cl100k_base编码器，这是GPT-4和ChatGPT使用的编码器
    enc = tiktoken.get_encoding("cl100k_base")
    
    # 测试文本
    text = "理解BPE分词原理对使用大型语言模型很有帮助！"
    # 将文本转换为token ID列表
    tokens = enc.encode(text)
    
    print(f"   原始文本: {text}")
    print(f"   编码后的tokens: {tokens}")  # 显示token ID列表
    print(f"   token数量: {len(tokens)}")  # 计算token数量
    print(f"   解码后的文本: {enc.decode(tokens)}")  # 将token ID列表转换回文本
    
    # 比较encode和encode_ordinary方法
    # encode_ordinary不会执行特殊tokens的处理，如BOS、EOS等
    tokens_ordinary = enc.encode_ordinary(text)
    print(f"   使用encode_ordinary的tokens: {tokens_ordinary}")
    print(f"   是否与encode结果相同: {tokens == tokens_ordinary}")  # 对比两种编码方式的结果
    print()

def different_encoders():
    print("2. 不同编码器比较:")
    
    text = "GPT模型使用tiktoken进行文本分词处理。"
    
    encoders = {
        "cl100k_base": tiktoken.get_encoding("cl100k_base"),  # GPT-4, ChatGPT
        "p50k_base": tiktoken.get_encoding("p50k_base"),      # GPT-3, Codex
        "r50k_base": tiktoken.get_encoding("r50k_base")       # GPT-2, 早期GPT-3
    }
    
    print(f"   测试文本: {text}")
    for name, enc in encoders.items():
        tokens = enc.encode(text)
        print(f"   {name}: {len(tokens)} tokens - {tokens}")
    print()

def model_specific_encoders():
    print("3. 特定模型的编码器:")
    
    text = "这是一个测试OpenAI模型分词的示例。"
    
    models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003", "text-embedding-ada-002"]
    
    for model in models:
        try:
            enc = tiktoken.encoding_for_model(model)
            tokens = enc.encode(text)
            print(f"   {model}: {len(tokens)} tokens - 使用编码器: {enc.name}")
        except KeyError:
            print(f"   {model}: 无法找到对应的编码器")
    print()

def inspect_tokens():
    print("4. 查看Token的具体内容:")
    
    enc = tiktoken.get_encoding("cl100k_base")
    text = "Hello, 你好世界!"
    tokens = enc.encode(text)
    
    print(f"   原始文本: {text}")
    print(f"   Token IDs: {tokens}")
    
    print("   每个token对应的字节和文本:")
    for token in tokens:
        bytes_repr = enc.decode_single_token_bytes(token)
        try:
            text_repr = bytes_repr.decode('utf-8')
            print(f"   Token ID: {token}, 字节: {bytes_repr}, 文本: '{text_repr}'")
        except UnicodeDecodeError:
            print(f"   Token ID: {token}, 字节: {bytes_repr}, 文本: [无法解码] (非完整UTF-8字节)")
    print()

def multilingual_comparison():
    print("5. 多语言Token数量对比:")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    texts = {
        "英文": "This is a test of the tokenizer.",
        "中文": "这是一个分词器的测试。",
        "日文": "これはトークナイザーのテストです。",
        "韩文": "이것은 토크나이저의 테스트입니다.",
        "俄文": "Это тест токенизатора.",
        "表情符号": "😀 👍 🚀 🌍 🎉"
    }
    
    for lang, text in texts.items():
        tokens = enc.encode(text)
        print(f"   {lang} ({len(text)}字符): {len(tokens)} tokens - {text}")
    print()

def batch_processing():
    print("6. 批处理示例:")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    texts = [
        "第一个句子很短。",
        "第二个句子稍微长一点，包含更多的内容。",
        "第三个句子是最长的，它包含了很多很多的文字内容，目的是为了测试批处理的效果。"
    ]
    
    # 单独编码
    start_time = time.time()
    tokens_individual = [enc.encode(text) for text in texts]
    individual_time = time.time() - start_time
    
    # 批量编码
    start_time = time.time()
    tokens_batch = enc.encode_batch(texts)
    batch_time = time.time() - start_time
    
    print(f"   单独编码用时: {individual_time:.6f}秒")
    print(f"   批量编码用时: {batch_time:.6f}秒")
    print(f"   速度提升: {individual_time/batch_time:.2f}倍")
    
    for i, (text, tokens) in enumerate(zip(texts, tokens_batch)):
        print(f"   文本 {i+1}: {text}")
        print(f"   Token数量: {len(tokens)}")
    print()

def visualize_tokenization():
    print("7. 可视化分词效果:")
    enc = tiktoken.get_encoding("cl100k_base")
    
    texts = [
        "Hello, 你好世界!",
        "理解BPE分词原理对使用大型语言模型很有帮助！",
        "GPT模型使用tiktoken进行文本分词处理。",
        "English and 中文混合的句子 with some special chars: @#$%"
    ]
    
    for text in texts:
        tokens = enc.encode(text)
        visualization = ""
        start_index = 0
        
        print(f"\n   原始文本: {text}")
        print(f"   Token数量: {len(tokens)}")
        
        # 方法1: 使用decode_single_token_bytes直接重建文本
        parts = []
        for token in tokens:
            byte_content = enc.decode_single_token_bytes(token)
            try:
                text_part = byte_content.decode("utf-8")
            except UnicodeDecodeError:
                text_part = f"[{byte_content.hex()}]"  # 以十六进制显示无法解码的字节
            parts.append(text_part)
        
        print(f"   分词结果: {' / '.join(parts)}")
        
        # 方法2: 对完整文本定位切分点
        cumulative_text = ""
        split_positions = []
        for token in tokens:
            byte_content = enc.decode_single_token_bytes(token)
            try:
                text_part = byte_content.decode("utf-8")
                cumulative_text += text_part
                split_positions.append(len(cumulative_text))
            except UnicodeDecodeError:
                # 处理无法解码的情况
                cumulative_text += "□"  # 使用占位符
                split_positions.append(len(cumulative_text))
        
        # 在原文中插入分隔符
        split_text = ""
        last_pos = 0
        for pos in split_positions:
            if pos > 0:
                split_text += text[last_pos:pos] + " / "
                last_pos = pos
        
        if last_pos < len(text):
            split_text += text[last_pos:]
            
        print(f"   分隔标记: {split_text}")
    print()



if __name__ == "__main__":
    main()
    
    
"""
===== TikToken 功能展示 =====

1. 基本编码和解码功能:
   原始文本: 理解BPE分词原理对使用大型语言模型很有帮助！
   编码后的tokens: [22649, 50338, 33, 1777, 17620, 6744, 235, 53229, 22649, 33764, 38129, 27384, 25287, 73981, 78244, 54872, 25287, 17599, 230, 19361, 13821, 106, 8239, 102, 6447]
   token数量: 25
   解码后的文本: 理解BPE分词原理对使用大型语言模型很有帮助！
   使用encode_ordinary的tokens: [22649, 50338, 33, 1777, 17620, 6744, 235, 53229, 22649, 33764, 38129, 27384, 25287, 73981, 78244, 54872, 25287, 17599, 230, 19361, 13821, 106, 8239, 102, 6447]
   是否与encode结果相同: True

2. 不同编码器比较:
   测试文本: GPT模型使用tiktoken进行文本分词处理。
   cl100k_base: 16 tokens - [38, 2898, 54872, 25287, 38129, 83, 1609, 5963, 72917, 17161, 22656, 17620, 6744, 235, 55642, 1811]
   p50k_base: 31 tokens - [38, 11571, 162, 101, 94, 161, 252, 233, 45635, 18796, 101, 83, 1134, 30001, 32573, 249, 26193, 234, 23877, 229, 17312, 105, 26344, 228, 46237, 235, 13783, 226, 49426, 228, 16764]
   r50k_base: 31 tokens - [38, 11571, 162, 101, 94, 161, 252, 233, 45635, 18796, 101, 83, 1134, 30001, 32573, 249, 26193, 234, 23877, 229, 17312, 105, 26344, 228, 46237, 235, 13783, 226, 49426, 228, 16764]

3. 特定模型的编码器:
   gpt-4: 15 tokens - 使用编码器: cl100k_base
   gpt-3.5-turbo: 15 tokens - 使用编码器: cl100k_base
   text-davinci-003: 30 tokens - 使用编码器: p50k_base
   text-embedding-ada-002: 15 tokens - 使用编码器: cl100k_base

4. 查看Token的具体内容:
   原始文本: Hello, 你好世界!
   Token IDs: [9906, 11, 220, 57668, 53901, 3574, 244, 98220, 0]
   每个token对应的字节和文本:
   Token ID: 9906, 字节: b'Hello', 文本: 'Hello'
   Token ID: 11, 字节: b',', 文本: ','
   Token ID: 220, 字节: b' ', 文本: ' '
   Token ID: 57668, 字节: b'\xe4\xbd\xa0', 文本: '你'
   Token ID: 53901, 字节: b'\xe5\xa5\xbd', 文本: '好'
   Token ID: 3574, 字节: b'\xe4\xb8', 文本: [无法解码] (非完整UTF-8字节)
   Token ID: 244, 字节: b'\x96', 文本: [无法解码] (非完整UTF-8字节)
   Token ID: 98220, 字节: b'\xe7\x95\x8c', 文本: '界'
   Token ID: 0, 字节: b'!', 文本: '!'

5. 多语言Token数量对比:
   英文 (32字符): 8 tokens - This is a test of the tokenizer.
   中文 (11字符): 10 tokens - 这是一个分词器的测试。
   日文 (17字符): 15 tokens - これはトークナイザーのテストです。
   韩文 (18字符): 19 tokens - 이것은 토크나이저의 테스트입니다.
   俄文 (22字符): 11 tokens - Это тест токенизатора.
   表情符号 (9字符): 13 tokens - 😀 👍 🚀 🌍 🎉

6. 批处理示例:
   单独编码用时: 0.000589秒
   批量编码用时: 0.002336秒
   速度提升: 0.25倍
   文本 1: 第一个句子很短。
   Token数量: 11
   文本 2: 第二个句子稍微长一点，包含更多的内容。
   Token数量: 19
   文本 3: 第三个句子是最长的，它包含了很多很多的文字内容，目的是为了测试批处理的效果。
   Token数量: 39

7. 可视化分词效果:

   原始文本: Hello, 你好世界!
   Token数量: 9
   分词结果: Hello / , /   / 你 / 好 / [e4b8] / [96] / 界 / !
   分隔标记: Hello / , /   / 你 / 好 / 世 / 界 / ! /  /

   原始文本: 理解BPE分词原理对使用大型语言模型很有帮助！
   Token数量: 25
   分词结果: 理 / 解 / B / PE / 分 / [e8af] / [8d] / 原 / 理 / 对 / 使用 / 大 / 型 / 语 / 言 / 模 / 型 / [e5be] / [88] / 有 / [e5b8] / [ae] / [e58a] / [a9] / ！
   分隔标记: 理 / 解 / B / PE / 分 / 词 / 原 / 理 / 对 / 使 / 用大 / 型 / 语 / 言 / 模 / 型 / 很 / 有 / 帮 / 助 / ！ /  /  /  /  /

   原始文本: GPT模型使用tiktoken进行文本分词处理。
   Token数量: 16
   分词结果: G / PT / 模 / 型 / 使用 / t / ik / token / 进行 / 文 / 本 / 分 / [e8af] / [8d] / 处理 / 。
   分隔标记: G / PT / 模 / 型 / 使用 / t / ik / token / 进行 / 文 / 本 / 分 / 词 / 处 / 理。 /  /

   原始文本: English and 中文混合的句子 with some special chars: @#$%
   Token数量: 19
   分词结果: English /  and /  中 / 文 / [e6b7] / [b7] / 合 / 的 / [e58f] / [a5] / 子 /  with /  some /  special /  chars / : /  @ / #$ / %
   分隔标记: English /  and /  中 / 文 / 混 / 合 / 的 / 句 / 子 /   / w / ith s / ome s / pecial c / hars:  / @ / #$ / % /  /
   
8. 解释解码失败的原因:
   在BPE分词中，有些token可能只包含UTF-8字符的部分字节
   例如中文字符'世'的UTF-8编码为 \xe4\xb8\x96，可能被分成两个token:
   - 前两个字节 \xe4\xb8
   - 最后一个字节 \x96

   原文: '世'
   对应的UTF-8字节: b'\xe4\xb8\x96'
   分词结果: [3574, 244]
   各token对应的字节:
   Token 1: ID=3574, 字节=b'\xe4\xb8'
   Token 2: ID=244, 字节=b'\x96'

   当我们尝试单独解码这些非完整的UTF-8字节序列时，会出现解码失败
   只有将所有字节重新组合，才能正确解码为完整的Unicode字符
   重新组合所有字节: b'\xe4\xb8\x96'
   组合后解码结果: '世'

"""
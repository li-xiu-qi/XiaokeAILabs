import re
from collections import defaultdict, Counter # 添加了 Counter 导入
import time # 导入 time 模块

def extract_frequencies(sequence):
    """
    给定一个字符串列表（代表句子或文档），计算每个词（已分割并添加</w>）的出现频率。
    """
    token_counter = Counter()
    print("--- Initial Tokenization ---")
    for item in sequence:
        # 将字符串转换为字符列表，用空格连接，并在末尾添加</w>表示词尾
        # 每个输入字符串被视为一个独立的序列单元
        tokens = ' '.join(list(item)) + ' </w>'
        token_counter[tokens] += 1
        print(f"Input: '{item}' -> Tokens: '{tokens}'")
    print("-" * 28)
    return token_counter

def frequency_of_pairs(frequencies):
    """
    给定一个词频字典 {token_string: count}，计算所有相邻子词对的出现频率。
    token_string 是像 "低 温 会 使 ... </w>" 这样的字符串。
    """
    pairs_count = Counter()
    for token_string, count in frequencies.items():
        chars = token_string.split() # 按空格分割成子词（token）列表
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i+1]) # 创建相邻对
            pairs_count[pair] += count # 累加该对的频率（乘以包含该对的 token_string 的频率）
    return pairs_count

def merge_vocab(merge_pair, vocab):
    """
    给定一个要合并的子词对 (token1, token2) 和一个词频字典 vocab {token_string: count}，
    将 vocab 中所有出现的 "token1 token2" 合并为 "token1token2"，并返回更新后的词频字典。
    """
    # 创建正则表达式，匹配独立的 pair，确保它们之间只有一个空格
    # 使用 re.escape 确保特殊字符（如 </w> 中的 /) 被正确处理
    # (?<!\S) 确保 token1 前面是空格或开头
    # (?!\S) 确保 token2 后面是空格或结尾
    re_pattern_str = r'(?<!\S)' + re.escape(merge_pair[0]) + r' ' + re.escape(merge_pair[1]) + r'(?!\S)'
    pattern = re.compile(re_pattern_str)
    merged_token = ''.join(merge_pair) # 合并后的新子词，例如 '温会'

    updated_tokens = {}
    for token_string, freq in vocab.items():
        # 使用 re.sub 替换所有匹配的 pair
        new_token_string = pattern.sub(merged_token, token_string)
        updated_tokens[new_token_string] = freq # 更新字典
    return updated_tokens

def encode_with_bpe(texts, iterations):
    """
    给定待分词的数据（字符串列表）以及最大合并次数，执行 BPE 算法，返回最终的词表。
    """
    # 1. 初始化词表：基于单个字符和词尾标记 </w>
    vocab_map = extract_frequencies(texts)
    print("Initial Vocab Map:", vocab_map)
    print("-" * 28 + "\n")


    merges = [] # 记录合并规则

    # 2. 迭代合并
    for i in range(iterations):
        print(f"--- Iteration {i+1}/{iterations} ---")
        # 2.1 计算当前词表中所有相邻子词对的频率
        pair_freqs = frequency_of_pairs(vocab_map)
        # print("Pair Frequencies:", pair_freqs) # 可选：打印当前轮次的频率

        # 2.2 如果没有可合并的对，则停止
        if not pair_freqs:
            print(f"No more pairs to merge. Stopped at iteration {i}.")
            break

        # 2.3 找到频率最高的对
        # pair_freqs.most_common(1) 返回 [( (token1, token2), freq )]
        # 我们需要处理平分的情况，简单实现取第一个 most_common 的结果
        # （注意：严格的 BPE 可能有更复杂的 tie-breaking 规则）
        most_common_pair, freq = pair_freqs.most_common(1)[0]

        # 检查最高频率是否大于 1 (或者可以设置 min_frequency 参数)
        # 在这个小例子中，频率为 1 的对很多，优先合并频率 > 1 的
        # 如果最高频率仅为 1，可能表示没有强烈的共现模式了（对于此小数据）
        # if freq <= 1 and len(pair_freqs) > 1:
        #      # 尝试找频率为1但可能更优的合并 (此简化代码不处理)
        #      print(f"Highest frequency is only {freq}. Potential stopping point for meaningful merges.")
        #      # 在此简单实现中，我们继续合并频率最高的，即使是 1
        #      # break # 或者可以选择在这里停止

        print(f"Highest frequency pair: {most_common_pair} with frequency {freq}")
        merges.append(most_common_pair) # 记录合并操作

        # 2.4 合并最高频对，更新词表
        vocab_map = merge_vocab(most_common_pair, vocab_map)
        print("Current Vocab Map:", vocab_map)
        print("-" * 28 + "\n")
        time.sleep(0.5) # 加入短暂延时，方便观察过程

    return vocab_map, merges

# --- 使用示例 ---
# 使用之前详细步骤中的中文例子
data = ["低温会使硅基芯片性能降低", "高温会使碳基芯片性能提高"]

# 设定合并次数，例如 6 次，以匹配之前的推演
num_merges = 6

# 执行 BPE 编码
final_bpe_vocab, learned_merges = encode_with_bpe(data, num_merges)

print("\n" + "=" * 30)
print("--- Final BPE Vocabulary Map ---")
for token_string, freq in final_bpe_vocab.items():
    print(f"'{token_string}': {freq}")

print("\n--- Learned Merge Rules (in order) ---")
for idx, pair in enumerate(learned_merges):
    print(f"{idx+1}: {pair}")
print("=" * 30)

# --- 关于下一步 ---
# 真正的 BPE 分词器还需要使用 `learned_merges`
# 来对新的、未见过的文本进行分词。
# 这个过程是：
# 1. 将新文本按字符分割 + </w>
# 2. 按照 `learned_merges` 的顺序，反复在文本中查找并替换对应的 pair。
# 这个示例代码主要演示了学习合并规则和构建词表的过程。

"""
--- Initial Tokenization ---
Input: '低温会使硅基芯片性能降低' -> Tokens: '低 温 会 使 硅 基 芯 片 性 能 降 低 </w>'
Input: '高温会使碳基芯片性能提高' -> Tokens: '高 温 会 使 碳 基 芯 片 性 能 提 高 </w>'
----------------------------
Initial Vocab Map: Counter({'低 温 会 使 硅 基 芯 片 性 能 降 低 </w>': 1, '高 温 会 使 碳 基 芯 片 性 能 提 高 </w>': 1})
----------------------------

--- Iteration 1/6 ---
Highest frequency pair: ('温', '会') with frequency 2
Current Vocab Map: {'低 温会 使 硅 基 芯 片 性 能 降 低 </w>': 1, '高 温会 使 碳 基 芯 片 性 能 提 高 </w>': 1}
----------------------------

--- Iteration 2/6 ---
Highest frequency pair: ('温会', '使') with frequency 2
Current Vocab Map: {'低 温会使 硅 基 芯 片 性 能 降 低 </w>': 1, '高 温会使 碳 基 芯 片 性 能 提 高 </w>': 1}
----------------------------

--- Iteration 3/6 ---
Highest frequency pair: ('基', '芯') with frequency 2
Current Vocab Map: {'低 温会使 硅 基芯 片 性 能 降 低 </w>': 1, '高 温会使 碳 基芯 片 性 能 提 高 </w>': 1}
----------------------------

--- Iteration 4/6 ---
Highest frequency pair: ('基芯', '片') with frequency 2
Current Vocab Map: {'低 温会使 硅 基芯片 性 能 降 低 </w>': 1, '高 温会使 碳 基芯片 性 能 提 高 </w>': 1}
----------------------------

--- Iteration 5/6 ---
Highest frequency pair: ('基芯片', '性') with frequency 2
Current Vocab Map: {'低 温会使 硅 基芯片性 能 降 低 </w>': 1, '高 温会使 碳 基芯片性 能 提 高 </w>': 1}
----------------------------

--- Iteration 6/6 ---
Highest frequency pair: ('基芯片性', '能') with frequency 2
Current Vocab Map: {'低 温会使 硅 基芯片性能 降 低 </w>': 1, '高 温会使 碳 基芯片性能 提 高 </w>': 1}
----------------------------


==============================
--- Final BPE Vocabulary Map ---
'低 温会使 硅 基芯片性能 降 低 </w>': 1
'高 温会使 碳 基芯片性能 提 高 </w>': 1

--- Learned Merge Rules (in order) ---
1: ('温', '会')
2: ('温会', '使')
3: ('基', '芯')
4: ('基芯', '片')
5: ('基芯片', '性')
6: ('基芯片性', '能')
==============================

"""
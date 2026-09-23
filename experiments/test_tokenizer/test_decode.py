"""
大模型流式输出中的BBPE解码处理

演示大模型推理框架如何在流式输出(streaming)过程中处理BBPE分词带来的UTF-8字符拆分问题
"""

import tiktoken
import time
from typing import List, Iterator, Optional


class BBPEStreamingDecoder:
    """模拟大模型推理框架中的流式解码器，处理BBPE字节级分词"""
    
    def __init__(self, encoding_name="cl100k_base"):
        """初始化解码器"""
        self.enc = tiktoken.get_encoding(encoding_name)
        self.buffer = b''  # 字节缓冲区，存储未完成解码的UTF-8字节序列
    
    def stream_decode(self, token_ids: Iterator[int]) -> Iterator[str]:
        """
        流式解码token ID序列，处理不完整字符的情况
        
        参数:
            token_ids: 一个生成token ID的迭代器，模拟大模型持续输出
            
        生成:
            解码后的字符串片段，确保每个片段都是有效的UTF-8字符序列
        """
        for token_id in token_ids:
            # 获取当前token对应的字节
            token_bytes = self.enc.decode_single_token_bytes(token_id)
            
            # 将新字节添加到缓冲区
            self.buffer += token_bytes
            
            # 尝试从缓冲区解码尽可能多的完整字符
            result = ""
            
            # 持续尝试解码，直到无法获取更多完整字符
            while True:
                char, consumed = self._try_decode_buffer()
                if not char:  # 无法解码更多完整字符
                    break
                    
                # 找到完整字符，添加到结果并从缓冲区移除
                result += char
                self.buffer = self.buffer[consumed:]
            
            # 如果此轮有可输出的内容，则生成它
            if result:
                yield result
    
    def _try_decode_buffer(self) -> tuple[Optional[str], int]:
        """
        尝试从缓冲区解码一个完整的UTF-8字符
        
        返回:
            (解码字符, 消耗字节数) 或 (None, 0)表示无法解码
        """
        if not self.buffer:
            return None, 0
            
        # 尝试从不同长度解码第一个字符
        # UTF-8编码可能是1-4字节长
        for i in range(1, min(5, len(self.buffer) + 1)):
            try:
                char = self.buffer[:i].decode('utf-8')
                return char, i
            except UnicodeDecodeError as e:
                # 如果这个长度解码失败，打印错误信息
                # print(f"解码错误: 尝试解码 {i} 字节 {self.buffer[:i].hex()}, 错误: {e}")
                # 尝试更长的序列
                continue
        
        # 所有尝试都失败，可能需要更多字节
        print(f"无法解码当前缓冲区内容: {self.buffer.hex()}, 需要更多字节")
        return None, 0

    
    def flush(self) -> Optional[str]:
        """
        处理缓冲区中剩余的字节，通常在流结束时调用
        
        返回:
            剩余字节解码的字符串，如果无法解码则返回None
        """
        if not self.buffer:
            return None
            
        try:
            # 尝试解码所有剩余字节
            result = self.buffer.decode('utf-8', errors='replace')
            self.buffer = b''
            return result
        except Exception:
            # 如果仍然失败，可以返回十六进制表示或替换字符
            result = f"[无法解码:{self.buffer.hex()}]"
            self.buffer = b''
            return result


def simulate_llm_streaming(text: str, delay: float = 0.3):
    """
    模拟大语言模型的流式输出
    
    参数:
        text: 要模拟输出的文本
        delay: 每个token之间的延迟(秒)
    """
    print("\n===== 模拟大语言模型流式输出 =====")
    print(f"原始文本: '{text}'")
    
    # 创建编码器和解码器
    enc = tiktoken.get_encoding("cl100k_base")
    decoder = BBPEStreamingDecoder()
    
    # 将文本编码为token IDs
    token_ids = enc.encode(text)
    # 打印token长度
    print(f"原始文本长度: {len(text.encode('utf-8'))} 字节")
    print("tokens length",len(token_ids))
    print(f"编码后的token IDs: {token_ids}")

    
    print(f"\n总共 {len(token_ids)} 个tokens:")
    print("-" * 40)
    
    # 模拟流式生成token
    cumulative_text = ""
    
    for i, chunk in enumerate(decoder.stream_decode(iter(token_ids))):
        time.sleep(delay)  # 模拟网络延迟或模型生成时间
        cumulative_text += chunk
        print(f"输出[{i+1}]: '{chunk}' (当前累积: '{cumulative_text}')")
    
    # 检查是否有剩余内容需要刷新
    remaining = decoder.flush()
    if remaining:
        cumulative_text += remaining
        print(f"最终刷新: '{remaining}' (最终结果: '{cumulative_text}')")
    
    print("-" * 40)
    print(f"最终输出: '{cumulative_text}'")
    print(f"原文一致: {text == cumulative_text}")


# 使用示例
if __name__ == "__main__":
    # 模拟包含被拆分字符的流式输出
    text = "Hello, 世界! 这是一个测试。"
    print("字符串长度",len(text))
    simulate_llm_streaming(text) 
    
    # 一个更复杂的例子，包含多个可能被拆分的中文字符
    text = "UTF-8编码的汉字被拆分时，需要正确处理。"
    print("字符串长度",len(text))
    simulate_llm_streaming(text, delay=0.2)
    
"""
字符串长度 18

===== 模拟大语言模型流式输出 =====
原始文本: 'Hello, 世界! 这是一个测试。'
原始文本长度: 36 字节
tokens length 13
编码后的token IDs: [9906, 11, 220, 3574, 244, 98220, 0, 33281, 247, 21043, 48044, 82805, 1811]

总共 13 个tokens:
----------------------------------------
输出[1]: 'Hello' (当前累积: 'Hello')
输出[2]: ',' (当前累积: 'Hello,')
输出[3]: ' ' (当前累积: 'Hello, ')
无法解码当前缓冲区内容: e4b8, 需要更多字节
输出[4]: '世' (当前累积: 'Hello, 世')
输出[5]: '界' (当前累积: 'Hello, 世界')
输出[6]: '!' (当前累积: 'Hello, 世界!')
无法解码当前缓冲区内容: e8bf, 需要更多字节
输出[7]: ' ' (当前累积: 'Hello, 世界! ')
输出[8]: '这' (当前累积: 'Hello, 世界! 这')
输出[9]: '是' (当前累积: 'Hello, 世界! 这是')
输出[10]: '一个' (当前累积: 'Hello, 世界! 这是一个')
输出[11]: '测试' (当前累积: 'Hello, 世界! 这是一个测试')
输出[12]: '。' (当前累积: 'Hello, 世界! 这是一个测试。')
----------------------------------------
最终输出: 'Hello, 世界! 这是一个测试。'
原文一致: True
字符串长度 22

===== 模拟大语言模型流式输出 =====
原始文本: 'UTF-8编码的汉字被拆分时，需要正确处理。'
原始文本长度: 56 字节
tokens length 20
编码后的token IDs: [8729, 12, 23, 31968, 16882, 9554, 21980, 231, 19113, 87743, 104, 26955, 228, 17620, 13646, 3922, 86206, 90091, 55642, 1811]

总共 20 个tokens:
----------------------------------------
输出[1]: 'UTF' (当前累积: 'UTF')
输出[2]: '-' (当前累积: 'UTF-')
输出[3]: '8' (当前累积: 'UTF-8')
输出[4]: '编' (当前累积: 'UTF-8编')
输出[5]: '码' (当前累积: 'UTF-8编码')
输出[6]: '的' (当前累积: 'UTF-8编码的')
无法解码当前缓冲区内容: e6b1, 需要更多字节
输出[7]: '汉' (当前累积: 'UTF-8编码的汉')
输出[8]: '字' (当前累积: 'UTF-8编码的汉字')
无法解码当前缓冲区内容: e8a2, 需要更多字节
输出[9]: '被' (当前累积: 'UTF-8编码的汉字被')
无法解码当前缓冲区内容: e68b, 需要更多字节
输出[10]: '拆' (当前累积: 'UTF-8编码的汉字被拆')
输出[11]: '分' (当前累积: 'UTF-8编码的汉字被拆分')
输出[12]: '时' (当前累积: 'UTF-8编码的汉字被拆分时')
输出[13]: '，' (当前累积: 'UTF-8编码的汉字被拆分时，')
输出[14]: '需要' (当前累积: 'UTF-8编码的汉字被拆分时，需要')
输出[15]: '正确' (当前累积: 'UTF-8编码的汉字被拆分时，需要正确')
输出[16]: '处理' (当前累积: 'UTF-8编码的汉字被拆分时，需要正确处理')
输出[17]: '。' (当前累积: 'UTF-8编码的汉字被拆分时，需要正确处理。')
----------------------------------------
最终输出: 'UTF-8编码的汉字被拆分时，需要正确处理。'
原文一致: True
"""
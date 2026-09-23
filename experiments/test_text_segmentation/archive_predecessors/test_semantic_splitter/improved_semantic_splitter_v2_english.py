"""
改进的英文语义分割器 V2
解决分块大小不均匀的问题，保持句子完整性
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import spacy
from dotenv import load_dotenv
from sentence_transformers import util
from embedding_models import get_default_embedding_model

# 加载环境变量
load_dotenv()

class ImprovedSemanticSplitterV2English:
    """
    改进的英文语义分割器 V2
    特点：
    1. 严格控制块大小
    2. 保持句子完整性
    3. 在合适的标点符号处分割
    4. 避免过度分割
    5. 处理英文文本特征
    """
    
    def __init__(
        self,
        embed_model=None,
        target_chunk_size=850,      # 目标块大小
        min_chunk_size=400,         # 最小块大小
        max_chunk_size=1200,        # 最大块大小
        similarity_threshold=0.35,  # 相似度阈值
        overlap_size=50,            # 重叠大小
    ):
        """
        初始化改进的语义分割器
        """
        if embed_model is None:
            self.embed_model = get_default_embedding_model()
        else:
            self.embed_model = embed_model
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_size = overlap_size  # 本次改进中未使用重叠大小，但保留以备将来使用

    def split_text(self, text: str) -> List[str]:
        """
        主要分割方法
        """
        print(f"开始分割文本，总长度: {len(text)} 字符")
        
        # 1. 预处理：按句子分割
        sentences = self._split_into_sentences(text)
        print(f"预处理完成，得到 {len(sentences)} 个句子")
        
        # 2. 按目标大小进行初始分组
        initial_chunks = self._create_initial_chunks(sentences)
        print(f"初始分组完成，得到 {len(initial_chunks)} 个初始块")
        
        # 3. 语义优化：调整分块边界
        optimized_chunks = self._semantic_optimize_chunks(initial_chunks)
        print(f"语义优化完成，最终得到 {len(optimized_chunks)} 个文本块")
        
        # 4. 验证和后处理
        final_chunks = self._post_process_chunks(optimized_chunks)
        
        return final_chunks
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子，保持句子完整性
        """
        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 统一空白字符
        text = text.strip()
        
        # 定义英文句子结尾标点符号
        sentence_endings = r'[.!?]+(?=\s+[A-Z]|\s*$)'
        
        # 按句子分割，保留分隔符
        sentences = []
        current_pos = 0
        
        for match in re.finditer(sentence_endings, text):
            end_pos = match.end()
            sentence = text[current_pos:end_pos].strip()
            if sentence and len(sentence) > 3:  # 过滤掉太短的片段
                sentences.append(sentence)
            current_pos = end_pos
        
        # 添加剩余部分（如果有）
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining and len(remaining) > 3:
                sentences.append(remaining)
        
        # 后处理：合并过短的句子
        filtered_sentences = []
        for sentence in sentences:
            if len(sentence) < 15 and filtered_sentences:  # 太短的句子合并到前面
                filtered_sentences[-1] += " " + sentence
            else:
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    def _create_initial_chunks(self, sentences: List[str]) -> List[str]:
        """
        根据目标大小创建初始块，严格保持句子完整性
        """
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果单个句子就超过最大大小，需要进一步分割
            if len(sentence) > self.max_chunk_size:
                # 先保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 分割长句子
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
                
            # 计算加上新句子后的总长度
            test_chunk = (current_chunk + " " + sentence) if current_chunk else sentence
            
            # 如果超过目标大小，检查是否应该分割
            if len(test_chunk) > self.target_chunk_size:
                # 如果当前块太小，或者加上新句子不会超过最大限制，就继续添加
                if len(current_chunk) < self.min_chunk_size or len(test_chunk) <= self.max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # 保存当前块，开始新块
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            else:
                # 正常添加句子
                current_chunk = test_chunk
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        分割过长的句子，在合适的标点符号处切分
        """
        if len(sentence) <= self.max_chunk_size:
            return [sentence]
            
        chunks = []
        
        # 定义可以作为分割点的英文标点符号（按优先级排序）
        split_patterns = [
            (r'[,;]', ','),       # 逗号、分号
            (r'[:]', ':'),        # 冒号
            (r'[\s]+', ' '),      # 空白字符
        ]
        
        # 尝试按不同的标点符号分割
        for pattern, joiner in split_patterns:
            if len(sentence) <= self.max_chunk_size:
                break
                
            # 按标点符号分割
            parts = re.split(f'({pattern})', sentence)
            if len(parts) > 1:
                current_chunk = ""
                
                i = 0
                while i < len(parts):
                    part = parts[i]
                    
                    # 检查是否是分隔符
                    if re.match(pattern, part):
                        current_chunk += part
                        i += 1
                        continue
                    
                    # 检查加上这部分是否会超过限制
                    test_chunk = current_chunk + part
                    if len(test_chunk) > self.target_chunk_size and current_chunk:
                        # 保存当前块
                        chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk = test_chunk
                    
                    i += 1
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果成功分割了，返回结果
                valid_chunks = [chunk for chunk in chunks if chunk.strip()]
                if len(valid_chunks) > 1:
                    return valid_chunks
                else:
                    chunks = []  # 重置，尝试下一个分割模式
        
        # 如果无法通过标点符号合理分割，按字符数强制分割
        chunks = []
        start = 0
        while start < len(sentence):
            end = min(start + self.target_chunk_size, len(sentence))
            
            # 如果不是最后一段，尝试在附近找到合适的分割点
            if end < len(sentence):
                # 在目标位置前后寻找合适的分割点
                best_split = end
                search_range = min(100, (end - start) // 4)  # 搜索范围
                
                for offset in range(search_range):
                    # 先向后找
                    pos = end + offset
                    if pos < len(sentence) and sentence[pos] in ',.!?;: ':
                        best_split = pos + 1
                        break
                    
                    # 再向前找
                    pos = end - offset
                    if pos > start and sentence[pos] in ',.!?;: ':
                        best_split = pos + 1
                        break
                
                chunks.append(sentence[start:best_split].strip())
                start = best_split
            else:
                chunks.append(sentence[start:end].strip())
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    def _semantic_optimize_chunks(self, chunks: List[str]) -> List[str]:
        """
        使用语义相似度优化分块
        """
        if len(chunks) <= 1:
            return chunks
        
        print("开始语义优化...")
        
        # 计算每个块的嵌入向量
        embeddings = self.embed_model.encode(chunks)
        
        # 计算相邻块之间的相似度并优化
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # 尝试与下一个块合并
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # 检查合并后的大小
                merged_size = len(current_chunk) + len(next_chunk) + 1  # +1 为空格
                
                if merged_size <= self.max_chunk_size:
                    # 计算语义相似度
                    similarity = self._calculate_similarity(embeddings[i], embeddings[i+1])
                    
                    # 如果相似度高于阈值，进行合并
                    if similarity > self.similarity_threshold:
                        merged_chunk = current_chunk + " " + next_chunk
                        optimized_chunks.append(merged_chunk)
                        i += 2  # 跳过下一个块
                        continue
            
            # 不合并，保留当前块
            optimized_chunks.append(current_chunk)
            i += 1
        
        return optimized_chunks
    
    def _calculate_similarity(self, embedding1, embedding2):
        """计算两个嵌入向量之间的余弦相似度"""
        sim = util.cos_sim(np.array(embedding1), np.array(embedding2))
        return float(sim.item())
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        后处理：确保所有块都符合大小要求
        """
        processed_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            # 如果块太小，尝试与前一个块合并
            if len(chunk) < self.min_chunk_size and processed_chunks:
                potential_merged = processed_chunks[-1] + " " + chunk
                if len(potential_merged) <= self.max_chunk_size:
                    processed_chunks[-1] = potential_merged
                    continue
            
            # 如果块太大，进一步分割
            if len(chunk) > self.max_chunk_size:
                sub_chunks = self._split_long_sentence(chunk)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks


def save_chunks_to_files(chunks: List[str], output_dir: str, prefix: str = "english_text"):
    """
    将分块结果保存到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n开始保存 {len(chunks)} 个文本块到 {output_dir}")
    
    # 统计信息
    sizes = [len(chunk) for chunk in chunks]
    print(f"\n分块统计信息:")
    print(f"平均大小: {np.mean(sizes):.1f} 字符")
    print(f"最小大小: {min(sizes)} 字符")
    print(f"最大大小: {max(sizes)} 字符")
    print(f"标准差: {np.std(sizes):.1f}")
    
    # 大小分布统计
    size_ranges = [
        (0, 300, "过小"),
        (300, 600, "较小"),
        (600, 900, "适中"),
        (900, 1200, "较大"),
        (1200, float('inf'), "过大")
    ]
    
    print(f"\n大小分布:")
    for min_size, max_size, label in size_ranges:
        count = sum(1 for size in sizes if min_size <= size < max_size)
        percentage = count / len(sizes) * 100
        print(f"{label} ({min_size}-{max_size if max_size != float('inf') else '∞'}): {count} 个 ({percentage:.1f}%)")
    
    # 保存文件
    for i, chunk in enumerate(chunks, 1):
        filename = f"{prefix}_chunk_{i:03d}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(chunk)
        
        print(f"保存文件: {filename} (长度: {len(chunk)} 字符)")
    
    print(f"\n所有文件已保存到: {output_dir}")


if __name__ == "__main__":
    # 测试改进的分割器
    
    # 读取英文文本文件
    text_file = "sample_english_text.txt"
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"读取文本文件: {text_file}")
        print(f"文本总长度: {len(text)} 字符")
        
        # 创建改进的分割器
        splitter = ImprovedSemanticSplitterV2English(
            target_chunk_size=850,
            min_chunk_size=400,
            max_chunk_size=1200,
            similarity_threshold=0.35
        )
        
        # 执行分割
        chunks = splitter.split_text(text)
        
        # 保存结果
        output_dir = "english_text_improved_splitting_results_v2"
        save_chunks_to_files(chunks, output_dir)
        
    else:
        print(f"文本文件 {text_file} 不存在")
        
        # 使用示例英文文本进行测试
        sample_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

        Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Modern deep learning techniques have led to significant improvements in NLP tasks. Transformer architectures, in particular, have revolutionized the field by enabling models to process sequences of data more effectively than previous approaches.

        Machine learning algorithms are essential for NLP systems. These algorithms learn patterns from large datasets of text and can then apply these patterns to new, unseen text. The advent of large language models has transformed how we approach various NLP tasks, from text classification to machine translation.

        Text preprocessing is a crucial step in any NLP pipeline. This involves cleaning the text, removing unnecessary characters, normalizing text, tokenization, and other steps to prepare the text for analysis. Effective preprocessing can significantly impact the performance of downstream NLP tasks.

        Applications of NLP are widespread and growing. They include machine translation, sentiment analysis, text summarization, question answering systems, chatbots, and many others. As the field continues to evolve, we can expect to see even more sophisticated applications that can better understand and generate human-like text.
        """
        
        print("使用示例英文文本进行测试")
        print(f"示例文本长度: {len(sample_text)} 字符")
        
        # 创建改进的分割器
        splitter = ImprovedSemanticSplitterV2English(
            target_chunk_size=400,  # 为示例文本使用较小的目标大小
            min_chunk_size=200,
            max_chunk_size=600,
            similarity_threshold=0.35
        )
        
        # 执行分割
        chunks = splitter.split_text(sample_text)
        
        # 保存结果
        output_dir = "sample_english_splitting_results"
        save_chunks_to_files(chunks, output_dir, "sample_english")

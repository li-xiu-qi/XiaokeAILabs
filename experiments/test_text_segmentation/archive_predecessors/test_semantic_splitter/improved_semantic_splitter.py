"""
改进的语义分割器
解决分块大小不均匀的问题，实现更合理的文本分割
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

class ImprovedSemanticSplitter:
    """
    改进的语义分割器
    特点：
    1. 严格控制块大小
    2. 避免过度分割
    3. 保持语义连贯性
    4. 处理中文文本特征
    """
    
    def __init__(
        self,
        embed_model=None,
        target_chunk_size=850,      # 目标块大小
        min_chunk_size=300,         # 最小块大小
        max_chunk_size=1200,        # 最大块大小
        similarity_threshold=0.3,   # 相似度阈值（降低以减少过度分割）
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
        self.overlap_size = overlap_size
    
    def split_text(self, text: str) -> List[str]:
        """
        主要分割方法
        """
        print(f"开始分割文本，总长度: {len(text)} 字符")
          # 1. 预处理：按段落分割
        sentences = self._split_into_paragraphs(text)
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
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        将文本分割成句子，保持句子完整性
        """
        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 统一空白字符
        text = text.strip()
        
        # 定义句子结尾标点符号
        sentence_endings = r'[。！？；]'
        
        # 按句子分割，保留分隔符
        sentences = []
        current_pos = 0
        
        for match in re.finditer(sentence_endings, text):
            end_pos = match.end()
            sentence = text[current_pos:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            current_pos = end_pos
        
        # 添加剩余部分（如果有）
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        # 过滤掉太短的句子（合并到前一个句子）
        filtered_sentences = []
        for sentence in sentences:
            if len(sentence) < 10 and filtered_sentences:  # 太短的句子合并到前面
                filtered_sentences[-1] += sentence
            else:
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    def _create_initial_chunks(self, sentences: List[str]) -> List[str]:
        """
        根据目标大小创建初始块，保持句子完整性
        """
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果单个句子就超过最大大小，需要进一步分割
            if len(sentence) > self.max_chunk_size:
                # 先保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 分割长句子
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                
            # 检查加上这个句子是否会超过目标大小
            elif len(current_chunk) + len(sentence) > self.target_chunk_size:
                # 如果当前块太小且不会超过最大限制，还是加上这个句子
                if len(current_chunk) < self.min_chunk_size and len(current_chunk) + len(sentence) <= self.max_chunk_size:
                    current_chunk += sentence
                else:
                    # 保存当前块，开始新块
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            else:
                # 正常添加句子
                current_chunk += sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        分割过长的句子，在合适的标点符号处切分
        """
        chunks = []
        
        # 定义可以作为分割点的标点符号（按优先级排序）
        split_patterns = [
            r'[，、]',      # 逗号、顿号
            r'[：；]',      # 冒号、分号
            r'[\s]+',       # 空白字符
        ]
        
        # 尝试按不同的标点符号分割
        for pattern in split_patterns:
            if len(sentence) <= self.max_chunk_size:
                break
                
            parts = re.split(f'({pattern})', sentence)
            if len(parts) > 1:
                current_chunk = ""
                
                for i, part in enumerate(parts):
                    if re.match(pattern, part):  # 是分隔符
                        current_chunk += part
                    else:  # 是内容
                        if len(current_chunk) + len(part) > self.target_chunk_size and current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += part
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return [chunk for chunk in chunks if chunk.strip()]
        
        # 如果无法通过标点符号分割，按字符数强制分割（保留最后的完整字符）
        if len(sentence) > self.max_chunk_size:
            chunks = []
            start = 0
            while start < len(sentence):
                end = start + self.target_chunk_size
                if end >= len(sentence):
                    chunks.append(sentence[start:])
                    break
                
                # 尝试在附近找到合适的分割点
                best_split = end
                for i in range(max(end - 50, start), min(end + 50, len(sentence))):
                    if sentence[i] in '，。！？；、：':
                        best_split = i + 1
                        break
                
                chunks.append(sentence[start:best_split])
                start = best_split
            
            return chunks
        
        return [sentence]
    
    def _semantic_optimize_chunks(self, chunks: List[str]) -> List[str]:
        """
        使用语义相似度优化分块
        """
        if len(chunks) <= 1:
            return chunks
        
        print("开始语义优化...")
        
        # 计算每个块的嵌入向量
        embeddings = self.embed_model.encode(chunks)
        
        # 计算相邻块之间的相似度
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # 尝试与下一个块合并
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # 检查合并后的大小
                merged_size = len(current_chunk) + len(next_chunk)
                
                if merged_size <= self.max_chunk_size:
                    # 计算语义相似度
                    similarity = self._calculate_similarity(embeddings[i], embeddings[i+1])
                    
                    # 如果相似度高于阈值，进行合并
                    if similarity > self.similarity_threshold:
                        merged_chunk = current_chunk + "。" + next_chunk
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
                if len(processed_chunks[-1]) + len(chunk) <= self.max_chunk_size:
                    processed_chunks[-1] += "。" + chunk
                    continue
            
            # 如果块太大，进一步分割
            if len(chunk) > self.max_chunk_size:
                sub_chunks = self._split_large_paragraph(chunk)
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks


def save_chunks_to_files(chunks: List[str], output_dir: str, prefix: str = "红楼梦"):
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
    
    # 读取红楼梦文本
    text_file = "红楼梦.txt"
    if os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"读取文本文件: {text_file}")
        print(f"文本总长度: {len(text)} 字符")
        
        # 创建改进的分割器
        splitter = ImprovedSemanticSplitter(
            target_chunk_size=850,
            min_chunk_size=300,
            max_chunk_size=1200,
            similarity_threshold=0.4  # 适中的相似度阈值
        )
        
        # 执行分割
        chunks = splitter.split_text(text)
        
        # 保存结果
        output_dir = "红楼梦_改进分割结果"
        save_chunks_to_files(chunks, output_dir)
        
    else:
        print(f"文本文件 {text_file} 不存在")

import os
# 在导入任何库之前设置环境变量，防止tokenizers并行性警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import spacy

from sentence_transformers import SentenceTransformer, util

class AdvancedSemanticSplitter:
    """
    高级语义切分器，完整实现SemanticDoubleMergingSplitterNodeParser的双重合并逻辑。
    可以接受自定义的句子列表作为输入，计算语义相似度并进行分组。
    """
    
    def __init__(
        self,
        embed_model=None,
        initial_threshold=0.4,   # 初始相似度阈值
        appending_threshold=0.5, # 附加阈值
        merging_threshold=0.5,   # 合并阈值
        max_chunk_size=500,      # 最大块大小
    ):
        """
        初始化高级语义切分器
        
        Args:
            embed_model: 嵌入模型，用于计算文本的向量表示
            initial_threshold: 初始相似度阈值，低于此阈值的句子会被分到不同块
            appending_threshold: 附加阈值，用于决定是否将句子附加到现有块
            merging_threshold: 合并阈值，用于决定是否合并两个相邻的块
            max_chunk_size: 最大块大小（字符数）
        """
        model_path = os.environ.get("EMBEDDING_MODEL_PATH", "all-MiniLM-L6-v2")
        self.embed_model = embed_model or SentenceTransformer(model_path)
        self.initial_threshold = initial_threshold
        self.appending_threshold = appending_threshold
        self.merging_threshold = merging_threshold
        self.max_chunk_size = max_chunk_size
    
    def calculate_similarity(self, embedding1, embedding2):
        """计算两个嵌入向量之间的余弦相似度"""
        # sentence-transformers util.pytorch_cos_sim 返回tensor
        sim = util.cos_sim(np.array(embedding1), np.array(embedding2))
        return float(sim.item())
    
    def process_sentences(self, sentences: List[str], document_id: str = None) -> List[str]:
        """
        处理自定义句子列表，实现完整的双重合并语义分组逻辑
        
        Args:
            sentences: 预先切分好的句子列表
            document_id: 可选的文档ID
            
        Returns:
            按语义相似度分组的文本块字符串列表
        """
        if not sentences:
            return []
        
        # 1. 为每个句子生成嵌入向量
        sentence_embeddings = self.embed_model.encode(sentences, convert_to_numpy=True)
        
        # 2. 第一阶段：初始分块 - 根据语义相似度将句子分成初始块
        initial_chunks = self._create_initial_chunks(sentences, sentence_embeddings)
        
        # 3. 第二阶段：附加操作 - 尝试将句子附加到已有的块
        appended_chunks = self._apply_appending(initial_chunks, sentence_embeddings)
        
        # 4. 第三阶段：合并操作 - 尝试合并相似度高的相邻块
        merged_chunks = self._apply_merging(appended_chunks, sentence_embeddings)
        
        # 5. 返回分组后的文本块字符串列表
        return self._create_text_chunks(merged_chunks, sentences)
    
    def _create_initial_chunks(self, sentences: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """
        第一阶段：初始分块
        根据语义相似度将句子分成初始块
        
        Args:
            sentences: 句子列表
            embeddings: 每个句子对应的嵌入向量
            
        Returns:
            初始块列表，每个块包含句子索引列表
        """
        chunks = []
        current_chunk = [0]  # 从第一个句子开始
        
        for i in range(1, len(sentences)):
            prev_embedding = embeddings[i-1]
            curr_embedding = embeddings[i]
            
            # 计算当前句子与前一句子的相似度
            similarity = self.calculate_similarity(prev_embedding, curr_embedding)
            
            # 如果相似度低于初始阈值，则开始新的块
            if similarity < self.initial_threshold:
                chunks.append(current_chunk)
                current_chunk = [i]
            else:
                current_chunk.append(i)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _apply_appending(self, chunks: List[List[int]], embeddings: np.ndarray) -> List[List[int]]:
        """
        第二阶段：附加操作
        尝试将一个块的句子附加到前一个块，如果语义相似度高于附加阈值
        
        Args:
            chunks: 初始块列表
            embeddings: 每个句子对应的嵌入向量
            
        Returns:
            附加操作后的块列表
        """
        if len(chunks) <= 1:
            return chunks
        
        result_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            prev_chunk = result_chunks[-1]
            
            # 计算当前块的第一个句子与前一个块的最后一个句子的相似度
            prev_sent_idx = prev_chunk[-1]
            curr_sent_idx = current_chunk[0]
            
            similarity = self.calculate_similarity(
                embeddings[prev_sent_idx],
                embeddings[curr_sent_idx]
            )
            
            # 如果相似度高于附加阈值，则将当前块附加到前一个块
            if similarity > self.appending_threshold:
                result_chunks[-1] = prev_chunk + current_chunk
            else:
                result_chunks.append(current_chunk)
        
        return result_chunks
    
    def _apply_merging(self, chunks: List[List[int]], embeddings: np.ndarray) -> List[List[int]]:
        """
        第三阶段：合并操作
        尝试合并相邻的块，如果它们的相似度高于合并阈值
        
        Args:
            chunks: 附加操作后的块列表
            embeddings: 每个句子对应的嵌入向量
            
        Returns:
            合并操作后的块列表
        """
        if len(chunks) <= 1:
            return chunks
        
        # 计算每个块的平均嵌入向量
        chunk_embeddings = []
        for chunk in chunks:
            chunk_embedding = np.mean([embeddings[idx] for idx in chunk], axis=0)
            chunk_embeddings.append(chunk_embedding)
        
        # 尝试合并相邻块
        i = 0
        result_chunks = []
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # 检查是否可以与下一个块合并
            if i + 1 < len(chunks):
                curr_embedding = chunk_embeddings[i]
                next_embedding = chunk_embeddings[i+1]
                
                similarity = self.calculate_similarity(curr_embedding, next_embedding)
                
                # 如果相似度高于合并阈值，则合并两个块
                if similarity > self.merging_threshold:
                    merged_chunk = current_chunk + chunks[i+1]
                    result_chunks.append(merged_chunk)
                    i += 2  # 跳过下一个块
                    continue
            
            # 如果不能合并或没有下一个块，则保留当前块
            result_chunks.append(current_chunk)
            i += 1
        
        return result_chunks
    
    def _create_text_chunks(self, chunks: List[List[int]], sentences: List[str]) -> List[str]:
        """
        根据分块索引返回文本块字符串列表
        """
        text_chunks = []
        for chunk in chunks:
            chunk_text = " ".join([sentences[idx] for idx in chunk])
            if len(chunk_text) > self.max_chunk_size:
                # 这里简单保持原样
                pass
            text_chunks.append(chunk_text)
        return text_chunks

# 加载spaCy模型(只加载一次以提高性能)
_SPACY_MODEL = None

def get_spacy_model():
    """获取spaCy模型，如果尚未加载则加载模型"""
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            # 首先尝试从环境变量获取模型路径
            model_path = os.environ.get("SPACY_MODEL_PATH")
            
            # 如果环境变量未设置，使用默认路径
            if not model_path:
                model_path = os.path.join(os.getcwd(), "xx_sent_ud_sm-3.8.0")
            
            print(f"尝试从 {model_path} 加载spaCy模型...")
            _SPACY_MODEL = spacy.load(model_path)
            print("模型加载成功!")
        except Exception as e:
            # 如果本地模型加载失败，尝试使用已安装的模型
            try:
                print("本地模型加载失败，尝试使用已安装的模型...")
                _SPACY_MODEL = spacy.load("xx_sent_ud_sm")
                print("已安装模型加载成功!")
            except Exception as e2:
                # 提供更详细的错误信息
                print(f"错误：无法加载spaCy模型: {e}")
                print(f"也无法加载已安装的模型: {e2}")
                print("请确保模型路径正确，并且包含所有必要的模型文件。")
                raise
    return _SPACY_MODEL

def custom_sentence_splitter(text: str) -> List[str]:
    """
    结合spaCy和正则表达式，增强中英文混合文本的句子切分能力
    
    Args:
        text: 要分割的文本
        
    Returns:
        切分后的句子列表
    """
    # 获取spaCy模型
    nlp = get_spacy_model()
    
    # 配置spaCy处理器以更好地处理中文
    nlp.max_length = 3000000  # 增加最大处理长度，如果文本很长
    
    # 步骤1: 先尝试使用spaCy进行基本分句
    doc = nlp(text)
    spacy_sentences = [sent.text.strip() for sent in doc.sents]

    # 步骤3: 进一步处理，拆分可能的段落（通过换行符）
    final_sentences = []
    for sent in spacy_sentences:
        # 按段落分隔符再次分割
        paragraph_splits = sent.split('\n\n')
        
        for paragraph in paragraph_splits:
            paragraph = paragraph.strip()
            if paragraph:  # 确保不是空段落
                # 如果有单个换行符，可能是段落内的分隔
                line_splits = paragraph.split('\n')
                for line in line_splits:
                    line = line.strip()
                    if line:  # 确保不是空行
                        final_sentences.append(line)
    
    return final_sentences

def read_text_file(file_path):
    """读取指定路径的文本文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dataclasses import dataclass
import json

@dataclass
class ChunkInfo:
    """存储块信息"""
    text: str
    start_token: int
    end_token: int
    embedding: np.ndarray = None

class LateChunkingProcessor:
    """迟分处理器 - 先编码整个文档，然后再分块"""
    
    def __init__(self, model_path: str, max_length: int = 8192):
        """
        初始化迟分处理器
        
        Args:
            model_path: BGE-M3模型路径
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # 加载模型和分词器
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on device: {self.device}")
    
    def encode_document(self, text: str) -> torch.Tensor:
        """
        编码整个文档
        
        Args:
            text: 输入文档文本
            
        Returns:
            文档的token级别的隐藏状态
        """
        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 获取隐藏状态
            outputs = self.model(**inputs, output_hidden_states=True)
            # 使用最后一层的隐藏状态
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
        return hidden_states, inputs['input_ids'], inputs['attention_mask']
    
    def create_chunks_by_sentences(self, text: str, chunk_size: int = 200) -> List[Tuple[str, int, int]]:
        """
        按句子创建块，返回每个块的文本和在原始token序列中的位置
        
        Args:
            text: 原始文本
            chunk_size: 每个块的大小（token数）
            
        Returns:
            [(chunk_text, start_token_idx, end_token_idx), ...]
        """
        # 分词获取token位置信息
        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.max_length)
        token_to_char = encoding.offset_mapping
        tokens = encoding['input_ids']
        
        chunks = []
        current_start = 1  # 跳过[CLS]
        
        while current_start < len(tokens) - 1:  # 保留[SEP]
            current_end = min(current_start + chunk_size, len(tokens) - 1)
            
            # 尽量在句子边界分割
            if current_end < len(tokens) - 1:
                # 寻找最近的句号或其他句子结束符
                for i in range(current_end, current_start, -1):
                    if i < len(token_to_char) and token_to_char[i] is not None:
                        char_start, char_end = token_to_char[i]
                        if char_end < len(text) and text[char_end-1:char_end+1] in ['。', '！', '？', '.', '!', '?']:
                            current_end = i + 1
                            break
            
            # 获取chunk对应的文本
            if current_start < len(token_to_char) and current_end <= len(token_to_char):
                start_char = token_to_char[current_start][0] if token_to_char[current_start] else 0
                end_char = token_to_char[current_end-1][1] if current_end > 0 and token_to_char[current_end-1] else len(text)
                chunk_text = text[start_char:end_char]
                
                chunks.append((chunk_text, current_start, current_end))
            
            current_start = current_end
        
        return chunks
    
    def extract_chunk_embeddings(self, hidden_states: torch.Tensor, 
                                attention_mask: torch.Tensor,
                                chunks: List[Tuple[str, int, int]]) -> List[ChunkInfo]:
        """
        从整体编码中提取每个块的embedding
        
        Args:
            hidden_states: 整个文档的隐藏状态
            attention_mask: 注意力掩码
            chunks: 块信息列表
            
        Returns:
            包含embedding的块信息列表
        """
        chunk_infos = []
        
        for chunk_text, start_idx, end_idx in chunks:
            # 提取块对应的隐藏状态
            chunk_hidden = hidden_states[0, start_idx:end_idx, :]  # [chunk_len, hidden_size]
            chunk_mask = attention_mask[0, start_idx:end_idx]  # [chunk_len]
            
            # 使用注意力掩码进行平均池化
            masked_hidden = chunk_hidden * chunk_mask.unsqueeze(-1).float()
            chunk_embedding = masked_hidden.sum(dim=0) / chunk_mask.sum().float()
            
            # 归一化
            chunk_embedding = F.normalize(chunk_embedding, p=2, dim=0)
            
            chunk_info = ChunkInfo(
                text=chunk_text,
                start_token=start_idx,
                end_token=end_idx,
                embedding=chunk_embedding.cpu().numpy()
            )
            chunk_infos.append(chunk_info)
        
        return chunk_infos
    
    def process_document(self, text: str, chunk_size: int = 200) -> List[ChunkInfo]:
        """
        完整的迟分处理流程
        
        Args:
            text: 输入文档
            chunk_size: 块大小
            
        Returns:
            处理后的块信息列表
        """
        print("Step 1: 编码整个文档...")
        hidden_states, input_ids, attention_mask = self.encode_document(text)
        
        print("Step 2: 创建文档块...")
        chunks = self.create_chunks_by_sentences(text, chunk_size)
        print(f"创建了 {len(chunks)} 个块")
        
        print("Step 3: 提取块级别的embedding...")
        chunk_infos = self.extract_chunk_embeddings(hidden_states, attention_mask, chunks)
        
        return chunk_infos
    
    def similarity_search(self, query: str, chunk_infos: List[ChunkInfo], top_k: int = 5) -> List[Tuple[ChunkInfo, float]]:
        """
        基于相似度搜索最相关的块
        
        Args:
            query: 查询文本
            chunk_infos: 块信息列表
            top_k: 返回top-k结果
            
        Returns:
            [(chunk_info, similarity_score), ...]
        """
        # 编码查询
        query_inputs = self.tokenizer(
            query,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
        
        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            # 使用[CLS]token的表示
            query_embedding = query_outputs.last_hidden_state[0, 0, :].cpu().numpy()
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 计算相似度
        similarities = []
        for chunk_info in chunk_infos:
            similarity = np.dot(query_embedding, chunk_info.embedding)
            similarities.append((chunk_info, float(similarity)))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def test_late_chunking():
    """测试迟分功能"""
    
    # 模型路径
    model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在 {model_path}")
        return
    
    # 创建处理器
    processor = LateChunkingProcessor(model_path)
    
    # 测试文档
    test_document = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    
    机器学习是人工智能的一个重要分支，它专注于使计算机系统能够从数据中自动学习和改进，而无需显式编程。机器学习算法通过分析大量数据来识别模式，并使用这些模式对新数据进行预测或分类。
    
    深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。常见的深度学习模型包括卷积神经网络（CNN）、循环神经网络（RNN）和Transformer等。
    
    自然语言处理（Natural Language Processing，NLP）是人工智能和语言学的交叉领域，专注于使计算机能够理解、解释和生成人类语言。NLP的应用包括机器翻译、情感分析、问答系统、文本摘要等。
    
    向量数据库是一种专门设计用于存储和检索高维向量数据的数据库系统。在人工智能应用中，向量数据库常用于存储文本、图像等数据的向量表示，支持快速的相似性搜索。常见的向量数据库包括Pinecone、Weaviate、Milvus等。
    """
    
    print("=" * 60)
    print("开始迟分测试")
    print("=" * 60)
    
    # 处理文档
    chunk_infos = processor.process_document(test_document, chunk_size=150)
    
    # 显示分块结果
    print("\n分块结果:")
    print("-" * 40)
    for i, chunk_info in enumerate(chunk_infos):
        print(f"块 {i+1} (tokens {chunk_info.start_token}-{chunk_info.end_token}):")
        print(f"文本: {chunk_info.text[:100]}...")
        print(f"Embedding shape: {chunk_info.embedding.shape}")
        print()
    
    # 测试相似性搜索
    queries = [
        "什么是深度学习？",
        "向量数据库有什么用？",
        "机器学习的应用领域",
        "自然语言处理包括哪些任务？"
    ]
    
    print("\n相似性搜索测试:")
    print("-" * 40)
    
    for query in queries:
        print(f"\n查询: {query}")
        results = processor.similarity_search(query, chunk_infos, top_k=3)
        for j, (chunk_info, score) in enumerate(results):
            print(f"  结果 {j+1} (相似度: {score:.4f}):")
            print(f"    {chunk_info.text[:80]}...")
        print()

def traditional_chunking_encode(processor, chunks: List[str]) -> List[np.ndarray]:
    """
    传统分块方法：分别编码每个块
    
    Args:
        processor: LateChunkingProcessor实例
        chunks: 文本块列表
        
    Returns:
        每个块的embedding列表
    """
    embeddings = []
    
    for chunk in chunks:
        # 分别编码每个块
        inputs = processor.tokenizer(
            chunk,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(processor.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = processor.model(**inputs)
            # 使用[CLS]token的表示
            chunk_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
            embeddings.append(chunk_embedding)
    
    return embeddings

def compare_with_traditional_chunking():
    """比较迟分和传统分块的效果"""
    
    model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
    
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在 {model_path}")
        return
    
    processor = LateChunkingProcessor(model_path)
    
    # 测试文档
    test_text = """
    大语言模型（Large Language Model，LLM）是一种基于深度学习的自然语言处理模型，具有数十亿甚至数千亿个参数。这些模型通过在大规模文本数据上进行预训练，学习语言的统计规律和语义知识。
    
    检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了信息检索和生成模型的技术。RAG系统首先从知识库中检索相关信息，然后将检索到的信息作为上下文提供给生成模型，以提高生成内容的准确性和相关性。
    """
    
    query = "RAG系统是如何工作的？"
    
    print("比较传统分块 vs 迟分:")
    print("=" * 50)
    
    # 编码查询
    print("编码查询...")
    query_inputs = processor.tokenizer(
        query,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    query_inputs = {k: v.to(processor.device) for k, v in query_inputs.items()}
    
    with torch.no_grad():
        query_outputs = processor.model(**query_inputs)
        query_embedding = query_outputs.last_hidden_state[0, 0, :].cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # 迟分方法
    print("\n1. 迟分方法:")
    chunk_infos = processor.process_document(test_text, chunk_size=100)
    late_results = processor.similarity_search(query, chunk_infos, top_k=len(chunk_infos))
    
    for i, (chunk_info, score) in enumerate(late_results):
        print(f"  结果 {i+1} (相似度: {score:.4f}): {chunk_info.text[:60]}...")
    
    # 传统分块方法
    print("\n2. 传统分块方法:")
    sentences = test_text.split('。')
    traditional_chunks = [s.strip() + '。' for s in sentences if s.strip()]
    
    print("  分别编码每个块...")
    traditional_embeddings = traditional_chunking_encode(processor, traditional_chunks)
    
    # 计算传统方法的相似度
    traditional_similarities = []
    for i, (chunk, embedding) in enumerate(zip(traditional_chunks, traditional_embeddings)):
        similarity = np.dot(query_embedding, embedding)
        traditional_similarities.append((chunk, float(similarity)))
    
    # 排序
    traditional_similarities.sort(key=lambda x: x[1], reverse=True)
    
    for i, (chunk, score) in enumerate(traditional_similarities):
        print(f"  结果 {i+1} (相似度: {score:.4f}): {chunk[:60]}...")
    


if __name__ == "__main__":
    # 运行测试
    test_late_chunking()
    
    print("\n" + "="*60)
    print("运行比较测试")
    print("="*60)
    
    # 运行比较测试
    compare_with_traditional_chunking()
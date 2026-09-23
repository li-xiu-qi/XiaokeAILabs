"""
嵌入模型封装模块
支持多种嵌入模型，包括硅基流动API和本地模型
"""

import os
from re import S
import time
from typing import List, Union, Protocol
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 加载环境变量
load_dotenv()


class EmbeddingModelProtocol(Protocol):
    """嵌入模型接口协议"""
    
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码句子为嵌入向量"""
        ...


class LocalSentenceTransformerModel:
    """本地SentenceTransformer模型包装器"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        return self.model.encode(sentences, **kwargs)


class GUIJI_BGE_M3_EmbeddingModel:
    """
    使用硅基流动API调用BAAI/bge-m3模型的嵌入类
    兼容SentenceTransformer的encode接口
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化BGE-M3嵌入模型
        
        Args:
            api_key: API密钥，如果为None则从环境变量获取
            base_url: API基础URL，如果为None则从环境变量获取
        """
        self.api_key = api_key or os.environ.get("GUIJI_API_KEY")
        self.base_url = base_url or os.environ.get("GUIJI_BASE_URL")
        
        if not self.api_key or not self.base_url:
            raise ValueError("API密钥和基础URL不能为空，请检查环境变量或传入参数")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model_name = "BAAI/bge-m3"   
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        使用API编码句子为嵌入向量
        
        Args:
            sentences: 要编码的句子或句子列表
            **kwargs: 其他参数（为了兼容性，实际不使用）
            
        Returns:
            嵌入向量数组
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 分批处理，硅基流动API限制批次大小最大为64
        batch_size = 32  # 使用较小的批次大小以确保稳定性
        all_embeddings = []
        
        print(f"正在处理 {len(sentences)} 个句子，分为 {(len(sentences) + batch_size - 1) // batch_size} 批次...")
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(sentences) + batch_size - 1) // batch_size
            
            print(f"处理第 {batch_num}/{total_batches} 批次 ({len(batch)} 个句子)...")
            
            # 重试机制处理速率限制
            max_retries = 3
            retry_delay = 1  # 初始延迟时间（秒）
            
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    # 提取嵌入向量
                    batch_embeddings = []
                    for item in response.data:
                        batch_embeddings.append(item.embedding)
                    
                    all_embeddings.extend(batch_embeddings)
                    break  # 成功后跳出重试循环
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # 检查是否是速率限制错误
                    if "rate_limit" in error_message.lower() or "429" in error_message or "too many requests" in error_message.lower():
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # 指数退避
                            print(f"遇到速率限制，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"达到最大重试次数，速率限制错误: {e}")
                            raise
                    
                    # 检查是否是批次大小错误
                    elif "batch size" in error_message and "maximum allowed" in error_message:
                        if batch_size > 1:
                            # 减小批次大小并重新处理当前批次
                            print(f"批次大小过大，从 {batch_size} 减少到 {batch_size // 2}")
                            batch_size = batch_size // 2
                            # 重新处理当前批次
                            remaining_sentences = sentences[i:]
                            remaining_embeddings = self._process_with_smaller_batch(remaining_sentences, batch_size)
                            all_embeddings.extend(remaining_embeddings)
                            break
                        else:
                            print(f"批次大小已经为1，仍然失败: {e}")
                            raise
                    else:
                        # 其他错误直接抛出
                        print(f"API调用失败: {e}")
                        raise
            
            # 添加小延迟避免过于频繁的请求
            time.sleep(0.1)
        
        if len(all_embeddings) != len(sentences):
            raise ValueError(f"嵌入向量数量({len(all_embeddings)})与输入句子数量({len(sentences)})不匹配")
        
        return np.array(all_embeddings)
    
    def _process_with_smaller_batch(self, sentences: List[str], batch_size: int) -> List[List[float]]:
        """
        使用更小的批次大小处理剩余句子
        """
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    batch_embeddings = []
                    for item in response.data:
                        batch_embeddings.append(item.embedding)
                    
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    error_message = str(e)
                    
                    if "rate_limit" in error_message.lower() or "429" in error_message:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"速率限制，等待 {wait_time} 秒...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise
                    else:
                        raise
            
            time.sleep(0.1)
        
        return all_embeddings


def load_embedding_model(model_type: str = "local", model_name: str = None, api_key: str = None, base_url: str = None):
    """
    根据类型加载嵌入模型
    model_type: "local" 或 "api"
    model_name: 本地模型名（如 "BAAI/bge-m3"）
    api_key, base_url: 远程API参数
    """
    if model_type == "local":
        if not model_name:
            raise ValueError("本地模型需要指定 model_name")
        return LocalSentenceTransformerModel(model_name)
    elif model_type == "api":
        return GUIJI_BGE_M3_EmbeddingModel(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

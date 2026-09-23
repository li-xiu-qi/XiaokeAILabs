"""
嵌入模型封装模块
支持多种嵌入模型，包括硅基流动API和本地模型
"""

import os
import time
from typing import List, Union
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 加载环境变量
load_dotenv()


class BGE_M3_EmbeddingModel:
    """
    使用硅基流动API调用BAAI/bge-m3模型的嵌入类
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


class LocalEmbeddingModel:
    """
    本地SentenceTransformer嵌入模型封装类
    """
    
    def __init__(self, model_name: str = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"):
        """
        初始化本地嵌入模型
        
        Args:
            model_name: 模型名称或路径
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            print(f"本地模型 {model_name} 加载成功")
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            raise
    
    def encode(self, sentences: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray:
        """
        使用本地模型编码句子为嵌入向量
        
        Args:
            sentences: 要编码的句子或句子列表
            convert_to_numpy: 是否转换为numpy数组
            
        Returns:
            嵌入向量数组
        """
        return self.model.encode(sentences, convert_to_numpy=convert_to_numpy)


class EmbeddingModelFactory:
    """
    嵌入模型工厂类，用于创建不同类型的嵌入模型
    """
    
    @staticmethod
    def create_model(model_type: str = "bge_m3", **kwargs):
        """
        创建嵌入模型
        
        Args:
            model_type: 模型类型，支持 "bge_m3" 和 "local"
            **kwargs: 模型特定的参数
            
        Returns:
            嵌入模型实例
        """
        if model_type == "bge_m3":
            try:
                return BGE_M3_EmbeddingModel(**kwargs)
            except Exception as e:
                print(f"创建BGE-M3模型失败: {e}")
                print("回退到本地模型...")
                return LocalEmbeddingModel()
        
        elif model_type == "local":
            model_name = kwargs.get("model_name", r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3")
            return LocalEmbeddingModel(model_name)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


def get_default_embedding_model(api_key: str = None, base_url: str = None):
    """
    获取默认的嵌入模型实例
    优先使用BGE-M3 API模型，失败时回退到本地模型
    
    Args:
        api_key: API密钥，如果为None则从环境变量获取
        base_url: API基础URL，如果为None则从环境变量获取
    
    Returns:
        嵌入模型实例
    """
    try:
        # 优先使用硅基流动的BGE-M3模型
        model = EmbeddingModelFactory.create_model("bge_m3", api_key=api_key, base_url=base_url)
        print("使用硅基流动API的BAAI/bge-m3模型")
        return model
    except Exception as e:
        print(f"无法初始化API模型，回退到本地模型: {e}")
        # 回退到本地模型
        model_path = os.environ.get("EMBEDDING_MODEL_PATH",  r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3")
        return EmbeddingModelFactory.create_model("local", model_name=model_path)


# 测试下api是否能用
if __name__ == "__main__":
    try:
        embedding_model = get_default_embedding_model()
        test_sentences = ["你好，世界！", "这是一个测试句子。", "BAAI/bge-m3模型工作正常。"]
        embeddings = embedding_model.encode(test_sentences)
        print("嵌入向量形状:", embeddings.shape)
    except Exception as e:
        print(f"测试嵌入模型失败: {e}")
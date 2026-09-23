"""
带缓存的嵌入模型封装模块
使用diskcache来缓存嵌入结果，避免重复计算
"""

import os
import hashlib
import json
from typing import List, Union, Optional
import numpy as np
import diskcache
from dotenv import load_dotenv
from embedding_models import get_default_embedding_model, EmbeddingModelFactory

# 加载环境变量
load_dotenv()


class CachedEmbeddingModel:
    """
    带缓存的嵌入模型包装器
    """
    
    def __init__(
        self, 
        base_model=None, 
        cache_dir: str = "./embedding_cache",
        cache_size_limit: int = 1024 * 1024 * 1024,  # 1GB缓存限制
        enable_cache: bool = True
    ):
        """
        初始化带缓存的嵌入模型
        
        Args:
            base_model: 基础嵌入模型，如果为None则使用默认模型
            cache_dir: 缓存目录路径
            cache_size_limit: 缓存大小限制（字节）
            enable_cache: 是否启用缓存
        """
        self.base_model = base_model or get_default_embedding_model()
        self.enable_cache = enable_cache
        
        if self.enable_cache:
            # 初始化缓存
            self.cache = diskcache.Cache(
                directory=cache_dir,
                size_limit=cache_size_limit,
                eviction_policy='least-recently-used'
            )
            print(f"缓存已初始化，目录: {cache_dir}, 大小限制: {cache_size_limit // (1024*1024)}MB")
            print(f"当前缓存中有 {len(self.cache)} 个条目")
        else:
            self.cache = None
    
    def _generate_cache_key(self, sentences: List[str], model_info: str = None) -> str:
        """
        为句子列表生成缓存键
        
        Args:
            sentences: 句子列表
            model_info: 模型信息（用于区分不同模型的缓存）
            
        Returns:
            缓存键
        """
        # 使用模型名称和句子内容生成哈希
        if model_info is None:
            model_info = getattr(self.base_model, 'model_name', 'default_model')
        
        content = {
            'model': model_info,
            'sentences': sentences
        }
        
        # 转换为JSON字符串并生成MD5哈希
        content_str = json.dumps(content, ensure_ascii=False, sort_keys=True)
        cache_key = hashlib.md5(content_str.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def _split_into_batches(self, sentences: List[str], batch_size: int = 32) -> List[List[str]]:
        """
        将句子列表分割成批次
        """
        batches = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batches.append(batch)
        return batches
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码句子为嵌入向量，使用缓存优化
        
        Args:
            sentences: 要编码的句子或句子列表
            **kwargs: 其他参数
            
        Returns:
            嵌入向量数组
        """
        # 统一处理输入格式
        if isinstance(sentences, str):
            sentences = [sentences]
        
        if not sentences:
            return np.array([])
        
        print(f"开始处理 {len(sentences)} 个句子的嵌入...")
        
        # 如果不启用缓存，直接调用基础模型
        if not self.enable_cache:
            return self.base_model.encode(sentences, **kwargs)
        
        model_info = getattr(self.base_model, 'model_name', 'default_model')
        
        # 第一步：检查每个句子的缓存状态
        sentence_cache_status = []
        cached_embeddings = {}
        uncached_sentences = []
        uncached_indices = []
        
        for i, sentence in enumerate(sentences):
            cache_key = self._generate_cache_key([sentence], model_info)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                # 缓存命中
                sentence_cache_status.append(True)
                cached_embeddings[i] = np.array(cached_result)[0]  # 取第一个（因为缓存的是单句列表）
            else:
                # 缓存未命中
                sentence_cache_status.append(False)
                uncached_sentences.append(sentence)
                uncached_indices.append(i)
        
        cache_hits = sum(sentence_cache_status)
        cache_misses = len(uncached_sentences)
        
        print(f"缓存检查完成: {cache_hits} 命中, {cache_misses} 未命中")
        
        # 第二步：对未缓存的句子进行批量处理
        uncached_embeddings = {}
        if uncached_sentences:
            print(f"开始处理 {len(uncached_sentences)} 个未缓存的句子...")
            
            # 分批处理未缓存的句子
            batch_size = 64
            batches = self._split_into_batches(uncached_sentences, batch_size)
            batch_start_idx = 0
            
            for batch_idx, batch in enumerate(batches):
                print(f"  处理第 {batch_idx + 1}/{len(batches)} 批次 ({len(batch)} 个句子)...")
                
                # 调用基础模型
                batch_embeddings = self.base_model.encode(batch, **kwargs)
                
                # 保存每个句子的嵌入到缓存和结果字典
                for local_idx, sentence in enumerate(batch):
                    global_idx = uncached_indices[batch_start_idx + local_idx]
                    embedding = batch_embeddings[local_idx]
                    
                    # 保存到结果字典
                    uncached_embeddings[global_idx] = embedding
                    
                    # 保存到缓存（单独缓存每个句子）
                    try:
                        cache_key = self._generate_cache_key([sentence], model_info)
                        self.cache.set(cache_key, [embedding.tolist()])
                    except Exception as e:
                        print(f"    保存句子到缓存失败: {e}")
                
                batch_start_idx += len(batch)
        
        # 第三步：按原始顺序重建完整的嵌入向量数组
        final_embeddings = []
        for i in range(len(sentences)):
            if i in cached_embeddings:
                final_embeddings.append(cached_embeddings[i])
            elif i in uncached_embeddings:
                final_embeddings.append(uncached_embeddings[i])
            else:
                raise ValueError(f"句子索引 {i} 既不在缓存中也不在新计算的结果中")
        
        final_embeddings = np.array(final_embeddings)
        
        # 打印缓存统计信息
        total_sentences = len(sentences)
        hit_rate = cache_hits / total_sentences * 100 if total_sentences > 0 else 0
        print(f"\n缓存统计:")
        print(f"  总句子数: {total_sentences}")
        print(f"  缓存命中: {cache_hits} ({hit_rate:.1f}%)")
        print(f"  缓存未命中: {cache_misses} ({100-hit_rate:.1f}%)")
        print(f"  当前缓存条目数: {len(self.cache)}")
        
        return final_embeddings
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache is not None:
            self.cache.clear()
            print("缓存已清空")
    
    def get_cache_info(self):
        """获取缓存信息"""
        if self.cache is not None:
            info = {
                'cache_size': len(self.cache),
                'cache_directory': self.cache.directory,
                'size_limit': self.cache.size_limit,
                'eviction_policy': self.cache.eviction_policy
            }
            return info
        return None
    
    def warmup_cache(self, sentences_list: List[List[str]]):
        """
        预热缓存：批量处理句子列表
        
        Args:
            sentences_list: 句子列表的列表
        """
        print("开始预热缓存...")
        
        for i, sentences in enumerate(sentences_list):
            print(f"预热批次 {i+1}/{len(sentences_list)}")
            self.encode(sentences)
        
        print("缓存预热完成")


class CachedEmbeddingModelFactory:
    """
    带缓存的嵌入模型工厂类
    """
    
    @staticmethod
    def create_cached_model(
        model_type: str = "bge_m3",
        cache_dir: str = "./embedding_cache",
        cache_size_limit: int = 1024 * 1024 * 1024,
        enable_cache: bool = True,
        **kwargs
    ):
        """
        创建带缓存的嵌入模型
        
        Args:
            model_type: 基础模型类型
            cache_dir: 缓存目录
            cache_size_limit: 缓存大小限制
            enable_cache: 是否启用缓存
            **kwargs: 基础模型的其他参数
            
        Returns:
            带缓存的嵌入模型实例
        """
        # 创建基础模型
        base_model = EmbeddingModelFactory.create_model(model_type, **kwargs)
        
        # 包装为带缓存的模型
        cached_model = CachedEmbeddingModel(
            base_model=base_model,
            cache_dir=cache_dir,
            cache_size_limit=cache_size_limit,
            enable_cache=enable_cache
        )
        
        return cached_model


def get_cached_embedding_model(
    cache_dir: str = "./embedding_cache",
    cache_size_limit: int = 1024 * 1024 * 1024,
    enable_cache: bool = True
):
    """
    获取默认的带缓存嵌入模型
    
    Args:
        cache_dir: 缓存目录
        cache_size_limit: 缓存大小限制（字节）
        enable_cache: 是否启用缓存
        
    Returns:
        带缓存的嵌入模型实例
    """
    return CachedEmbeddingModelFactory.create_cached_model(
        model_type="bge_m3",
        cache_dir=cache_dir,
        cache_size_limit=cache_size_limit,
        enable_cache=enable_cache
    )


# 测试缓存功能
if __name__ == "__main__":
    # 创建带缓存的模型
    cached_model = get_cached_embedding_model(
        cache_dir="./test_embedding_cache",
        enable_cache=True
    )
    
    # 测试句子
    test_sentences = [
        "这是第一个测试句子。",
        "这是第二个测试句子。",
        "这是第三个测试句子。",
        "这是第四个测试句子。"
    ]
    
    print("第一次调用（应该缓存未命中）:")
    embeddings1 = cached_model.encode(test_sentences)
    print(f"嵌入向量形状: {embeddings1.shape}")
    
    print("\n第二次调用（应该缓存命中）:")
    embeddings2 = cached_model.encode(test_sentences)
    print(f"嵌入向量形状: {embeddings2.shape}")
    
    # 验证结果一致性
    print(f"\n结果一致性检查: {np.allclose(embeddings1, embeddings2)}")
    
    # 打印缓存信息
    cache_info = cached_model.get_cache_info()
    print(f"\n缓存信息: {cache_info}")

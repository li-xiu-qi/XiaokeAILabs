# -*- coding: utf-8 -*-
"""
使用《红楼梦》文本测试迟分(Late Chunking)和传统分割方法的对比
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from dataclasses import dataclass
import time

# 添加项目根目录到路径
project_root = r"C:\Users\k\Documents\project\programming_project\python_project\importance\XiaokeAILabs"
sys.path.append(project_root)

# 导入迟分处理器
from datas.test_late_chunking.test_late_chunking import LateChunkingProcessor, ChunkInfo

def load_hongloumeng_text():
    """加载《红楼梦》文本"""
    hongloumeng_path = os.path.join(project_root, "datas", "test_late_chunking", "红楼梦.txt")
    
    if not os.path.exists(hongloumeng_path):
        print(f"错误：找不到《红楼梦》文件: {hongloumeng_path}")
        return None
    
    try:
        with open(hongloumeng_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"成功加载《红楼梦》文本，总长度: {len(content)} 字符")
        return content
    
    except Exception as e:
        print(f"读取《红楼梦》文件时出错: {e}")
        return None

def extract_sample_text(full_text: str, start_chars: int = 1000, length: int = 8000) -> str:
    """从完整文本中提取样本用于测试，默认提取8000字符"""
    if not full_text:
        return ""
    
    # 从指定位置开始提取
    sample = full_text[start_chars:start_chars + length]
    
    # 尝试在句子边界结束
    for end_char in ['。', '！', '？']:
        last_sentence_end = sample.rfind(end_char)
        if last_sentence_end > length * 0.8:  # 至少保留80%的长度
            sample = sample[:last_sentence_end + 1]
            break
    
    return sample.strip()

def traditional_chunking_encode(processor: LateChunkingProcessor, chunks: List[str]) -> List[np.ndarray]:
    """
    传统分块方法：分别编码每个块
    
    Args:
        processor: LateChunkingProcessor实例
        chunks: 文本块列表
        
    Returns:
        每个块的embedding列表
    """
    embeddings = []
    
    print(f"  正在编码 {len(chunks)} 个传统分块...")
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"    进度: {i+1}/{len(chunks)}")
        
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

def create_traditional_chunks(text: str, chunk_size: int = 800) -> List[str]:
    """创建传统的文本分块，默认块大小为800字符"""
    # 按句子分割
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in ['。', '！', '？', '…']:
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # 如果最后还有未完成的句子
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # 将句子组合成块
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def test_search_quality(query: str, processor: LateChunkingProcessor, 
                       late_chunk_infos: List[ChunkInfo], 
                       traditional_chunks: List[str], 
                       traditional_embeddings: List[np.ndarray]) -> None:
    """比较两种方法的搜索质量"""
    
    print(f"\n查询: '{query}'")
    print("-" * 50)
    
    # 编码查询
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
    
    # 迟分方法搜索
    late_results = processor.similarity_search(query, late_chunk_infos, top_k=3)
    
    print("🔥 迟分方法结果:")
    for i, (chunk_info, score) in enumerate(late_results):
        preview = chunk_info.text.replace('\n', ' ')[:100]
        print(f"  {i+1}. [相似度: {score:.4f}] {preview}...")
    
    # 传统方法搜索
    traditional_similarities = []
    for chunk, embedding in zip(traditional_chunks, traditional_embeddings):
        similarity = np.dot(query_embedding, embedding)
        traditional_similarities.append((chunk, float(similarity)))
    
    traditional_similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📚 传统分块结果:")
    for i, (chunk, score) in enumerate(traditional_similarities[:3]):
        preview = chunk.replace('\n', ' ')[:100]
        print(f"  {i+1}. [相似度: {score:.4f}] {preview}...")

def run_hongloumeng_test():
    """运行《红楼梦》测试"""
    
    print("=" * 80)
    print("🏛️  《红楼梦》迟分 vs 传统分割对比测试")
    print("=" * 80)
    
    # 加载文本
    full_text = load_hongloumeng_text()
    if not full_text:
        return
      # 提取测试样本 (使用更长的文本)
    test_text = extract_sample_text(full_text, start_chars=500, length=8000)
    print(f"\n📖 测试文本长度: {len(test_text)} 字符")
    print(f"📝 文本预览: {test_text[:200]}...")
    
    # 初始化模型
    model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型路径不存在 {model_path}")
        return
    
    print(f"\n🚀 正在加载模型: {model_path}")
    processor = LateChunkingProcessor(model_path, max_length=8192)
    
    # 使用单一块大小进行测试
    chunk_size = 800    # 使用单一块大小进行测试
    chunk_size = 800
    
    print(f"\n" + "="*60)
    print(f"📏 测试块大小: {chunk_size} tokens")
    print("="*60)
    
    # 迟分方法
    print("\n⏱️  迟分方法处理中...")
    start_time = time.time()
    late_chunk_infos = processor.process_document(test_text, chunk_size=chunk_size)
    late_time = time.time() - start_time
    
    print(f"✅ 迟分完成 - 耗时: {late_time:.2f}秒, 生成块数: {len(late_chunk_infos)}")
    
    # 传统分块方法
    print("\n⏱️  传统分块方法处理中...")
    start_time = time.time()
    traditional_chunks = create_traditional_chunks(test_text, chunk_size=chunk_size)
    traditional_embeddings = traditional_chunking_encode(processor, traditional_chunks)
    traditional_time = time.time() - start_time
    
    print(f"✅ 传统分块完成 - 耗时: {traditional_time:.2f}秒, 生成块数: {len(traditional_chunks)}")
    
    # 性能对比
    print(f"\n📊 性能对比:")
    print(f"  • 迟分方法: {late_time:.2f}秒")
    print(f"  • 传统方法: {traditional_time:.2f}秒")
    print(f"  • 速度提升: {traditional_time/late_time:.2f}x" if late_time > 0 else "  • 无法计算速度提升")
    
    # 测试搜索质量
    queries = [
        "贾宝玉的性格特点",
        "林黛玉进贾府",
        "贾雨村的仕途经历",
        "荣国府的繁华",
        "甄士隐的故事"
    ]
    
    print(f"\n🔍 搜索质量测试 (块大小: {chunk_size})")
    for query in queries:
        test_search_quality(query, processor, late_chunk_infos, 
                          traditional_chunks, traditional_embeddings)
    
    print(f"\n💾 块信息统计:")
    print(f"  迟分方法:")
    print(f"    - 平均块长度: {np.mean([len(info.text) for info in late_chunk_infos]):.1f} 字符")
    print(f"    - 最短块: {min([len(info.text) for info in late_chunk_infos])} 字符")
    print(f"    - 最长块: {max([len(info.text) for info in late_chunk_infos])} 字符")
    
    print(f"  传统方法:")
    print(f"    - 平均块长度: {np.mean([len(chunk) for chunk in traditional_chunks]):.1f} 字符")
    print(f"    - 最短块: {min([len(chunk) for chunk in traditional_chunks])} 字符")
    print(f"    - 最长块: {max([len(chunk) for chunk in traditional_chunks])} 字符")

def test_different_text_samples():
    """测试不同的文本样本"""
    
    print("\n" + "="*80)
    print("📚 不同文本片段测试")
    print("="*80)
    
    full_text = load_hongloumeng_text()
    if not full_text:
        return
    
    # 测试不同章节的片段
    sample_configs = [
        {"start": 1000, "length": 2000, "name": "第一章开头"},
        {"start": 5000, "length": 2000, "name": "第一章中段"},
        {"start": 15000, "length": 2000, "name": "第二章内容"},
    ]
    
    model_path = r"C:\Users\k\Desktop\BaiduSyncdisk\baidu_sync_documents\hf_models\bge-m3"
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型路径不存在 {model_path}")
        return
    
    processor = LateChunkingProcessor(model_path, max_length=4096)
    
    for config in sample_configs:
        print(f"\n📖 测试片段: {config['name']}")
        print("-" * 40)
        
        sample_text = extract_sample_text(full_text, config['start'], config['length'])
        print(f"文本长度: {len(sample_text)} 字符")
          # 迟分处理
        late_chunks = processor.process_document(sample_text, chunk_size=800)
        
        # 传统分块
        traditional_chunks = create_traditional_chunks(sample_text, chunk_size=800)
        
        print(f"迟分块数: {len(late_chunks)}, 传统块数: {len(traditional_chunks)}")
        
        # 简单搜索测试
        query = "贾宝玉"
        if query in sample_text:
            late_results = processor.similarity_search(query, late_chunks, top_k=1)
            if late_results:
                print(f"查询'{query}' - 迟分最佳匹配相似度: {late_results[0][1]:.4f}")

if __name__ == "__main__":
    try:
        # 运行主测试
        run_hongloumeng_test()
        
        # 运行不同样本测试
        test_different_text_samples()
        
        print("\n" + "="*80)
        print("🎉 测试完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

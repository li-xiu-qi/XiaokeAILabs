#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试改进后的缓存策略
"""

import os
import time
import numpy as np
from cached_embedding_models import get_cached_embedding_model

def test_cache_strategy():
    """测试改进后的缓存策略"""
    
    # 创建带缓存的模型
    print("创建带缓存的嵌入模型...")
    cached_model = get_cached_embedding_model(
        cache_dir="./test_cache_strategy",
        enable_cache=True
    )
    
    # 清空缓存开始测试
    cached_model.clear_cache()
    
    # 测试句子
    test_sentences = [
        "红楼梦是中国古典文学四大名著之一。",
        "作者曹雪芹用细腻的笔触描绘了贾府的兴衰。",
        "林黛玉是书中的女主角之一。",
        "贾宝玉与林黛玉的爱情故事令人动容。",
        "王熙凤是贾府的管家奶奶。",
        "这部小说反映了封建社会的衰落。",
        "书中有很多经典的诗词。",
        "红楼梦的艺术价值极高。"
    ]
    
    print(f"\n=== 第一次调用（全部缓存未命中）===")
    start_time = time.time()
    embeddings1 = cached_model.encode(test_sentences)
    time1 = time.time() - start_time
    print(f"第一次调用耗时: {time1:.2f}秒")
    print(f"嵌入向量形状: {embeddings1.shape}")
    
    print(f"\n=== 第二次调用（全部缓存命中）===")
    start_time = time.time()
    embeddings2 = cached_model.encode(test_sentences)
    time2 = time.time() - start_time
    print(f"第二次调用耗时: {time2:.2f}秒")
    print(f"嵌入向量形状: {embeddings2.shape}")
    
    # 验证结果一致性
    is_consistent = np.allclose(embeddings1, embeddings2, rtol=1e-6)
    print(f"\n结果一致性检查: {is_consistent}")
    
    if is_consistent:
        print("✅ 缓存策略正确：两次调用结果完全一致")
    else:
        print("❌ 缓存策略有问题：两次调用结果不一致")
    
    # 测试部分缓存的情况
    print(f"\n=== 第三次调用（部分缓存命中）===")
    mixed_sentences = [
        "红楼梦是中国古典文学四大名著之一。",  # 已缓存
        "这是一个新的测试句子。",                # 未缓存
        "贾宝玉与林黛玉的爱情故事令人动容。",    # 已缓存
        "另一个全新的句子用于测试。",            # 未缓存
        "书中有很多经典的诗词。",                # 已缓存
    ]
    
    start_time = time.time()
    embeddings3 = cached_model.encode(mixed_sentences)
    time3 = time.time() - start_time
    print(f"第三次调用耗时: {time3:.2f}秒")
    print(f"嵌入向量形状: {embeddings3.shape}")
    
    # 验证顺序正确性
    print(f"\n=== 验证顺序正确性 ===")
    # 分别获取已知句子的嵌入
    sentence1_embedding = cached_model.encode([test_sentences[0]])  # "红楼梦是中国古典文学四大名著之一。"
    sentence4_embedding = cached_model.encode([test_sentences[3]])  # "贾宝玉与林黛玉的爱情故事令人动容。"
    sentence7_embedding = cached_model.encode([test_sentences[6]])  # "书中有很多经典的诗词。"
    
    # 检查在混合调用中的顺序是否正确
    order_check1 = np.allclose(embeddings3[0], sentence1_embedding[0])
    order_check2 = np.allclose(embeddings3[2], sentence4_embedding[0])
    order_check3 = np.allclose(embeddings3[4], sentence7_embedding[0])
    
    print(f"第1个句子顺序正确: {order_check1}")
    print(f"第3个句子顺序正确: {order_check2}")
    print(f"第5个句子顺序正确: {order_check3}")
    
    all_order_correct = order_check1 and order_check2 and order_check3
    if all_order_correct:
        print("✅ 顺序检查通过：缓存和新计算的结果按正确顺序排列")
    else:
        print("❌ 顺序检查失败：结果顺序有问题")
    
    # 性能对比
    print(f"\n=== 性能对比 ===")
    print(f"第一次调用（无缓存）: {time1:.2f}秒")
    print(f"第二次调用（全缓存）: {time2:.2f}秒")
    print(f"第三次调用（部分缓存）: {time3:.2f}秒")
    
    if time2 < time1:
        speedup = time1 / time2
        print(f"缓存加速比: {speedup:.2f}x")
    
    # 打印最终缓存信息
    cache_info = cached_model.get_cache_info()
    print(f"\n=== 最终缓存信息 ===")
    print(f"缓存条目数: {cache_info['cache_size']}")
    print(f"缓存目录: {cache_info['cache_directory']}")

if __name__ == "__main__":
    test_cache_strategy()

# -*- coding: utf-8 -*-
"""
红楼梦文本语义分割测试脚本
"""

import os
# 在导入任何库之前设置环境变量，防止tokenizers并行性警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from typing import List
from test_semantic_splitter import SemanticSplitter, custom_sentence_splitter


def read_hongloumeng_text(file_path: str) -> str:
    """读取红楼梦文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return ""


def test_hongloumeng_splitting():
    """测试红楼梦文本的语义分割"""
    print("=" * 80)
    print("开始测试红楼梦文本的语义分割")
    print("=" * 80)
    
    # 文件路径
    file_path = "红楼梦.txt"
    
    # 读取文本
    print(f"正在读取文件: {file_path}")
    text = read_hongloumeng_text(file_path)
    
    if not text:
        print("无法读取文本文件，请检查文件路径")
        return
    
    print(f"文本总长度: {len(text)} 字符")
    print(f"文本前500字符: {text[:500]}...")
    print()
    
    # 第一步：句子分割
    print("第一步：使用spaCy进行句子分割...")
    start_time = time.time()
    
    try:
        sentences = custom_sentence_splitter(text)
        sentence_split_time = time.time() - start_time
        
        print(f"句子分割完成，用时: {sentence_split_time:.2f} 秒")
        print(f"总共分割出 {len(sentences)} 个句子")
        
        # 显示前10个句子作为示例
        print("\n前10个分割出的句子:")
        for i, sent in enumerate(sentences[:10]):
            print(f"{i+1:2d}: {sent}")
        
        print(f"\n后10个分割出的句子:")
        for i, sent in enumerate(sentences[-10:], len(sentences)-9):
            print(f"{i:2d}: {sent}")
            
    except Exception as e:
        print(f"句子分割失败: {e}")
        print("可能需要安装spaCy模型：python -m spacy download xx_sent_ud_sm")
        return
    
    print("\n" + "="*60)
    
    # 第二步：语义分割（使用较小的文本片段进行测试）
    print("第二步：使用语义分割器进行文本分块...")
    
    # 为了演示，我们只使用前100个句子进行语义分割
    test_sentences = sentences[:100]
    print(f"使用前 {len(test_sentences)} 个句子进行语义分割测试")
    
    try:
        # 初始化语义分割器
        splitter = SemanticSplitter(
            initial_threshold=0.4,    # 初始相似度阈值
            appending_threshold=0.5,  # 附加阈值
            merging_threshold=0.5,    # 合并阈值
            max_chunk_size=1000,      # 最大块大小
        )
        
        start_time = time.time()
        semantic_chunks = splitter.process_sentences(test_sentences)
        semantic_split_time = time.time() - start_time
        
        print(f"语义分割完成，用时: {semantic_split_time:.2f} 秒")
        print(f"总共分割出 {len(semantic_chunks)} 个语义块")
        
        # 显示分割结果统计
        chunk_lengths = [len(chunk) for chunk in semantic_chunks]
        print(f"\n语义块长度统计:")
        print(f"  平均长度: {sum(chunk_lengths)/len(chunk_lengths):.1f} 字符")
        print(f"  最短块: {min(chunk_lengths)} 字符")
        print(f"  最长块: {max(chunk_lengths)} 字符")
        
        # 显示前5个语义块
        print(f"\n前5个语义块内容:")
        for i, chunk in enumerate(semantic_chunks[:5]):
            print(f"\n【块 {i+1}】(长度: {len(chunk)} 字符)")
            print(f"{chunk}")
            print("-" * 40)
            
    except Exception as e:
        print(f"语义分割失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("测试完成！")
    
    # 总结
    print(f"\n总结:")
    print(f"  原始文本长度: {len(text)} 字符")
    print(f"  句子分割数量: {len(sentences)} 个句子")
    print(f"  测试句子数量: {len(test_sentences)} 个句子")
    print(f"  语义块数量: {len(semantic_chunks)} 个块")
    print(f"  句子分割用时: {sentence_split_time:.2f} 秒")
    print(f"  语义分割用时: {semantic_split_time:.2f} 秒")


def test_different_thresholds():
    """测试不同阈值参数对分割效果的影响"""
    print("\n" + "="*80)
    print("测试不同阈值参数对分割效果的影响")
    print("="*80)
    
    file_path = "红楼梦.txt"
    text = read_hongloumeng_text(file_path)
    
    if not text:
        return
    
    # 句子分割
    sentences = custom_sentence_splitter(text)
    test_sentences = sentences[:50]  # 使用前50个句子快速测试
    
    # 测试不同的阈值组合
    threshold_configs = [
        {"initial": 0.3, "appending": 0.4, "merging": 0.4, "name": "低阈值(更多块)"},
        {"initial": 0.4, "appending": 0.5, "merging": 0.5, "name": "中等阈值"},
        {"initial": 0.6, "appending": 0.7, "merging": 0.7, "name": "高阈值(更少块)"},
    ]
    
    for config in threshold_configs:
        print(f"\n测试配置: {config['name']}")
        print(f"  初始阈值: {config['initial']}")
        print(f"  附加阈值: {config['appending']}")  
        print(f"  合并阈值: {config['merging']}")
        
        try:
            splitter = SemanticSplitter(
                initial_threshold=config['initial'],
                appending_threshold=config['appending'],
                merging_threshold=config['merging'],
                max_chunk_size=800,
            )
            
            chunks = splitter.process_sentences(test_sentences)
            chunk_lengths = [len(chunk) for chunk in chunks]
            
            print(f"  结果: {len(chunks)} 个块")
            print(f"  平均长度: {sum(chunk_lengths)/len(chunk_lengths):.1f} 字符")
            print(f"  长度范围: {min(chunk_lengths)} - {max(chunk_lengths)} 字符")
            
        except Exception as e:
            print(f"  测试失败: {e}")


if __name__ == "__main__":
    # 确保在正确的目录中运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 运行主要测试
    test_hongloumeng_splitting()
    
    # 运行阈值对比测试
    test_different_thresholds()

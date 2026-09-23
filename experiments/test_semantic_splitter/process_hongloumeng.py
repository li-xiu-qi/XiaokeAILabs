#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红楼梦文本语义分割处理脚本
使用语义分割器将红楼梦文本分割成语义连贯的块，并保存到多个文件中
"""

import os
import time
from pathlib import Path
from test_semantic_splitter import SemanticSplitter, custom_sentence_splitter, read_text_file


def create_output_directory(base_dir: str, output_dir: str = "红楼梦_分割结果") -> Path:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录
        output_dir: 输出目录名称
        
    Returns:
        输出目录路径
    """
    output_path = Path(base_dir) / output_dir
    output_path.mkdir(exist_ok=True)
    return output_path


def save_chunks_to_files(chunks: list, output_dir: Path, prefix: str = "chunk"):
    """
    将分割的文本块保存到多个txt文件中
    
    Args:
        chunks: 分割后的文本块列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    print(f"\n开始保存 {len(chunks)} 个文本块到 {output_dir}")
    
    for i, chunk in enumerate(chunks, 1):
        filename = f"{prefix}_{i:03d}.txt"
        file_path = output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        
        print(f"保存文件: {filename} (长度: {len(chunk)} 字符)")
    
    print(f"\n所有文件已保存到: {output_dir}")


def analyze_chunks(chunks: list):
    """
    分析分割结果的统计信息
    
    Args:
        chunks: 分割后的文本块列表
    """
    if not chunks:
        print("没有分割结果可分析")
        return
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    print("\n=== 分割结果统计 ===")
    print(f"总块数: {len(chunks)}")
    print(f"平均长度: {sum(chunk_lengths) / len(chunk_lengths):.1f} 字符")
    print(f"最短块: {min(chunk_lengths)} 字符")
    print(f"最长块: {max(chunk_lengths)} 字符")
    print(f"总字符数: {sum(chunk_lengths)} 字符")
    
    # 长度分布统计
    length_ranges = [
        (0, 200), (200, 400), (400, 600), 
        (600, 800), (800, 1000), (1000, 1200), 
        (1200, 1500), (1500, float('inf'))
    ]
    
    print("\n长度分布:")
    for min_len, max_len in length_ranges:
        if max_len == float('inf'):
            count = sum(1 for length in chunk_lengths if length >= min_len)
            print(f"  {min_len}+ 字符: {count} 块")
        else:
            count = sum(1 for length in chunk_lengths if min_len <= length < max_len)
            print(f"  {min_len}-{max_len-1} 字符: {count} 块")


def main():
    """主函数"""
    print("=== 红楼梦语义分割处理 ===")
    
    # 文件路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hongloumeng_file = os.path.join(current_dir, "红楼梦.txt")
    
    # 检查文件是否存在
    if not os.path.exists(hongloumeng_file):
        print(f"错误: 找不到文件 {hongloumeng_file}")
        return
    
    # 创建输出目录
    output_dir = create_output_directory(current_dir)
    
    try:
        # 1. 读取红楼梦文本
        print(f"正在读取文件: {hongloumeng_file}")
        text = read_text_file(hongloumeng_file)
        print(f"文件读取成功，总字符数: {len(text)}")
        
        # 2. 进行句子分割
        print("\n正在进行句子分割...")
        start_time = time.time()
        sentences = custom_sentence_splitter(text)
        sentence_time = time.time() - start_time
        print(f"句子分割完成，共 {len(sentences)} 个句子，耗时: {sentence_time:.2f}秒")
        
        # 显示前几个句子作为示例
        print("\n前5个句子示例:")
        for i, sent in enumerate(sentences[:5], 1):
            print(f"  {i}: {sent[:50]}{'...' if len(sent) > 50 else ''}")
        
        # 3. 初始化语义分割器，设置块大小为850字符
        print("\n正在初始化语义分割器...")
        splitter = SemanticSplitter(
            initial_threshold=0.4,
            appending_threshold=0.5,
            merging_threshold=0.5,
            max_chunk_size=850  # 设置最大块大小为850字符
        )
        
        # 4. 进行语义分割
        print("\n正在进行语义分割...")
        start_time = time.time()
        semantic_chunks = splitter.process_sentences(sentences)
        semantic_time = time.time() - start_time
        print(f"语义分割完成，耗时: {semantic_time:.2f}秒")
        
        # 5. 分析分割结果
        analyze_chunks(semantic_chunks)
        
        # 6. 保存分割结果到文件
        save_chunks_to_files(semantic_chunks, output_dir, "红楼梦_chunk")
        
        # 7. 创建统计信息文件
        stats_file = output_dir / "分割统计.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("红楼梦语义分割统计信息\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"原文总字符数: {len(text)}\n")
            f.write(f"句子总数: {len(sentences)}\n")
            f.write(f"分割后块数: {len(semantic_chunks)}\n")
            f.write(f"句子分割耗时: {sentence_time:.2f}秒\n")
            f.write(f"语义分割耗时: {semantic_time:.2f}秒\n\n")
            
            chunk_lengths = [len(chunk) for chunk in semantic_chunks]
            f.write(f"平均块长度: {sum(chunk_lengths) / len(chunk_lengths):.1f} 字符\n")
            f.write(f"最短块: {min(chunk_lengths)} 字符\n")
            f.write(f"最长块: {max(chunk_lengths)} 字符\n\n")
            
            f.write("长度分布:\n")
            length_ranges = [
                (0, 200), (200, 400), (400, 600), 
                (600, 800), (800, 1000), (1000, 1200), 
                (1200, 1500), (1500, float('inf'))
            ]
            
            for min_len, max_len in length_ranges:
                if max_len == float('inf'):
                    count = sum(1 for length in chunk_lengths if length >= min_len)
                    f.write(f"  {min_len}+ 字符: {count} 块\n")
                else:
                    count = sum(1 for length in chunk_lengths if min_len <= length < max_len)
                    f.write(f"  {min_len}-{max_len-1} 字符: {count} 块\n")
        
        print(f"\n统计信息已保存到: {stats_file}")
        print(f"\n处理完成！所有文件已保存到: {output_dir}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

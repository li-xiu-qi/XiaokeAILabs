"""
红楼梦语义分割测试脚本
使用硅基流动的BAAI/bge-m3模型进行语义分割
"""

import os
import time
from test_semantic_splitter import SemanticSplitter, custom_sentence_splitter, read_text_file


def main():
    """主函数：测试红楼梦文本的语义分割"""
    
    # 文件路径
    hongloumeng_file = "红楼梦.txt"
    
    print("=" * 60)
    print("红楼梦语义分割测试")
    print("=" * 60)
    
    # 1. 读取红楼梦文本
    print("1. 读取红楼梦文本...")
    try:
        text = read_text_file(hongloumeng_file)
        print(f"文本长度: {len(text)} 字符")
    except FileNotFoundError:
        print(f"错误：找不到文件 {hongloumeng_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 2. 使用句子分割器进行初步分割
    print("\n2. 进行句子分割...")
    start_time = time.time()
    
    try:
        sentences = custom_sentence_splitter(text)
        sentence_split_time = time.time() - start_time
        print(f"句子分割完成，耗时: {sentence_split_time:.2f}秒")
        print(f"分割出的句子数量: {len(sentences)}")
        
        # 显示前5个句子
        print("\n前5个句子示例:")
        for i, sent in enumerate(sentences[:5]):
            print(f"{i+1}. {sent[:100]}{'...' if len(sent) > 100 else ''}")
            
    except Exception as e:
        print(f"句子分割时出错: {e}")
        return
    
    # 3. 初始化语义分割器
    print("\n3. 初始化语义分割器...")
    try:
        # 调整参数以适合中文小说
        splitter = SemanticSplitter(
            initial_threshold=0.3,   # 降低初始阈值，让更多相关句子归为一组
            appending_threshold=0.4, # 附加阈值
            merging_threshold=0.4,   # 合并阈值
            max_chunk_size=800,      # 增大块大小以适合小说段落
        )
        print("语义分割器初始化成功")
    except Exception as e:
        print(f"初始化语义分割器时出错: {e}")
        return
    
    # 4. 执行语义分割（处理部分句子避免API调用过多）
    print("\n4. 执行语义分割...")
    
    # 为了演示，只处理前100个句子
    test_sentences = sentences[:100]
    print(f"处理前 {len(test_sentences)} 个句子进行演示...")
    
    start_time = time.time()
    try:
        chunks = splitter.process_sentences(test_sentences)
        semantic_split_time = time.time() - start_time
        
        print(f"语义分割完成，耗时: {semantic_split_time:.2f}秒")
        print(f"分割出的语义块数量: {len(chunks)}")
        
    except Exception as e:
        print(f"语义分割时出错: {e}")
        return
    
    # 5. 显示分割结果
    print("\n5. 分割结果展示:")
    print("-" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\n【块 {i+1}】(长度: {len(chunk)} 字符)")
        # 显示每个块的前200个字符
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(preview)
        print("-" * 40)
        
        # 只显示前10个块
        if i >= 9:
            print(f"... 还有 {len(chunks) - 10} 个块")
            break
    
    # 6. 统计信息
    print("\n6. 统计信息:")
    print(f"原文长度: {len(text)} 字符")
    print(f"句子数量: {len(sentences)}")
    print(f"处理的句子数: {len(test_sentences)}")
    print(f"语义块数量: {len(chunks)}")
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    print(f"平均块长度: {sum(chunk_lengths) / len(chunk_lengths):.1f} 字符")
    print(f"最大块长度: {max(chunk_lengths)} 字符")
    print(f"最小块长度: {min(chunk_lengths)} 字符")
    
    print(f"\n总耗时: {sentence_split_time + semantic_split_time:.2f}秒")


if __name__ == "__main__":
    main()

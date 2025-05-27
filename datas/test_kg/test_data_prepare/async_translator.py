# -*- coding: utf-8 -*-
"""
异步红楼梦繁体转简体翻译器 - 支持200个并发
"""

import os
import asyncio
import json
import time
from typing import List
from dotenv import load_dotenv
from fallback_openai_client import AsyncFallbackOpenAIClient 

load_dotenv(dotenv_path=".env")


def _split_text_into_smart_chunks_helper(
    text_block: str, target_chunk_size: int = 512, max_chunk_size: int = 4000
) -> List[str]:
    sub_chunks: List[str] = []
    if not text_block.strip():
        return sub_chunks

    sentence_end_punctuations = ["。", "！", "？", "”", "’", "…", ".", "!", "?"]
    lines = text_block.splitlines(keepends=True)
    current_chunk_lines: List[str] = []
    current_char_count = 0

    def ends_with_punctuation(line_list: List[str]) -> bool:
        if not line_list:
            return False
        for line_idx in range(len(line_list) - 1, -1, -1):
            last_line_stripped = line_list[line_idx].rstrip()
            if last_line_stripped:
                return last_line_stripped[-1] in sentence_end_punctuations
        return False

    i = 0
    while i < len(lines):
        current_chunk_lines.append(lines[i])
        current_char_count += len(lines[i])

        # 达到目标块大小，尝试在目标块附近优先切分
        if current_char_count >= target_chunk_size:
            # 优先在当前块内的目标区间附近找切分点
            split_at_line_idx_in_current_chunk = -1
            # 允许浮动到max_chunk_size
            search_end = len(current_chunk_lines) - 1
            search_start = 0
            # 只在目标区间到最大区间内找切分点
            for k in range(search_end, -1, -1):
                chunk_so_far = "".join(current_chunk_lines[:k+1])
                if len(chunk_so_far) < target_chunk_size:
                    break
                if len(chunk_so_far) <= max_chunk_size and ends_with_punctuation(current_chunk_lines[:k+1]):
                    split_at_line_idx_in_current_chunk = k
                    break
            # 如果没找到合适切点，允许块长度到max_chunk_size
            if split_at_line_idx_in_current_chunk == -1 and current_char_count >= max_chunk_size:
                split_at_line_idx_in_current_chunk = len(current_chunk_lines) - 1

            if split_at_line_idx_in_current_chunk != -1:
                content = "".join(current_chunk_lines[:split_at_line_idx_in_current_chunk+1]).strip()
                if content:
                    sub_chunks.append(content)
                current_chunk_lines = current_chunk_lines[split_at_line_idx_in_current_chunk+1:]
                current_char_count = sum(len(l) for l in current_chunk_lines)
        i += 1

    # 处理剩余部分
    if current_chunk_lines:
        content = "".join(current_chunk_lines).strip()
        if content:
            sub_chunks.append(content)
    return sub_chunks


# 修改后的简单文本分割函数 (目标512，最大4000)
def simple_text_splitter(full_text: str, target_chunk_size: int = 512, max_chunk_size: int = 4000) -> List[str]:
    chunks: List[str] = []
    if not full_text.strip():
        chunks.append("")
        return chunks

    chunks.extend(_split_text_into_smart_chunks_helper(full_text, target_chunk_size, max_chunk_size))

    if not chunks and full_text.strip():
        chunks.append(full_text.strip())
    elif not chunks and not full_text.strip():
        chunks.append("")

    return chunks


class AsyncTraditionalToSimplifiedTranslator:
    """异步繁体到简体的翻译器"""

    def __init__(self, max_concurrent: int = 200):
        """
        初始化异步翻译器
        Args:
            max_concurrent: 最大并发数
        """
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        zhipu_base_url = os.getenv("ZHIPU_BASE_URL")
        zhipu_model_name = "glm-4-flash"

        guiji_api_key = os.getenv("GUIJI_API_KEY")
        guiji_base_url = os.getenv("GUIJI_BASE_URL")
        guiji_model_name = "THUDM/GLM-4-9B-0414"

        if not zhipu_api_key or not zhipu_base_url:
            raise ValueError(
                "错误：主 API (ZHIPU_API_KEY, ZHIPU_BASE_URL) 环境变量未完全设置。"
            )

        self.client = AsyncFallbackOpenAIClient(
            primary_api_key=zhipu_api_key,
            primary_base_url=zhipu_base_url,
            primary_model_name=zhipu_model_name,
            fallback_api_key=guiji_api_key,
            fallback_base_url=guiji_base_url,
            fallback_model_name=guiji_model_name,
        )

        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.client.__aenter__()  # 调用封装客户端的 __aenter__
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.client.__aexit__(
            exc_type, exc_val, exc_tb
        )  # 调用封装客户端的 __aexit__

    async def translate_chunk(
        self, chunk: str, chunk_index: int, total_chunks: int
    ) -> str:
        """
        异步翻译单个文本块
        参数:
            chunk: 要翻译的文本块
            chunk_index: 当前块的索引
            total_chunks: 总块数
        返回:
            翻译后的文本块
        """
        async with self.semaphore:  # 控制并发数
            # 检查 chunk.content 是否为空或仅包含空白字符
            if not chunk or chunk.isspace():
                print(f"ℹ️ 跳过空文本块 [{chunk_index+1}/{total_chunks}]")
                return chunk
            try:
                prompt_content = f"""请将以下繁体中文文本转换为简体中文，保持原文的文学风格和韵味，不要改变任何内容的含义：

{chunk}

要求：
1. 只进行繁体到简体的转换
2. 标点符号同样需要转换为简体中文的形式
3. 保持古典文学的语言风格
4. 不要添加任何解释或注释
5. 直接输出转换后的简体中文文本
"""
                full_prompt = f"{prompt_content}"

                messages = [
                    {
                        "role": "system",
                        "content": "你是一个专业的中文繁简转换助手，专门将繁体中文转换为简体中文。",
                    },
                    {"role": "user", "content": full_prompt},
                ]

                try:
                    # 使用封装的客户端进行 API 调用
                    completion = await self.client.chat_completions_create(
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.1,
                        top_p=0.7,
                        # model 参数由 AsyncFallbackOpenAIClient 内部处理
                    )
                    translated_content = completion.choices[0].message.content.strip()
                    # 可以在此处根据 completion.response_ms 或其他返回信息判断是哪个 API 成功的
                    # 为了简化，我们假设成功即是主 API 或备用 API 之一
                    print(f"✅ 已翻译 (API) [{chunk_index+1}/{total_chunks}]")
                    return translated_content
                # 异常处理现在由 AsyncFallbackOpenAIClient 内部处理一部分，这里捕获最终未能处理的异常
                except Exception as e:
                    print(
                        f"❌ API 调用最终失败 [{chunk_index+1}/{total_chunks}]: {type(e).__name__} - {str(e)}"
                    )

                # 如果 API 调用最终失败 (包括主 API 和备用 API)，则使用原文
                print(f"ℹ️ 翻译失败，使用原文 [{chunk_index+1}/{total_chunks}]")
                return chunk
            except Exception as e:  # 捕获 translate_chunk 方法内的其他所有未预期异常
                print(
                    f"❌ 翻译块处理失败 [{chunk_index+1}/{total_chunks}]: {type(e).__name__}: {str(e)}"
                )
                return chunk

    async def translate_chunks(self, chunks: List[str]) -> List[str]:
        """
        异步翻译所有文本块
        参数:
            chunks: 文本块列表
        返回:
            翻译后的文本块列表
        """
        print(f"🚀 开始异步翻译 {len(chunks)} 个文本块，并发数: {self.max_concurrent}")
        start_time = time.time()

        # 创建异步任务
        tasks = [
            self.translate_chunk(chunk, i, len(chunks))
            for i, chunk in enumerate(chunks)
        ]

        # 并发执行所有翻译任务
        translated_chunks = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        final_chunks = []
        for i, result in enumerate(translated_chunks):
            if isinstance(result, Exception):
                print(f"❌ 任务 {i+1} 执行异常: {result}")
                final_chunks.append(chunks[i])
            else:
                final_chunks.append(result)
            # 新增：打印输入输出长度对比
            input_len = len(chunks[i]) if i < len(chunks) else 0
            output_len = len(result) if not isinstance(result, Exception) else input_len
            print(f"块 {i+1}: 输入长度={input_len}，输出长度={output_len}")

        end_time = time.time()
        print(f"✅ 所有文本块翻译完成！耗时: {end_time - start_time:.2f} 秒")
        return final_chunks


def save_translated_text(chunks: List[str], output_path: str):
    """
    保存翻译后的文本
    参数:
        chunks: 翻译后的文本块列表
        output_path: 输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk)
            f.write("\n")
    print(f"💾 翻译结果已保存到: {output_path}")


async def main():
    """主函数"""
    print("🏮 异步红楼梦繁体转简体翻译工具")
    print(f"🚀 支持 200 个并发请求")
    print("=" * 50)

    # 输入输出文件路径
    input_file = r"./红楼梦繁体版本.txt"
    output_file = r"./红楼梦简体版本_async.txt"
    target_chunk_size = 512      # 目标块大小
    max_chunk_size = 4000        # 最大块大小

    # 读取原文
    print("📖 读取原文...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ 成功读取文件，总字符数: {len(text)}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 分割文本
    print("\n📑 分割文本...")
    print(f"📏 目标块大小: {target_chunk_size} 字符，最大块大小: {max_chunk_size} 字符")
    chunks = simple_text_splitter(text, target_chunk_size, max_chunk_size)
    print(f"✅ 文本分割完成，共 {len(chunks)} 个块")

    # 显示分割结果
    print("\n📋 分割结果预览:")
    for i, chunk in enumerate(chunks[:5]):  # 只显示前5个
        display_content = chunk if len(chunk) <= 30 else chunk[:30] + "..."
        print(f"  块 {i+1}: '{display_content}'")
    if len(chunks) > 5:
        print(f"  ... 还有 {len(chunks) - 5} 个块")

    # 异步翻译文本
    print("\n🔄 开始异步翻译...")
    async with AsyncTraditionalToSimplifiedTranslator(max_concurrent=200) as translator:
        translated_chunks = await translator.translate_chunks(chunks)

    # 保存结果
    print("\n💾 保存翻译结果...")
    save_translated_text(translated_chunks, output_file)

    print("\n🎉 翻译完成！")
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())

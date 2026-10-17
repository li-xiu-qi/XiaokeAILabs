# -*- coding: utf-8 -*-
"""
利用大模型API清理红楼梦每一回的噪声和排版
"""

import os
import asyncio
from typing import List
from fallback_openai_client import AsyncFallbackOpenAIClient
import re

def _split_text_into_smart_chunks_helper(
    text_block: str, target_chunk_size: int = 1500, max_chunk_size: int = 3500
) -> list:
    """
    按照结尾符号和目标长度智能分割文本为多个块
    """
    sub_chunks = []
    if not text_block.strip():
        return sub_chunks

    sentence_end_punctuations = ["。", "！", "？", "”", "’", "…", ".", "!", "?"]
    lines = text_block.splitlines(keepends=True)
    current_chunk_lines = []
    current_char_count = 0

    def ends_with_punctuation(line_list):
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

        if current_char_count >= target_chunk_size:
            split_at_line_idx_in_current_chunk = -1
            search_end = len(current_chunk_lines) - 1
            for k in range(search_end, -1, -1):
                chunk_so_far = "".join(current_chunk_lines[:k+1])
                if len(chunk_so_far) < target_chunk_size:
                    break
                if len(chunk_so_far) <= max_chunk_size and ends_with_punctuation(current_chunk_lines[:k+1]):
                    split_at_line_idx_in_current_chunk = k
                    break
            if split_at_line_idx_in_current_chunk == -1 and current_char_count >= max_chunk_size:
                split_at_line_idx_in_current_chunk = len(current_chunk_lines) - 1

            if split_at_line_idx_in_current_chunk != -1:
                content = "".join(current_chunk_lines[:split_at_line_idx_in_current_chunk+1]).strip()
                if content:
                    sub_chunks.append(content)
                current_chunk_lines = current_chunk_lines[split_at_line_idx_in_current_chunk+1:]
                current_char_count = sum(len(l) for l in current_chunk_lines)
        i += 1

    if current_chunk_lines:
        content = "".join(current_chunk_lines).strip()
        if content:
            sub_chunks.append(content)
    return sub_chunks

async def clean_chapter_content(chapter_title: str, chapter_content: str, client: AsyncFallbackOpenAIClient, idx: int, total: int, semaphore: asyncio.Semaphore) -> str:
    """
    使用大模型API清理单个章节内容（带分块和并发控制，先判断是否需要清理）
    """
    chunks = _split_text_into_smart_chunks_helper(chapter_content, target_chunk_size=1500, max_chunk_size=3500)
    cleaned_chunks = []
    for chunk_idx, chunk in enumerate(chunks):
        async with semaphore:
            # 第一步：让大模型判断是否需要清理，要求输出原因和结果
            judge_prompt = f"""请判断下面的文本是否包含如下内容：版权声明、页码、注释、无关噪声、排版混乱等非正文内容。
<输出格式要求>
原因：（使用10-40字简要说明你的判断理由,不需要清理则为null）
结果：（只输出“需要清理”或“不需要清理”）

文本如下：
{chunk}
"""
            judge_messages = [
                {"role": "system", "content": "你是专业的中文古典小说文本清理助手。"},
                {"role": "user", "content": judge_prompt}
            ]
            try:
                judge_result = await client.chat_completions_create(
                    messages=judge_messages,
                    max_tokens=400,
                    temperature=0.0,
                    top_p=0.6,
                )
                judge_reply = judge_result.choices[0].message.content.strip()
                m = re.search(r"结果[:：]\s*(需要清理|不需要清理)", judge_reply)
                if m:
                    result_str = m.group(1)
                    need_clean = (result_str == "需要清理")
                else:
                    print(f"⚠️ 判断回复无法识别，默认跳过清理流程，回复内容: {judge_reply}")
                    need_clean = False
            except Exception as e:
                print(f"⚠️ 判断第{idx+1}回第{chunk_idx+1}块是否需要清理时出错: {e}，默认跳过清理流程")
                need_clean = False
            # 打印下chunk的长度
            print(f"🔍 判断第{idx+1}/{total}回，第{chunk_idx+1}/{len(chunks)}块 | 长度: {len(chunk)} | 结果: {result_str} | 原因: {judge_reply}")
            if need_clean:
                # 第二步：真正清理
                prompt = f"""你是红楼梦文本清理助手。请对以下文本进行清理：
1. 删除所有版权声明、页码、注释、无关噪声，仅保留正文内容。
2. 修复排版错误，使文本段落清晰、格式规范。
3. 不要添加任何解释或注释，只输出清理后的正文。

章节标题：{chapter_title}
原始文本如下：
{chunk}
"""
                messages = [
                    {"role": "system", "content": "你是专业的中文古典小说文本清理助手。"},
                    {"role": "user", "content": prompt}
                ]
                try:
                    completion = await client.chat_completions_create(
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.1,
                        top_p=0.7,
                    )
                    cleaned = completion.choices[0].message.content.strip()
                    print(f"✅ 已清理第{idx+1}/{total}回，第{chunk_idx+1}/{len(chunks)}块 | 原长度: {len(chunk)}，清理后: {len(cleaned)}")
                    cleaned_chunks.append(cleaned)
                except Exception as e:
                    print(f"❌ 清理第{idx+1}回第{chunk_idx+1}块失败: {e}")
                    cleaned_chunks.append(chunk)
            else:
                print(f"➡️ 跳过清理第{idx+1}/{total}回，第{chunk_idx+1}/{len(chunks)}块（无需清理） | 长度: {len(chunk)}")
                cleaned_chunks.append(chunk)
    # 打印每回整体长度对比
    cleaned_text = "\n".join(cleaned_chunks)
    print(f"📏 回目《{chapter_title}》清理前总长度: {len(chapter_content)}，清理后总长度: {len(cleaned_text)}")
    return cleaned_text

async def main():
    input_file = "./红楼梦简体版本_async.txt"
    output_dir = "./红楼梦_按回清理"
    max_concurrent = 200  # 最大并发数
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取全书文本
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 按回分割（复用split_by_chapter.py的正则逻辑）
    pattern = re.compile(r'^第[一二三四五六七八九十百千\d]+回[^\n]*', re.MULTILINE)
    matches = list(pattern.finditer(text))
    chapters = []
    for idx, match in enumerate(matches):
        start = match.start()
        if idx > 0:
            prev_match = matches[idx-1]
            chapter_title = prev_match.group(0)
            chapter_content = text[prev_match.start():start]
            chapters.append((chapter_title, chapter_content.strip()))
    if matches:
        last_match = matches[-1]
        chapter_title = last_match.group(0)
        chapter_content = text[last_match.start():]
        chapters.append((chapter_title, chapter_content.strip()))

    print(f"共检测到 {len(chapters)} 回，开始清理...")

    # 冗余设计：主API和备用API均从环境变量读取
    zhipu_api_key = os.getenv("ZHIPU_API_KEY")
    zhipu_base_url = os.getenv("ZHIPU_BASE_URL")
    zhipu_model_name = "glm-4-flash"

    guiji_api_key = os.getenv("GUIJI_API_KEY")
    guiji_base_url = os.getenv("GUIJI_BASE_URL")
    guiji_model_name = "THUDM/GLM-4-9B-0414"

    client = AsyncFallbackOpenAIClient(
        primary_api_key=zhipu_api_key,
        primary_base_url=zhipu_base_url,
        primary_model_name=zhipu_model_name,
        fallback_api_key=guiji_api_key,
        fallback_base_url=guiji_base_url,
        fallback_model_name=guiji_model_name,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async with client:
        tasks = []
        for idx, (title, content) in enumerate(chapters):
            tasks.append(clean_chapter_content(title, content, client, idx, len(chapters), semaphore))
        cleaned_list: List[str] = await asyncio.gather(*tasks)

        # 保存清理后的章节
        for idx, cleaned in enumerate(cleaned_list):
            filename = f"chapter_{idx+1:03d}.txt"
            out_path = os.path.join(output_dir, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"已保存清理后文件: {out_path}")

    print("全部章节清理完成！")

if __name__ == "__main__":
    asyncio.run(main())

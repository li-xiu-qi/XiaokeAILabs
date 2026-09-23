# -*- coding: utf-8 -*-
"""
按“回”分割红楼梦简体文本，每回为一个块
"""

import re
import os

def split_by_chapter(text: str):
    """
    按“第X回”分割文本，返回每回的标题和内容
    """
    # 只匹配行首的“第X回”，如“第十二回”，且必须独占一行或紧跟标题内容
    pattern = re.compile(r'^第[一二三四五六七八九十百千\d]+回[^\n]*', re.MULTILINE)
    chapters = []
    matches = list(pattern.finditer(text))
    for idx, match in enumerate(matches):
        start = match.start()
        if idx > 0:
            prev_match = matches[idx-1]
            chapter_title = prev_match.group(0)
            chapter_content = text[prev_match.start():start]
            chapters.append((chapter_title, chapter_content.strip()))
    # 最后一回
    if matches:
        last_match = matches[-1]
        chapter_title = last_match.group(0)
        chapter_content = text[last_match.start():]
        chapters.append((chapter_title, chapter_content.strip()))
    return chapters

def save_chapters(chapters, output_dir):
    """
    保存每一回为单独的txt文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, (title, content) in enumerate(chapters, 1):
        filename = f"chapter_{idx:03d}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"已保存: {filepath}，标题: {title}，长度: {len(content)} 字符")

def split_text_into_smart_chunks_helper(
    text_block: str, target_chunk_size: int = 512, max_chunk_size: int = 4000
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

        # 达到目标块大小，尝试在目标块附近优先切分
        if current_char_count >= target_chunk_size:
            split_at_line_idx_in_current_chunk = -1
            search_end = len(current_chunk_lines) - 1
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

def main():
    input_file = "./红楼梦简体版本_async.txt"
    output_dir = "./红楼梦_按回分割"
    print("读取简体红楼梦文本...")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"总长度: {len(text)} 字符")
    chapters = split_by_chapter(text)
    print(f"共分割出 {len(chapters)} 回")
    # save_chapters(chapters, output_dir)
    print("分割完成！")

if __name__ == "__main__":
    main()

import re
from typing import List, Dict, Optional, Union

class Chunk:
    """
    文本块数据结构，包含唯一ID、标题、内容、层级、合并来源等信息。
    """
    def __init__(
        self,
        chunk_id: Union[int, str],
        title: str,
        content: str,
        level: int,
        original_chunk_ids: Optional[List[Union[int, str]]] = None
    ):
        self.chunk_id = chunk_id  # 唯一标识符
        self.title = title        # 块标题
        self.content = content    # 块内容
        self.level = level        # Markdown 标题级别（0=无标题，1=#，2=##，…）
        self.original_chunk_ids = original_chunk_ids or []  # 合并来源

    def to_dict(self):
        """
        转换为字典，便于后续处理。
        """
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "original_chunk_ids": self.original_chunk_ids
        }

class HybridChunker:
    """
    支持 Markdown 结构化与递归长度分割的混合分块器，适配中英文。
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        markdown_headers: Optional[List[str]] = None,
        default_separators: Optional[List[str]] = None,
        language: str = "zh"
    ):
        """
        初始化分块器，设置分块参数。
        """
        self.chunk_size = chunk_size
        self.markdown_headers = markdown_headers or ["#", "##", "###"]
        self.language = language
        self.chunk_counter = 0

        # 设置分隔符和分句符号，自动适配中英文
        if default_separators is not None:
            self.separators = default_separators
        else:
            if language == "en":
                self.separators = ["\n\n", "\n", ".", " "]
                self.sentence_punctuations = [".", "!", "?"]
            else:
                self.separators = ["\n\n", "\n", "。", "！", "？", " "]
                self.sentence_punctuations = ["。", "！", "？"]

    def _next_chunk_id(self):
        """
        生成下一个唯一块ID。
        """
        self.chunk_counter += 1
        return self.chunk_counter

    def split_by_markdown_headers(self, text: str) -> List[Dict]:
        """
        按 Markdown 标题结构对文本进行初步分块。
        """
        header_pattern = re.compile(
            rf"^({'|'.join(re.escape(h) for h in self.markdown_headers)})\s*(.*)", re.MULTILINE
        )
        lines = text.splitlines()
        chunks = []
        current_title = ""
        current_level = 0
        current_content = []
        current_chunk_id = self._next_chunk_id()

        def flush_chunk():
            """
            将当前内容刷新为一个块。
            """
            nonlocal current_content, current_title, current_level, current_chunk_id
            content = "\n".join(current_content).strip()
            if content or current_title or current_level == 0:
                chunks.append(Chunk(
                    chunk_id=current_chunk_id,
                    title=current_title,
                    content=content,
                    level=current_level
                ))
            current_content = []

        for line in lines:
            m = header_pattern.match(line)
            if m:
                # 新标题，先保存上一个块
                flush_chunk()
                current_title = m.group(2).strip()
                current_level = m.group(1).count("#")
                current_chunk_id = self._next_chunk_id()
            else:
                current_content.append(line)
        flush_chunk()
        # 移除空内容的块
        return [c.to_dict() for c in chunks if c.content.strip()]

    def recursive_character_split(self, text: str) -> List[str]:
        """
        递归按分隔符将文本分割为不超过 chunk_size 的块，优先自然边界。
        """
        if len(text) <= self.chunk_size:
            return [text]
        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                # 恢复分隔符
                parts_with_sep = []
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        parts_with_sep.append(part + sep)
                    else:
                        parts_with_sep.append(part)
                # 合并为不超过 chunk_size 的块
                chunks = []
                buf = ""
                for part in parts_with_sep:
                    if len(buf) + len(part) <= self.chunk_size:
                        buf += part
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = part
                if buf:
                    chunks.append(buf)
                # 递归处理超长块
                result = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size and sep != self.separators[-1]:
                        result.extend(self.recursive_character_split(chunk))
                    else:
                        result.append(chunk)
                return result
        # 若没有分隔符可用，硬切
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def sentence_split(self, text: str) -> List[str]:
        """
        按句子边界分割文本，并合并为不超过 chunk_size 的块。
        """
        if self.language == "en":
            pattern = r'([.!?])'
        else:
            pattern = r'([。！？])'
        sents = []
        buf = ""
        for seg in re.split(pattern, text):
            if not seg:
                continue
            buf += seg
            if seg in self.sentence_punctuations:
                sents.append(buf)
                buf = ""
        if buf.strip():
            sents.append(buf)
        # 合并为不超过 chunk_size 的块
        merged = []
        cur = ""
        for sent in sents:
            if len(cur) + len(sent) <= self.chunk_size:
                cur += sent
            else:
                if cur:
                    merged.append(cur)
                cur = sent
        if cur:
            merged.append(cur)
        return merged

    def hybrid_chunk(self, text: str) -> List[Dict]:
        """
        主分块流程，结合标题分割和递归分割。
        """
        self.chunk_counter = 0  # 每次分块前重置计数器
        header_chunks = self.split_by_markdown_headers(text)
        result = []
        for chunk in header_chunks:
            content = chunk["content"]
            if len(content) <= self.chunk_size:
                result.append(chunk)
            else:
                # 超长则递归分割
                sub_chunks = self.recursive_character_split(content)
                for idx, sub in enumerate(sub_chunks):
                    part_title = (
                        f'{chunk["title"]} Part {idx+1}/{len(sub_chunks)}'
                        if chunk["title"] else f'Part {idx+1}/{len(sub_chunks)}'
                    )
                    result.append({
                        "chunk_id": f'{chunk["chunk_id"]}-{idx+1}',
                        "title": part_title,
                        "content": sub,
                        "level": chunk["level"],
                        "original_chunk_ids": [chunk["chunk_id"]]
                    })
        return result

    def merge_chunks_by_size(self, chunks: List[Dict], target_chunk_size: int) -> List[Dict]:
        """
        将相邻块合并，目标为 target_chunk_size，尊重结构层级和标题。
        若即将合并的块的 level 比当前合并块中所有块的 level 都小，则不能合并，需另起新块。
        合并后chunk_id重新排序为连续整数。
        修正：level=0（无标题）块合并时，遇到下一个有标题的块，必须立即断开，不能合并到有标题的块里。
        """
        merged = []
        buf = []
        buf_len = 0
        for chunk in chunks:
            chunk_len = len(chunk["content"])
            chunk_level = chunk["level"]
            # 修正：如果buf为level=0（无标题），且当前chunk为有标题（level>0），则立即断开
            if buf and buf[0]["level"] == 0 and chunk_level > 0:
                merged.append(self._merge_chunks_by_id(buf))
                buf = []
                buf_len = 0
            # 若当前buf非空，且即将合并的块level比buf中所有块的level都小，则不能合并
            elif buf and chunk_level < min(c["level"] for c in buf):
                merged.append(self._merge_chunks_by_id(buf))
                buf = []
                buf_len = 0
            # 检查合并后是否超长
            elif buf and buf_len + chunk_len > target_chunk_size:
                merged.append(self._merge_chunks_by_id(buf))
                buf = []
                buf_len = 0
            buf.append(chunk)
            buf_len += chunk_len
        if buf:
            merged.append(self._merge_chunks_by_id(buf))
        # 合并后重新分配chunk_id为连续整数
        for idx, chunk in enumerate(merged, 1):
            chunk["chunk_id"] = idx
        return merged

    def _merge_chunks_by_id(self, chunk_dicts: List[Dict]) -> Dict:
        """
        按指定 chunk_id 列表合并块，采用第一个块的标题和级别，合并内容与来源。
        不负责分配最终 chunk_id，由外部统一排序赋值。
        """
        if not chunk_dicts:
            return {}
        title = chunk_dicts[0]["title"]
        level = chunk_dicts[0]["level"]
        content = "\n".join([c["content"] for c in chunk_dicts])
        original_chunk_ids = []
        for c in chunk_dicts:
            if "original_chunk_ids" in c and c["original_chunk_ids"]:
                original_chunk_ids.extend(c["original_chunk_ids"])
            else:
                original_chunk_ids.append(c["chunk_id"])
        # 不分配最终chunk_id，由外部分配
        return {
            # "chunk_id": ...  # 由外部赋值
            "title": title,
            "content": content,
            "level": level,
            "original_chunk_ids": original_chunk_ids
        }

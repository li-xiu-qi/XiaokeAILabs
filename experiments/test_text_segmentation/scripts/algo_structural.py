# -*- coding: utf-8 -*-
"""
算法 3：结构感知分块（Markdown / 代码 / 表格）

这类内容不能按句子切。代码块的语义完整性取决于语法结构，
表格的完整性取决于行，Markdown 章节的完整性取决于标题层级。

规则（按优先级从高到低）：
1. 代码围栏 ``` / ~~~ 整体成块，内部绝不切开
2. 表格（连续的 | 行）整体成块
3. Markdown 标题作为硬边界，按标题层级聚合成节（section）
   节超过 max_tokens 时才在节内按段落二次切分
4. 其余文本按段落 \n\n 累积，超过 max_tokens 时切

源：archive_predecessors/test_hybrid_chunking/document_split.py 与 markdown_split.py（早期前身），
原实现依赖 HuixiangDou 的 yaml 元数据流，这里抽成纯规则版，
去掉 yaml 依赖，保留「结构优先于长度」的核心。
"""
import re

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder

_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|")
_HEADING = re.compile(r"^(#{1,6})\s+\S")


class StructuralSplitter(BaseSplitter):
    name = "structural"
    kind = "structural"

    def __init__(self, max_tokens=512, min_tokens=48, keep_code_intact=True):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.keep_code_intact = keep_code_intact

    def _tok_count(self, tok, s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def split(self, text):
        enc = TextEncoder.get()
        tok = enc.tok
        lines = text.split("\n")
        bounds = []
        pos = 0

        # 先切出原子单元：代码块 / 表格 / 段落 / 标题
        units = []   # (kind, text)
        i = 0
        while i < len(lines):
            ln = lines[i]
            if self.keep_code_intact and _FENCE.match(ln):
                j = i + 1
                while j < len(lines) and not _FENCE.match(lines[j]):
                    j += 1
                j = min(j + 1, len(lines))
                units.append(("code", "\n".join(lines[i:j])))
                i = j
                continue
            if _TABLE_ROW.match(ln) and i + 1 < len(lines) and _TABLE_ROW.match(lines[i + 1]):
                j = i
                while j < len(lines) and _TABLE_ROW.match(lines[j]):
                    j += 1
                units.append(("table", "\n".join(lines[i:j])))
                i = j
                continue
            if _HEADING.match(ln):
                units.append(("heading", ln))
                i += 1
                continue
            if not ln.strip():
                i += 1
                continue
            # 普通段落：累积到空行或下一个结构标记
            j = i
            buf = []
            while j < len(lines) and lines[j].strip():
                if _FENCE.match(lines[j]) or _TABLE_ROW.match(lines[j]) \
                        or _HEADING.match(lines[j]):
                    break
                buf.append(lines[j])
                j += 1
            units.append(("para", "\n".join(buf)))
            i = j

        # 组装：标题作为硬边界；代码块与表格整块追加；段落累积到上限
        # 无结构标记的纯文本（如 Wikipedia 段落）会走 para 累积分支
        cur_start = 0
        cur_tokens = 0
        for kind, utext in units:
            n = self._tok_count(tok, utext)
            if kind == "heading":
                if cur_tokens >= self.min_tokens:
                    bounds.append(Boundary(pos=pos, reason=f"标题边界: {utext[:20]}"))
                    cur_start = pos
                    cur_tokens = 0
                pos += n
                continue
            if kind in ("code", "table"):
                pos += n
                cur_tokens += n
                # 代码块/表格本身就是硬块：超过上限则在此强制切开，
                # 否则整块作为独立块（不参与累积），避免超长块
                if n > self.max_tokens:
                    bounds.append(Boundary(pos=pos, reason=f"{kind} 超上限强制切"))
                    cur_tokens = 0
                continue
            # para
            if cur_tokens + n > self.max_tokens and cur_tokens >= self.min_tokens:
                bounds.append(Boundary(pos=pos, reason="段落累积超上限"))
                cur_start = pos
                cur_tokens = 0
            pos += n
            cur_tokens += n

        total = self._tok_count(tok, text)
        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)

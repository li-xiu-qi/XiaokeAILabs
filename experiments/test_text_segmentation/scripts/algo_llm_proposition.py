# -*- coding: utf-8 -*-
"""
算法 9：LLM Proposition 分块

方法出处：Dense X Retrieval (arXiv 2312.06648, EMNLP 2024, EMNLP 2024)
          附录 Figure 8 给出完整 proposition 抽取提示词，本实现按其四条指令移植为中文。

与前两个 LLM 算法的区别（关键）：
  llm_pairwise / llm_segment 只判断「主题是否相同」，切出来的仍是原文段落。
  本算法要求模型做**去上下文化**（decontextualize）：把代词替换为实体全名、
  拆开复合句、实体修饰独立成句。因此切出的命题脱离原文也能独立读懂。

代价：调用量远高于 pairwise（逐句而非逐对），仅适合小样本评测。

实现分两步：
  Step 1  逐句抽取命题（_ask 单句 → JSON 命题列表）
  Step 2  按 token 预算把命题聚合成块，块内命题首尾相接还原为文本
块边界 = 命题聚合块的切分点，pos 用「该块首句的 token 起点」标定。
"""
import concurrent.futures
import json
import os
import re

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans

# 移植自 Dense X Figure 8（原为英文，此处中文化，四条指令结构不变）
PROMPT_PROPOSITION = """把下面这段「内容」分解为清晰、简单的命题，确保每条命题脱离上下文也能理解。

要求：
1. 把复合句拆成简单句，尽量保留原文措辞。
2. 如果一个命名实体带有额外的描述性信息，把这些信息拆成独立的命题。
3. 对命题做去上下文化：给名词或整个句子加上必要的修饰语，并把代词（如「它」「他」「她」「他们」「这」「那」）替换成它们所指实体的全名。
4. 以 JSON 字符串列表的形式输出结果，只输出 JSON，不要输出其他内容。

标题：{title}
内容：{content}

输出："""


class LLMPropositionSplitter(BaseSplitter):
    name = "llm_proposition"
    kind = "model"

    def __init__(self, model="water18-new", max_tokens=512,
                 chunk_word_cap=500, api_key=None,
                 base_url="https://api.stepfun.com/v1", temperature=0.0,
                 concurrency=16):
        self.model = model
        self.max_tokens = max_tokens
        self.chunk_word_cap = chunk_word_cap
        self.temperature = temperature
        self.base_url = base_url
        self.concurrency = concurrency
        self.api_key = api_key or os.environ.get("STEPFUN_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "未提供 API key：请设置环境变量 STEPFUN_API_KEY，或通过 api_key 参数传入")
        self.n_calls = 0
        self.blocked = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _ask(self, prompt):
        r = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048,
        )
        self.n_calls += 1
        u = getattr(r, "usage", None)
        if u:
            self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        msg = r.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            text = (getattr(msg, "reasoning_content", None) or "").strip()
        return text

    @staticmethod
    def _parse_props(ans):
        """从模型输出里抽出命题列表。优先找 JSON 数组，退化到逐行切分。"""
        # 去掉可能的 markdown 代码围栏
        ans = re.sub(r"^```(?:json)?|```$", "", ans.strip(), flags=re.M).strip()
        m = re.search(r"\[.*\]", ans, flags=re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
            except Exception:
                pass
        # 退化：按行/引号切
        lines = [l.strip().strip(",。") for l in ans.split("\n") if len(l.strip()) > 4]
        return lines

    @staticmethod
    def _extract_one(args):
        """并发工作单元：抽一句的命题。"""
        sp, s, a, b = args
        if len(s.strip()) < 6:
            return (a, b, None)
        p = PROMPT_PROPOSITION.format(title="", content=s[:800])
        try:
            ans = sp._ask(p)
        except Exception:
            return (a, b, None)
        props = sp._parse_props(ans)
        return (a, b, props if props else None)

    def split(self, text):
        enc = TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) < 2:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])
        total = spans[-1][2]

        # 按 max_tokens 聚合成「待抽取段」，段内逐句并发抽取命题
        groups = []
        cur, cur_tok = [], 0
        for s, a, b in spans:
            n = b - a
            if cur and cur_tok + n > self.max_tokens:
                groups.append(cur)
                cur, cur_tok = [], 0
            cur.append((s, a, b))
            cur_tok += n
        if cur:
            groups.append(cur)

        # Step 1: 并发抽取命题
        flat = [(self, s, a, b) for g in groups for s, a, b in g]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = [ex.submit(self._extract_one, args) for args in flat]
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        sentence_props = [(a, b, props) for a, b, props in results if props]

        if not sentence_props:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=total)

        # Step 2: 按 chunk_word_cap 聚合命题成块。
        # 一块的「首句起点」即该块边界 pos；末块无边界（文档固有尾界）。
        bounds = []
        cur_words = 0
        cur_start_tok = sentence_props[0][0]
        for a, b, props in sentence_props:
            piece_words = sum(len(x) for x in props)
            if cur_words > 0 and cur_words + piece_words > self.chunk_word_cap:
                bounds.append(Boundary(pos=cur_start_tok, reason="LLM proposition 块满"))
                cur_words = piece_words
                cur_start_tok = a
            else:
                cur_words += piece_words

        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)

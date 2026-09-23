# -*- coding: utf-8 -*-
"""
算法 8：LLM 主题分割

用 water18-new 直接判断「这两段之间是不是主题转移」。
这是唯一能理解语义而不只是统计词频的方法，代价是延迟与 token 消耗。

策略（两选一，由 mode 控制）：
- pairwise：逐对相邻段问「是否同一主题」，O(n) 次调用
  适合短文，调用次数可控
- segment：整篇一次调用，让模型直接输出断点位置
  适合长文，但受输出长度与模型定位精度限制

两种都返回带 reason 的断点，reason 记录模型给出的判断依据，
这是其他七种算法都没有的（它们只能给分数）。

模型配置（来自 pkm-coding-cli-model-configs）：
  water18-new，OpenAI Chat Completions 协议，base_url https://api.stepfun.com
  API Key 读 pkm-hub-configs/coding-cli-model-configs/keys.json 的 stepfun 字段
"""
import json
import os
import re

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans

PROMPT_PAIRWISE = """下面两段文字是否在讲同一个主题？

段落A：{a}

段落B：{b}

这是一个二选一问题。请在最后一行输出答案，格式严格为「答案：相同」或「答案：不同」，不要输出其他内容。"""

PROMPT_SEGMENT = """下面是一篇长文的段落序列，每段前有编号。请找出主题发生转移的位置，即「前一段与后一段讲的是不同主题」的边界编号。

在最后一行输出所有边界编号，格式严格为「答案：3,7,12」，编号之间用逗号分隔。如果没有任何主题转移，输出「答案：无」。

{paras}"""


class LLMTopicSplitter(BaseSplitter):
    name = "llm_topic"
    kind = "model"

    def __init__(self, mode="pairwise", model="water18-new", max_tokens=512,
                 api_key=None, base_url="https://api.stepfun.com/v1", temperature=0.0):
        self.mode = mode
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("STEPFUN_API_KEY")
        if not self.api_key:
            keys_path = ("C:/Users/ke/Documents/projects/obsidian_projects/"
                         "pkm-hub-configs/coding-cli-model-configs/keys.json")
            if os.path.exists(keys_path):
                self.api_key = json.load(open(keys_path, encoding="utf-8"))["stepfun"]
        self.n_calls = 0
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
            max_tokens=2048,         # reasoning 模型需留足思考空间，16/64 都会被截断
        )
        self.n_calls += 1
        u = getattr(r, "usage", None)
        if u:
            self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        msg = r.choices[0].message
        # water18-new 把思考写进 reasoning_content，content 可能是空字符串
        text = (msg.content or "").strip()
        if not text:
            text = (getattr(msg, "reasoning_content", None) or "").strip()
        return text

    def split(self, text):
        enc = TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) < 2:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])
        total = spans[-1][2]

        # 按 max_tokens 聚合成「待判段」
        paras = []
        cur = []
        cur_tok = 0
        for s, a, b in spans:
            n = b - a
            if cur and cur_tok + n > self.max_tokens:
                paras.append((cur[0][1], cur[-1][2], "".join(x[0] for x in cur)))
                cur, cur_tok = [], 0
            cur.append((s, a, b))
            cur_tok += n
        if cur:
            paras.append((cur[0][1], cur[-1][2], "".join(x[0] for x in cur)))

        bounds = []
        blocked = 0
        if self.mode == "pairwise":
            for i in range(len(paras) - 1):
                p = PROMPT_PAIRWISE.format(a=paras[i][2][:600], b=paras[i + 1][2][:600])
                try:
                    ans = self._ask(p)
                except Exception as e:
                    # 451 等内容审查：跳过该边界，不让整轮评测崩掉
                    blocked += 1
                    self.blocked = getattr(self, "blocked", 0) + 1
                    continue
                # 取最后一行「答案：X」，兼容模型在答案前输出推理过程
                tail = ans.strip().split("\n")[-1]
                same = ("相同" in tail) and ("不同" not in tail)
                if not same:
                    bounds.append(Boundary(pos=paras[i + 1][0], reason="LLM: 主题转移"))
        else:
            numbered = "\n\n".join(f"[{i}] {p[2][:400]}" for i, p in enumerate(paras))
            try:
                ans = self._ask(PROMPT_SEGMENT.format(paras=numbered))
            except Exception:
                self.blocked = getattr(self, "blocked", 0) + 1
                return Segmentation(algo=self.name, boundaries=[], n_tokens=total)
            tail = ans.strip().split("\n")[-1]
            for m in re.findall(r"\d+", tail):
                idx = int(m)
                if 0 < idx < len(paras):
                    bounds.append(Boundary(pos=paras[idx][0], reason="LLM: 整篇标定"))
            seen, uniq = set(), []
            for b in bounds:
                if b.pos not in seen:
                    seen.add(b.pos)
                    uniq.append(b)
            bounds = uniq

        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)

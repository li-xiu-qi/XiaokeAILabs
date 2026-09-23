"""Nimble 的分题型决策打分器：单次前向，读取候选 token 的 logit。

给定一段 state 与一个分题型 schema（choice / 布尔 / 量表评分），
返回每个选项经校准的概率。不生成文本。

自适应设备：CUDA（bf16）或 Apple Silicon MPS（bf16 权重，fp32 logit）。
"""
import json
from pathlib import Path

import torch

MAX_CHOICES = 26


def _pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_model(model_id_or_path, device=None):
    """加载 merge 后的（或基座）模型。返回 (model, tokenizer, device)。"""
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    dev, dtype = device or _pick_device()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id_or_path, dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(dev)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, dev


def score(model, tok, state, schema, temperature=1.0, max_input_tokens=4096):
    """给一次决策打分。

    state:  str（上下文/证据）
    schema: {"decision": {"description": ..., "type": "enum"|"boolean",
                          "choices": [...], "choice_descriptions": {...}}}
    temperature: softmax 前用该值除 logit（按题型校准）
    返回 {"prediction", "probabilities", "logits"}，以选项值为键。
    """
    from jev_schema import prepare_prompts, choice_key  # vendored prompt builder

    prepared = prepare_prompts(tok, state, schema, max_input_tokens)
    i = prepared.names.index("decision")
    ids = torch.tensor([prepared.full_ids[i]], device=next(model.parameters()).device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()[0]

    cand = prepared.candidate_ids[i]
    choices = prepared.choices[i]
    picked = logits[cand]
    scaled = picked / temperature
    probs = torch.softmax(scaled, -1)
    best = int(scaled.argmax())
    return {
        "prediction": choices[best],
        "probabilities": {choice_key(c): float(p) for c, p in zip(choices, probs.tolist())},
        "logits": {choice_key(c): float(v) for c, v in zip(choices, picked.tolist())},
    }


def apply_temperature(probabilities, logits, kind_temperature):
    """对已有分布重新校准：softmax(logits / T)。"""
    import math
    mx = max(logits.values())
    exps = {k: math.exp(v / kind_temperature - mx / kind_temperature) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}
#!/usr/bin/env python3
"""诊断 Qwen3-VL-4B tokenizer 的 chat template 调用形态。不加载模型，秒级。"""
import os, traceback
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

M = "Qwen/Qwen3-VL-4B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
print("tokenizer class:", type(tok).__name__)
print("has apply_chat_template:", hasattr(tok, "apply_chat_template"))
print("chat_template 存在:", getattr(tok, "chat_template", None) is not None)
ct = getattr(tok, "chat_template", None)
if ct:
    print("chat_template 前 200 字:", repr(ct[:200]))

P = "把下面的用户消息分成唯一一类。\n类别：\n- billing: 账单、扣费、退款\n\n消息：我这个月的订阅被扣了两次款。\n类别："

cases = [
    ("A: content=list-of-dict(type/text)", lambda: tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": P}]}],
        add_generation_prompt=True, return_tensors="pt", tokenize=True)),
    ("B: content=纯字符串", lambda: tok.apply_chat_template(
        [{"role": "user", "content": P}],
        add_generation_prompt=True, return_tensors="pt", tokenize=True)),
    ("C: 单字符串消息", lambda: tok.apply_chat_template(
        P, add_generation_prompt=True, return_tensors="pt", tokenize=True)),
    ("D: messages + tokenize=False", lambda: tok.apply_chat_template(
        [{"role": "user", "content": P}],
        add_generation_prompt=True, tokenize=False)),
    ("E: apply_chat_template 后手动 tokenize", lambda: tok(
        tok.apply_chat_template([{"role": "user", "content": P}],
                                add_generation_prompt=True, tokenize=False),
        return_tensors="pt")),
]

for name, fn in cases:
    print("\n" + "=" * 70)
    print("[%s]" % name)
    try:
        r = fn()
        if isinstance(r, dict):
            print("  OK dict keys=%s shape=%s" % (list(r.keys()), r["input_ids"].shape))
        elif isinstance(r, str):
            print("  OK str 长度=%d 前 160=%r" % (len(r), r[:160]))
        else:
            print("  OK tensor shape=%s" % tuple(r.shape))
            print("  解码:", repr(tok.decode(r[0])[:200]))
    except Exception as e:
        print("  失败: %r" % e)
        traceback.print_exc()

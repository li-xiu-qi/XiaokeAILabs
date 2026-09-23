#!/usr/bin/env python3
"""Qwen3-VL-4B 对同一分类任务的耗时基准。单卡 CUDA。

对比对象是 Laya（判别式）在 laya-latency.py 里测出的 16.98 ms(EN) / 8.25 ms(ZH)。
本篇回答「同样这件事，生成式大模型要多久」。

口径对齐 laya-latency.py：
  - 同一台机器、同一个环境、bfloat16、batch=1、greedy
  - 输入文本逐字照抄 laya-latency.py 的 EN/ZH 两个样本
  - 任务与四个 category 定义逐字照抄
  - 同样先预热再计时，全部取分位数

LLM 侧必须比 Laya 多拆两个量，否则「延迟」不可解释：
  - prefill：一次 forward 不出 token，等价于 TTFT，随输入长度走
  - full   ：prefill + 自回归解码到 max_new_tokens
  - decode = full - prefill，随输出长度走

测两档 prompt，因为「让 LLM 只回一个词」是它的最优情形，不测就等于高估它：
  - minimal：只准输出类别名（decode 最短，LLM 的最有利上限）
  - json   ：输出 JSON 对象（工程常见形态，decode 更长）

用 AutoTokenizer 而不是 processor，避免 VL 路径往 prompt 里塞 image placeholder
token，那会让 prefill 时间随虚拟图像膨胀，不代表纯文本任务。
"""
import json, os, statistics, time

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
N = int(os.environ.get("BENCH_N", "30"))
WARMUP = 3

print("host:", os.uname().nodename)
print("model:", MODEL, "| N =", N, "| warmup =", WARMUP)

import torch

EN = "I was charged twice for my subscription this month, please refund the duplicate charge."
ZH = "我这个月的订阅被扣了两次款，请把重复的那笔退给我。"

CATS = {
    "billing": "invoices, payments, refunds",
    "technical": "bugs, outages, system errors",
    "sales": "pricing, new contracts",
    "other": "praise, greetings, unrelated chat",
}


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def report(tag, xs):
    xs = sorted(xs)
    print("%-52s n=%-3d min %8.2f  p50 %8.2f  p90 %8.2f  p99 %8.2f  max %9.2f  ms" % (
        tag, len(xs), xs[0] * 1000, pct(xs, 50) * 1000, pct(xs, 90) * 1000,
        pct(xs, 99) * 1000, xs[-1] * 1000))


def build_prompt(style, lang, text):
    cat_lines = "\n".join("- %s: %s" % (k, v) for k, v in CATS.items())
    if lang == "en":
        head = ("Classify the customer message into exactly one category.\n"
                "Categories:\n%s\n\nMessage: %s\n" % (cat_lines, text))
        tail = "Category:" if style == "minimal" else (
            'Return only a JSON object: {"choice": "<category>", "reason": "<short>"}')
    else:
        cat_lines_zh = ("- billing: 账单、扣费、退款\n"
                        "- technical: 故障、报错、系统异常\n"
                        "- sales: 价格、合同、采购\n"
                        "- other: 表扬、问候、无关闲聊")
        head = ("把下面的用户消息分成唯一一类。\n类别：\n%s\n\n消息：%s\n" % (cat_lines_zh, text))
        tail = "类别：" if style == "minimal" else '只返回一个 JSON 对象：{"choice": "<类别>", "reason": "<简短理由>"}'
    return head + tail


def load():
    # AutoModel 对 VL 架构返回的是 backbone（Qwen3VLModel），没有 generate。
    # 逐个试带 generation mixin 的包装类，取第一个加载成功的，避免来回返工。
    from transformers import AutoTokenizer
    candidates = ["AutoModelForImageTextToText", "AutoModelForVision2Seq",
                  "AutoModelForCausalLM", "AutoModelForTextToTextGeneration"]
    import transformers
    tok = AutoTokenizer.from_pretrained(MODEL)
    last = None
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            t0 = time.time()
            model = cls.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
            model.eval()
            print("[load] 使用 %s，耗时 %.2f s" % (name, time.time() - t0))
            return tok, model, time.time() - t0
        except Exception as e:
            last = "%s: %r" % (name, e)
            print("[load] %s 失败，%s" % (name, last))
    raise RuntimeError("没有可用的生成类。最后错误: %s" % last)


def encode(tok, prompt, use_template):
    """返回 (ids_dict, 使用的编码方式)。

    坑：apply_chat_template(return_tensors="pt") 返回的是 BatchFeature，
    它不是 dict 子类，isinstance(x, dict) 判否，接着访问 .shape 会由
    BatchFeature.__getattr__ 抛出无消息的 AttributeError。
    必须用 hasattr(x, "keys") 判，不能 isinstance dict。
    """
    if use_template:
        variants = [
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            [{"role": "user", "content": prompt}],
        ]
        for messages in variants:
            try:
                out = tok.apply_chat_template(messages, add_generation_prompt=True,
                                              return_tensors="pt", tokenize=True)
                if hasattr(out, "keys") and "input_ids" in out.keys():
                    ids = {k: (v.to("cuda") if hasattr(v, "to") else v)
                           for k, v in out.items()}
                elif hasattr(out, "dim"):
                    ids = {"input_ids": out.to("cuda")}
                else:
                    continue
                if ids["input_ids"].dim() == 2:
                    return ids, "chat_template"
            except Exception as e:
                print("  [encode] chat_template 变体失败: %r" % e)
        print("  [encode] chat_template 全部失败，回落裸 prompt")
    ids = tok(prompt, return_tensors="pt").to("cuda")
    return ids, "raw"


def bench(model, tok, prompt, max_new, use_template=True):
    ids, mode = encode(tok, prompt, use_template)
    n_prompt = int(ids["input_ids"].shape[1])
    out = {}
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model(**ids)
        torch.cuda.synchronize()
        out["prefill"] = time.perf_counter() - t0

        torch.cuda.synchronize(); t1 = time.perf_counter()
        gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        out["full"] = time.perf_counter() - t1
    n_new = int(gen.shape[1] - n_prompt)
    out["decode"] = out["full"] - out["prefill"]
    out["n_prompt"] = n_prompt
    out["n_new"] = n_new
    out["mode"] = mode
    out["text"] = tok.decode(gen[0][n_prompt:], skip_special_tokens=True)
    return out


def run_set(model, tok, style, lang, text, max_new, label, use_template=True):
    prompt = build_prompt(style, lang, text)
    pre, ful, dec, tot_new, empty = [], [], [], 0, 0
    sample_text, mode = "", ""
    for i in range(N + WARMUP):
        r = bench(model, tok, prompt, max_new, use_template)
        if i < WARMUP:
            continue
        pre.append(r["prefill"]); ful.append(r["full"]); dec.append(r["decode"])
        tot_new += r["n_new"]
        if not r["text"].strip():
            empty += 1
        if i == WARMUP:
            sample_text, mode = r["text"], r["mode"]
    n_prompt = r["n_prompt"]
    avg_new = tot_new / N
    avg_dec = sum(dec) / N
    print("\n" + "-" * 100)
    print("[%s] style=%s 编码=%s max_new=%d  prompt_tokens=%d  平均生成 token=%.1f  空输出 %d/%d" % (
        label, style, mode, max_new, n_prompt, avg_new, empty, N))
    print("  样例输出:", repr(sample_text[:120]))
    report("  prefill（一次前向，等价 TTFT）", pre)
    report("  full（prefill + 解码）", ful)
    report("  decode（full - prefill）", dec)
    if avg_new > 1:
        print("  %-52s %8.2f ms/token" % ("  每 token 解码（均值）", avg_dec / (avg_new - 1) * 1000))
    return {"label": label, "style": style, "lang": lang, "mode": mode,
            "n_prompt": n_prompt, "empty": empty,
            "avg_new": tot_new / N, "max_new": max_new,
            "prefill_p50": pct(pre, 50) * 1000, "prefill_p99": pct(pre, 99) * 1000,
            "full_p50": pct(ful, 50) * 1000, "full_p99": pct(ful, 99) * 1000,
            "decode_p50": pct(dec, 50) * 1000, "sample": sample_text[:120]}


def vram():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3
    return 0, 0


def main():
    tok, model, load_s = load()
    print("\n[加载] %.2f s | 常驻 %.2f GB 峰值 %.2f GB" % ((load_s,) + vram()))
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)

    results = []
    # max_new 给足：minimal 允许完整类别名，json 允许完整 JSON 对象。
    # 上一轮 json 用 64 却只吐 1 token，是裸 prompt 下模型直接吐 EOS，不是长度受限。
    for lang, text in (("en", EN), ("zh", ZH)):
        results.append(run_set(model, tok, "minimal", lang, text, 12, "短输入 %s" % lang))
        results.append(run_set(model, tok, "json", lang, text, 192, "短输入 %s" % lang))

    print("\n" + "=" * 100)
    print("[汇总] 全部为毫秒，p50")
    print("%-12s %-9s %-14s %-8s %-9s %-9s %-9s %-9s" % (
        "场景", "prompt风格", "编码", "prompt长度", "生成token", "prefill", "full", "decode"))
    for r in results:
        print("%-12s %-9s %-14s %-8d %-9.1f %-9.1f %-9.1f %-9.1f" % (
            r["label"], r["style"], r["mode"], r["n_prompt"], r["avg_new"],
            r["prefill_p50"], r["full_p50"], r["decode_p50"]))
    print("\n[收尾] 常驻 %.2f GB 峰值 %.2f GB" % vram())
    with open(os.environ.get("BENCH_OUT", "qwen4b-result.json"), "w") as f:
        json.dump({"model": MODEL, "load_s": load_s, "results": results}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

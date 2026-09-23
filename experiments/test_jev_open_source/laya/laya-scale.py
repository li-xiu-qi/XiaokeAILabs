#!/usr/bin/env python3
"""Laya 多题合并扩展基准：题数 1 到 256，题目互不相同。单卡 CUDA。

对齐 laya-latency.py 的 [D] 段口径（英文 checkpoint，Router 路径），
把题数上界从 16 推到 256，检验单题边际成本的衰减拐点与激活显存的增长。
题面池共 12 个不同问题，覆盖 choice / score / noul 三种题型，指令领域
彼此错开，题数超过池大小时循环取样，避免同题复制把边际成本测低。
"""
import os, time, statistics, json

os.environ.setdefault("HF_HUB_OFFLINE", "1")

print("host:", os.uname().nodename, "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

import torch
from laya import Router

def avail_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024

# 12 个互不相同的题面，指令领域错开，避免同题复制的分布失真
TEMPLATES = [
    {"type": "choice", "instructions": "Which department should handle this request?",
     "criteria": {"billing": "invoices, payments, refunds",
                  "technical": "bugs, outages, system errors",
                  "sales": "pricing, new contracts",
                  "other": "everything else"}},
    {"type": "choice", "instructions": "What is the customer's primary intent?",
     "criteria": {"refund": "wants money back for a charge",
                  "cancel": "wants to end the subscription",
                  "upgrade": "wants a higher tier or more seats",
                  "inquiry": "asks about features or availability"}},
    {"type": "choice", "instructions": "What is the sentiment of this message?",
     "criteria": {"angry": "complaints, threats, strong negative wording",
                  "neutral": "factual statements without emotion",
                  "positive": "praise, thanks, satisfaction"}},
    {"type": "choice", "instructions": "How should this ticket be prioritized?",
     "criteria": {"low": "cosmetic issues, no business impact",
                  "normal": "standard request, no deadline",
                  "high": "affects a paying customer's workflow",
                  "urgent": "outage, data loss, legal or security impact"}},
    {"type": "choice", "instructions": "Which product area does this concern?",
     "criteria": {"billing_area": "invoices, payment methods, tax",
                  "api": "endpoints, keys, rate limits, webhooks",
                  "dashboard": "charts, filters, exports, slow UI",
                  "mobile": "iOS or Android app behaviour"}},
    {"type": "score", "instructions": "How urgent is this request?",
     "criteria": ["not urgent, purely informational",
                  "needs attention soon but not blocking",
                  "blocking the customer's work right now"]},
    {"type": "score", "instructions": "How satisfied is the customer overall?",
     "criteria": ["very dissatisfied, wants to leave",
                  "mixed feelings, some friction",
                  "satisfied, minor suggestions",
                  "highly satisfied, explicit praise",
                  "delighted, offers to recommend"]},
    {"type": "noul", "instructions": "Does the customer explicitly request a refund?"},
    {"type": "noul", "instructions": "Does the customer threaten to cancel or churn?"},
    {"type": "choice", "instructions": "Which language is the message written in?",
     "criteria": {"english": "English text",
                  "spanish": "Spanish text",
                  "german": "German text",
                  "french": "French text"}},
    {"type": "choice", "instructions": "Which support channel produced this message?",
     "criteria": {"email": "written as an email thread",
                  "chat": "short live-chat style turns",
                  "phone": "transcript of a phone call",
                  "social": "public social media mention"}},
    {"type": "noul", "instructions": "Does the message contain a legal or compliance threat?"},
]

def build_questions(n):
    qs = {}
    for i in range(n):
        t = dict(TEMPLATES[i % len(TEMPLATES)])
        qs["q%03d" % i] = t
    return qs

STATE = {"body": "I was charged twice for my subscription this month, please refund the duplicate charge."}
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
REPS = {1: 10, 2: 10, 4: 10, 8: 10, 16: 10, 32: 5, 64: 5, 128: 3, 256: 3}

print("MemAvailable 基线: %.1f GB" % avail_gb())
router = Router(device="cuda", preload=True, max_loaded=3)
base_alloc = torch.cuda.memory_allocated() / 1024**3
print("Router 就绪，常驻 CUDA: %.2f GB，当前可用 %.1f GB" % (base_alloc, avail_gb()))

r0 = router.route(STATE, build_questions(1))
print("路由: model=%s reason=%s" % (r0.model, r0.reason))

print("\n%s" % ("=" * 100))
print("%6s %10s %10s %12s %12s %12s %10s" % ("题数", "总时长p50", "单题ms", "激活峰值MB", "常驻GB", "MemAvail差", "返回题数"))
print("%s" % ("=" * 100))

results = []
for n in NS:
    qs = build_questions(n)
    avail_before = avail_gb()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    try:
        res = router.predict(STATE, qs)
        torch.cuda.synchronize()
        warm = (time.time() - t) * 1000
    except Exception as e:
        print("%6d 首次调用失败: %s: %s" % (n, type(e).__name__, str(e)[:200]))
        break
    n_out = len(res["answers"])
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for _ in range(REPS[n]):
        t = time.time()
        res = router.predict(STATE, qs)
        torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    peak_mb = (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 1024**2
    lat.sort()
    p50 = statistics.median(lat)
    peak_gb = torch.cuda.memory_allocated() / 1024**3
    avail_after = avail_gb()
    print("%6d %10.2f %10.2f %12.1f %12.2f %12.1f %10d" % (
        n, p50, p50 / n, peak_mb, peak_gb, avail_after - avail_before, n_out))
    results.append({
        "n": n, "total_p50_ms": round(p50, 3), "per_q_ms": round(p50 / n, 3),
        "activation_peak_mb": round(peak_mb, 2), "resident_gb": round(peak_gb, 3),
        "reps": REPS[n], "n_out": n_out,
        "lat_min": round(lat[0], 3), "lat_max": round(lat[-1], 3),
    })

print("\n[收尾] 常驻 CUDA: %.2f GB，MemAvailable: %.1f GB" % (
    torch.cuda.memory_allocated() / 1024**3, avail_gb()))
with open(os.path.expanduser("~/laya-scale-result.json"), "w") as f:
    json.dump({"host": os.uname().nodename, "state": STATE, "results": results}, f, ensure_ascii=False, indent=2)
print("结果已写入 ~/laya-scale-result.json")

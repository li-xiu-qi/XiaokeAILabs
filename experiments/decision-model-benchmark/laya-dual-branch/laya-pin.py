#!/usr/bin/env python3
"""指定调用某个 checkpoint 的三种模式实测。单卡 CUDA。

Router.predict 在 0.3.5 即支持 per-request 覆盖（源码确认）：
  router.predict(state, questions, model="multilingual")   单请求指定
  router = Router(default="multilingual")                  全局默认
另测强灌：中文输入强制 model="english"，看置信度表现。
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

def alloc_gb():
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

ZH = {"body": "我这个月的订阅被扣了两次款，请把重复的那笔退给我。"}
Q = {"department": {
    "type": "choice",
    "instructions": "Which category does this message belong to?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "praise, greetings, unrelated chat",
    },
}}

print("MemAvailable 基线: %.1f GB" % avail_gb())
router = Router(device="cuda", preload=True, max_loaded=3)
print("Router 就绪，常驻 CUDA %.2f GB" % alloc_gb())

CASES = [
    ("不指定（自动路由）", {}),
    ("model=multilingual", {"model": "multilingual"}),
    ("model=english（强灌）", {"model": "english"}),
    ("model=typed-decisions", {"model": "typed-decisions"}),
]

results = []
print("\n%s" % ("=" * 96))
print("%-26s %10s %12s %10s %s" % ("模式", "p50 ms", "choice", "conf", "routing.reason"))
print("%s" % ("=" * 96))
for name, kw in CASES:
    router.predict(ZH, Q, **kw)  # warmup
    lat = []
    for _ in range(10):
        t = time.time()
        res = router.predict(ZH, Q, **kw)
        torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    lat.sort()
    a = res["answers"]["department"]
    rec = {"mode": name, "p50_ms": round(statistics.median(lat), 2),
           "choice": a["choice"], "confidence": round(a["confidence"], 4),
           "routed_to": res["routing"]["model"], "reason": res["routing"]["reason"]}
    results.append(rec)
    print("%-26s %10.2f %12s %10.4f %s" % (name, rec["p50_ms"], a["choice"], a["confidence"], rec["reason"][:60]))

# default 全局默认另起一个 Router 对照
router2 = Router(device="cuda", default="multilingual", max_loaded=3)
router2.predict(ZH, Q)
lat = []
for _ in range(10):
    t = time.time()
    res2 = router2.predict(ZH, Q)
    torch.cuda.synchronize()
    lat.append((time.time() - t) * 1000)
lat.sort()
a = res2["answers"]["department"]
rec = {"mode": "Router(default=multilingual)", "p50_ms": round(statistics.median(lat), 2),
       "choice": a["choice"], "confidence": round(a["confidence"], 4),
       "routed_to": res2["routing"]["model"], "reason": res2["routing"]["reason"]}
results.append(rec)
print("%-26s %10.2f %12s %10.4f %s" % (rec["mode"], rec["p50_ms"], a["choice"], a["confidence"], rec["reason"][:60]))

print("\n[收尾] CUDA: %.2f GB，MemAvailable: %.1f GB" % (alloc_gb(), avail_gb()))
with open(os.path.expanduser("~/laya-pin-result.json"), "w") as f:
    json.dump({"host": os.uname().nodename, "results": results}, f, ensure_ascii=False, indent=2)
print("结果已写入 ~/laya-pin-result.json")

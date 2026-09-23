#!/usr/bin/env python3
"""锁定单个模型 vs Router 自动路由：调用路径对照。单卡 CUDA。

回答一个调用方式问题：不经过 Router，能否直接用 Agent 指定模型？
以及锁 multilingual 后，中文与英文输入各自的延迟、输出与常驻显存。

路径：
  A  Agent("convaiinnovations/laya", subfolder="multilingual")  bundle 内子目录
  C  Router(default="multilingual", max_loaded=1, preload=True)  单模型常驻
  D  Router(device="cuda", preload=True, max_loaded=3)           全自动（对照）
独立仓 laya-multilingual 不在本地缓存，离线加载会失败，跳过（offline 限定）。
"""
import os, time, statistics, json

os.environ.setdefault("HF_HUB_OFFLINE", "1")

print("host:", os.uname().nodename, "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

import torch
from laya import Agent, Router

def avail_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024

def alloc_gb():
    return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

ZH = {"body": "我这个月的订阅被扣了两次款，请把重复的那笔退给我。"}
EN = {"body": "I was charged twice for my subscription this month, please refund the duplicate charge."}
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

def bench(tag, obj, expects_routing):
    torch.cuda.empty_cache()
    out = {"tag": tag}
    for lang, state in (("zh", ZH), ("en", EN)):
        obj.predict(state, Q)  # warmup
        lat = []
        for _ in range(10):
            t = time.time()
            res = obj.predict(state, Q)
            torch.cuda.synchronize()
            lat.append((time.time() - t) * 1000)
        lat.sort()
        a = res["answers"]["department"]
        rec = {"p50_ms": round(statistics.median(lat), 2),
               "choice": a["choice"],
               "confidence": round(a["confidence"], 4)}
        if expects_routing:
            rec["routed_to"] = res["model"]
        out[lang] = rec
    out["cuda_resident_gb"] = round(alloc_gb(), 2)
    return out

print("MemAvailable 基线: %.1f GB" % avail_gb())
results = []

# A. Agent + subfolder（bundle 内 multilingual）
t0 = time.time()
agent_a = Agent("convaiinnovations/laya", device="cuda", subfolder="multilingual")
load = time.time() - t0
r = bench("A_Agent_subfolder", agent_a, False)
r["load_s"] = round(load, 2)
print(json.dumps(r, ensure_ascii=False))
results.append(r)
del agent_a
torch.cuda.empty_cache()

# C. Router 锁定默认 + 单模型常驻
t0 = time.time()
router_c = Router(device="cuda", default="multilingual", max_loaded=1, preload=True)
load = time.time() - t0
r = bench("C_Router_default_zh_max1", router_c, True)
r["load_s"] = round(load, 2)
print(json.dumps(r, ensure_ascii=False))
results.append(r)
del router_c
torch.cuda.empty_cache()

# D. Router 全自动（对照）
t0 = time.time()
router_d = Router(device="cuda", preload=True, max_loaded=3)
load = time.time() - t0
r = bench("D_Router_auto_max3", router_d, True)
r["load_s"] = round(load, 2)
print(json.dumps(r, ensure_ascii=False))
results.append(r)
del router_d
torch.cuda.empty_cache()

print("\n[收尾] CUDA: %.2f GB，MemAvailable: %.1f GB" % (alloc_gb(), avail_gb()))
with open(os.path.expanduser("~/laya-direct-result.json"), "w") as f:
    json.dump({"host": os.uname().nodename, "results": results}, f, ensure_ascii=False, indent=2)
print("结果已写入 ~/laya-direct-result.json")

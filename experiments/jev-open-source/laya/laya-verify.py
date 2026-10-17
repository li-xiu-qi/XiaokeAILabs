#!/usr/bin/env python3
"""Laya 输入输出校验。单卡 CUDA。先 dump 全结构，不猜字段名。"""
import json, time, os, statistics

print("host:", os.uname().nodename)

import laya
from laya import Agent, Router
import torch

print("laya:", laya.__version__, "| torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

def avail_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024

base = avail_gb()
print("MemAvailable 基线: %.1f GB" % base)
if base < 20:
    print("!! 可用内存低于 20GB，中止")
    raise SystemExit(1)

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

t0 = time.time()
agent = Agent("convaiinnovations/laya", device="cuda")
print("\nenglish checkpoint 加载: %.1f s" % (time.time() - t0))

res = agent.predict(state, questions)

print("\n########## 1. 输出结构 ##########")
print("顶层 keys:", sorted(res.keys()))
print("answers keys:", sorted(res["answers"].keys()))
for k, v in res["answers"].items():
    print("  %-16s type=%-7s value_keys=%s" % (k, questions[k]["type"], sorted(v.keys())))

print("\n########## 2. 四种题型的完整输出（原样 dump）##########")
for name in ("department", "urgency", "churn_risk"):
    print("\n--- %s (type=%s) ---" % (name, questions[name]["type"]))
    print(json.dumps(res["answers"][name], ensure_ascii=False, indent=2))

print("\n########## 3. 取值与自检 ##########")
a = res["answers"]
d = a["department"]
print("department.choice      :", d.get("choice"))
print("department.confidence  :", d.get("confidence"))

# 找 choice 的概率字段：任何值为 dict 且键是 criteria 名的字段
crit = set(questions["department"]["criteria"])
for k, v in d.items():
    if isinstance(v, dict) and set(v.keys()) >= crit:
        s = sum(v.values())
        print("choice 概率字段名      :", k)
        print("choice 概率            :", {kk: round(vv, 4) for kk, vv in v.items()})
        print("choice 概率和          : %.6f  %s" % (s, "OK(≈1)" if abs(s - 1) < 1e-4 else "!! 非1"))
        break
else:
    print("!! 未找到 choice 概率字段，d 的字段:", list(d.keys()))

u = a["urgency"]
print("urgency.score          :", u.get("score"))
for k, v in u.items():
    if isinstance(v, (list, tuple)) and len(v) == 3:
        print("urgency 分布字段名      :", k, "->", [round(x, 4) for x in v], "sum=%.4f" % sum(v))
print("urgency.confidence     :", u.get("confidence"))

print("churn_risk.noul        :", a["churn_risk"].get("noul"))
print("refund_requested.noul  :", a["refund_requested"].get("noul"))
print("churn_risk.confidence  :", a["churn_risk"].get("confidence"))

print("\n########## 4. 延迟（10 次，4 题合并单次 forward）##########")
lat = []
for _ in range(10):
    t = time.time()
    agent.predict(state, questions)
    lat.append((time.time() - t) * 1000)
lat.sort()
print("min %.1f | p50 %.1f | max %.1f | mean %.1f ms" %
      (lat[0], statistics.median(lat), lat[-1], statistics.mean(lat)))

print("\n########## 5. Router 模式 ##########")
r = Router(device="cuda")
rres = r.predict(state, questions)
print("routing 完整:", json.dumps(rres.get("routing"), ensure_ascii=False, indent=2))
print("routed department:", rres["answers"]["department"]["choice"],
      "conf:", round(rres["answers"]["department"]["confidence"], 4))

print("\n--- 不跑 forward 的路由判断 ---")
try:
    print("英语态 :", r.route(state, questions).reason)
except Exception as e:
    print("route() 失败:", type(e).__name__, e)
try:
    print("德语态 :", r.route({"body": "Der Kunde wurde zweimal belastet"}, questions).reason)
except Exception as e:
    print("route() 失败:", type(e).__name__, e)

print("\n########## 6. 内存 ##########")
after = avail_gb()
print("基线 %.1f GB -> 现在 %.1f GB（+%.1f GB）" % (base, after, base - after))
print("CUDA 峰值: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))

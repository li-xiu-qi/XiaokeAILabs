#!/usr/bin/env python3
"""Laya 每次推理延迟基准。单卡 CUDA。

拆成六个可分离的量，避免把不同开销混成一个「延迟」：
  A. 模型加载（冷启动）
  B. 路由判定（router.route()，不跑前向，纯 Python 特征判断）
  C. 单题前向：英文 checkpoint vs 多语言 checkpoint
  D. 题目数扩展：1/2/4/8/16 题合并一次 predict，看单题边际成本
  E. Router 路径 vs 直接 Agent 路径，隔离路由开销
  F. 显存与统一内存常驻
"""
import json, os, statistics, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("host:", os.uname().nodename, "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

import torch
from laya import Agent, Router

def avail_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024

def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]

def report(tag, xs, unit="ms"):
    xs = sorted(xs)
    print("%-42s n=%-4d min %7.2f  p50 %7.2f  p90 %7.2f  p99 %7.2f  max %8.2f  均值 %7.2f %s" % (
        tag, len(xs), xs[0] * (1000 if unit == "s" else 1), pct(xs, 50), pct(xs, 90),
        pct(xs, 99), xs[-1] * (1000 if unit == "s" else 1),
        statistics.mean(xs) * (1000 if unit == "s" else 1), unit))

print("MemAvailable 基线: %.1f GB" % avail_gb())

EN = ("I was charged twice for my subscription this month, please refund the duplicate charge.")
ZH = ("我这个月的订阅被扣了两次款，请把重复的那笔退给我。")

CRITERIA = {
    "billing": "invoices, payments, refunds",
    "technical": "bugs, outages, system errors",
    "sales": "pricing, new contracts",
    "other": "praise, greetings, unrelated chat",
}
Q1 = {"intent": {"type": "choice",
                 "instructions": "Which category does this message belong to?",
                 "criteria": CRITERIA}}
QN = {("q%d" % i): dict(Q1["intent"]) for i in range(16)}

# ============ A. 冷启动 ============
print("\n" + "=" * 96)
print("[A] 模型加载（冷启动，无缓存预热）")
print("=" * 96)
t0 = time.time()
agent = Agent("convaiinnovations/laya", device="cuda")
t_en_load = time.time() - t0
print("Agent(english) 单 checkpoint 加载      : %6.2f s" % t_en_load)

t0 = time.time()
router = Router(device="cuda", preload=True, max_loaded=3)
t_router_ready = time.time() - t0
print("Router(preload=True, 三 checkpoint 常驻): %6.2f s" % t_router_ready)
print("Router 就绪后当前可用内存              : %.1f GB" % avail_gb())

if torch.cuda.is_available():
    print("CUDA 已分配                            : %.2f GB" % (torch.cuda.memory_allocated() / 1024**3))
    print("CUDA 峰值                              : %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

# ============ B. 路由判定开销（不跑前向） ============
print("\n" + "=" * 96)
print("[B] 路由判定开销：router.route() 单独调用，不含任何前向")
print("=" * 96)
route_dt = []
for _ in range(2000):
    st = {"body": EN}
    t = time.time(); router.route(st, Q1); route_dt.append((time.time() - t) * 1000)
report("英文输入 route() 判定", route_dt)
route_dt_zh = []
for _ in range(2000):
    st = {"body": ZH}
    t = time.time(); router.route(st, Q1); route_dt_zh.append((time.time() - t) * 1000)
report("中文输入 route() 判定", route_dt_zh)

# ============ C. 单题前向，按语言 ============
print("\n" + "=" * 96)
print("[C] 单题前向（1 个 choice，4 个 criteria），各 300 次")
print("=" * 96)
for tag, body in (("英文 → english", EN), ("中文 → multilingual", ZH)):
    lat = []
    for _ in range(300):
        st = {"body": body}
        t = time.time(); res = router.predict(st, Q1); lat.append((time.time() - t) * 1000)
    report("%s（经 Router）" % tag, lat)
    used = res["routing"]["model"]
    print("      ^^ 实际落到 %s，置信度 %.3f，choice=%s" % (
        used, res["answers"]["intent"]["confidence"], res["answers"]["intent"]["choice"]))

# ============ E. Router 路径 vs 直接调 Agent 路径，隔离路由开销 ============
print("\n" + "=" * 96)
print("[E] 同一条英文输入：Router 路径 vs 直接 Agent 路径")
print("=" * 96)
lat = []
for _ in range(300):
    st = {"body": EN}
    t = time.time(); router.predict(st, Q1); lat.append((time.time() - t) * 1000)
report("经 Router（英文分流到 english）", lat)
lat = []
for _ in range(300):
    st = {"body": EN}
    t = time.time(); agent.predict(st, Q1); lat.append((time.time() - t) * 1000)
report("直接用 Agent（跳过分流）", lat)

# ============ D. 题目数扩展 ============
print("\n" + "=" * 96)
print("[D] 一次 predict 塞 N 个题，看单题边际成本（英文 checkpoint）")
print("=" * 96)
print("%-8s %10s %10s %10s" % ("题数", "总时长ms", "单题ms", "单前向判定"))
prev_total = None
for n in (1, 2, 4, 8, 16):
    lat = []
    for _ in range(100):
        st = {"body": EN}
        qs = {k: v for k, v in list(QN.items())[:n]}
        t = time.time(); agent.predict(st, qs); lat.append((time.time() - t) * 1000)
    total = statistics.median(lat)
    print("%-8d %10.2f %10.2f %10s" % (n, total, total / n,
          ("%.2f" % (total - prev_total)) if prev_total is not None else "基准"))
    prev_total = total

# ============ F. 资源占用 ============
print("\n" + "=" * 96)
print("[F] 资源占用")
print("=" * 96)
if torch.cuda.is_available():
    print("CUDA 当前分配   : %.2f GB" % (torch.cuda.memory_allocated() / 1024**3))
    print("CUDA 峰值       : %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("MemAvailable     : %.1f GB" % avail_gb())
print("引擎线程数       :", torch.get_num_threads())

# 温度系数告警复现（官方 multilingual 校准缺失的现场证据）
print("\n" + "=" * 96)
print("[附] multilingual checkpoint 出厂温度系数告警（首次加载时打印）")
print("=" * 96)
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = router.models["multilingual"] if "multilingual" in getattr(router, "models", {}) else None
    if m is not None:
        print("reload multilingual 观察告警…")
    for x in w:
        print("  %s: %s" % (x.category.__name__, str(x.message)[:160]))

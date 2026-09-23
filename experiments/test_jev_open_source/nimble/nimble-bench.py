#!/usr/bin/env python3
"""Bespoke Nimble 推理实测。单卡 CUDA。

三件事：
  A. 能不能加载、能不能出结果（输入输出契约）
  B. 延迟基准，与 laya 同口径（单次 predict 的 p50/p90/p99）
  C. 中文能不能用（Nimble README 说 language: en，只有英文）
"""
import json
import os
import statistics
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
print("host:", os.uname().nodename)

import sys
sys.path.insert(0, os.path.expanduser("~/nimble"))

import torch
from nimble.scoring.cuda_scorer import CudaCandidateScorer

from pathlib import Path

CFG = json.loads(Path("~/nimble-config.json").expanduser().read_text())
MODEL_PATH = CFG["model_path"]
MODEL_ID = CFG["model_id"]
BASE_REV = CFG["base_revision"]
print("model_path:", MODEL_PATH)

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

def report(tag, xs):
    xs = sorted(xs)
    if not xs:
        print(tag, "无样本")
        return
    print("%-40s n=%-4d min %8.2f  p50 %8.2f  p90 %8.2f  p99 %8.2f  max %9.2f  均值 %8.2f ms" % (
        tag, len(xs), xs[0], pct(xs, 50), pct(xs, 90), pct(xs, 99), xs[-1], statistics.mean(xs)))

print("MemAvailable 基线: %.1f GB" % avail_gb())

SCHEMA = {
    "priority": {
        "type": "enum",
        "choices": ["HIGH", "LOW"],
        "description": "Current urgency of this request.",
        "choice_descriptions": {
            "HIGH": "A critical business operation is currently blocked.",
            "LOW": "An optional enhancement with no current business impact.",
        },
    },
    "requires_review": {
        "type": "boolean",
        "description": "Whether customers are unable to complete a purchase.",
    },
}

EN_CTX = ("The payment service is down for all customers. "
          "Nobody can complete a purchase and revenue is being lost every minute.")
ZH_CTX = ("支付服务对所有客户不可用。没有人能完成购买，每分钟都在损失收入。")

EN_CASES = [
    "The payment service is down for all customers, nobody can complete a purchase.",
    "We would like to add a dark mode option to the settings page someday.",
    "Checkout returns a 500 error for every card and orders cannot be placed.",
    "It would be nice if the export dialog remembered the last folder used.",
    "The login page times out and customers cannot reach their accounts at all.",
    "Consider adding a tooltip explaining what the annual plan discount covers.",
    "Refunds are failing and customers are being charged without receiving service.",
    "A nice improvement would be sorting the invoice list by date.",
]

# ============ A. 契约 ============
print("\n" + "=" * 92)
print("[A] 加载与输出契约")
print("=" * 92)
t0 = time.time()
scorer = CudaCandidateScorer(MODEL_PATH, MODEL_ID, BASE_REV)
print("CudaCandidateScorer 加载: %.2f s，当前可用 %.1f GB" % (time.time() - t0, avail_gb()))

if torch.cuda.is_available():
    print("CUDA 已分配: %.2f GB  峰值: %.2f GB" % (
        torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3))

res = scorer.score(EN_CTX, SCHEMA)
print("\noutput  :", res["output"])
print("fields  :", json.dumps(res["fields"], ensure_ascii=False, indent=2))

# ============ B. 延迟 ============
print("\n" + "=" * 92)
print("[B] 延迟基准，英文，与 laya 同口径（单次 score 调用）")
print("=" * 92)
# 预热
for c in EN_CASES[:2]:
    scorer.score(c, SCHEMA)
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

lat = []
for i in range(120):
    c = EN_CASES[i % len(EN_CASES)]
    t = time.time()
    scorer.score(c, SCHEMA)
    lat.append((time.time() - t) * 1000)
report("单次 score（2 字段）", lat)

# ============ C. 中文 ============
print("\n" + "=" * 92)
print("[C] 中文场景（README 声明 language: en）")
print("=" * 92)
zh_lat = []
for i in range(60):
    c = ZH_CTX
    t = time.time()
    r = scorer.score(c, SCHEMA)
    zh_lat.append((time.time() - t) * 1000)
report("单次 score 中文", zh_lat)
print("中文输出:", r["output"])
print("中文各字段分数:", json.dumps(r["fields"], ensure_ascii=False))

print("\n" + "=" * 92)
print("[F] 资源占用")
print("=" * 92)
if torch.cuda.is_available():
    print("CUDA 当前分配: %.2f GB" % (torch.cuda.memory_allocated() / 1024**3))
    print("CUDA 峰值    : %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("MemAvailable  : %.1f GB" % avail_gb())

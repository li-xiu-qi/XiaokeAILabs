#!/usr/bin/env python3
"""Laya 中英推理效果对比。单卡 CUDA。

平行测试集：同一语义各给中英一句，走 Router 自动路由，算准确率、延迟、置信度。
另做一组强灌实验：中文输入直接打 english checkpoint，检验非拉丁字符崩塌。
"""
import json, time, statistics, os

print("host:", os.uname().nodename)
import laya
from laya import Agent, Router
import torch

def avail_gb():
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024

BASE = avail_gb()
print("MemAvailable 基线: %.1f GB" % BASE)

# ---------- 平行测试集：同一语义，中英各一句 ----------
# label: 期望类别
CASES = [
    # billing 计费退款
    ("billing", "I was charged twice for my subscription this month, please refund the duplicate.",
               "我这个月的订阅被扣了两次款，请把重复的那笔退给我。"),
    ("billing", "The invoice shows an amount I did not agree to, I want it corrected.",
               "发票上的金额不是我认可的数额，我希望改正。"),
    ("billing", "Can you switch me to the annual plan and refund the difference?",
               "能帮我换成年付套餐并退还差价吗？"),
    ("billing", "My card was declined but the payment still went through.",
               "我的卡显示被拒，但钱还是扣走了。"),
    ("billing", "I need a VAT invoice for last quarter's payments.",
               "我需要上个季度付款的增值税发票。"),
    # technical 技术故障
    ("technical", "The app crashes every time I open the export dialog.",
                  "每次打开导出对话框应用就崩溃。"),
    ("technical", "Webhook deliveries stopped arriving after yesterday's update.",
                  "昨天的更新之后 webhook 推送就不再到达了。"),
    ("technical", "I get a 500 error when uploading files larger than 20MB.",
                  "上传超过 20MB 的文件时报 500 错误。"),
    ("technical", "Two-factor codes are rejected even though they are correct.",
                  "两步验证码是对的却一直被拒绝。"),
    ("technical", "The dashboard shows no data since Monday morning.",
                  "周一早上开始仪表盘就不显示数据了。"),
    # sales 销售咨询
    ("sales", "Do you offer discounts for non-profit organizations?",
               "非营利组织有折扣吗？"),
    ("sales", "What is the price difference between the Pro and Team tiers?",
               "Pro 版和 Team 版的价格差是多少？"),
    ("sales", "Is there an education plan for university labs?",
               "大学实验室有教育版套餐吗？"),
    ("sales", "Can I get a quote for two hundred seats?",
               "两百个席位能给我一个报价吗？"),
    ("sales", "Do you have a reseller program in Southeast Asia?",
               "东南亚有代理商计划吗？"),
    # other 其他
    ("other", "Just wanted to say the team is doing excellent work.",
               "你们的团队做得很好，致谢。"),
    ("other", "Congratulations on the launch, looking forward to the roadmap.",
               "恭喜发布，期待后续路线图。"),
    ("other", "Is your office hiring for summer internships?",
               "你们办公室招暑期实习生吗？"),
    ("other", "I liked the talk your founder gave last week.",
               "我喜欢上周你们创始人的演讲。"),
    ("other", "Please add my name to the newsletter list.",
               "请把我加入邮件订阅名单。"),
]

CRITERIA = {
    "billing": "invoices, charges, refunds, payment methods",
    "technical": "bugs, outages, errors, integrations not working",
    "sales": "pricing, discounts, quotes, plans, reselling",
    "other": "praise, greetings, hiring, unrelated chat",
}
Q = {"intent": {"type": "choice",
                "instructions": "Which category does this message belong to?",
                "criteria": CRITERIA}}

LABELS = list(CRITERIA.keys())

# ---------- 1. Router 自动路由，中英各跑一遍 ----------
print("\n" + "=" * 70)
print("[1] Router 自动路由：中英平行对比")
print("=" * 70)

t0 = time.time()
router = Router(device="cuda", preload=True, max_loaded=3)
print("Router(preload=True) 就绪: %.1f s，当前可用 %.1f GB" % (time.time() - t0, avail_gb()))

results = {"en": [], "zh": []}
routed_to = {"en": {}, "zh": {}}
routing_reasons = {"en": set(), "zh": set()}

for lang, idx in (("en", 1), ("zh", 2)):
    lat = []
    for gold, *texts in CASES:
        body = texts[idx - 1]
        st = {"body": body}
        t = time.time()
        res = router.predict(st, Q)
        dt = (time.time() - t) * 1000
        lat.append(dt)
        a = res["answers"]["intent"]
        mdl = res["routing"]["model"]
        routed_to[lang][mdl] = routed_to[lang].get(mdl, 0) + 1
        routing_reasons[lang].add(res["routing"].get("reason", ""))
        pred = a["choice"]
        results[lang].append(dict(gold=gold, pred=pred, conf=a["confidence"],
                                  probs=a["probabilities"], model=mdl, ms=dt))
    results[lang + "_lat"] = lat

for lang in ("en", "zh"):
    r = results[lang]
    ok = sum(1 for x in r if x["pred"] == x["gold"])
    lat = results[lang + "_lat"]
    conf = [x["conf"] for x in r]
    conf_right = [x["conf"] for x in r if x["pred"] == x["gold"]]
    conf_wrong = [x["conf"] for x in r if x["pred"] != x["gold"]]
    print("\n--- %s ---" % ("英文" if lang == "en" else "中文"))
    print("  准确率      : %d/%d = %.1f%%" % (ok, len(r), 100.0 * ok / len(r)))
    print("  路由到      : %s" % routed_to[lang])
    print("  路由理由    : %s" % [s for s in routing_reasons[lang]])
    print("  延迟        : p50 %.1f ms, min %.1f, max %.1f" %
          (statistics.median(lat), min(lat), max(lat)))
    print("  置信度均值  : %.3f（答对 %.3f / 答错 %.3f）" %
          (statistics.mean(conf),
           statistics.mean(conf_right) if conf_right else -1,
           statistics.mean(conf_wrong) if conf_wrong else -1))
    # 逐类
    for lb in LABELS:
        sub = [x for x in r if x["gold"] == lb]
        if sub:
            o = sum(1 for x in sub if x["pred"] == lb)
            print("    %-10s %d/%d" % (lb, o, len(sub)))

# 逐条错例
for lang in ("en", "zh"):
    wrong = [x for x in results[lang] if x["pred"] != x["gold"]]
    if wrong:
        print("\n  [%s] 答错 %d 条:" % (lang, len(wrong)))
        for x in wrong[:8]:
            print("    期望 %-10s 实得 %-10s 置信 %.3f  概率分布 %s" %
                  (x["gold"], x["pred"], x["conf"],
                   {k: round(v, 2) for k, v in x["probs"].items()}))

# ---------- 2. 强灌实验：中文打 english checkpoint ----------
print("\n" + "=" * 70)
print("[2] 强灌实验：中文输入直接打 english checkpoint")
print("=" * 70)
en_agent = Agent("convaiinnovations/laya", device="cuda")  # english, subfolder=None
zh_correct = zh_wrong = 0
conf_all = []
for gold, en_t, zh_t in CASES:
    res = en_agent.predict({"body": zh_t}, Q)
    a = res["answers"]["intent"]
    conf_all.append(a["confidence"])
    if a["choice"] == gold:
        zh_correct += 1
    else:
        zh_wrong += 1
print("  中文 + english checkpoint: 准确率 %d/%d = %.1f%%, 置信度均值 %.3f" %
      (zh_correct, len(CASES), 100.0 * zh_correct / len(CASES), statistics.mean(conf_all)))
print("  对照：同一 Router  multilingual 下中文准确率见 [1]")

# ---------- 3. noul 题型中英对比 ----------
print("\n" + "=" * 70)
print("[3] noul 题型中英对比")
print("=" * 70)
QN = {"threat": {"type": "noul", "instructions": "Does the message threaten to cancel or charge back?"}}
NOUL_CASES = [
    (True,  "If this is not fixed today I will cancel and dispute the charge.",
            "今天不解决我就取消订阅并拒付这笔款。"),
    (True,  "Refund now or I will file a chargeback with my bank.",
            "立刻退款，否则我去银行拒付。"),
    (False, "Please let me know when the export feature is back.",
            "导出功能恢复了请告诉我一声。"),
    (False, "Thanks, the issue resolved itself after the restart.",
            "谢谢，重启后问题自己好了。"),
    (True,  "This is the third outage, I am done, cancel my account.",
            "这是第三次故障，我受够了，注销我的账户。"),
]
for lang, idx in (("en", 1), ("zh", 2)):
    rows = []
    for gold, *t in NOUL_CASES:
        res = router.predict({"body": t[idx - 1]}, QN)
        v = res["answers"]["threat"]["noul"]
        rows.append((gold, v, res["routing"]["model"]))
    print("  --- %s ---  路由: %s" % ("英文" if lang == "en" else "中文", set(r[2] for r in rows)))
    for g, v, _ in rows:
        flag = "对" if (v >= 0.5) == g else "错"
        print("    真值 %-5s noul=%.3f  %s" % (g, v, flag))
    tp = sum(1 for g, v, _ in rows if g and v >= 0.5)
    fn = sum(1 for g, v, _ in rows if g and v < 0.5)
    fp = sum(1 for g, v, _ in rows if not g and v >= 0.5)
    tn = sum(1 for g, v, _ in rows if not g and v < 0.5)
    print("    TP %d FN %d FP %d TN %d" % (tp, fn, fp, tn))

print("\n[内存] 基线 %.1f GB -> 现在 %.1f GB（占用 %.1f GB），CUDA 峰值 %.2f GB"
      % (BASE, avail_gb(), BASE - avail_gb(), torch.cuda.max_memory_allocated() / 1024 ** 3))
print("\n[口径] 样本量小（每题 20 条、noul 5 条），只说明方向，不构成基准对比。")

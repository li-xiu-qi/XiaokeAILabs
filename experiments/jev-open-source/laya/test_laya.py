#!/usr/bin/env python3
"""Laya 最小可运行示例：一条输入、一个问题，跑通完整调用链。

对应教程第三步。用 Router 自动分流，输入英文工单，打印路由去向与判定结果。
首次运行会自动下载三个 checkpoint，耗时约一分钟；国内网络先设
HF_ENDPOINT=https://hf-mirror.com。
"""
from laya import Router

# 初始化路由器，指定使用 GPU 并预加载三套权重
router = Router(device="cuda", preload=True, max_loaded=3)

# 1. 待分析的输入文本
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel your plan."
}

# 2. 定义判定问题与候选选项描述
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    }
}

# 3. 执行前向推理
result = router.predict(state, questions)

# 4. 打印路由去向与判定结果
print("路由到：", result["routing"]["model"])
print("判定结果：", result["answers"]["department"]["choice"],
      "置信度：", round(result["answers"]["department"]["confidence"], 4))

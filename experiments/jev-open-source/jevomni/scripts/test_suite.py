#!/usr/bin/env python3
"""Jev-Omni 综合测试套件：单次加载模型，跑完六组测试。

测试组：
1. question_types：boolean、choice、score 三种题型
2. option_scale：2/4/8/16/20/32 候选，验证 20 以上的质量边界
3. zh_business：15 例中文工单分类，四类金标
4. stability：同一文本用例重复多次，判定翻转率与概率波动
5. multimodal：图像、音频、视频三份独立素材的内容理解，加图像重复稳定性
6. text_length：短、约 500、约 2k token 三档

结果写 results/test_suite_results.json。日志按规范重定向保存。
需要先运行 download_assets.py 准备三份素材。
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import (
    setup_environment,
    patch_ffmpeg,
    default_paths,
    ensure_jev_omni_importable,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stability-runs", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    setup_environment(args.models_dir)
    patch_ffmpeg()

    paths = default_paths()
    for name in ("image", "audio", "video"):
        if not os.path.isfile(paths[name]):
            sys.exit(f"缺少 {name} 素材：{paths[name]}，请先运行 download_assets.py")

    snapshot_path = ensure_jev_omni_importable(args.models_dir)
    if snapshot_path is None:
        sys.exit("在缓存中找不到 Jev-Omni，请先运行 download_models.py")
    from jev_omni import load_jev_omni

    print("加载模型...", flush=True)
    t0 = time.perf_counter()
    clf = load_jev_omni(device=args.device)
    print(f"加载完成，用时 {time.perf_counter() - t0:.0f} 秒", flush=True)

    results = {}

    def run(**kwargs):
        return clf.predict(**kwargs)

    # ---------- 1. 三种题型 ----------
    print("测试组 1：题型覆盖", flush=True)
    base_state = "Hi, we were billed twice for March. Please refund the duplicate today."
    t_results = {}

    r_bool = run(state=base_state,
                 question="Does this message convey urgency?",
                 options=["No", "Yes"])
    t_results["boolean"] = {"prediction": r_bool["prediction"],
                            "confidence": round(r_bool["confidence"], 4)}

    r_choice = run(state=base_state,
                   question="Which department should handle this?",
                   options=["Billing", "Technical", "Sales", "Other"])
    t_results["choice"] = {"prediction": r_choice["prediction"],
                           "confidence": round(r_choice["confidence"], 4),
                           "distribution": {k: round(v, 4)
                                            for k, v in r_choice["probabilities"].items()}}

    r_score = run(state=base_state,
                  question="How urgent is this on a 1-5 scale?",
                  options=["Not urgent", "Slightly urgent", "Moderately urgent",
                           "Very urgent", "Extremely urgent"])
    t_results["score"] = {"prediction": r_score["prediction"],
                          "confidence": round(r_score["confidence"], 4),
                          "distribution": {k: round(v, 4)
                                           for k, v in r_score["probabilities"].items()}}
    results["question_types"] = t_results
    print(json.dumps(t_results, ensure_ascii=False), flush=True)

    # ---------- 2. 选项规模 ----------
    print("测试组 2：选项规模 2/4/8/16/20/32", flush=True)
    scale_results = {}
    option_pool = [f"option-{i}" for i in range(32)]
    # 把正确类语义固定为第一项，其余为无干扰描述
    for n in (2, 4, 8, 16, 20, 32):
        options = ["refund and billing issue"] + option_pool[1:n]
        r = run(state="Customer requests a refund for a duplicate charge.",
                question="What is this about?", options=options)
        scale_results[f"n={n}"] = {
            "prediction_index": r["prediction_index"],
            "top1_correct": r["prediction_index"] == 0,
            "confidence": round(r["confidence"], 4),
            "entropy_probe": round(r["probabilities"][r["prediction"]], 4),
        }
    results["option_scale"] = scale_results
    print(json.dumps(scale_results, ensure_ascii=False), flush=True)

    # ---------- 3. 中文业务 15 例 ----------
    print("测试组 3：中文业务 15 例", flush=True)
    zh_question = "这个请求应该由哪个部门处理？"
    zh_options = ["账务", "技术", "销售", "其他"]
    zh_cases = [
        ("我这个月信用卡被扣了两次费，请尽快把多扣的钱退给我", "账务"),
        ("发票开错了，抬头需要修改后重新开", "账务"),
        ("这个月的账单金额和我实际使用的对不上", "账务"),
        ("会员自动续费扣了费，但我已经提前取消了", "账务"),
        ("付款的时候一直提示支付失败，银行卡没有问题", "账务"),
        ("App 一直闪退，根本打不开，重启手机也没用", "技术"),
        ("登录的时候提示服务器错误 502", "技术"),
        ("账号明明密码正确却一直登录不上去", "技术"),
        ("你们的产品支持 SSO 单点登录吗", "技术"),
        ("导出报表的时候总是失败，文件下载不下来", "技术"),
        ("想了解企业版一年多少钱，团队采购有没有折扣", "销售"),
        ("我们公司打算采购 200 个账号，怎么签合同", "销售"),
        ("现在用的是基础版，想咨询怎么升级到专业版", "销售"),
        ("想申请你们产品的免费试用，流程是什么", "销售"),
        ("你们的客服电话怎么一直没人接", "其他"),
    ]
    zh_results = []
    correct = 0
    for text, gold in zh_cases:
        r = run(state=text, question=zh_question, options=zh_options)
        ok = r["prediction"] == gold
        correct += ok
        zh_results.append({"gold": gold, "prediction": r["prediction"],
                           "correct": ok, "confidence": round(r["confidence"], 4)})
    results["zh_business"] = {"accuracy": f"{correct}/{len(zh_cases)}",
                              "cases": zh_results}
    print(f"中文准确率 {correct}/{len(zh_cases)}", flush=True)

    # ---------- 4. 稳定性 ----------
    print(f"测试组 4：同一用例重复 {args.stability_runs} 次", flush=True)
    stab_kw = dict(state="Customer requests a refund for a duplicate charge.",
                   question="What is this about?",
                   options=["Billing", "Technical", "Sales", "Other"])
    preds, confs, dists = [], [], []
    for _ in range(args.stability_runs):
        r = run(**stab_kw)
        preds.append(r["prediction"])
        confs.append(r["confidence"])
        dists.append([round(r["probabilities"][o], 4) for o in stab_kw["options"]])
    unique_preds = sorted(set(preds))
    stability = {
        "runs": args.stability_runs,
        "unique_predictions": unique_preds,
        "prediction_changes": len(unique_preds) > 1,
        "confidence_mean": round(statistics.mean(confs), 4),
        "confidence_stdev": round(statistics.pstdev(confs), 4),
        "confidence_range": [round(min(confs), 4), round(max(confs), 4)],
        "first_distribution": dists[0],
        "last_distribution": dists[-1],
    }
    results["stability_en_choice"] = stability
    print(json.dumps({k: v for k, v in stability.items()
                      if k not in ("first_distribution", "last_distribution")},
                     ensure_ascii=False), flush=True)

    # 中文稳定性同样测，验证跨模态观察是否是普遍现象
    preds_zh, confs_zh = [], []
    stab_zh = dict(state="我这个月信用卡被扣了两次费，请退款",
                   question=zh_question, options=zh_options)
    for _ in range(args.stability_runs):
        r = run(**stab_zh)
        preds_zh.append(r["prediction"])
        confs_zh.append(r["confidence"])
    results["stability_zh_choice"] = {
        "unique_predictions": sorted(set(preds_zh)),
        "prediction_changes": len(set(preds_zh)) > 1,
        "confidence_mean": round(statistics.mean(confs_zh), 4),
        "confidence_stdev": round(statistics.pstdev(confs_zh), 4),
        "confidence_range": [round(min(confs_zh), 4), round(max(confs_zh), 4)],
    }
    print("中文稳定性：", json.dumps(results["stability_zh_choice"],
                                     ensure_ascii=False), flush=True)

    # ---------- 5. 多模态理解 ----------
    print("测试组 5：图像、音频、视频内容理解", flush=True)
    multi = {"image": None, "audio": None, "video": None,
             "multimodal_stability": {}}

    # 图像：虎斑猫特写，正确答案为猫
    ri = run(state="一张动物照片。",
             question="画面的主体是什么？",
             options=["猫", "狗", "人物", "汽车"],
             media=paths["image"], modality="image")
    multi["image"] = {
        "gold": "猫",
        "prediction": ri["prediction"],
        "correct": ri["prediction"] == "猫",
        "confidence": round(ri["confidence"], 4),
        "distribution": {k: round(v, 4) for k, v in ri["probabilities"].items()},
    }

    # 音频：两人英文对话讨论排球队，正确答案为人声对话
    ra = run(state="一段约 12 秒的录音。",
             question="音频里是否包含人声？",
             options=["人声对话", "仅音乐", "静音"],
             media=paths["audio"], modality="audio")
    multi["audio"] = {
        "gold": "人声对话",
        "prediction": ra["prediction"],
        "correct": ra["prediction"] == "人声对话",
        "confidence": round(ra["confidence"], 4),
        "distribution": {k: round(v, 4) for k, v in ra["probabilities"].items()},
    }

    # 视频：年轻人先按太阳穴表现疲惫，第 9 秒戴眼镜低头阅读
    rv = run(state="一段约 19 秒的人物近景片段。",
             question="视频里的人主要在做什么？",
             options=["按太阳穴表现疲惫", "吃东西", "跑步", "睡觉"],
             media=paths["video"], modality="video")
    multi["video"] = {
        "gold": "按太阳穴表现疲惫",
        "prediction": rv["prediction"],
        "correct": rv["prediction"] == "按太阳穴表现疲惫",
        "confidence": round(rv["confidence"], 4),
        "distribution": {k: round(v, 4) for k, v in rv["probabilities"].items()},
    }

    # 多模态稳定性：同一图像重复多次，观察判定与置信度是否稳定
    img_stab_kw = dict(state="一张动物照片。",
                       question="画面的主体是什么？",
                       options=["猫", "狗", "人物", "汽车"],
                       media=paths["image"], modality="image")
    img_preds, img_confs = [], []
    for _ in range(args.stability_runs):
        r = run(**img_stab_kw)
        img_preds.append(r["prediction"])
        img_confs.append(r["confidence"])
    multi["multimodal_stability"]["image_repeat"] = {
        "runs": args.stability_runs,
        "unique_predictions": sorted(set(img_preds)),
        "prediction_changes": len(set(img_preds)) > 1,
        "confidence_mean": round(statistics.mean(img_confs), 4),
        "confidence_stdev": round(statistics.pstdev(img_confs), 4),
        "confidence_range": [round(min(img_confs), 4), round(max(img_confs), 4)],
    }

    results["multimodal"] = multi
    print(json.dumps({k: multi[k] for k in ("image", "audio", "video")},
                     ensure_ascii=False), flush=True)
    print("图像稳定性：", json.dumps(multi["multimodal_stability"]["image_repeat"],
                                     ensure_ascii=False), flush=True)

    # ---------- 6. 文本长度 ----------
    print("测试组 6：文本长度三档", flush=True)
    length_results = {}
    for tag, state in (
        ("short", "The deployment window is closed."),
        ("about_500_tokens", "Operations note. " * 35 + "The deployment window is closed."),
        ("about_2k_tokens", "Quarterly operations note. " * 150
                            + "The deployment window is closed."),
    ):
        r = run(state=state,
                question="Is the deployment window still open?",
                options=["Yes", "No"])
        length_results[tag] = {
            "prediction": r["prediction"],
            "correct": r["prediction"] == "No",
            "confidence": round(r["confidence"], 4),
        }
    results["text_length"] = length_results
    print(json.dumps(length_results, ensure_ascii=False), flush=True)

    # ---------- 保存 ----------
    os.makedirs(paths["results"], exist_ok=True)
    out_path = os.path.join(paths["results"], "test_suite_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n全部测试完成，结果保存：{out_path}", flush=True)


if __name__ == "__main__":
    main()

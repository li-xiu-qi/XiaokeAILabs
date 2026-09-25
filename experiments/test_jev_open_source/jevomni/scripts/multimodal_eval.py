#!/usr/bin/env python3
"""Jev-Omni 多模态批量评估：单次加载模型，跑完图像、音频两组分类。

目的：在不止一份素材上统计多模态分类准确率，而不是只看单个样例。
- 图像：assets/eval/images/ 下 10 张图，统一用 10 个候选标签，金标即文件名
- 音频：assets/eval/audio/ 下 3 段，候选为人声对话/音乐/环境声，金标即文件名
- 视频：基础集的 person-reading.mp4 单独测一次动作理解

需要先运行 download_assets.py --full。
结果写 results/multimodal_eval.json，并打印汇总。
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

IMAGE_LABELS = ["猫", "狗", "汽车", "人物", "自行车", "鸟", "船", "椅子", "建筑", "花"]
# 评估图片按英文文件名保存，这里给出标签到文件名（无扩展名）的映射
IMAGE_FILE_KEY = {
    "猫": "cat", "狗": "dog", "汽车": "car", "人物": "person",
    "自行车": "bicycle", "鸟": "bird", "船": "boat", "椅子": "chair",
    "建筑": "building", "花": "flower",
}
AUDIO_LABELS = ["人声对话", "音乐", "环境声"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def evaluate_group(clf, files, labels, question, modality, state):
    cases = []
    for fp, gold in files:
        r = clf.predict(state=state, question=question, options=labels,
                        media=fp, modality=modality)
        cases.append({
            "file": os.path.basename(fp),
            "gold": gold,
            "prediction": r["prediction"],
            "correct": r["prediction"] == gold,
            "confidence": round(r["confidence"], 4),
        })
    return cases


def summarize(cases):
    n = len(cases)
    correct = sum(c["correct"] for c in cases)
    confs = [c["confidence"] for c in cases]
    return {
        "n": n,
        "correct": correct,
        "accuracy": f"{correct}/{n}",
        "confidence_mean": round(statistics.mean(confs), 4),
        "confidence_min": round(min(confs), 4),
        "confidence_max": round(max(confs), 4),
    }


def main():
    args = parse_args()
    setup_environment(args.models_dir)
    patch_ffmpeg()

    paths = default_paths()
    root = os.path.dirname(paths["image"])  # assets
    eval_images = os.path.join(root, "eval", "images")
    eval_audio = os.path.join(root, "eval", "audio")
    for d in (eval_images, eval_audio):
        if not os.path.isdir(d):
            sys.exit(f"缺少扩展素材目录 {d}，请先运行 download_assets.py --full")

    snapshot_path = ensure_jev_omni_importable(args.models_dir)
    if snapshot_path is None:
        sys.exit("在缓存中找不到 Jev-Omni，请先运行 download_models.py")
    from jev_omni import load_jev_omni

    print("加载模型...", flush=True)
    t0 = time.perf_counter()
    clf = load_jev_omni(device=args.device)
    print(f"加载完成，用时 {time.perf_counter() - t0:.0f} 秒", flush=True)

    results = {}

    # 图像：文件名（去扩展名）即金标
    image_files = []
    for lab in IMAGE_LABELS:
        fp = os.path.join(eval_images, f"{IMAGE_FILE_KEY[lab]}.jpg")
        if os.path.isfile(fp):
            image_files.append((fp, lab))
    print(f"\n图像分类 {len(image_files)} 张", flush=True)
    img_cases = evaluate_group(
        clf, image_files, IMAGE_LABELS,
        "这张图片的主体属于哪一类？", "image", "一张待分类的图片。")
    results["image"] = {"summary": summarize(img_cases), "cases": img_cases}
    print(json.dumps(results["image"]["summary"], ensure_ascii=False), flush=True)
    for c in img_cases:
        if not c["correct"]:
            print("  错误：", json.dumps(c, ensure_ascii=False), flush=True)

    # 音频
    audio_gold = {"speech": "人声对话", "music": "音乐", "ambient": "环境声"}
    audio_files = []
    for fname, gold in audio_gold.items():
        fp = os.path.join(eval_audio, f"{fname}.mp3")
        if os.path.isfile(fp):
            audio_files.append((fp, gold))
    print(f"\n音频分类 {len(audio_files)} 段", flush=True)
    aud_cases = evaluate_group(
        clf, audio_files, AUDIO_LABELS,
        "这段音频属于哪一类？", "audio", "一段待分类的音频。")
    results["audio"] = {"summary": summarize(aud_cases), "cases": aud_cases}
    print(json.dumps(results["audio"]["summary"], ensure_ascii=False), flush=True)
    for c in aud_cases:
        print("  ", json.dumps(c, ensure_ascii=False), flush=True)

    # 视频动作理解
    print("\n视频动作理解 1 个", flush=True)
    rv = clf.predict(
        state="一段约 19 秒的人物近景片段。",
        question="视频里的人主要在做什么？",
        options=["按太阳穴表现疲惫", "吃东西", "跑步", "睡觉"],
        media=paths["video"], modality="video")
    results["video"] = {
        "gold": "按太阳穴表现疲惫",
        "prediction": rv["prediction"],
        "correct": rv["prediction"] == "按太阳穴表现疲惫",
        "confidence": round(rv["confidence"], 4),
    }
    print(json.dumps(results["video"], ensure_ascii=False), flush=True)

    os.makedirs(paths["results"], exist_ok=True)
    out_path = os.path.join(paths["results"], "multimodal_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估完成，结果保存：{out_path}", flush=True)


if __name__ == "__main__":
    main()

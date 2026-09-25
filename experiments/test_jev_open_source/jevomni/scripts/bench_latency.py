#!/usr/bin/env python3
"""Jev-Omni 四模态延迟基准。

模型只加载一次，每个模态预热后计时，输出 p50/p90/p99。
结果同时写入 results/latency_results.json。口径为进程内直调，
不含预处理脚本开销与网络时间，与官方公布的 H200 数字口径一致。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime import (
    setup_environment,
    patch_ffmpeg,
    default_paths,
    ensure_jev_omni_importable,
)


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(values):
    return {
        "count": len(values),
        "p50": round(percentile(values, 0.5), 2),
        "p90": round(percentile(values, 0.9), 2),
        "p99": round(percentile(values, 0.99), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", default=None, help="权重缓存目录，需与下载时一致")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", type=int, default=30, help="每个模态的计时次数")
    return p.parse_args()


def time_predict(clf, kwargs, runs, warmup):
    for _ in range(warmup):
        clf.predict(**kwargs)
    values = []
    for _ in range(runs):
        t0 = time.perf_counter()
        clf.predict(**kwargs)
        values.append((time.perf_counter() - t0) * 1000)
    return summarize(values)


def main():
    args = parse_args()
    runs = args.runs
    setup_environment(args.models_dir)
    patch_ffmpeg()

    paths = default_paths()
    for name in ("video", "image", "audio"):
        if not os.path.isfile(paths[name]):
            sys.exit(f"缺少 {name} 素材：{paths[name]}，请先准备素材")

    snapshot_path = ensure_jev_omni_importable(args.models_dir)
    if snapshot_path is None:
        sys.exit("在缓存中找不到 Jev-Omni，请先运行 download_models.py")
    from jev_omni import load_jev_omni

    print("加载模型...")
    mem_before = mem_available_gb()
    t0 = time.perf_counter()
    clf = load_jev_omni(device=args.device)
    load_seconds = round(time.perf_counter() - t0, 1)
    print(f"加载完成，用时 {load_seconds} 秒")

    results = {
        "environment": {
            "device": args.device,
            "load_seconds": load_seconds,
            "mem_available_before_gb": mem_before,
            "mem_available_after_gb": mem_available_gb(),
        },
        "latency_ms": {},
    }

    text_short = dict(
        state="The meeting starts at 10 AM. It is now 9 AM.",
        question="Has the meeting started?",
        options=["Yes", "No"],
    )
    text_long = dict(
        state="Quarterly operations note. " * 150 + "The deployment window is closed.",
        question="Is the deployment window still open?",
        options=["Yes", "No"],
    )
    image_kw = dict(
        state="一张动物照片。",
        question="画面的主体是什么？",
        options=["猫", "狗", "人物", "汽车"],
        media=paths["image"],
        modality="image",
    )
    audio_kw = dict(
        state="一段约 12 秒的录音。",
        question="音频里是否包含人声？",
        options=["人声对话", "仅音乐", "静音"],
        media=paths["audio"],
        modality="audio",
    )
    video_kw = dict(
        state="一段约 19 秒的人物近景片段。",
        question="视频里的人主要在做什么？",
        options=["按太阳穴表现疲惫", "吃东西", "跑步", "睡觉"],
        media=paths["video"],
        modality="video",
    )

    bench_cases = [
        ("text_short", text_short, 5),
        ("text_long_about_2k_tokens", text_long, 3),
        ("image", image_kw, 3),
        ("audio_13s", audio_kw, 3),
        ("video_16frames", video_kw, 2),
    ]

    for name, kwargs, warmup in bench_cases:
        print(f"基准测试：{name}（{runs} 次）...")
        results["latency_ms"][name] = time_predict(clf, kwargs, runs, warmup)
        r = results["latency_ms"][name]
        print(f"  p50={r['p50']} p90={r['p90']} p99={r['p99']}")

    os.makedirs(paths["results"], exist_ok=True)
    out_path = os.path.join(paths["results"], "latency_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""下载 Jev-Omni 全部权重与 Gemma 4 底座。

默认下载到 HuggingFace 标准缓存（load_jev_omni 从缓存加载）。
用 --target-dir 可下载到实验目录下的 models/，此时需要配合 quickstart
与 bench 脚本的环境变量使用（脚本会自动设置 HF_HOME）。

国内网络用 --mirror 走 hf-mirror。
"""
import argparse
import os
import sys

JEV_OMNI_ID = "akhilaaa3/Jev-Omni"
GEMMA_ID = "google/gemma-4-12B-it"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mirror", action="store_true", help="使用 hf-mirror 镜像")
    p.add_argument(
        "--target-dir",
        default=None,
        help="下载到指定目录（默认 HuggingFace 缓存）。建议填实验目录下的 models",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # hf-mirror 对 Xet 存储返回 401，统一关闭。
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    if args.target_dir:
        target = os.path.abspath(args.target_dir)
        os.makedirs(target, exist_ok=True)
        # 直接指定 Hub 缓存目录，snapshot_download 不带 local_dir 时会按缓存结构
        # 落在这里，load_jev_omni 也从同一缓存读取。quickstart 与 bench 脚本
        # 需要用相同的 --models-dir 才能找到权重。
        os.environ["HF_HUB_CACHE"] = target
        print(f"HF_HUB_CACHE 设为 {target}")

    from huggingface_hub import snapshot_download

    print(f"下载 Jev-Omni 决策权重（约 48 GB）...")
    p1 = snapshot_download(JEV_OMNI_ID)
    print(f"完成：{p1}")

    print(f"下载 Gemma 4 12B 底座（约 24 GB）...")
    p2 = snapshot_download(GEMMA_ID)
    print(f"完成：{p2}")

    print("全部权重下载完成。")


if __name__ == "__main__":
    sys.exit(main())

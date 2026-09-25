#!/usr/bin/env python3
"""下载多模态测试素材：从实验仓库的 GitHub Release Assets 拉取并校验。

素材不进 git，仓库本体内只留脚本和文档。素材托管在 Release：
tag `assets/test_jev_open_source`，下载后按 SHA256SUMS 校验哈希。

默认下载基础集（quickstart 用）：
- cat.jpg：虎斑猫特写
- conversation.mp3：两人英文对话，约 12 秒
- person-reading.mp4：人物近景，约 19 秒

加 --full 额外下载批量评估集（multimodal_eval 用）：
- eval/ 下 10 类图片和人声、音乐、环境声三段音频

网络：直连失败时用 --proxy 指定本地代理，例如 --proxy http://127.0.0.1:7890；
或用 --gh-prefix 走 GitHub 加速前缀。--only 可只下载某一类。
"""
import argparse
import hashlib
import os
import sys
import urllib.request

OWNER = "li-xiu-qi"
REPO = "XiaokeAILabs"
TAG = "assets/test_jev_open_source"

# asset 文件名 -> 本地相对 assets 目录的落位路径
BASIC = {
    "cat.jpg": "cat.jpg",
    "conversation.mp3": "conversation.mp3",
    "person-reading.mp4": "person-reading.mp4",
}

EVAL = {
    "eval-cat.jpg": "eval/images/cat.jpg",
    "eval-dog.jpg": "eval/images/dog.jpg",
    "eval-car.jpg": "eval/images/car.jpg",
    "eval-person.jpg": "eval/images/person.jpg",
    "eval-bicycle.jpg": "eval/images/bicycle.jpg",
    "eval-bird.jpg": "eval/images/bird.jpg",
    "eval-boat.jpg": "eval/images/boat.jpg",
    "eval-chair.jpg": "eval/images/chair.jpg",
    "eval-building.jpg": "eval/images/building.jpg",
    "eval-flower.jpg": "eval/images/flower.jpg",
    "eval-speech.mp3": "eval/audio/speech.mp3",
    "eval-music.mp3": "eval/audio/music.mp3",
    "eval-ambient.mp3": "eval/audio/ambient.mp3",
}

# --only 过滤：只保留某一类文件
ONLY_KIND = {
    "image": (".jpg",),
    "audio": (".mp3",),
    "video": (".mp4",),
}


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    experiment_root = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets-dir",
                   default=os.path.join(experiment_root, "assets"),
                   help="素材根目录，默认实验目录下 assets/")
    p.add_argument("--full", action="store_true", help="额外下载批量评估集")
    p.add_argument("--only", choices=["image", "audio", "video"],
                   help="只下载某一类素材")
    p.add_argument("--proxy", default=None,
                   help="下载代理，例如 http://127.0.0.1:7890")
    p.add_argument("--gh-prefix", default=None,
                   help="GitHub 加速前缀，拼在下载地址前，例如 https://ghfast.top/")
    p.add_argument("--force", action="store_true",
                   help="已存在也重新下载并校验")
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


def make_opener(proxy):
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def base_download_url(args):
    url = (f"https://github.com/{OWNER}/{REPO}/releases/download/"
           f"{TAG}/")
    if args.gh_prefix:
        url = args.gh_prefix.rstrip("/") + "/" + url
    return url


def fetch_sha256sums(opener, args):
    url = base_download_url(args) + "SHA256SUMS"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=args.timeout) as r:
        text = r.read().decode("utf-8")
    sums = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip()] = parts[0].strip().lower()
    return sums


def download_one(asset_name, local_rel, expected_hash, opener, args):
    target = os.path.join(args.assets_dir, local_rel.replace("/", os.sep))
    if os.path.isfile(target) and not args.force:
        actual = hashlib.sha256(open(target, "rb").read()).hexdigest()
        if actual == expected_hash:
            print(f"已存在且校验通过，跳过：{local_rel}")
            return True
    os.makedirs(os.path.dirname(target), exist_ok=True)
    url = base_download_url(args) + asset_name
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = target + ".part"
    try:
        with opener.open(req, timeout=args.timeout) as r, open(tmp, "wb") as f:
            f.write(r.read())
        actual = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if actual != expected_hash:
            print(f"  校验失败：{local_rel}（哈希不符）")
            os.remove(tmp)
            return False
        os.replace(tmp, target)
    except Exception as e:
        print(f"  下载失败：{local_rel}：{e}")
        if os.path.isfile(tmp):
            os.remove(tmp)
        return False
    print(f"  完成：{local_rel}")
    return True


def main():
    args = parse_args()
    opener = make_opener(args.proxy)

    print("读取 SHA256SUMS ...")
    try:
        sums = fetch_sha256sums(opener, args)
    except Exception as e:
        print(f"无法获取校验清单：{e}")
        print("可加 --proxy 或 --gh-prefix 重试。")
        return 1

    assets = dict(BASIC)
    if args.full:
        assets.update(EVAL)
    if args.only:
        exts = ONLY_KIND[args.only]
        assets = {a: p for a, p in assets.items() if a.lower().endswith(exts)}

    missing_hash = [a for a in assets if a not in sums]
    if missing_hash:
        print("校验清单缺少：", ", ".join(missing_hash))
        return 1

    failed = []
    for asset_name, local_rel in assets.items():
        if not download_one(asset_name, local_rel, sums[asset_name], opener, args):
            failed.append(asset_name)

    if failed:
        print("\n下载或校验失败：", ", ".join(failed))
        return 1
    print(f"\n素材准备完成（{len(assets)} 份），全部通过 SHA256 校验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

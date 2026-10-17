# -*- coding: utf-8 -*-
"""跨模态检索实验的数据集获取脚本。

设计约束（2026-09-26 实测）：
  - DGX Spark（192.168.1.170）没有外网出口，IPv4 与 IPv6 均不通，
    huggingface.co 与 zenodo.org 全部超时。本机 HF 模型缓存是之前
    从别处搬进去的，不是本机下载的。
  - 因此本脚本不在 DGX 上执行下载，而是在有外网的中继机（ke 主机）
    执行，产物再推到 DGX。脚本放在 DGX 是为了让实验环境自描述：
    任何人看到 datasets/ 都能查到每份数据的来源、体积、许可与获取命令。
  - 数据集只落 DGX，不在中继机长期保留。中继机的 staging 目录传完即清。

数据源分三类，脚本统一处理：
  hf      HuggingFace 数据集仓库，走 resolve 直链，curl 断点续传
  http    普通 HTTPS 直链（Karpathy 划分的标注文件）
  gdrive  Google Drive，需要 gdown；装不上就跳过并在 manifest 记缺口

用法（在中继机执行）：
  python fetch_crossmodal_datasets.py --dest <staging_dir> --all
  python fetch_crossmodal_datasets.py --dest <staging_dir> --only flickr30k aro_vg
  python fetch_crossmodal_datasets.py --verify <dir>     # 只校验不下载

产物结构：
  <dest>/flickr30k/images/*.jpg
  <dest>/flickr30k/flickr_annotations_30k.csv
  <dest>/aro/vgr_annotations.json  vga_annotations.json  vgr_vga_images/
  <dest>/aro/flickr30k_order.json
  <dest>/fsd50k/parquet/*.parquet
  <dest>/manifest.json
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

# ------------------------------------------------------------------ 清单

# 期望字节数用于校验。标注为 None 的不校验大小，只校验存在与非空。
SOURCES = [
    {
        "name": "flickr30k_images",
        "kind": "hf",
        "repo": "nlphuji/flickr30k",
        "file": "flickr30k-images.zip",
        "dest": "flickr30k/flickr30k-images.zip",
        "expect_bytes": 4605000000,  # 约 4.39 GB，允许 5% 误差
        "tol": 0.05,
        "license": "CC BY 2.0（Flickr 用户授权，仅研究用）",
        "use": "图文检索主基准，31K 图 × 5 caption；同时作 ARO Flickr30k-Order 的图像源",
    },
    {
        "name": "flickr30k_annotations",
        "kind": "hf",
        "repo": "nlphuji/flickr30k",
        "file": "flickr_annotations_30k.csv",
        "dest": "flickr30k/flickr_annotations_30k.csv",
        "expect_bytes": 13500000,
        "tol": 0.10,
        "license": "CC BY 2.0",
        "use": "Flickr30k 的 caption 标注（文件名, caption 编号, caption）",
    },
    {
        "name": "aro_vgr_annotations",
        "kind": "gdrive",
        "gid": "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3",
        "dest": "aro/vgr_annotations.json",
        "expect_bytes": None,
        "license": "沿用 Visual Genome 与 GQA 的许可",
        "use": "ARO Visual Genome Relation：交换两实体关系顺序构造的硬负 caption",
    },
    {
        "name": "aro_vga_annotations",
        "kind": "gdrive",
        "gid": "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw",
        "dest": "aro/vga_annotations.json",
        "expect_bytes": None,
        "license": "沿用 Visual Genome 与 GQA 的许可",
        "use": "ARO Visual Genome Attribution：交换对象与属性归因构造的硬负 caption",
    },
    {
        "name": "aro_vg_images",
        "kind": "gdrive",
        "gid": "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9",
        "dest": "aro/vgr_vga_images.zip",
        "expect_bytes": 1073741824,  # README 称 1GB
        "tol": 0.20,
        "license": "沿用 Visual Genome",
        "use": "VG-Relation 与 VG-Attribution 共用的图像子集",
    },
    {
        "name": "aro_flickr30k_order",
        "kind": "http",
        "url": "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json",
        "dest": "aro/flickr30k_order.json",
        "expect_bytes": None,
        "license": "Karpathy 划分，研究用",
        "use": "ARO Flickr30k-Order：打乱词序的硬负 caption，图像用 flickr30k_images",
    },
    {
        "name": "fsd50k_parquet",
        "kind": "hf_repo",
        "repo": "nateraw/fsd50k",
        "subdir": "data",
        "dest": "fsd50k/parquet",
        "expect_bytes": 25080000000,  # 约 23.34 GB
        "tol": 0.05,
        "license": "CC BY 4.0（FSD50K 官方标注）/ 音频来自 Freesound，各片段许可不一，仅研究用",
        "use": "音频侧检索基准，多标签声音事件，音频以二进制内嵌 parquet",
    },
]


# ------------------------------------------------------------------ 工具

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def sha256_head(path, blocks=8):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for _ in range(blocks):
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def curl_resume(url, out_path):
    """curl 断点续传单文件。已存在且完整则跳过。"""
    tmp = out_path + ".part"
    cmd = ["curl", "-sSL", "-C", "-", "--retry", "3", "--retry-delay", "2",
           "--connect-timeout", "20", "-o", tmp, url]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        # -C - 对已完成的文件会报 416，此时文件已完整
        if os.path.exists(tmp):
            log(f"  curl 返回 {r.returncode}，但 .part 已存在，按已完成处理")
        else:
            raise RuntimeError(f"curl 失败 code={r.returncode}")
    shutil.move(tmp, out_path)


def fetch_hf_file(src, dest_root):
    out = os.path.join(dest_root, src["dest"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and not src.get("expect_bytes"):
        log(f"  已存在，跳过 {src['dest']}")
        return out
    url = f"https://huggingface.co/datasets/{src['repo']}/resolve/main/{src['file']}"
    log(f"  下载 {url}")
    curl_resume(url, out)
    return out


def fetch_http(src, dest_root):
    out = os.path.join(dest_root, src["dest"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        log(f"  已存在，跳过 {src['dest']}")
        return out
    log(f"  下载 {src['url']}")
    curl_resume(src["url"], out)
    return out


def fetch_gdrive(src, dest_root):
    """Google Drive 下载。优先 curl 直下，失败再退回 gdown。

    直下端点 drive.usercontent.google.com 对标注这类小文件直接可用，不必装
    gdown（runtime venv 没有 pip，装不了）。大文件若返回 HTML 确认页则交给
    gdown，两者都没有就记 missing。
    """
    out = os.path.join(dest_root, src["dest"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        log(f"  已存在，跳过 {src['dest']}")
        return out
    url = (f"https://drive.usercontent.google.com/download?id={src['gid']}"
           f"&export=download&confirm=t")
    tmp = out + ".part"
    r = subprocess.run(["curl", "-sSL", "-C", "-", "--connect-timeout", "20",
                        "--max-time", "1800", "-o", tmp, url])
    ok = False
    if r.returncode == 0 and os.path.exists(tmp):
        with open(tmp, "rb") as f:
            head = f.read(512)
        # Google Drive 对大文件会返回 HTML 确认页，据此判断是否需要 gdown
        if b"<html" not in head.lower() and b"<!doctype" not in head.lower():
            ok = True
    if ok:
        shutil.move(tmp, out)
        return out
    if os.path.exists(tmp):
        os.remove(tmp)
    if shutil.which("gdown") is not None:
        log("  直下拿到确认页，退回 gdown")
        r2 = subprocess.run(["gdown", "--id", src["gid"], "-O", out, "--no-cookies"])
        if r2.returncode == 0 and os.path.exists(out):
            return out
    log(f"  两种方式都失败，跳过 {src['name']}")
    return None


def fetch_hf_repo(src, dest_root):
    """整目录下载（HF 数据集的子目录，多个 parquet 分片）。"""
    out_dir = os.path.join(dest_root, src["dest"])
    os.makedirs(out_dir, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{src['repo']}/tree/main/{src['subdir']}"
    import urllib.request
    with urllib.request.urlopen(api, timeout=30) as r:
        entries = json.load(r)
    files = [e["path"] for e in entries if e["type"] == "file"]
    log(f"  {src['repo']}/{src['subdir']} 共 {len(files)} 个文件")
    done = 0
    for path in files:
        name = os.path.basename(path)
        out = os.path.join(out_dir, name)
        if os.path.exists(out):
            done += 1
            continue
        url = f"https://huggingface.co/datasets/{src['repo']}/resolve/main/{path}"
        curl_resume(url, out)
        done += 1
        if done % 8 == 0:
            log(f"  进度 {done}/{len(files)}")
    return out_dir


FETCHERS = {"hf": fetch_hf_file, "http": fetch_http,
            "gdrive": fetch_gdrive, "hf_repo": fetch_hf_repo}


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


# ------------------------------------------------------------------ 主流程

def run(dest_root, only):
    os.makedirs(dest_root, exist_ok=True)
    targets = [s for s in SOURCES if not only or s["name"] in only]
    manifest = {"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dest_root": os.path.abspath(dest_root), "items": []}
    for src in targets:
        log(f"=== {src['name']} ({src['kind']}) ===")
        fetcher = FETCHERS[src["kind"]]
        try:
            path = fetcher(src, dest_root)
        except Exception as e:
            log(f"  失败：{e}")
            path = None
        rec = dict(src)
        rec.pop("kind", None)
        if path and os.path.exists(path):
            size = dir_size(path) if os.path.isdir(path) else os.path.getsize(path)
            rec["actual_bytes"] = size
            rec["status"] = "ok"
            exp = src.get("expect_bytes")
            if exp:
                tol = src.get("tol", 0.05)
                lo, hi = exp * (1 - tol), exp * (1 + tol)
                rec["size_in_range"] = lo <= size <= hi
                if not rec["size_in_range"]:
                    log(f"  警告：大小 {human(size)} 不在期望 [{human(lo)}, {human(hi)}]")
            rec["sha256_head"] = None if os.path.isdir(path) else sha256_head(path)
        else:
            rec["actual_bytes"] = 0
            rec["status"] = "missing"
        manifest["items"].append(rec)

    mpath = os.path.join(dest_root, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    ok = sum(1 for i in manifest["items"] if i["status"] == "ok")
    log(f"完成 {ok}/{len(manifest['items'])}，manifest 写入 {mpath}")


def verify(dest_root):
    mpath = os.path.join(dest_root, "manifest.json")
    if not os.path.exists(mpath):
        log("没有 manifest.json")
        return 1
    manifest = json.load(open(mpath, encoding="utf-8"))
    bad = 0
    for it in manifest["items"]:
        p = os.path.join(dest_root, it["dest"])
        if not os.path.exists(p):
            log(f"缺失 {it['dest']}")
            bad += 1
            continue
        size = dir_size(p) if os.path.isdir(p) else os.path.getsize(p)
        exp = it.get("expect_bytes")
        flag = ""
        if exp:
            tol = it.get("tol", 0.05)
            if not (exp * (1 - tol) <= size <= exp * (1 + tol)):
                flag = f"  大小异常，期望约 {human(exp)}"
                bad += 1
        log(f"OK {it['dest']}  {human(size)}{flag}")
    log("全部通过" if bad == 0 else f"{bad} 项有问题")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./crossmodal_data")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="+", default=[])
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify(a.dest))
    if not (a.all or a.only):
        ap.error("需要 --all 或 --only <name...>")
    run(a.dest, set(a.only))


if __name__ == "__main__":
    main()

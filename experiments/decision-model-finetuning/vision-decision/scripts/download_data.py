"""下载 ScienceQA 数据集。

ScienceQA 非 gated，许可信息请核对数据集卡片。图像内嵌在 parquet 里，
不需要额外拉 COCO 注释文件，这是选它而不选 VQAv2 的原因。

先落 test split，训练集与测试集的图片按 sha256 去重，见 prepare_data.py。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="derek-thomas/ScienceQA")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dest", default=str(ROOT / "data" / "all"))
    ap.add_argument("--mirror", action="store_true", help="Use a mirror unless HF_ENDPOINT is already set")
    args = ap.parse_args()

    import os

    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=False)
    print(f"[download] {args.repo} split={args.split} -> {dest}")
    p = snapshot_download(
        args.repo,
        repo_type="dataset",
        allow_patterns=[f"*{args.split}*"],
        local_dir=str(dest),
    )
    print(f"[done] {p}")
    for f in sorted(dest.rglob("*.parquet")):
        print(f"[file] {f} {f.stat().st_size / 2**20:.1f}MB")


if __name__ == "__main__":
    sys.exit(main())

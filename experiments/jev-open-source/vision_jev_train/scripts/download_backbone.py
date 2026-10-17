"""下载 Qwen3.5 多模态底座。

默认 2B。0.8B 更小但语言能力偏弱，做候选判策不够用；4B 权重 8.8 GB，
在 128 GB 统一内存的训练机上实测能把 sshd 打到无响应，需要先跑通内存
闸门再考虑。先用 2B 跑通链路。

国内网络加 --mirror 走 hf-mirror。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = {
    "2b": "Qwen/Qwen3.5-2B",
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="2b", choices=sorted(CANDIDATES))
    ap.add_argument("--dest", default=str(ROOT / "models"))
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    import os

    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    repo = CANDIDATES[args.size]
    dest = str(Path(args.dest) / repo.split("/")[-1])
    print(f"[download] {repo} -> {dest}")
    p = snapshot_download(repo, local_dir=dest, revision=args.revision)
    print(f"[done] {p}")

    total = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"[size] {total / 2**30:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())

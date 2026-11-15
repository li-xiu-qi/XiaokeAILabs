"""下载 Qwen3.5 多模态底座。

默认 2B。0.8B 可用于较小规模的流程检查；4B 及更大模型需要更高内存。
模型、输入长度与 batch 均会影响训练时的内存需求。

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
    ap.add_argument("--mirror", action="store_true", help="Use a mirror unless HF_ENDPOINT is already set")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    import os

    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    repo = CANDIDATES[args.size]
    dest = Path(args.dest) / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=False)
    print(f"[download] {repo} -> {dest}")
    p = snapshot_download(repo, local_dir=str(dest), revision=args.revision)
    print(f"[done] {p}")

    total = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"[size] {total / 2**30:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())

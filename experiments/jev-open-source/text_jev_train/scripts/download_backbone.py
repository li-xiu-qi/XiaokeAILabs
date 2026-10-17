"""下载 Qwen3-0.6B 底座到本地目录（约 1.5 GB）。

需要 huggingface_hub。国内网络加 --mirror，自动关闭 Xet 通道（镜像站不支持）。
默认落到 models/Qwen3-0.6B-Base，可被 --target-dir 覆盖。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "Qwen/Qwen3-0.6B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", action="store_true", help="走 hf-mirror 国内镜像")
    parser.add_argument(
        "--target-dir",
        default=str(ROOT / "models" / "Qwen3-0.6B-Base"),
        help="底座落盘目录",
    )
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少 huggingface_hub，先 pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(target),
        revision="main",
    )
    print(f"backbone ready: {path}")


if __name__ == "__main__":
    main()

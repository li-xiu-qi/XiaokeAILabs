"""下载 typed-decisions 公开数据集（两个 parquet，共约 0.8 MB）。

默认走 HuggingFace 官方源；国内网络加 --mirror 走 hf-mirror。
文件逐份校验字节数与 sha256，校验失败立即报错退出。
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "all"

REPO_ID = "LocalLLaMA/typed-decisions"
REVISION = "ea9306458d6e9563628369a3d1e72e362fb381d2"

FILES = {
    "train-00000-of-00001.parquet": {
        "bytes": 598824,
        "sha256": "46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5",
    },
    "test-00000-of-00001.parquet": {
        "bytes": 222140,
        "sha256": "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c",
    },
}


def build_url(name: str, mirror: bool) -> str:
    base = (
        "https://hf-mirror.com"
        if mirror
        else "https://huggingface.co"
    )
    return f"{base}/datasets/{REPO_ID}/resolve/{REVISION}/all/{name}"


def download(name: str, mirror: bool, dest: Path) -> Path:
    url = build_url(name, mirror)
    target = dest / name
    print(f"downloading {name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "agentjev-train/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read()
    meta = FILES[name]
    if len(content) != meta["bytes"]:
        print(f"  byte count mismatch: got {len(content)}, expect {meta['bytes']}", file=sys.stderr)
        raise ValueError(f"byte count mismatch for {name}")
    digest = hashlib.sha256(content).hexdigest()
    if digest != meta["sha256"]:
        print(f"  sha256 mismatch: got {digest}", file=sys.stderr)
        raise ValueError(f"sha256 mismatch for {name}")
    target.write_bytes(content)
    print(f"  ok, {len(content)} bytes, sha256 verified")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", action="store_true", help="走 hf-mirror 国内镜像")
    parser.add_argument("--out-dir", type=Path, default=DEST, help="数据输出目录（默认 data/all）")
    args = parser.parse_args()
    dest = args.out_dir.resolve()
    # Check the whole set before making a network request, including dangling symlinks.
    existing = [dest / name for name in FILES if os.path.lexists(dest / name)]
    if existing:
        parser.error("目标已存在，未下载或覆盖：" + ", ".join(map(str, existing)))

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=dest) as temp_dir:
        staged = Path(temp_dir)
        for name in FILES:
            download(name, args.mirror, staged)
        try:
            for name in FILES:
                # Same-filesystem hard link publishes a complete verified file without
                # replacing a target that appeared since the initial check.
                os.link(staged / name, dest / name)
        except BaseException as exc:
            # Never delete a target: another process may have replaced it.
            print("发布中断：已发布的完整文件会保留；检查目标目录或选择新目录重试。", file=sys.stderr)
            if isinstance(exc, FileExistsError):
                parser.exit(1, f"发布失败：目标在下载期间出现，未覆盖：{exc.filename2 or exc.filename}\n")
            if isinstance(exc, OSError):
                parser.exit(1, f"发布失败：无法创建硬链接，请检查输出文件系统是否支持硬链接及写入权限：{exc}\n")
            raise
    print(json.dumps({"dest": str(dest), "files": list(FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

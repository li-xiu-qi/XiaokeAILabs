#!/usr/bin/env python3
"""提交前拦截超大文件，防止把不该进版本库的资产推到 GitHub。

为什么需要这个：`.gitignore` 能忽略路径和扩展名，但**无法按文件大小匹配**。
检查脚本可由仓库自带钩子调用，也可直接运行。

本脚本是 pre-commit 的钩子入口，也可以手工跑：
    python scripts/check_large_files.py            # 查已暂存文件
    python scripts/check_large_files.py --all      # 查工作树全部受跟踪文件
    python scripts/check_large_files.py --limit 5  # 改阈值（MB）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_LIMIT_MB = 10.0

# 明确允许超过阈值的例外：报告证据图确需高分辨率的，逐条登记，注明理由。
# 宁可在这里多写一行，也不要把阈值调大——阈值一松，所有大文件都跟着进来。
ALLOWLIST: dict[str, str] = {}


def staged_files() -> list[str]:
    """已暂存的、非删除的文件。"""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
        capture_output=True, check=True,
    ).stdout
    return [os.fsdecode(f) for f in out.split(b"\0") if f]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    return [os.fsdecode(f) for f in out.split(b"\0") if f]


def sizes_of(paths: list[str]) -> dict[str, int]:
    """批量取文件在索引中的大小。

    一次 cat-file --batch-check 拿全部，不逐个起子进程——逐文件调用在
    1800 个受跟踪文件上要跑几十秒，钩子里不可接受。
    """
    if not paths:
        return {}
    # 先按 NUL 分隔读取索引路径，再按对象 ID 查询大小；路径可含中文或换行。
    index = subprocess.run(
        ["git", "ls-files", "--stage", "-z"], capture_output=True, check=True
    ).stdout
    objects = {}
    for entry in index.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        _, oid, stage = metadata.split()
        if stage == b"0":
            objects[os.fsdecode(path)] = oid
    indexed_paths = [path for path in paths if path in objects]
    lines = []
    if indexed_paths:
        payload = b"\n".join(objects[path] for path in indexed_paths) + b"\n"
        proc = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectsize)"],
            input=payload, capture_output=True, check=True,
        )
        lines = proc.stdout.splitlines()
    if len(lines) != len(indexed_paths):
        raise RuntimeError("无法取得暂存区文件的大小")
    result: dict[str, int] = {}
    for path, line in zip(indexed_paths, lines):
        try:
            result[path] = int(line.strip())
        except ValueError as exc:
            raise RuntimeError(f"无法取得暂存区文件的大小: {path}") from exc
    # 未入索引（新文件未 add 等）时退回读磁盘。
    for path in paths:
        if path in result:
            continue
        try:
            result[path] = Path(path).stat().st_size
        except OSError:
            result[path] = 0
    return result


def human(n: int) -> str:
    return f"{n / 1048576:.1f} MB" if n >= 1048576 else f"{n / 1024:.0f} KB"


def main() -> int:
    ap = argparse.ArgumentParser(description="拦截超大文件，防止推慢公开仓")
    ap.add_argument("--all", action="store_true", help="检查全部受跟踪文件，而非仅暂存区")
    ap.add_argument("--limit", type=float, default=DEFAULT_LIMIT_MB, help="阈值，单位 MB")
    ap.add_argument("files", nargs="*", help="pre-commit 传入的文件列表")
    args = ap.parse_args()

    limit = int(args.limit * 1048576)
    candidates = args.files or (tracked_files() if args.all else staged_files())
    candidates = [f for f in candidates if f and f not in ALLOWLIST]

    sizes = sizes_of(candidates)
    offenders = sorted(((f, sz) for f, sz in sizes.items() if sz > limit), key=lambda x: -x[1])

    if not offenders:
        return 0

    offenders.sort(key=lambda x: -x[1])
    print(f"\n检出 {len(offenders)} 个超过 {args.limit:g} MB 的文件：\n", file=sys.stderr)
    for f, sz in offenders:
        print(f"  {human(sz):>9s}  {f}", file=sys.stderr)
    print(
        "\n这些文件不该进版本库。按类型处理：\n"
        "  - 训练数据 / 数据集      → 不入库，README 写清上游来源与重建脚本\n"
        "  - 模型权重 / checkpoint  → 走 GitHub Release，仓里只留下载脚本与 SHA256\n"
        "  - 报告证据图（照片类）    → 转 JPEG，实测 PNG→JPEG(q92) 可省 75%，PSNR 40+dB\n"
        "  - 源码级证据图（曲线/截图）→ 保留 PNG，必要时裁掉多余留白\n"
        "确需例外时，把路径加进本文件的 ALLOWLIST 并写明理由。\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

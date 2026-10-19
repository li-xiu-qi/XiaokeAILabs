#!/usr/bin/env python3
"""提交前拦截超大文件，防止把不该进版本库的资产推到 GitHub。

为什么需要这个：`.gitignore` 能忽略路径和扩展名，但**无法按文件大小匹配**。
历史教训是具体的——本仓本地 .git 一度 608MB，其中
  - 411MB：浮世绘实验的原始训练图（单张最大 24MB，路径是对的，
    只是那个目录当时还没被 data/ 规则覆盖）
  - 33.6MB：20news 的 pickle / npz 数据集（*.npz 早被规则覆盖但入库更早；
    *.pkl.gz 从未被任何规则覆盖）
根因不是规则没写，而是**写了规则的钩子从没装过**：`.pre-commit-config.yaml`
里配着 check-added-large-files，但 `.git/hooks/pre-commit` 不存在，
所以那条规则从未执行过一次。

本脚本是 pre-commit 的钩子入口，也可以手工跑：
    python scripts/check_large_files.py            # 查已暂存文件
    python scripts/check_large_files.py --all      # 查工作树全部受跟踪文件
    python scripts/check_large_files.py --limit 5  # 改阈值（MB）
"""
from __future__ import annotations

import argparse
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
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.splitlines() if f.strip()]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [f for f in out.splitlines() if f.strip()]


def sizes_of(paths: list[str]) -> dict[str, int]:
    """批量取文件在索引中的大小。

    一次 cat-file --batch-check 拿全部，不逐个起子进程——逐文件调用在
    1800 个受跟踪文件上要跑几十秒，钩子里不可接受。
    """
    if not paths:
        return {}
    payload = "\n".join(f":{p}" for p in paths)
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectsize)"],
        input=payload, capture_output=True, text=True,
    )
    result: dict[str, int] = {}
    lines = proc.stdout.splitlines()
    for path, line in zip(paths, lines):
        try:
            result[path] = int(line.strip())
        except ValueError:
            # 未入索引（新文件未 add 等）时退回读磁盘
            try:
                result[path] = Path(path).stat().st_size
            except OSError:
                result[path] = 0
    # 行数对不上时（有文件取不到），剩下的逐个兜底
    for path in paths[len(lines):]:
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

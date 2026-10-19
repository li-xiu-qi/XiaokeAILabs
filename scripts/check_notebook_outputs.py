#!/usr/bin/env python3
"""拦住带执行输出的 notebook，避免执行结果反复涨大仓库。

为什么需要：notebook 的 outputs 里常含 base64 内嵌图片和完整数据表回显，
单个可到 1MB 以上（本仓最大一个 1.06MB，清掉输出后只剩 57KB）。
更麻烦的是输出每次重跑都会变，导致整个文件在 diff 里被当作整体重写，
历史里堆积大量几乎相同的快照——本仓清历史时发现 36 个旧 notebook 快照
占了 5.6MB，全是同一份代码的不同输出。

用法：
    python scripts/check_notebook_outputs.py            # 查已暂存
    python scripts/check_notebook_outputs.py --all      # 查全部受跟踪
    python scripts/check_notebook_outputs.py a.ipynb    # 查指定文件
清输出：
    jupyter nbconvert --clear-output --inplace <file>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def staged_names() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.splitlines() if f.endswith(".ipynb")]


def tracked_names() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f.endswith(".ipynb")]


def content_of(path: str) -> bytes | None:
    """优先读索引内容，退回磁盘——钩子里要检查的是即将提交的版本。"""
    proc = subprocess.run(["git", "show", f":{path}"], capture_output=True)
    if proc.returncode == 0:
        return proc.stdout
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def output_bytes(nb: dict) -> int:
    """只统计非空 outputs。序列化空列表会得到 2 字节，不能拿它判断有无输出。"""
    total = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs")
        if outputs:
            total += len(json.dumps(outputs, ensure_ascii=False).encode())
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="拦截带执行输出的 notebook")
    ap.add_argument("--all", action="store_true", help="查全部受跟踪文件")
    ap.add_argument("files", nargs="*", help="pre-commit 传入的文件列表")
    args = ap.parse_args()

    candidates = args.files or (tracked_names() if args.all else staged_names())
    offenders = []
    for path in candidates:
        if not path.endswith(".ipynb"):
            continue
        raw = content_of(path)
        if raw is None:
            continue
        try:
            nb = json.loads(raw)
        except json.JSONDecodeError:
            continue
        size = output_bytes(nb)
        if size > 0:
            offenders.append((path, size, len(raw)))

    if not offenders:
        return 0

    offenders.sort(key=lambda x: -x[1])
    print(f"\n检出 {len(offenders)} 个 notebook 带执行输出：\n", file=sys.stderr)
    for path, out_sz, total in offenders:
        # 输出很小时按字节显示，否则 0 KB 会让人以为检查器算错了
        shown = f"{out_sz} B" if out_sz < 1024 else f"{out_sz / 1024:.1f} KB"
        print(f"  输出 {shown:>9s} / 总大小 {total / 1024:>.0f} KB  {path}", file=sys.stderr)
    print(
        "\nnotebook 输出是执行结果不是源码，会随重跑变化并持续放大仓库历史。\n"
        "清掉再提交：\n"
        "  jupyter nbconvert --clear-output --inplace <file>\n"
        "  # 或 python scripts/clear_notebook_outputs.py\n"
        "需要在仓库里保留关键输出时，把结果另存为图片或表格放进 docs/，\n"
        "并在 README 说明复现命令。\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

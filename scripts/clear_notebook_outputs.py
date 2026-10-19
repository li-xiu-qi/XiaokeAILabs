#!/usr/bin/env python3
"""一键批量清空 notebook 执行输出（等价于 nbconvert --clear-output，不依赖 jupyter）。

用法：
    python scripts/clear_notebook_outputs.py            # 清全部受跟踪的 ipynb
    python scripts/clear_notebook_outputs.py a.ipynb    # 清指定文件
    python scripts/clear_notebook_outputs.py --dry-run  # 只报告，不改文件
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def tracked_notebooks() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f.endswith(".ipynb")]


def clear(path: str, dry_run: bool) -> int:
    """清空并返回省下的字节数；无输出可清时返回 0。"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return 0
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    if not changed:
        return 0

    if not dry_run:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return len(raw) - len(
        json.dumps(nb, ensure_ascii=False, indent=1).encode() + b"\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="清空 notebook 执行输出")
    ap.add_argument("files", nargs="*", help="指定文件；省略则处理全部受跟踪 notebook")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不修改")
    args = ap.parse_args()

    targets = args.files or tracked_notebooks()
    total = 0
    for path in targets:
        saved = clear(path, args.dry_run)
        if saved > 0:
            total += saved
            print(f"  {saved / 1024:>8.0f} KB  {path}")
    verb = "可省" if args.dry_run else "已省"
    print(f"\n{verb} {total / 1048576:.2f} MB")
    if args.dry_run and total:
        print("去掉 --dry-run 实际执行")
    return 0


if __name__ == "__main__":
    sys.exit(main())

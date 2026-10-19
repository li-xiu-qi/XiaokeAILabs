#!/bin/sh
# 安装提交前闸门。克隆仓库后跑一次即可：
#     bash scripts/install-hooks.sh
#
# 为什么不依赖 pre-commit：本仓实测 .pre-commit-config.yaml 从未生效——
# .git/hooks/pre-commit 一直不存在，check-added-large-files 一次都没跑过，
# .git 因此涨到 608MB（411MB 原始训练图 + 33.6MB 数据集）。
# 规则只要依赖「每台机器自己装 pre-commit 再 install」，实际就等于没有规则。
#
# 本脚本把钩子从版本库拷进 .git/hooks/，只用 git 与 python 标准库。

set -e
cd "$(git rev-parse --show-toplevel)"

if [ ! -f scripts/hooks/pre-commit ]; then
    echo "找不到 scripts/hooks/pre-commit，请确认在仓库根目录执行" >&2
    exit 1
fi

cp scripts/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo "已安装 .git/hooks/pre-commit"
echo "  拦截规则一：单文件 > 10MB（scripts/check_large_files.py）"
echo "  拦截规则二：notebook 带执行输出（scripts/check_notebook_outputs.py）"
echo ""
echo "自检：python scripts/check_large_files.py --all"

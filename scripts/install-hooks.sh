#!/bin/sh
# 安装提交前闸门。克隆仓库后跑一次即可：
#     bash scripts/install-hooks.sh
#
# 自带钩子负责文件大小和 notebook 输出检查；其他格式检查需单独运行。
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

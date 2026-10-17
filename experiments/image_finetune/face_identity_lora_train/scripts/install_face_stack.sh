#!/usr/bin/env bash
# 在 proart 上装人脸识别评估栈（走阿里云 pypi 镜像）
set -u
PY=/home/ke/jevomni-env/bin/python

echo "=== 试装 onnxruntime ==="
"$PY" -m pip install --quiet --disable-pip-version-check \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    onnxruntime 2>&1 | tail -4

echo "=== 试装 insightface ==="
"$PY" -m pip install --quiet --disable-pip-version-check \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    insightface 2>&1 | tail -6

echo "=== 验证 ==="
"$PY" -c "
import onnxruntime
print('onnxruntime', onnxruntime.__version__)
try:
    import insightface
    print('insightface', insightface.__version__)
except Exception as e:
    print('insightface 失败:', str(e)[:100])
"
echo "INSTALL_DONE"

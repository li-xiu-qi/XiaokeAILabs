#!/usr/bin/env bash
# 测 HF 通路 + 找 CelebA 里样本量足够的身份
set -u
echo "=== 连通性 ==="
curl -s -o /dev/null -w "hf_api %{http_code}\n" --max-time 25 "https://huggingface.co/api/datasets/flwrlabs/celeba"
curl -s -o /dev/null -w "hf_cdn %{http_code}\n" --max-time 25 "https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/README.md"
curl -s -o /dev/null -w "ds_server %{http_code}\n" --max-time 25 "https://datasets-server.huggingface.co/valid"

echo
echo "=== 身份分布统计 ==="
curl -s --max-time 60 "https://datasets-server.huggingface.co/statistics?dataset=flwrlabs%2Fceleba&config=img_align%2Bidentity%2Battr&split=train" | head -c 1500
echo
echo "STATS_DONE"

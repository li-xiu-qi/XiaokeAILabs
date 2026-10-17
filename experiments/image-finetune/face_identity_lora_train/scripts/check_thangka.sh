#!/usr/bin/env bash
echo "=== 当前时间 ==="
date
echo "=== 最新 4 个条目 ==="
ls -t /home/ke/thangka_fetch/ | head -4
echo "=== jpg 数量 ==="
ls /home/ke/thangka_fetch/*.jpg | wc -l
echo "=== meta.json ==="
ls -la /home/ke/thangka_fetch/meta.json
cat /home/ke/thangka_fetch/meta.json | head -c 200
echo
echo "=== 抓取进程 ==="
ps -eo pid,etime,pcpu,cmd | grep fetch_thangka | grep -v grep

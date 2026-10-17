#!/usr/bin/env bash
# 拉 v2 最佳检查点的生成图 + 汇总两轮扫描数据到本地
mkdir -p /mnt/c/Users/ke/facev2
cp /home/ke/face_ckpt_scan_v2/step_v2/s00.jpg /mnt/c/Users/ke/facev2/v2_best_a.jpg 2>/dev/null
cp /home/ke/face_ckpt_scan_v2/step_v2/s03.jpg /mnt/c/Users/ke/facev2/v2_best_b.jpg 2>/dev/null
cp /home/ke/face_ckpt_scan_v2/step_000000600/s00.jpg /mnt/c/Users/ke/facev2/v2_600_a.jpg 2>/dev/null
cp /home/ke/face_ckpt_scan/step_000001000/s00.jpg /mnt/c/Users/ke/facev2/v1_1000_a.jpg 2>/dev/null
cp /home/ke/face_ckpt_scan/step_000001000/s03.jpg /mnt/c/Users/ke/facev2/v1_1000_b.jpg 2>/dev/null
cp /home/ke/face_ckpt_scan_v2/scan.json /mnt/c/Users/ke/facev2/scan_v2.json
cp /home/ke/face_ckpt_scan/scan.json /mnt/c/Users/ke/facev2/scan_v1.json
cp /home/ke/face_baseline/baseline.json /mnt/c/Users/ke/facev2/baseline.json
ls -la /mnt/c/Users/ke/facev2/

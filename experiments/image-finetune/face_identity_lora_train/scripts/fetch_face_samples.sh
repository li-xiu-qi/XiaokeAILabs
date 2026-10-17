#!/usr/bin/env bash
S=/home/ke/XiaokeAILabs/experiments/image-finetune/face_identity_lora_train/output_face/face_identity_sdxl_lora/samples
mkdir -p /mnt/c/Users/ke/facesamp
cp "$S"/1792236638108__000000500_0.jpg /mnt/c/Users/ke/facesamp/s500_a.jpg
cp "$S"/1792236649520__000000500_1.jpg /mnt/c/Users/ke/facesamp/s500_b.jpg
cp "$S"/1792236663852__000000500_2.jpg /mnt/c/Users/ke/facesamp/s500_c.jpg
ls -la /mnt/c/Users/ke/facesamp/

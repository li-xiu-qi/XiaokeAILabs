#!/usr/bin/env bash
# 单人身份 LoRA 训练：备数据（含 caption）→ 上传配置 → 起训
set -eu
EX=/home/ke/XiaokeAILabs/experiments/image_finetune/kuniyoshi_style_lora_train
CFG=face_identity_sdxl_lora.yaml
FT=/home/ke/face_train

echo "=== 备训练数据 ==="
rm -rf "$FT"; mkdir -p "$FT"
cp /home/ke/celeb_person_14/train/*.jpg "$FT"/
# 每张图配同样的 caption：trigger + 类词
for f in "$FT"/*.jpg; do
  echo "vkqzm, a photo of a man" > "${f%.jpg}.txt"
done
echo "训练图 $(ls "$FT"/*.jpg | wc -l) 张，caption $(ls "$FT"/*.txt | wc -l) 个"
ls "$FT" | head -4

echo
echo "=== 起训 ==="
cd "$EX"
source /home/ke/jevomni-env/bin/activate
python /home/ke/ai-toolkit-main/run.py "config/$CFG" 2>&1 | tail -200

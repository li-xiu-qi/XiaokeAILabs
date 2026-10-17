# Wan2.2-TI2V-5B LoRA 微调学习实验

在 DGX Spark（GB10，aarch64，121GB 统一内存）上学习视频生成模型的 LoRA 微调链路。
完整结论与数据分析见 [`docs/Wan2.2-TI2V-5B-LoRA微调学习实验报告.md`](docs/Wan2.2-TI2V-5B-LoRA微调学习实验报告.md)。

## 目录结构

```
config/     训练配置（smoke / 第一轮 / 第二轮修正轮）
data/       训练数据集，不在本仓。见下方"数据位置"
scripts/    数据生成、数据集汇编、训练启动、评估脚本
docs/       实验报告与评估帧网格图（assets/）
output/     第一轮训练产物（checkpoint、采样、评估）
output_v2/  第二轮训练产物
```

## 数据位置

训练数据全量只在 DGX Spark 本地，不进版本库、不留 Windows 工作树
（`.gitignore` 规则 `experiments/video-finetune/*/data/`）。
三个数据集由底模自产合成，共 22 段 clip 加 caption，约 445 MB：

| 目录 | 内容 |
|---|---|
| `data/rzqduck_clips/` | 第一轮，12 段 |
| `data/rzqduck_clips_v2/` | 第二轮修正，6 段补充 |
| `data/rzqduck_clips_v3_full/` | 第三轮配平，11 水 + 11 陆 |

Spark 路径 `/home/ke/XiaokeAILabs/experiments/video-finetune/wan22_ti2v_lora_train/data/`。
换机复现时按下方复现步骤用 `gen_dataset*.py` 重新生成，无需拷贝。
latent 缓存（`_latent_cache/`）是 ai-toolkit 的训练副产物，按 dataset folder
自动重建，本实验曾达 417 MB，只在训练机留存。

## 复现步骤

环境：DGX Spark，`/home/ke/jevomni-env`，底模已落盘 `/home/ke/models/Wan2.2-TI2V-5B-Diffusers`。

```bash
cd ~/XiaokeAILabs/experiments/video-finetune/wan22_ti2v_lora_train
export HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com

# 1. 生成训练数据（12 段，每段约 114 秒）
python scripts/gen_dataset.py --out data/rzqduck_clips

# 2. 冒烟验证（4 步）
python /home/ke/ai-toolkit-main/run.py config/rzqduck_wan22_ti2v_smoke.yaml

# 3. 第一轮正式训练（300 步，约 34 分钟）
bash scripts/run_train.sh rzqduck_wan22_ti2v_lora

# 4. 评估（before/after 各 4 段，约 10 分钟）
python scripts/eval_wan_lora.py \
  --lora output/rzqduck_wan22_ti2v_lora_v1/rzqduck_wan22_ti2v_lora_v1.safetensors \
  --out output/eval

# 第二轮（caption 极简化 + 扩充陆地场景 + 400 步）
python scripts/gen_dataset_v2.py --out data/rzqduck_clips_v2   # 补 6 段非水景
python scripts/build_dataset_v2.py                             # 汇编 18 段并重写 caption
python /home/ke/ai-toolkit-main/run.py config/rzqduck_wan22_ti2v_lora_v2.yaml
python scripts/eval_wan_lora_v2style.py \
  --lora output_v2/rzqduck_wan22_ti2v_lora_v2/rzqduck_wan22_ti2v_lora_v2.safetensors \
  --out output_v2/eval_v2style

# 第三轮（数据集严格配平 11 水 + 11 陆 + 600 步压力测试）
python scripts/gen_dataset_v3.py --out data/rzqduck_clips_v3   # 补 4 段水景
python scripts/build_dataset_v3.py                             # 汇编 22 段并核对配比
python /home/ke/ai-toolkit-main/run.py config/rzqduck_wan22_ti2v_lora_v3.yaml
python scripts/eval_wan_lora_v2style.py \
  --lora output_v3/rzqduck_wan22_ti2v_lora_v3/rzqduck_wan22_ti2v_lora_v3.safetensors \
  --out output_v3/eval_v2style
python scripts/eval_wan_lora.py \
  --lora output_v3/rzqduck_wan22_ti2v_lora_v3/rzqduck_wan22_ti2v_lora_v3.safetensors \
  --out output_v3/eval_v1style
```

## 关键结论速览

- 视频 LoRA 全链路在 aarch64 上零障碍跑通，训练期内存仅 21-25GB，推理峰值约 104GB，全程无 OOM
- 触发词绑定呈现场景依赖性：水景最强（黄鸭稳定复现，600 步后对长短 prompt 均激活），纸船场景弱，厨房场景在三轮迭代（300/400/600 步 + 配平）后始终不绑定，判定为结构性阻断
- caption 极简化（去掉显式外观描述）显著强化生僻 token 的绑定
- 推理 prompt 风格要与训练 caption 格式对齐（400 步时陆地场景尤其明显，600 步后水景脱敏）
- 训练数据是底模自产合成数据，结论不外推到真实视频；评测为单 seed 单次，场景间存在采样波动

# Wan2.2-TI2V-5B LoRA 微调学习实验

在 DGX Spark（GB10，aarch64，121GB 统一内存）上学习视频生成模型的 LoRA 微调链路。
完整结论与数据分析见 [`docs/Wan2.2-TI2V-5B-LoRA微调学习实验报告.md`](docs/Wan2.2-TI2V-5B-LoRA微调学习实验报告.md)。

## 目录结构

```
config/     训练配置（smoke / 第一轮 / 第二轮修正轮）
data/       训练数据集（底模自产合成视频 + caption）
scripts/    数据生成、数据集汇编、训练启动、评估脚本
docs/       实验报告与评估帧网格图（assets/）
output/     第一轮训练产物（checkpoint、采样、评估）
output_v2/  第二轮训练产物
```

## 复现步骤

环境：DGX Spark，`/home/ke/jevomni-env`，底模已落盘 `/home/ke/models/Wan2.2-TI2V-5B-Diffusers`。

```bash
cd ~/XiaokeAILabs/experiments/wan22_ti2v_lora_train
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
```

## 关键结论速览

- 视频 LoRA 全链路在 aarch64 上零障碍跑通，训练期内存仅 21-24GB，推理峰值约 104GB，全程无 OOM
- 触发词绑定呈现场景依赖性：水景最强（双鸭 + 红领结红点），厨房场景始终不绑定
- caption 极简化（去掉显式外观描述）显著强化生僻 token 的绑定
- 推理 prompt 风格必须与训练 caption 格式对齐，否则概念不激活
- 训练数据是底模自产合成数据，结论不外推到真实视频

# 用教师生成图像训练轻量学生

一个大模型生成的图片，能否让小模型在特定任务上比自生成数据训练得更好？我们使用 128 条陶瓷杯产品图提示，将 SDXL 教师生成的数据与 SD 1.5 学生自生成的数据做对照。两组从同一学生基座开始，以相同训练预算微调，再用 32 个留出场景、每场景两个种子进行盲评。两轮 AI 评审的结果支持该场景下教师数据组优于自生成数据组；详见 [结果与限制](RESULTS.md)。

这属于用合成输出监督学生的实验，不匹配教师内部去噪预测。数据包含训练、开发与测试提示及种子：`train.jsonl`、`development.jsonl`、`test.jsonl`。每条给出主体颜色、背景、视角、光照、完整提示和种子；训练与测试的属性组合没有交集，不能排除模型预训练时见过陶瓷杯。

## 步骤

1. 先在开发集比较教师和原始学生。教师 1024 像素生成后以 Pillow LANCZOS 缩至 512；学生直接生成 512。两者均为 30 步、guidance 7.5。按照单杯、杯身颜色、背景、严重伪影四项盲评；有效配对至少 28、教师成功至少 20、优势至少 4 才继续。本地冻结记录没有第三方可信时间戳。
2. 对同一组 128 条训练提示分别生成教师图和学生自生成图。`prepare_mug_training_data.py` 验证 128 个 ID、提示、种子、图像尺寸，记录原图与训练图哈希；两组不按质量筛选。
3. 学生从同一 SD 1.5 基座以 LoRA 训练，batch 1、梯度累积 4、400 步、学习率 1e-4、seed 42、FP16。`run_experiment.sh` 依次生成训练图、准备数据、训练两组、为四组生成留出图。脚本拒绝覆盖已有目录，中断后不能直接当断点续跑。
4. 两轮独立 AI 盲评分别记录四项 0、1、unknown；前三项全为 1、严重伪影为 0 才算成功，任何 unknown 使复合结果为 unknown。第一轮由 8 个 AI 代理分批完成，第二轮由另一个 AI 代理完成。相同测试场景的两个种子不应当成两个独立场景。

## 本地运行

实际运行采用 NVIDIA GB10、driver 580.173.02、PyTorch 2.14.0+cu130、diffusers 0.40.0、transformers 5.17.0、accelerate 1.15.0、datasets 5.0.1、peft 0.20.0、Pillow 12.3.0。该版本记录不是完整依赖锁；更换软硬件不保证逐位重现。基座是 [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 快照 `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` 和 [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 快照 `462165984030d82259a11f4367a4eed129e94a7b`。分别遵守模型许可、使用限制和衍生物分发要求。

准备 Linux、可用 GPU、模型和依赖，并取得 diffusers 训练脚本 `examples/text_to_image/train_text_to_image_lora.py`；本次脚本 SHA-256 为 `d1958e329d01f5878a545de675b94e0b3dc86b6625d2db1eaffd08eee5684da5`，不能仅凭目录名证明源码 commit。修改为本机路径后执行：

```bash
export PYTHON=/path/to/python
export WORK=/path/to/new-run
export STUDENT_MODEL=/path/to/sd15-snapshot
export TEACHER_MODEL=/path/to/sdxl-snapshot
export TRAIN_SCRIPT=/path/to/diffusers/examples/text_to_image/train_text_to_image_lora.py
bash run_experiment.sh
```

公开版脚本已做语法检查；路径参数化后的版本尚未重新完成 GPU 全流程，实际完成的运行使用了相同核心命令的设备脚本。系统可用内存低于 50 GiB 时守护进程停止任务，该阈值不是 GPU 显存阈值。基础模型大权重未逐个计算哈希。

## 复算公开评分

本包包含评分、完成评审后公开的匿名映射及图像 SHA-256，但不含图像，所以读者不能只凭本包核验图像内容及哈希。以下用 Python 3.10 以上重算分数与 32 场景结果：

```bash
python recompute.py --manifest test.jsonl --manifest-sha256 2e45677b4f920765f128e2a85d94f92bfdb7aa580946ca4e56d846b430753e87 --review1 review-1.csv --key1 key-1.json --review2 review-2.csv --key2 key-2.json --out replay.json
```

`mug-final-summary.json` 是该命令生成的结果。评审代理记录使用匿名编号；原始图像、训练数据和 LoRA 权重未随本包发布。

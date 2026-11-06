# 用教师视频微调小模型：水波纹场景探索

让较小的视频生成模型专注一个视觉场景，能否改变它的生成结果？本实验让 Wan2.2 TI2V 5B 生成水波纹视频，再用其中六条视频对 Wan2.1 T2V 1.3B 做多帧 LoRA 微调。我们保留训练前后相同提示词与随机种子的四组留出视频，观察结果如何变化。

**结果：**200 步训练和四组前后对照均已完成。LoRA 被加载，画面确实改变，但四组样本没有提供足够证据证明整体质量提高：其中一组训练前后均缺少清晰波纹。六条训练视频是事后筛选所得，这是一项探索，不能当作预先注册方案验证成功，也不能外推至其他视频场景。

## 为什么要做场景蒸馏

视频生成模型通常需要同时处理内容、风格和跨帧变化。如果实际应用只需要一种有限场景，可以尝试让大模型生成训练样本，再训练较小的模型学习该场景的视觉分布。这样做可能降低部署负担，但教师样本中的错误也会进入学生训练，因此必须检查数据与留出输出。

这里的“蒸馏”指**基于教师生成样本的监督微调**：学生看到的是教师输出的视频和对应提示词，而不是教师网络的 logits、内部特征或去噪轨迹。它不能证明学生复现了教师的内部能力，更不能仅凭参数量推断推理速度。

## 输入、输出与训练目标

每条训练记录是一对 `(提示词, 视频)`：提示词描述固定镜头下水面波纹，视频是教师生成的 49 帧片段。训练时，视频由学生自身的 VAE 编码为视频 latent，文本编码器处理提示词；LoRA 更新视频生成网络的少量权重。推理时输入提示词和随机种子，输出连续的视频帧。

LoRA 把预训练权重的更新限制为低秩矩阵：

$$W' = W + \frac{\alpha}{r}BA,$$

其中 $r$ 是秩，$\alpha$ 是缩放系数；本次均为 8。视频扩散训练对带噪 latent 的预测误差进行优化，具体目标和噪声参数化取决于训练框架实现；不能把训练 loss 当作画质或运动准确率。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 教师 | Wan2.2 TI2V 5B，ComfyUI 适配权重；512 × 512、49 帧、24 fps、30 步 |
| 学生 | Wan2.1 T2V 1.3B，LoRA 秩 8；480 × 480、49 帧、200 步 |
| 学生对照 | 训练前后均为 480 × 480、49 帧、30 步、CFG 5；每对使用相同提示词和 seed |
| 数据 | 预定教师训练样本 8 条，检查后 2 条不符合要求；本次仅事后选取其余 6 条进行探索 |
| 留出 | 4 个未参与训练的提示词与固定 seed；全部保留，包括生成不理想的结果 |

教师与学生使用不同模型及分辨率，因此这里不把两者生成速度或单张画面质量当作公平的性能排名。学生训练实际看到的 latent 形状为 `[1, 16, 13, 60, 60]`，时间维有 13 个 latent 帧，说明本次训练包含多帧信息。

## 如何复现思路

1. 准备教师、学生模型与可运行的视频训练环境。记录各组件版本、权重来源和哈希，并确认模型许可。教师参考 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 与 [ComfyUI 模型文件](https://huggingface.co/Comfy-Org/Wan-2.2)；学生参考 [Wan2.1](https://github.com/Wan-Video/Wan2.1)。
2. 在生成前固定训练与留出提示词及种子。为每条教师视频保存原始输出、提示词、种子、工作流和运行日志；分别检查帧数、画面可辨性与损坏情况。不要因为测试结果不好而更换留出样本。
3. 每个训练视频旁放置同名 `.txt` 提示词，以训练工具读取视频与文本。设置 49 帧、24 fps、480 像素分辨率、LoRA 秩 8、batch size 1、学习率 $10^{-4}$ 和 200 步，并保存中间检查点。训练前先验证时间维大于 1，避免误把单帧训练当成视频训练。
4. 训练前对全部留出提示生成基线；训练后加载 LoRA，以完全相同的提示词、种子和学生采样参数再生成一次。检查 adapter 已注入，再逐组观察视频内容、运动连续性和画面伪影。

不同训练框架的数据加载、调度器和 LoRA 权重格式可能不同，上述参数是本次实验条件，不能直接保证在另一套环境中得到相同结果。

### 使用实验脚本

本目录提供学生侧数据准备、配置生成、推理和逐帧对比脚本。**原始教师训练视频及本次 LoRA 权重未随仓库完整公开**，因此读者可以直接复核已发布的视频结果，也可以用自己生成的教师视频按下列流程重做；无法只凭本仓库逐字节重建本次训练。示例中的 `ai-toolkit` 表示自行安装的 [AI Toolkit](https://github.com/ostris/ai-toolkit)。

```bash
# 在本实验目录执行；先自行准备教师 MP4，文件名以 train-00 等 ID 开头。
python scripts/prepare_data.py --manifest config/prompts.json --source /path/to/teacher-clips \
  --output ./work/train --ids train-00 train-02 train-03 train-04 train-06 train-07

# 生成本机路径对应的训练配置，再调用已安装的 AI Toolkit。
python scripts/make_config.py --template config/train.yaml.template --data ./work/train \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers --output ./work/checkpoints \
  --config ./work/train.yaml
python /path/to/ai-toolkit/run.py ./work/train.yaml

# 相同提示词、seed 和推理配置分别生成训练前后的留出视频。
python scripts/generate.py --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --manifest config/prompts.json --output ./work/baseline
python scripts/generate.py --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --manifest config/prompts.json --output ./work/lora --lora /path/to/adapter.safetensors

# 复核仓库中已经提供的 4 组结果；输出哈希和逐帧像素差。
python scripts/compare.py --baseline results/baseline --lora results/lora
```

脚本需 Python、NumPy、PyAV；生成还需 PyTorch、Diffusers、Transformers 和 PEFT，以及可运行的 CUDA 环境。实测环境中的主要版本为 PyTorch `2.14.0+cu130`、Diffusers `0.39.0.dev0`、Transformers `5.5.3`、PEFT `0.21.2`、PyAV `19.0.1`；使用的是开发版 Diffusers，且当时 AI Toolkit 的源码快照未记录 commit，因此环境不能由版本号完整锁定。脚本中的参数化整理与结果复核已分别检查，重新运行完整 200 步训练尚未做端到端验证。实际运行前请根据设备资源设置内存监控和中断策略。

### 视频与许可

四组均提供训练前和训练后完整输出，可直接审阅：

| 留出组 | 学生基线 | LoRA 后 |
| --- | --- | --- |
| 00 | [视频](results/baseline/eval-00.mp4) | [视频](results/lora/eval-00.mp4) |
| 01 | [视频](results/baseline/eval-01.mp4) | [视频](results/lora/eval-01.mp4) |
| 02 | [视频](results/baseline/eval-02.mp4) | [视频](results/lora/eval-02.mp4) |
| 03 | [视频](results/baseline/eval-03.mp4) | [视频](results/lora/eval-03.mp4) |

视频由本次实验的文本提示生成，不含实拍人物。教师和学生模型卡分别标注 [Wan2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 与 [Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) 为 Apache-2.0；实际运行的教师文件来自 [Comfy-Org 的适配发布](https://huggingface.co/Comfy-Org/Wan-2.2)。使用和再分发模型时仍应遵守相应许可及署名要求。仓库中的视频是实验输出，不包含模型权重。

[历史逐帧统计与视频 SHA-256](results/frame-comparison.json) 保留了当时的结果。重新运行 `compare.py` 时，8 个视频哈希均匹配，但另一环境算得的四组平均逐帧像素差约为 12.74、2.44、8.86、11.50，与历史的 12.72、2.35、8.92、11.43 存在差异；原因尚未定位，数值不应视为严格复现。像素值范围为 0–255，像素差只是输出差异，不是质量评分。

## 四组留出观察

以下是对训练前后每组视频全部 49 帧的非盲人工观察，不构成主观评分或统计检验：

| 留出组 | 训练前后观察 |
| --- | --- |
| 00 | 两者均出现扩散的圆环；LoRA 后圆环更细，后段逐渐淡出。 |
| 01 | 两者都没有清晰可辨的圆形波纹。 |
| 02 | 两者均有同心纹；LoRA 后纹路更小、更淡。 |
| 03 | 两者都有池面纹理；运动方向和物理可信度难以判断。 |

四对视频逐帧像素有差异，但像素差不能证明运动更自然。49 帧在 24 fps 下约为两秒，也不能评价长时间一致性。更可靠的后续评估需要更多独立场景、盲评、重复种子和明确的运动质量指标。

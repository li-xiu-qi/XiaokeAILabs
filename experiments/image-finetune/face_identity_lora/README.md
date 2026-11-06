# 用 LoRA 微调 SDXL 的人物身份

[English](README.en.md) · [图片微调入口](../README.md)

给通用图像生成模型一小组人物照片和一个触发词，能否让它生成带有该人物特征的画面？本实验以一个身份为例，介绍数据准备、LoRA 训练、生成和人脸特征评估。该方法可用于经授权的人物内容制作，是否可用仍取决于身份相似度、生成质量和测试场景。

## 结果与限制

历史记录采用 CelebA 的一个身份，20 张训练照片、10 张留出参考照片。ArcFace 将人脸编码为向量，以生成图和目标参考向量的余弦相似度衡量接近程度。均值只包含检测到人脸的生成图。

| 条件 | 分辨率 | 检出数 | 目标相似度均值 |
|---|---:|---:|---:|
| 未微调基线 | 768×768 | 12/20 | −0.0172 |
| LoRA 500 步 | 512×512 | 6/10 | 0.3110 |
| LoRA 1000 步 | 512×512 | 7/10 | 0.3232 |
| LoRA 2000 步 | 512×512 | 9/10 | 0.2461 |

来源：[基线记录](docs/evidence/baseline.json)及[检查点记录](docs/evidence/checkpoint_scan.json)。这些汇总由旧脚本结果转录，原始生成图片、模型 revision、依赖版本和完整训练日志没有全部保存。表中两个分辨率不同，**不能把基线与 LoRA 分数相减视为微调带来的提升**。扫描结果显示不同检查点的观测均值不同，但每步有效样本只有 6 至 9 张、仅一个身份，不能确定最佳步数，也不能确定下降是否由过拟合造成。

1000 步记录中，同一批可检测生成图对目标和一个其他身份的平均相似度分别为 0.3232 和 0.0546，说明此批次更接近目标参考；这不能证明人物身份可靠识别或证明对其他身份同样有效。20 张训练照片对目标测试参考的相似度均值为 0.7159，但这不是生成结果的理论上限，不应据此计算「学会百分之多少」。原图为 178×218，低分辨率值得关注，但现有实验未控制原图分辨率，因此不能断言它是主要瓶颈。

训练图和测试参考按照片划分，来源仍是同一身份：同一拍摄环境或近似照片可能让结果偏乐观。该实验没有多身份、跨场景或人工身份评价；人脸未检出的生成图也不应被忽略。

## 原理与输入输出

SDXL 从噪声出发，在文本条件引导下逐步去噪生成图像。LoRA 不直接更新全部权重，而用低秩增量近似部分权重的变化：

$$W_{\mathrm{adapted}} = W_0 + \frac{\alpha}{r}BA$$

其中 $W_0$ 是冻结的原权重，$A$ 和 $B$ 是训练的低秩矩阵，$r$ 是秩；实际实现的缩放还取决于训练框架。训练输入为图像及 caption，训练过程对加噪图像及文本条件计算去噪目标，具体预测目标与损失应以所用工具和版本为准。本次配置训练 UNet，不训练文本编码器。

推理时输入文字提示词、随机种子及 LoRA 权重，输出生成图片。触发词是 caption 的一部分，不是内置的身份数据库；模型是否真的学到身份，需要在未见照片、不同提示词和同条件底模对照中验证。

## 数据与配置

- 数据：CelebA 单个身份，20 张训练、10 张测试参考；数据按照片而非身份划分。[数据集协议](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)限定非商业研究并禁止进一步发布或分发数据集，使用时也应核对人物授权。
- 底模：SDXL base；本次未保留模型 revision。LoRA rank 32、alpha 32、dropout 0.05。
- 训练：512×512、batch 1、学习率 1e-4、2000 步，每 500 步保存一次；详细配置见 [YAML](config/face_identity_lora.yaml)。
- 评估：RetinaFace 检测后用 ArcFace 编码；目标、负对照与同人参考的原始聚合数据见上面的 JSON。InsightFace 权重及底模均有各自许可。

## 操作路径

以下命令应从本目录运行；需要提前安装相应 Python 依赖、获得数据和模型权重，并按训练环境检查 YAML 中的数据路径。逐张原始训练图与模型权重不随仓库发布，现有记录无法确保精确复跑。评估脚本拒绝复用已有输出目录；如需再跑，设置新的 `FACE_OUT_DIR`。

```bash
python scripts/build_dataset.py --parquet /path/to/celeba_shard0.parquet --also-negative
python scripts/check_dataset.py
AI_TOOLKIT=/path/to/ai-toolkit/run.py bash scripts/train_lora.sh

# 全局参数放在子命令之前；同分辨率下重新生成基线和检查点
python scripts/score_generations.py --size 512 baseline
python scripts/score_generations.py --size 512 lora --lora /path/to/adapter.safetensors
python scripts/score_generations.py --size 512 scan --run-dir /path/to/checkpoints
```

评估脚本现在显式配置 DPM-Solver++；历史结果在脚本重构前生成，本次未用修正后的脚本重新生成旧数据。要评估微调效果，应重新运行同尺寸、采样器、随机种子和提示词模板匹配的底模与 LoRA 对照，并报告所有生成图片及检测失败数。

主要脚本：`build_dataset.py` 导出数据与 caption，`check_dataset.py` 检查样本，`train_lora.sh` 调用训练器，`score_generations.py` 生成并评估。训练器、模型和评估权重的使用条件应分别查看上游许可。

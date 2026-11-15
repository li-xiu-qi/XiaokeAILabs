# 图像微调：用少量照片学习特定人物

[English](README.en.md)

从少量照片出发，给已有的图像生成模型加上对特定人物的生成能力，并用可量化的指标检验是否真的学到了。讲解与结论见[图像微调与量化评估](../../docs/zh/multimodal/人脸身份LoRA微调与量化评估.md)。

| 目录 | 内容 |
|---|---|
| [face-identity-lora](face-identity-lora/README.md) | 用 CelebA 单人 20 张照片微调 SDXL 的 LoRA，以 ArcFace 相似度和正、负对照评估身份特异性 |

[assets](assets/) 里是 SDXL LoRA 流程示意图。

## 环境准备

本目录没有共用的 Python 依赖清单；图像训练、推理和人脸评估使用的依赖不同。当前 [人脸身份 LoRA 子实验](face-identity-lora/README.md#环境准备)提供独立环境、PyTorch CUDA 安装、其余依赖、模型与数据准备命令。请按子实验 README 操作，不要在本目录自行拼装环境。

## 须知

- 训练数据不随仓发布：CelebA 的数据协议限制非商业研究使用并禁止分发图片，来源与协议见 [face-identity-lora](face-identity-lora/README.md)。训练数据只放在执行训练的机器上。
- 本目录只保留收敛后的脚本、配置、报告与证据图。已废弃的子实验不再保留在仓库中。

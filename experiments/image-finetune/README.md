# 图像微调：用少量照片学习特定人物

[English](README.en.md)

从少量照片出发，给已有的图像生成模型加上对特定人物的生成能力，并用可量化的指标检验是否真的学到了。讲解与结论见[图像微调与量化评估](../../docs/zh/multimodal/人脸身份LoRA微调与量化评估.md)。

| 目录 | 内容 |
|---|---|
| [face-identity-lora](face-identity-lora/README.md) | 用 CelebA 单人 20 张照片微调 SDXL 的 LoRA，以 ArcFace 相似度和正、负对照评估身份特异性 |

[assets](assets/) 里是 SDXL LoRA 流程示意图。

运行前请进入具体子实验，按其环境准备说明安装依赖并准备模型和数据。

## 须知

- 训练数据不随仓发布：CelebA 的数据协议限制非商业研究使用并禁止分发图片，来源与协议见 [face-identity-lora](face-identity-lora/README.md)。训练数据只放在执行训练的机器上。
- 本目录只保留收敛后的脚本、配置、报告与证据图。已废弃的子实验不再保留在仓库中。

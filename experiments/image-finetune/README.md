# 图片微调学习线

> 本目录是图片生成模型 LoRA 微调的学习线索引：在做什么实验、结论是什么、数据在哪。
> 每个实验一个子目录，自包含（依赖、数据说明、文档、脚本自成一套）。

在 DGX Spark（GB10，121 GB 统一内存）与 proart-temp（RTX Spark N1X，WSL2 arm64）上，
用 ai-toolkit 对图片生成模型做 LoRA 微调。

## 在做的实验

| 实验 | 任务 | 结论 | 报告 |
|---|---|---|---|
| [face_identity_lora_train](face_identity_lora_train/) | 单人身份 LoRA：让底模画出一个它画不出来的人 | 微调有效：ArcFace 相似度 −0.017 → 0.3232，负对照 0.0546，净差 0.27；瓶颈是数据分辨率不是训练参数 | [README](face_identity_lora_train/) |

![微调前后对比](face_identity_lora_train/docs/assets_celeb/face_compare.jpg)

## 已终止并归档的实验

两个风格实验（通用浮世绘、歌川国芳个人风格）已终止，代码与数据随目录删除，
完整报告不随本仓发布。它们的结论是当前实验的前提：

- **ukiyoe_style_lora_train**（六轮）：数据分辨率封顶风格保真度，作用强于数据量
  （107 张 1024² 全面超过 150 张 256²）；风格强度 20 epoch 见顶
- **kuniyoshi_style_lora_train**（两轮 + 多 seed 复评）：训练量在 200 到 800 步之间
  不是敏感变量；微调净贡献不显著（+0.0092，p=0.238）；数据质量才是根因

三条结论共同指向同一个判据：**底模已经能生成的，微调就没有可测量的赢面。**
当前实验就是按这条判据重新选的靶子。

## 数据位置

训练数据只在训练机本地，不进版本库（`.gitignore` 规则 `experiments/image-finetune/*/data/`）。
人脸实验的数据是 Hugging Face 的 CelebA，从 `flwrlabs/celeba` 的 parquet 按
`celeb_id` 切出，复现脚本见该实验的 `scripts/pick_celeb.sh` 与 `export_celeb_person.sh`。

## 相关目录

- 视频微调：`../video-finetune/`
- 检索与重排：`../retrieval-and-rerank/`

# 图片微调学习线

在 DGX Spark（GB10，121 GB 统一内存）与 proart-temp（RTX Spark N1X，WSL2 arm64）上，
用 ai-toolkit 对图片生成模型做 LoRA 微调。本目录按「一个实验一个子目录」组织。

## 在做的实验

| 实验 | 任务 | 结论 | 报告 |
|---|---|---|---|
| [face_identity_lora_train](face_identity_lora_train/) | 单人身份 LoRA，证明微调对底模盲区有效 | 微调有效：ArcFace 相似度 −0.017 → 0.3232；瓶颈是数据分辨率不是训练参数 | 见该目录 README |

## 已终止并归档的实验

两个风格实验（通用浮世绘、歌川国芳个人风格）已终止，代码与数据随目录删除，
报告归档在 `pkm-hub-lab/image-style-lora/_归档/`。它们的结论是当前实验的前提：

- **ukiyoe_style_lora_train**（六轮）：数据分辨率封顶风格保真度，作用强于数据量
  （107 张 1024² 全面超过 150 张 256²）；风格强度 20 epoch 见顶
- **kuniyoshi_style_lora_train**（两轮 + 多 seed 复评）：训练量在 200 到 800 步之间
  不是敏感变量；微调净贡献不显著（+0.0092，p=0.238）；数据质量才是根因

三条结论共同指向同一个判据：**底模已经能生成的，微调就没有可测量的赢面。**
当前实验就是按这条判据重新选的靶子。

## 方法论与规范

结论、判据、数据审查规范的唯一权威版在私有库：

```
pkm-hub-lab/image-style-lora/
  README.md                    实验定位、当前进度、网络实况
  notes/数据审查规范.md         上游不信任原则、四项机器检查、抽样级别、判定门槛
  notes/人脸身份实验设计.md     三对照设计、指标定义、事先写死的判定门槛
  notes/底模能力探针.md         候选靶子用 3 seed 探针判「底模会不会」
  notes/数据选择复盘.md         两个风格实验失败的根因
  _归档/                       两个旧实验的完整报告
```

## 数据位置

训练数据只在训练机本地，不进版本库（`.gitignore` 规则 `experiments/image_finetune/*/data/`）。
人脸实验的数据是 Hugging Face 的 CelebA，从 `flwrlabs/celeba` 的 parquet 按
`celeb_id` 切出，复现脚本见该实验的 `scripts/pick_celeb.sh` 与 `export_celeb_person.sh`。

## 相关目录

- 视频微调：`../video_finetune/`
- 检索与重排：`../retrieval_and_rerank/`
- 偏好训练（决策模型，有 benchmark）：`../preference_training/`

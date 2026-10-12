# 图片微调学习线

在 DGX Spark（GB10，121 GB 统一内存）上，用 ai-toolkit 对图片生成模型做 LoRA 微调。
本目录按「一个实验一个子目录」组织，与 `../test_jev_open_source/` 的多实验结构一致。

## 在库内的实验

| 实验 | 任务 | 结论 | 报告 |
|---|---|---|---|
| [ukiyoe_style_lora_train](ukiyoe_style_lora_train/) | 通用浮世绘风格 LoRA，六轮迭代 | 风格成功迁移，六轮配方优化有据；C 列未超过提示词工程天花板 | [docs/浮世绘风格LoRA微调学习实验报告.md](ukiyoe_style_lora_train/docs/浮世绘风格LoRA微调学习实验报告.md) |
| [kuniyoshi_style_lora_train](kuniyoshi_style_lora_train/) | 单一艺术家（歌川国芳）风格 LoRA，五列评估 | 主目标未达成，微调未超过点名艺术家的提示词；但 epoch 扫描推翻了 30 epoch 的选择 | [docs/国芳风格LoRA微调学习实验报告.md](kuniyoshi_style_lora_train/docs/国芳风格LoRA微调学习实验报告.md) |

## 已移出本仓的实验

橡皮鸭主体 LoRA 两轮已移出本仓，迁入私有归档
`pkm-hub-lab/_归档/XiaokeAILabs-废弃实验/test_rubber_duck_subject_lora/`。
迁出前确认 `origin/main` 不含 `image_finetune`，内容从未推送，无需改写历史。

## 这条线的验收标准

微调实验的立项判据、三层验收线、基线阶梯、人眼主判的完整标准，
写在 pkm-hub 主库：

```
pkm-hub/Projects/001_AI与工具/AI技术研究/02_模型与训练/
  微调实验的验收标准-问题导向、人眼主判与基线阶梯.md
```

国芳实验是第一个按该标准预注册验收判据再开跑的实验。ukiyoe 实验早于该标准，
它的报告结论需按新标准重读（缺 D 列点名艺术家这一档，真实差距比当时报的更大）。

## 数据位置

训练数据全量只在 DGX Spark 本地，不进版本库、不留 Windows 工作树
（`.gitignore` 规则 `experiments/image_finetune/*/data/`）。
上游是 Wikimedia Commons 的公有领域扫描件，各实验 README 给了可重新抓取的分类名。

## 相关目录

- 视频微调：`../video_finetune/`
- 检索与重排：`../retrieval_and_rerank/`
- 偏好训练（决策模型，有 benchmark）：`../preference_training/`

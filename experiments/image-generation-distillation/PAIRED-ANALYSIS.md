# 图像蒸馏评审的类别与配对变化

学生微调前后对象与数量符合提示词的评分从 24/32 增至 27/32，净增加的 3 条全部来自相机：该类从 3/8 增至 6/8，其中 4 条改善、1 条下降。运动鞋为 5/8 对 5/8，香水瓶和太阳镜均为 8/8 对 8/8。这个结果提醒我们，整体分数的变化并非四类商品均有提高。

这些数据是对已有单次 AI 缩略图评分的事后分组，而不是新增评审。每个类别仅 8 张，使用同一模板和固定种子；无法据此推断其他商品、真实业务、人类偏好或证明蒸馏造成改善。每个对象共有不同背景和光线的提示，评分也包含数量要求；各提示并非随机抽取的独立用户需求。

| 指标 | 改善 | 下降 | 不变 | 探索性双侧精确符号检验 |
|---|---:|---:|---:|---:|
| 对象与数量 | 6 | 3 | 23 | 0.508 |
| 背景 | 3 | 1 | 28 | 0.625 |
| 主体完整 | 7 | 2 | 23 | 0.180 |
| 明显伪影减少 | 6 | 1 | 25 | 0.125 |

检验只以变化方向计数，基于各配对变化在零假设下可交换的假设；样本共享模板、评审者相同，假设不一定成立。这是对四个已观察指标的探索性计算，未预先注册、未校正多重比较，不是对训练有效性的确认。未发现统计证据也不等于证明没有效果。

## 复算

在实验目录下使用标准库运行：

```bash
python scripts/analyze_paired_review.py --prompts evidence/prompts/heldout.jsonl \
  --key evidence/review-key.json --scores evidence/review-scores.csv \
  --out paired-review-analysis.json
```

脚本核验全部提示和三个模型角色各自对应唯一评分，拒绝缺失值，并拒绝覆盖输出。完整分组与计算结果见 `evidence/paired-review-analysis.json`；原始图片和 LoRA 权重不随仓库提供。

## English

The count of outputs meeting the object-and-quantity criterion rose from 24/32 to 27/32. All three net gains came from cameras: 3/8 before versus 6/8 after, with four improvements and one deterioration. Sneakers remained 5/8, and perfume bottles and sunglasses remained 8/8 each. This is a post-hoc breakdown of one AI review of thumbnails, not a new evaluation. Eight controlled prompts per object and a shared template cannot establish broader improvement or causality.

The table above reports paired improvements, deteriorations, unchanged outcomes, and exploratory two-sided exact sign-test values. These assume exchangeable paired changes; shared templates and one reviewer weaken that assumption. Four outcomes were examined post hoc without multiple-testing correction, so the figures are not confirmatory. Use the command above to reconstruct the breakdown from the released prompts, identity mapping, and scores; the original full-resolution images and LoRA weights are not distributed here.

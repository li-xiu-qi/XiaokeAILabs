# 08 全链路评估报告

> 职责声明：本报告来自 37KB 玩具合成语料上的历史运行（2.07M 参数、128 token 上下文），测的是全链路流程与教学现象，没测真实数据上的语言能力；后续只有讲解链路流程时可引用，任何能力结论都不要引用本文件。


同一套评估跑在四个 checkpoint 上：聊天问答命中率（SFT 留出集）与算术 exact-match（训练集算式进过预训练语料，留出集算式完全未见）。

| 阶段 | step | 参数量 | 聊天问答命中率 | 算术 exact-match（训练/留出） |
|---|---|---|---|---|
| pretrain | 250 | 2,065,536 | 0% | 0% / 0% |
| midtrain | 120 | 2,065,536 | 0% | 0% / 0% |
| sft | 250 | 2,065,536 | 25% | 21% / 4% |
| rl | 80 | 2,065,536 | 25% | 26% / 2% |

## 预训练：Muon vs AdamW（同 steps 同种子）

| 优化器 | best val loss（步数） | final val loss | train loss 末段 |
|---|---|---|---|
| Muon+AdamW | 1.3424（step 49） | 2.4978 | 0.0924 |
| AdamW | 1.4086（step 149） | 1.6475 | 0.8083 |

Muon 的 best val 更低且出现更早（优化更快）；继续训练后两者 final val 都上扬，Muon 更早过拟合，9K token 的微型语料上这是预期行为。


## RL（GRPO）效果

| 指标 | SFT（RL 前） | RL 后 |
|---|---|---|
| 算术 exact-match（训练集） | 21% | 26% |
| 算术 exact-match（留出集） | 4% | 2% |
| 聊天问答命中率（回归） | 40% | 40% |


各阶段 loss 曲线：pretrain_loss.png、midtrain_loss.png、sft_loss.png、rl_reward.png（与本报告同目录）；生成样例见 generation-samples-toy-synthetic.md。

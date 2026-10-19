# 图片微调：SDXL LoRA

对 SDXL 做 LoRA 微调，每个子目录一个实验，自包含配置、数据说明、脚本与证据。

| 实验 | 靶子 | 结论 |
|---|---|---|
| [face_identity_lora](face_identity_lora/) | 一张具体的脸——底模画不出来的那个人 | 20 张照片、2000 步即可学会该身份：ArcFace 相似度从 −0.017 到 0.3232，负对照 0.0546，净差 0.27；天花板受源图分辨率限制而非训练参数 |

![微调前后对比](face_identity_lora/docs/evidence/face_compare.jpg)

左起：真实照片（训练时未见过）、底模输出、微调后输出、训练过量的反例。四列同提示词、
同种子、同采样器。

## 靶子选错，微调就没有可测量的赢面

图片模型微调最容易白做的地方，是拿它去画底模本来就会画的东西——那样无论怎么调参，
测出来的提升都在噪声里。所以靶子的选择本身就是结论的一部分：**只有当目标超出底模
现有能力时，微调才有可测量的赢面。**

人脸身份是这样一个靶子。提示词描述不出一张具体的脸，而 20 张照片可以。这让「学到
没有」变成一个能算出数的问题，而不是凭肉眼说像不像。

具体测量方式、分辨率的限制、样本量带来的不确定性，见该实验的 README。

## 数据不进版本库，由脚本从 CelebA 确定性导出

人脸实验用 Hugging Face 的 CelebA（`flwrlabs/celeba`），从 parquet 按 `celeb_id`
切出单个身份，交替划分训练 / 测试，全部由脚本完成：

```bash
python face_identity_lora/scripts/build_dataset.py --parquet <shard>.parquet --also-negative
```

同一 shard、同一身份，重跑得到同样的 20 张训练 / 10 张测试。

## 环境

Python 3.10+ 与一张 CUDA GPU。训练用
[ai-toolkit](https://github.com/ostris/ai-toolkit)；评估用 insightface 的 ArcFace
与 RetinaFace，权重从 InsightFace `buffalo_l` 发布包获取，本仓不转存。

## 相关目录

- 视频微调：`../video-finetune/`
- 检索与重排：`../retrieval-and-rerank/`

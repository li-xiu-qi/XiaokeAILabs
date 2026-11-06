# 未参与教师选择的合成页面复测

本次针对上级实验的三个既有模型，在此前没有用于训练、调参或教师选择的新页面上比较检索结果。数据由 [evaluate.py](evaluate.py) 用种子 20261108 生成，共 64 页、每页 3 个查询，每次从相同 64 页中寻找已知配对页。脚本核验新旧数据的实体、字段值和图片 SHA-256 均不重叠。

| 模型 | Recall@1 | Recall@5 |
|---|---:|---:|
| 原始 ColSmol | 149/192 = 0.7760 | 186/192 = 0.9688 |
| 监督微调 | 154/192 = 0.8021 | 187/192 = 0.9740 |
| 监督微调 + 蒸馏 | 151/192 = 0.7865 | 186/192 = 0.9688 |

相对监督组，蒸馏组有 2 条查询进入首位，5 条退出首位，净少 3 条。这个结果没有显示蒸馏带来提升；单次训练、单一教师和单次数据生成都不足以证明普遍结论。每张页面的三个查询相互关联，不能把 192 条视为完全独立样本。

新页面与原实验共享页面版式、字段、查询模板，但改用此前没有出现的六位数实体与数值。它可以检验原结论在新合成样本上的表现，不能证明对真实文档、不同版式或语义匹配的泛化。教师和模型的选择仍使用过前一组测试集，这一新集合没有参与此前选择。

本次使用 NVIDIA DGX Spark，和主实验相同的 ColSmol 底模、初始 adapter 及两份训练后的 adapter，推理使用 BF16、ColPali Engine 0.3.13。PyTorch 峰值分配 3,021,570,048 字节；独立监控观测到的最低系统可用内存为 90,478,160 KiB。数字为单次运行，不能视为通用设备要求。

[results.json](results.json) 保存匿名逐条排名；[recompute.py](recompute.py) 可从排名计算表中指标和配对变化。原始合成图片、旧数据及训练后 adapter 均未随本目录分发；因此读者可以重新计算指标，但目前不能仅凭仓库复现模型推理。可用已保存的 adapter、本实验脚本和原始 synthetic/records.json 重建相同流程；图片像素依赖字体版本，排名也可能受软件和设备影响。

## Independent synthetic holdout

The three existing models were evaluated on 64 newly generated pages and 192 queries, none used for previous training, tuning or teacher selection. `evaluate.py` uses seed 20261108 and checks that entity identifiers, field values and page SHA-256 hashes do not overlap the earlier synthetic data. All models rank the same 64 candidates for each query; the matching page is the single known positive.

| Model | Recall@1 | Recall@5 |
|---|---:|---:|
| Original ColSmol | 149/192 = 0.7760 | 186/192 = 0.9688 |
| Supervised fine-tuning | 154/192 = 0.8021 | 187/192 = 0.9740 |
| Supervised + distillation | 151/192 = 0.7865 | 186/192 = 0.9688 |

Relative to the supervised model, distillation moved two queries into first place and five out. This run shows no improvement from distillation. One training seed, one teacher and one generated dataset cannot establish a general result; each page contributes three correlated queries.

The new pages share the earlier layout, fields and query templates, but use new six-digit entities and values. They are a fresh synthetic holdout, not evidence of generalization to real documents, other layouts or semantic matching. Earlier test data did influence teacher and model selection; this holdout did not.

The run used DGX Spark, BF16 and ColPali Engine 0.3.13 with the same base model and adapters as the parent experiment. Peak PyTorch allocation was 3,021,570,048 bytes and the lowest observed system available memory was 90,478,160 KiB. These measurements from one run are not general hardware requirements.

[results.json](results.json) contains anonymized per-query ranks, and `python recompute.py` checks the table and paired changes. Synthetic images, the original dataset and trained adapters are not distributed here. The ranks can be recomputed, but the model inference cannot be reproduced from this repository alone. Rebuilding the full run requires the saved adapters and the earlier `synthetic/records.json`; font versions can change image pixels, and software or device differences can change rankings.

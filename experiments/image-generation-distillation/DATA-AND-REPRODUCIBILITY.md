# 数据与复现清单

这份清单区分已保存的运行证据和重新执行所需的材料。提示词可以原样恢复，但训练图片、评估图片全集和 LoRA 权重未随仓库发布；即便固定种子，不同软硬件生成的图片也未必逐字节一致。公开文件足以复算已记录的评分汇总，不能独立检查所有图片或精确重跑原训练。

## 数据如何构成

本实验没有下载外部照片；教师根据自编英文提示词合成商品图。每行包括 ID、提示词、商品类别、背景、光线和随机种子。下列两份清单由 `scripts/make_prompts.py` 生成，逐条与原始教师生成记录的对应字段比对一致：

| 数据 | 条数 | 用途 | 清单 |
|---|---:|---|---|
| 训练提示 | 64 | 8 个训练对象 × 4 种背景 × 2 种光线；教师每条生成 1 张图供学生训练 | [train.jsonl](evidence/prompts/train.jsonl) |
| 留出提示 | 32 | 4 个未出现在训练集的对象 × 相同背景与光线；教师和两个学生版本分别生成 | [heldout.jsonl](evidence/prompts/heldout.jsonl) |

训练对象为 ceramic mug、glass bottle、leather wallet、wristwatch、desk lamp、headphones、teapot、backpack；留出对象为 sneaker、perfume bottle、camera、sunglasses。两组对象无交集，但共享提示模板与场景要素。每个 split 的种子都从 10000 开始；相同整数种子跨模型不意味着相同噪声或可逐像素对比。

重新生成清单：

```bash
python scripts/make_prompts.py --out data/prompts
```

教师生成的单图记录、提示、采样参数与时间见 [训练元数据](evidence/teacher-train-metadata.jsonl) 和 [留出元数据](evidence/teacher-heldout-metadata.jsonl)；生成脚本另生成 `metadata.jsonl`，其中的 `file_name` 和 `text` 用作训练图片与文本对应。教师图也可能错误，不应把生成标签当作真值。

## 运行参数与证据

| 项目 | 本次记录 |
|---|---|
| 教师 | SDXL base 1.0，revision `462165984030d82259a11f4367a4eed129e94a7b`，EulerDiscreteScheduler |
| 学生 | SD 1.5，revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`，PNDMScheduler |
| 生成 | 512 × 512、30 步、guidance 7.5，FP16；教师与学生各自使用记录的 scheduler |
| 训练 | 64 张图片、单卡 batch 1、累积 4、400 个优化步、学习率 1e-4、FP16、梯度检查点、seed 42 |
| 训练产物 | LoRA SHA-256 `aae5734d3ecaab8570bdc87f13189f4ed273c60a0b46110e634f7473b7c09fb7`；仓库不含权重 |
| 运行软件 | PyTorch 2.14.0+cu130、Diffusers 0.39.0.dev0；完整锁定依赖与训练脚本源码提交未保存于公开证据 |

具体 scheduler 配置和推理版本见[教师 manifest](evidence/teacher-run-manifest.json)、[微调前 manifest](evidence/student-before-manifest.json)、[微调后 manifest](evidence/student-after-manifest.json)；[训练输出](evidence/train-output.txt)记录样本数和优化步数。README 中训练命令描述此次配置，但仅凭现有公开记录无法完整核验训练脚本提交、所有默认参数和完整依赖，因此不应当作逐位重现承诺。

## 评估与共享边界

每条留出提示对应教师、学生微调前与微调后三张图，形成 96 个匿名评分项。一名独立 AI 评审看缩略图，判断对象及数量、背景、主体完整、明显伪影四项。原始[评分](evidence/review-scores.csv)、[揭盲映射](evidence/review-key.json)和[汇总](evidence/review-summary.json)公开；原始 96 张图未公开，故读者无法只凭本仓核实单图评分。单评审、32 条提示、每条一个种子且共享模板，不能证明泛化或人类偏好改善。

自编提示词和脚本未包含第三方图片。教师与学生各有模型许可和使用限制，分别参见 [SDXL 模型卡](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 与 [SD 1.5 模型卡](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)；使用和重新分发权重、适配器或生成内容之前应依据实际版本的许可证判断。仓库提供的少量示例是该实验的生成结果，不代表对模型许可作额外授权。

# 相机商品图后续盲评

前一次 32 条留出评审里，学生在对象与数量评分上净增 3 条，全部来自 8 条相机提示。这里用另外 8 条相机提示各配两个种子，检查这种观察是否在一组新的表述与随机数上重复。相机类别是看过前次结果后选定的，因此这是事后驱动的探索性复测，不是预先计划的独立确认试验。

同一名独立 AI 评审在不知道三组身份的情况下，通过接触表给全部 48 图评分；每图缩略到最长边不超过 360 像素。首要指标为画面恰好有一台可辨认相机；另观察指定相机颜色、背景颜色与明显伪影。`1` 表示符合或有伪影，`0` 表示不符合或无伪影，`?` 表示无法判断。本次 192 个评分均有 0 或 1。

| 评分 | 教师 | 学生原模型 | 学生 + LoRA |
|---|---:|---:|---:|
| 一台可辨认相机 | 12/16 | 16/16 | 16/16 |
| 相机颜色符合 | 11/16 | 10/16 | 9/16 |
| 背景颜色符合 | 11/16 | 8/16 | 9/16 |
| 有明显伪影，越少越好 | 8/16 | 6/16 | 2/16 |

学生原模型在首要指标已达到 16/16，配对全部不变，未重复观察到该指标的提升。相机颜色有 2 条改善、3 条下降；背景 1 条改善；明显伪影 4 条减少，没有新增。颜色和伪影判断属于单次 AI 主观评分，不等同人工质量测评；缩略图也可能漏掉局部瑕疵。16 张对应 8 条提示、各两个固定种子，样本不是 16 条独立用户需求。教师在这组评分中的表现不是其总体能力排名。

DGX Spark 上使用同一 PyTorch 2.14.0+cu130、Diffusers 0.40.0、512×512、30 步和 guidance 7.5，原学生、LoRA 学生与教师分别逐张生成 16 张。学生采用 PNDM，教师采用 Euler，跨架构的相同 seed 也不表示相同噪声。教师和学生的模型 revision 分别为 `462165984030d82259a11f4367a4eed129e94a7b` 和 `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`；LoRA 的 SHA-256 为 `aae5734d3ecaab8570bdc87f13189f4ed273c60a0b46110e634f7473b7c09fb7`。本次使用与原实验不同的 Diffusers 版本，因此不能视为原输出的复现。教师模型的 Euler 和学生 PNDM 也使教师结果不能视为受控速度或质量排序。

首次使用旧环境时因缺少依赖未进入模型生成；第二次原模型生成完成，但 LoRA 环境缺少 PEFT，未进入 LoRA 生成。最终在独立安装 PEFT 的环境中从头完成三组。三组的持续内存监测最低分别为 87.047、87.024 和 81.429 GiB；指标是整机可用内存，不是单模型显存占用。48 张图片在设备和本地按 SHA-256 逐一一致。

[新提示](prompts.jsonl) · [匿名评分](scores.csv) · [揭盲映射](identity-key.json) · [汇总](summary.json) · [评测计划](protocol.md)

从本目录复算，输出文件不得预先存在：

```bash
python summarize.py --review scores.csv --key identity-key.json --out summary-recomputed.json
```

映射公开后可重算统计，不再用于盲评。完整图像和 LoRA 权重未随仓库分发，因此读者可核对评分汇总，不能仅凭仓库独立复核逐图判断或完整重跑原图；模型权重与生成脚本参照上级实验。

## English

The previous 32-prompt review showed a net gain of three on object-and-count compliance, all in the camera category. This follow-up uses eight new camera prompts and two fixed seeds each. Camera was chosen after examining the earlier scores, so this is an exploratory follow-up rather than an independently preregistered confirmation.

One independent AI reviewer scored all 48 randomly anonymized images on contact sheets with a maximum image side of 360 pixels. The primary outcome was exactly one recognizable camera. The table above shows the teacher, original student and LoRA student respectively: 12/16, 16/16 and 16/16 on that outcome. All 16 student pairs were unchanged. Color improved in two and deteriorated in three; background improved in one; obvious artifacts disappeared in four and appeared in none. These subjective thumbnail scores are not a human quality assessment. Two outputs share each prompt, so the 16 images are not 16 independent user requirements.

All three arms used the same GPU environment, PyTorch 2.14.0+cu130, Diffusers 0.40.0, 512 pixels, 30 steps and guidance 7.5. The teacher used Euler and the student PNDM; equal seeds across architectures do not imply equal noise. This software version differs from the earlier run. The LoRA hash and model revisions are listed above. The first two incomplete attempts are described above; the final comparison was regenerated from scratch in one environment with PEFT available. The 48 device and local image hashes matched.

Prompts, blind scores, identity mapping, the protocol and a standard-library summarizer are linked above. The complete images and LoRA are not distributed, so the repository supports score recomputation but not independent review of every image or exact end-to-end reproduction.

## 匿名接触表 / Anonymous contact sheets

[1](contact-sheets/sheet-01.jpg) · [2](contact-sheets/sheet-02.jpg) · [3](contact-sheets/sheet-03.jpg) · [4](contact-sheets/sheet-04.jpg) · [5](contact-sheets/sheet-05.jpg) · [6](contact-sheets/sheet-06.jpg) · [7](contact-sheets/sheet-07.jpg) · [8](contact-sheets/sheet-08.jpg)

这些图是评审所用匿名接触表的 JPEG 压缩副本，带有匿名编号与提示词，不是原始生成图。压缩可能影响细节判断。

These are JPEG-compressed copies of the anonymized contact sheets used in review, with identifiers and prompts. They are not original generated images; compression may alter fine details.

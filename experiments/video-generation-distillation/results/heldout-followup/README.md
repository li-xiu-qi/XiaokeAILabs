# 水波纹学生模型的新留出对照

此前的四组留出视频不足以判断结果是否稳定。这里使用四条新的水波纹描述，每条固定两个此前未用过的种子，比较原学生模型与已有的 200 步 LoRA，合计八对视频。所有提示和种子在生成前写入 `manifest.json`；每对使用相同提示、种子、学生模型和采样设置。

两组都是 480×480、49 帧、24 fps、30 步、CFG 5。学生为 Wan2.1 T2V 1.3B；LoRA 来自教师生成的六条水波视频的事后筛选训练。16 段视频的哈希与生成日志一致，均可解码为 49 帧。运行时可用内存最低约 74 GiB。训练权重未在本目录发布。

一名 AI 评审在不查看模型身份的条件下，以每 8 帧和每秒 2 帧抽帧制作分镜并比较。揭盲后，8 对中 LoRA 获偏好 3 对、原学生获偏好 2 对、相当 1 对、无法判断 2 对。评审没有播放完整视频，所有“连续向外扩散”判断均记为不确定，因此这些结果无法评价连续运动、闪烁或证明质量提升。四条提示各使用两个种子，样本和模板相关，不能当作八个独立场景或推广到真实应用。

`baseline/` 与 `lora/` 保留全部视频；`manifest.json` 是预先固定的输入；`review-protocol.md`、`blind-review-results.md` 保留揭盲前的评审要求与逐条记录；`paired-results.json` 是揭盲映射、哈希和配对汇总。

## English

Four new prompt phrasings with two previously unused seeds each form eight held-out pairs. For each pair, the base student and the existing 200-step LoRA used the same prompt, seed, model and sampling settings: 480×480, 49 frames, 24 fps, 30 steps and CFG 5. The LoRA was trained on a post-hoc selection of six teacher-generated water videos. All 16 video hashes match the generation logs and every clip decodes to 49 frames; minimum available system memory was approximately 74 GiB. Trained weights are not distributed here.

One AI reviewer compared anonymized contact sheets sampled every eight frames and at two frames per second, without inspecting model identity. After unblinding, the reviewer preferred the LoRA in three pairs and the base in two, marked one tie and two inconclusive. The reviewer did not watch complete clips and marked all continuous outward-motion judgments uncertain. This small, template-related sample cannot establish better video quality, temporal continuity, or generalization.

The two video directories contain every output, `manifest.json` records the pre-fixed inputs, the review documents preserve the blinded protocol and individual judgments, and `paired-results.json` records the unblinded mapping and hashes.

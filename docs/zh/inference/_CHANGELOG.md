# 配图升级记录

## 2026-11-12

本次调整覆盖 `docs/zh/inference/` 下 6 篇文档，目标是让每张图只承载一个主题，并优先使用论文原图。

- `分词机制与生成成本.md`：加入 Sawada 与 Goyal 的 BPE merge-list 原论文 Figure 1，保留字节编解码流与四类算法对比的本地重绘。
- `分词器训练与词表规模权衡.md`：保留词表规模权衡和 CPU 流式训练两张本地重绘。现有论文素材没有同时覆盖这两个主题。
- `分词器效果评测与跨词表BPB计算.md`：保留多语域压缩率、词表利用率和 BPB 分母审计的本地重绘。没有强配不匹配的论文图。
- `大模型预训练原理与扩展律.md`：加入 Muon Figure 3 与 Figure 4、MiniCPM WSD Figure 5、Kaplan Figure 1、Chinchilla Figure 3；把原先混合图拆成本地 Muon 收敛、GB10 数据扩展和梯度累积三张图。
- `预训练阶段评测与能力基准.md`：用 Schaeffer Figure 2 替换旧复合图，并把 GB10 连续指标与离散命中率保留为独立本地重绘。
- `模型量化等级与精度相变.md`：把旧复合图拆为仿射阶梯和位宽密度两张本地重绘，加入 MXFP4/NVFP4 Figure 1、LLM.int8 Figure 2 与 Figure 3。

论文图均保留原始 PNG，图下注明论文来源、Figure 编号和许可状态。Muon 与 Schaeffer 图为 CC BY-NC-ND 4.0，MiniCPM、BPE、LLM.int8 图为 CC BY 4.0，Kaplan、Chinchilla、MXFP4/NVFP4 图按 arXiv 非独占分发许可记录。CC BY-NC-ND 和 arXiv 非独占许可不等同于第三方图复用授权，发布前仍需人工复核。

被替换的旧混合图已通过 safe-delete 移入回收站：`pretrain-muon-convergence-and-scaling`、`pretrain-continuous-vs-discrete-metrics`、`quantization-staircase-and-outliers` 的 PNG 与 SVG。仍被文档引用的自绘图保留。

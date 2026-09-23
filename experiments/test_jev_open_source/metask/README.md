# Jev 系列实验：metask-jev (4B) 篇

本目录对应 `wayfind/metask-jev-4b-policy-mix` 模型的实机评测与复现工程。

该模型基于 Qwen3.5-4B 微调，技术核心是把候选答案约束为单个 Token（A 到 Z），直接读取输出层对应 Token 的 Logits 并归一化为概率。它在中文分类测试集上 20 条全对，是四组方案里唯一的满分，常驻显存 8.58 GB。

## 文件组成与作用

- `metask-install.sh`：自动化部署脚本。负责创建 Python 独立运行环境、拉取模型权重（约 8.5 GB），并执行 Token 编码完整性校验（确保 A 到 Z 均为单 Token，防止破坏解码映射契约）。
- `metask-bench.py`：推理性能与业务准确率测试脚本。评测涵盖接口协议契约、单次延迟基准、中英文工单平行测试、温度系数校准前后差异以及显存占用监控。
- `metask-temperature.json`：上游提供的各题型置信度校准温度系数。
- `metask-jev_scorer.py` 与 `metask-jev_schema.py`：上游核心评分器与 Schema 校验逻辑的工程留档副本。

## 部署与复现步骤

按顺序执行初始化与测试脚本：

```bash
bash metask/metask-install.sh
python metask/metask-bench.py
```

**运行依赖说明**：`metask-bench.py` 运行时会直接调用安装脚本初始化在 `~/metask-jev` 的评分器模块。因此必须完整执行 `metask-install.sh` 初始化环境与权重后，方可启动测试。

## 实测指标与数据说明

在本机测试环境下的实测数据：

- **端到端延迟**：单次分类判定的中位数 133.5 ms（英文）与 133.7 ms（中文）。
- **业务准确率**：英文工单 20 条命中 19 条；中文工单 20 条全对。
- **显存占用**：常驻 8.58 GB。

**关于上游基准数据**：
上游 README 列出的 ECE 校准误差（0.114 降至 0.040）与 JevBench 排名我们没有复现，排名也是上游自测插值，未经过官方评测。选型时以本仓实测数字为准，完整取舍说明见《实验报告.md》。

# Jev 系列实验：metask-jev (4B) 篇

[English](README.en.md)

本目录对应 `wayfind/metask-jev-4b-policy-mix` 模型的实机评测与复现工程。

该模型基于 Qwen3.5-4B 微调，技术核心是把候选答案约束为单个 Token（A 到 Z），直接读取输出层对应 Token 的 Logits 并归一化为概率。它在中文分类测试集上 20 条全对，是四组方案里唯一的满分，常驻显存 8.58 GB。

## 文件组成与作用

- `metask-install.sh`：自动化部署脚本。负责创建 Python 独立运行环境、拉取模型权重（约 8.5 GB），并执行 Token 编码完整性校验（确保 A 到 Z 均为单 Token，防止破坏解码映射契约）。
- `metask-bench.py`：推理性能与业务准确率测试脚本。评测涵盖接口协议契约、单次延迟基准、中英文工单平行测试、温度系数校准前后差异以及显存占用监控。
- `metask-temperature.json`：上游提供的各题型置信度校准温度系数。
- `metask-jev_scorer.py` 与 `metask-jev_schema.py`：上游核心评分器与 Schema 校验逻辑的工程留档副本。

## 不下载模型的输入检查

评分器使用的 schema 校验代码只依赖 Python 标准库，可在 CPU 上先检查输入结构。以下命令从本目录运行；它只验证本地输入校验，不加载模型，也不能替代下方的 GPU 基准：

```bash
python - <<'PY'
import importlib

schema_tools = importlib.import_module("metask-jev_schema")
schema = {
    "priority": {"type": "enum", "description": "ticket priority", "choices": ["low", "high"]},
    "urgent": {"type": "boolean", "description": "is urgent"},
}
schema_tools.validate_schema(schema)
assert schema_tools.choices_for(schema["urgent"]) == [False, True]
try:
    schema_tools.validate_schema({"priority": {"type": "enum", "description": "priority", "choices": []}})
except ValueError:
    print("valid schema accepted, empty choices rejected")
else:
    raise AssertionError("empty choices must fail")
PY
```

2026-11-13 在 Linux ARM64、Python 3.12.3 的新建虚拟环境中运行以上检查，进程退出码为 0，有效 schema 通过且空候选被拒绝。这个检查未安装模型依赖，也未运行模型或性能基准。

## 部署与复现步骤

脚本需要 Bash、Python 3、curl、网络和足够的模型存储空间；历史性能测试使用 CUDA。安装脚本会在 `~/metask-jev` 创建或复用虚拟环境、安装和升级包，从上游 main 下载评分器，并下载约 8.5 GB 模型后执行一次推理。除 PyTorch 外的直接依赖列在 `requirements.txt`，大部分未锁版本；上游代码和模型也未锁定，不能保证重建相同环境。Linux 下默认先尝试 CUDA 12.1 的 PyTorch 下载源，需核对是否适配自己的设备。已有同名目录时应先检查其中内容。此安装路径尚未在全新环境端到端验证。

从本目录按顺序执行初始化与测试脚本：

```bash
bash metask-install.sh
"$HOME/metask-jev/.venv/bin/python" metask-bench.py
```

**运行依赖说明**：`metask-bench.py` 运行时会直接调用安装脚本初始化在 `~/metask-jev` 的评分器模块。安装脚本在子进程中启用虚拟环境，退出后不会改变当前 shell，因此测试命令显式使用该虚拟环境的解释器。必须完整执行 `metask-install.sh` 初始化环境与权重后，方可启动测试。

## 实测指标与数据说明

在本机测试环境下的实测数据：

- **端到端延迟**：单次分类判定的中位数 133.5 ms（英文）与 133.7 ms（中文）。
- **业务准确率**：英文工单 20 条命中 19 条；中文工单 20 条全对。
- **显存占用**：常驻 8.58 GB。

**关于上游基准数据**：
上游 README 列出的 ECE 校准误差（0.114 降至 0.040）与 JevBench 排名我们没有复现，排名也是上游自测插值，未经过官方评测。选型时以本仓实测数字为准，完整取舍说明见[基准报告](../../../docs/zh/decision-models/判别式决策模型基准实测.md)。

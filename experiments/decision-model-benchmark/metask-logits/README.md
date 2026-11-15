# Jev 系列实验：metask-jev (4B) 篇

[English](README.en.md)

本目录对应 `wayfind/metask-jev-4b-policy-mix` 模型的实机评测与复现工程。

该模型基于 Qwen3.5-4B 微调，技术核心是把候选答案约束为单个 Token（A 到 Z），直接读取输出层对应 Token 的 Logits 并归一化为概率。2026-11-15 在 NVIDIA GB10 上的新环境复跑中，中文测试集 20 条全对，CUDA 显存峰值为 8.65 GB。该结果只反映这组短测试题。

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

脚本需要 Linux、Bash、Python 3.12、网络、CUDA 版 PyTorch 和约 8.5 GB 模型存储空间；最低内存尚未验证。2026-11-15 已在 NVIDIA GB10 / Linux aarch64、Python 3.12.3 的新建虚拟环境中完成安装与 GPU 基准。安装器默认在 `~/metask-jev` 创建或复用虚拟环境，也可用 `METASK_DIR` 指向独立位置。评分器从仓内固定副本安装，不再下载上游 `main`。本次模型固定为 `wayfind/metask-jev-4b-policy-mix@ea20fe85b28733b1522721dec119bec50947a869`；直接依赖版本固定在 `requirements.txt`，PyTorch 单独按设备安装，并由约束文件防止其他依赖替换它。模型与缓存分别由 `METASK_DIR`、`HF_HOME` 控制。

从本目录按顺序执行初始化与测试脚本：

```bash
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/out/run-001"
mkdir -p "$XL_OUT_DIR"
export METASK_DIR="$XL_DATA_DIR/metask-jev"
export HF_HOME="$XL_DATA_DIR/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"  # 网络可直连官方 Hub 时可改为 https://huggingface.co
export METASK_MODEL_REVISION="ea20fe85b28733b1522721dec119bec50947a869"
export TORCH_REQUIREMENT="torch==2.15.0+cu132"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu132"
bash metask-install.sh
"$METASK_DIR/.venv/bin/python" metask-bench.py | tee "$XL_OUT_DIR/benchmark.log"
```

这里的 PyTorch 命令适用于本次记录的 NVIDIA GB10 / Linux aarch64 组合；其他设备需先按 PyTorch 官方说明选可用构建，再设置 `TORCH_REQUIREMENT` 与 `TORCH_INDEX_URL`。`XL_DATA_DIR`、`HF_HOME` 和 `METASK_DIR` 需指向新的隔离目录，避免复用其他实验的环境或缓存。安装脚本在子进程中启用虚拟环境，退出后不会改变当前 shell，因此基准命令显式使用该环境的解释器。安装器会创建依赖约束、检查依赖一致性、复制固定评分代码、下载指定模型版本并运行一次 GPU 自测；随后基准输出保存在日志中。

PyTorch `2.15.0+cu132` 在 Linux aarch64 上固定依赖 NVIDIA `cuSPARSELt 0.8.1`。该组件的 NVIDIA `manylinux2014_sbsa` 轮子标签会被当前 pip 的 `pip check` 报为平台不匹配；安装器只在版本、架构和错误文本都完全匹配时执行额外检查，确认该 ARM 库能加载且 CUDA 设备可用。其他依赖冲突仍会让安装失败。实测还提示 `causal_conv1d` 与 `flash-linear-attention` 未安装，会退回较慢的 PyTorch 实现；本次延迟结果包含这一条件。

`requirements.txt` 固定直接依赖，传递依赖仍由 pip 解析。本次新环境安装、模型类导入、单例 GPU 自测和下方基准均已完成。

## 实测指标与数据说明

2026-11-15 在 NVIDIA GB10、Linux aarch64、Python 3.12.3、PyTorch `2.15.0+cu132`、Transformers `5.22.2` 上的新环境实测：

- **加载与输出**：模型加载 56.19 秒；A–Z 均为单 Token；GPU 自测能够返回有效概率。
- **单次延迟**：英文 4 选项判定 120 次，中位数 98.20 ms、P90 99.33 ms、P99 100.19 ms。未校准温度 `T=1` 的 40 次中位数为 98.69 ms。
- **中英分类**：固定测试集各 20 条，英文命中 19 条（95%），中文命中 20 条（100%）；两组 P50 分别为 98.9 ms 和 97.7 ms。
- **显存占用**：当前分配 8.55 GB，峰值 8.65 GB。主机可用内存从加载前 117.4 GB 到测试后 107.3 GB；最低需求仍未确定。
- **测试限制**：中英文各只有 20 条；`causal_conv1d` 和 `flash-linear-attention` 未安装，模型使用较慢的参考实现。温度检查只展示 4 个样例的置信度与熵变化，没有重新计算上游 ECE。

可核对的公开摘要见[本次复跑记录](evidence/2026-11-15-gpu-rerun.json)。

**关于上游基准数据**：
上游 README 列出的 ECE 校准误差（0.114 降至 0.040）与 JevBench 排名我们没有复现，排名也是上游自测插值，未经过官方评测。选型时以本仓实测数字为准，完整取舍说明见[基准报告](../../../docs/zh/decision-models/判别式决策模型基准实测.md)。

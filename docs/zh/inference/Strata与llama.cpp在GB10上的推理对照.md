# Strata 对 llama.cpp：Qwen3.8-Flash-Next IQ3_S 在 GB10 上的性能对照

> **阅读导航**：[实验入口](../../../experiments/strata-arm64/README.md) · [DGX Spark 推理](DGX-Spark大模型推理与并发.md)

**职责声明**：本报告测量了 Strata 0.1.47（启用 MTP）与 llama.cpp 0.4.0-dev（无 MTP）在同一台 DGX Spark、同一份 IQ3_S 权重下的单请求推理性能，用于评估 Strata 完整方案（MoE 分层存储 + 专家缓存 + 投机解码）相对于 llama.cpp 基础方案的实际收益。**这不是同条件开关 MTP 的公平对照**：Strata 的 IQ 原生包强制要求 MTP（`--spec >= 2`），而 llama.cpp 当前版本无法加载该模型的 MTP 架构（`qwen4exp-mtp`）。不测量并发扩展、长上下文（>8K）或不同量化格式的表现，相关结论不外推到 x86 独立显卡。

## 测试环境

| 项目 | 配置 |
|------|------|
| 硬件 | DGX Spark，GB10，20 核 ARM64，121 GiB 统一内存 |
| Strata | 0.1.47（PR #3622, commit `d891f006aaec64024513a1e55856afd992baaa94`）；既有记录为 expert-cache 4096，与现存启动日志不一致，实际参数待核 |
| llama.cpp | 0.4.0-dev（build 1, commit `ccc3646`），GNU 13.3.0，Linux aarch64，`-ngl 40 -t 8 -c 8192`，无 MTP |
| 模型 | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S，83.62 GB，2 个 GGUF 分片（SHA-256 见 README 版本锁定节） |
| CUDA | 13.0 |
| 协议 | 9 组场景，temperature 0.7，单请求，context 8192 |
| 实验日期 | 2026-10-26 |

## 性能对照

| 测试场景 | Strata 生成 tok/s | llama.cpp 生成 tok/s | 加速比 |
|----------|------------------|---------------------|--------|
| 预热（不计入下方平均） | 34.2 | 10.96 | 3.1x |
| 短输入短输出（算术） | 40.8 | 15.52 | 2.6x |
| 短输入短输出（常识） | 31.3 | 15.56 | 2.0x |
| 短输入长输出（写诗 200t） | 20.5 | 12.99 | 1.6x |
| 中输入中输出（解释+举例） | 33.1 | 14.54 | 2.3x |
| 长输入短输出（总结） | 47.3 | 15.91 | 3.0x |
| 重复测试 1 | 37.5 | 12.87 | 2.9x |
| 重复测试 2 | 41.7 | 16.01 | 2.6x |
| 重复测试 3 | 33.2 | 13.37 | 2.5x |

除预热外，8 条请求逐条速度比的算术平均约 2.4，范围约 1.6–3.0。两条请求的生成 token 数不同，且多条输出被长度上限截断；这些数字是不同配置下的引擎报告速度，不代表相同输出质量或等量工作的端到端加速。

## 机制分析

Strata 在本次测试启用了专家缓存与 MTP，llama.cpp 配置未启用该模型的 MTP。两项实现和配置同时变化，不能将速度差单独归因于其中任何机制。

逐请求 `draft_n` 与 `draft_n_accepted` 见原始记录。要分别测量投机解码与缓存的贡献，还需在可运行的配置内设计控制变量对照。现有启动日志中 `expert_slots=10701`、`expert_cache_mib=20798`、`spec=6`，与报告列出的参数不同，尚不能确认是否属于同次基准运行。

## 语义正确性

| 测试 | 期望 | 实际 | 结果 |
|------|------|------|------|
| 算术：1 + 1 | 2 | 两组输出均被长度上限截断 | 未通过完整回答检查 |
| 常识：中国首都 | 北京 | 两组均含「北京」，Strata 输出被截断 | 不能据此判断完整回答 |

现存 9 条请求不含 17 × 23 的原始结果；不能据此声称两引擎均通过该测试。语义质量需要单独评估。

## 内存与稳定性

- 既有记录称最低 MemAvailable 为 61 GiB，未发生 mem-guard 处置；逐时采样日志尚待核对。
- 既有记录称 Strata 的 auto 模式出现 OOM，llama.cpp 的 `-ngl 99` 启动失败、`-ngl 40` 可运行；相应失败日志尚待核对。

## 局限

1. **MTP 对比不同口径**：Strata 的 IQ 原生包（`data/packs/iq3_s`）强制要求 `--spec >= 2`，无法关闭 MTP 做无投机基线；llama.cpp 加载 `mtp-q2_0.gguf` 时报错 `unknown model architecture: 'qwen4exp-mtp'`，无法启用同模型 MTP。因此本报告是"Strata 完整方案 vs llama.cpp 基础方案"的对比，不能精确拆分"专家缓存"与"MTP"各自的贡献比例，也不能回答"同开 MTP 时谁更快"。
2. 仅测试单请求，未测并发；Strata 的 `--parallel` 参数可能进一步拉开差距，但不在本次范围。
3. llama.cpp 的 `-ngl 40` 是手动调优值，不同值可能影响对比结果；未做参数扫描。
4. 启动日志提到 CPU 路径不使用 AVX2，但没有跨 CPU 架构对照，无法确认其为性能瓶颈或推断其他平台上的加速比。

## 原始数据

- [strata-results.jsonl](../../../experiments/strata-arm64/evidence/strata-results.jsonl)：Strata 9 组完整响应
- [llama-results.jsonl](../../../experiments/strata-arm64/evidence/llama-results.jsonl)：llama.cpp 9 组完整响应
- [bench-output.txt](../../../experiments/strata-arm64/evidence/bench-output.txt)：Strata 基准汇总
- [llama-bench-output.txt](../../../experiments/strata-arm64/evidence/llama-bench-output.txt)：llama.cpp 基准汇总
- [engine-startup.txt](../../../experiments/strata-arm64/evidence/engine-startup.txt)：Strata 引擎启动日志（与基准的对应关系尚待核实）

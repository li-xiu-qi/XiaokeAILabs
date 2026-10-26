# Strata 对 llama.cpp：Qwen3.8-Flash-Next IQ3_S 在 GB10 上的性能对照

**职责声明**：本报告测量了 Strata 0.1.47 与 llama.cpp 0.4.0-dev 在同一台 DGX Spark、同一份 IQ3_S 权重下的单请求推理性能，用于评估 MoE 分层存储与投机解码在统一内存硬件上的实际收益。不测量并发扩展、长上下文（>8K）或不同量化格式的表现，相关结论不外推到 x86 独立显卡。

## 测试环境

| 项目 | 配置 |
|------|------|
| 硬件 | DGX Spark，GB10，20 核 ARM64，121 GiB 统一内存 |
| Strata | 0.1.47（PR #3622, commit `d891f006aaec64024513a1e55856afd992baaa94`），expert-cache 4096（10.16 GiB VRAM） |
| llama.cpp | 0.4.0-dev（build 1, commit `ccc3646`），GNU 13.3.0，Linux aarch64，`-ngl 40 -t 8 -c 8192`，无 MTP |
| 模型 | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S，83.62 GB，2 个 GGUF 分片（SHA-256 见 README 版本锁定节） |
| CUDA | 13.0 |
| 协议 | 9 组场景，temperature 0.7，单请求，context 8192 |
| 实验日期 | 2026-10-26 |

## 性能对照

| 测试场景 | Strata 生成 tok/s | llama.cpp 生成 tok/s | 加速比 |
|----------|------------------|---------------------|--------|
| 预热 | 34.2 | 10.96 | 3.1x |
| 短输入短输出（算术） | 40.8 | 15.52 | 2.6x |
| 短输入短输出（常识） | 31.3 | 15.56 | 2.0x |
| 短输入长输出（写诗 200t） | 20.5 | 12.99 | 1.6x |
| 中输入中输出（解释+举例） | 33.1 | 14.54 | 2.3x |
| 长输入短输出（总结） | 47.3 | 15.91 | 3.0x |
| 重复测试 1 | 37.5 | 12.87 | 2.9x |
| 重复测试 2 | 41.7 | 16.01 | 2.6x |
| 重复测试 3 | 33.2 | 13.37 | 2.5x |

**平均加速比约 2.5 倍**，范围 1.6-3.1 倍。

## 机制分析

Strata 的优势主要来自三个层面：

1. **MoE 稀疏激活 + 专家缓存**：常用专家驻留 GPU（本实验 5341 slots，10.16 GiB VRAM），避免每次推理重复从内存加载全部专家。llama.cpp 无此机制，每次需要重新调度。
2. **MTP 投机解码**：Strata 的 draft 层接受率 50-70%，意味着约一半 token 由轻量 draft 模型预先猜中，主模型只需验证而非完整计算。
3. **分层存储与 CPU/GPU 混合调度**：Strata 将不常用专家留在内存，按需调用，降低 VRAM 压力；llama.cpp 的 `-ngl 40` 只能静态分配层到 GPU，灵活性较低。

## 语义正确性

| 测试 | 期望 | 实际 | 结果 |
|------|------|------|------|
| 算术：17 × 23 | 391 | 391 | ✅ |
| 常识：中国首都 | 北京 | 北京 | ✅ |

两引擎均返回正确答案，finish_reason 为 stop，content 完整。

## 内存与稳定性

- 测试期间最低 MemAvailable：61 GiB（远高于 20 GiB 安全线）
- 无 mem-guard 处置事件
- Strata 需手动设置 expert-cache（auto 模式导致 NVIDIA 驱动 OOM），llama.cpp 需手动调整 `-ngl`（99 过高导致启动失败，40 可运行）

## 局限

1. 仅测试单请求，未测并发；Strata 的 `--parallel` 参数可能进一步拉开差距，但不在本次范围。
2. llama.cpp 的 `-ngl 40` 是手动调优值，不同值可能影响对比结果；未做参数扫描。
3. ARM CPU 无 AVX2 是共同瓶颈，两引擎均受影响；在 x86+AVX2 平台上，绝对速度可能更高，但加速比趋势预计一致。
4. 未测试 Strata 无 MTP 的独立表现，无法精确拆分"专家缓存"与"MTP"各自的贡献比例。

## 原始数据

- `docs/evidence/strata-results.jsonl`：Strata 9 组完整响应
- `docs/evidence/llama-results.jsonl`：llama.cpp 9 组完整响应
- `docs/evidence/bench-output.log`：Strata 基准汇总
- `docs/evidence/llama-bench-output.log`：llama.cpp 基准汇总
- `docs/evidence/engine-startup.log`：Strata 引擎启动日志（含硬件检测与内存分配）

# Strata 在 DGX Spark 上的统一内存推理实验

## 要回答的问题

同一台 DGX Spark（GB10，统一内存）、同一个模型（Qwen3.8-Flash-Next IQ3_S）、同一套压测协议下：

1. Strata 的 ARM64 社区移植能否在 GB10（aarch64、sm_121）上完成构建与推理；
2. Strata 完整方案（MoE 稀疏激活 + 专家缓存 + MTP 投机解码）相比 llama.cpp 基础方案能带来多少实际加速；
3. 统一内存架构下，Strata 的 expert-cache 自动分配是否可靠，需要怎样的手动调优。

**口径说明**：Strata 的 IQ 原生包强制要求 MTP（`--spec >= 2`），llama.cpp 无法加载该模型的 MTP 架构，因此对比为"Strata 启用 MTP vs llama.cpp 无 MTP"，不是同条件开关投机的公平对照。

## 实验对象

| 项 | 配置 |
|---|---|
| Strata 来源 | PR #3622，`ferstar/Strata` 的 `gb10-022` 分支 |
| 固定提交 | `d891f006aaec64024513a1e55856afd992baaa94` |
| 模型 | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`，IQ3_S（83.62 GB，2 个 GGUF 分片） |
| 硬件 | DGX Spark，GB10，20 核 ARM64，121 GiB 统一内存，sm_121 |
| 软件 | Ubuntu 24.04，CUDA 13.0，源码构建 |
| 上下文 | 8,192 token，单请求，关闭视觉 |
| 采样 | temperature 0.7，max_tokens 按场景 20-200 |
| Strata 投机 | `--spec 4 --spec-min-p 0.5 --mtp <path>`（强制启用，无法关闭） |
| 对照组 | llama.cpp 0.4.0-dev（commit ccc3646），`-ngl 40 -t 8 -c 8192`，无 MTP（架构不支持该模型 MTP） |

## 版本锁定

以下版本信息用于确保实验可追溯。后续复现时若组件版本不同，结果可能不可直接比较。

| 组件 | 版本标识 |
|------|----------|
| Strata 引擎 | 0.1.47（PR #3622, commit `d891f006aaec64024513a1e55856afd992baaa94`） |
| llama.cpp | 0.4.0-dev（build 1, commit `ccc3646`），GNU 13.3.0，Linux aarch64，构建目录 `build-force-cublas` |
| 模型分片 1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`，SHA-256: `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| 模型分片 2 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`，SHA-256: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| CUDA | 13.0（`/usr/local/cuda/bin/nvcc`） |
| 实验日期 | 2026-10-26 |

## 协议

9 组测试场景：短输入短输出、短输入长输出、中输入中输出、长输入短输出、重复稳定性测试。记录原生 prefill/decode tok/s、输出 token 数、完成原因；Strata 另记录 MTP 草稿接受率。语义冒烟包含算术题（17×23=391）与常识题（中国首都=北京）。

## 环境

DGX Spark（GB10，121 GiB 统一内存，273 GB/s），Strata 源码构建（aarch64，CUDA 13.0），llama.cpp `build-force-cublas`。测试前停止其他大模型服务，避免资源竞争。

## 状态

- [x] 引擎构建（2026-10-26）：aarch64 + CUDA 13.0 + sm_121，strata 0.1.47
- [x] 模型部署：IQ3_S 83.62 GB，SHA-256 校验通过
- [x] 语义冒烟：17×23=391 ✅，中国首都=北京 ✅
- [x] Strata 基准：生成 20.5-47.3 tok/s，prompt 49.1-166.3 tok/s，MTP 接受率 50-70%
- [x] llama.cpp 对照：生成 11.0-16.0 tok/s，prompt 21.3-83.2 tok/s
- [x] 内存安全：最低 61 GiB 可用，无 mem-guard 事件
- [x] 对比报告 `reports/01-strata对llamacpp-iq3s.md`：Strata 生成速度为 llama.cpp 的 1.6-3.1 倍（平均约 2.5 倍）

原始数据在 `docs/evidence/`（逐请求 JSON，含 timings 与 draft 统计）。

## 关键发现

1. **统一内存下 MoE 分层存储确实有效**：Strata 通过专家缓存 + MTP 实现约 2.5 倍加速，证明稀疏激活与分层调度在 GB10 上有实际收益。
2. **expert-cache auto 不可靠**：GB10 统一内存架构下，auto 模式导致 NVIDIA 驱动 OOM（`NV_ERR_NO_MEMORY`），必须手动限容（本实验用 4096，约 10 GiB VRAM）。
3. **ARM CPU 无 AVX2 是瓶颈**：GB10 的 ARM CPU 走 ggml-cpu 回退路径，专家计算效率低于 x86+AVX2，限制了整体速度进一步提升。

## 复现

见 `scripts/` 下的准备与基准脚本。模型权重需自行从 Hugging Face 或镜像下载，不进本仓。

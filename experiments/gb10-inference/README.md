# GB10 单机大模型极限推理与引擎评测总组

本目录汇聚在 **NVIDIA DGX Spark（GB10，aarch64，121.6 GiB 统一内存，273 GB/s 带宽，sm_121）** 上的系列极限推理实验。

旨在回答三个工业落地的关键问题：
1. **超大参数 MoE 能否在百 GB 级单卡上开箱落地？**
2. **非 x86 / sm_121a 新架构下，第三方自研引擎（NInfer / Strata）能否移植并跑出宣称性能？**
3. **投机解码（MTP / DFlash）在不同硬件与后端下何时为正收益、何时为负陷阱？**

---

## 📊 子实验全景对照

| 子实验目录 | 评测对象 | 核心技术点与架构 | 核心实测指标 | 关键发现与避坑提示 |
|---|---|---|---|---|
| [**ds4-flash-2bit/**](ds4-flash-2bit/) | DeepSeek-V4-Flash 2-bit (299B MoE) | 2-bit VQ 量化，常驻 94.69G，vLLM 0.23 | **20.16 tok/s** (decode) | ⚠️ **MTP 负收益陷阱**：MTP K=2 仅 13.42 tok/s (-33%)，因 vLLM 拒绝 drafter FULL cudagraph |
| [**ninfer-27b/**](ninfer-27b/) | Qwen3.8-27B (W4A4 + DFlash2 + Vision) | NInfer GB10 移植，NVFP4 / FP8 对照 | C8 并发 **132.1~223.7 tok/s** | 突破官方限定 `sm_120a`，证实统一内存下 **NVFP4 W4A4 实测超越 FP8** |
| [**qwen36-35b-ab/**](qwen36-35b-ab/) | Qwen3.6-35B-A3B 双引擎 A/B | llama.cpp MTP2 vs NInfer MTP3 / DFlash7 | 单请求 **1.37x**；C8 并发 **223.7 vs 54.8 tok/s** | llama.cpp 单槽位限制致高并发串行化；DFlash7 低接受率下亏成本 |
| [**strata-dgx/**](strata-dgx/) | Qwen3.8-Flash-Next IQ3_S | Strata (稀疏激活 + 专家缓存 + MTP) 对照 llama.cpp | 相比 llama.cpp 提速 **2.5x** | 统一内存下需手动调优 expert-cache，自动分配易显存溢出 |

---

## 🛠️ 统一硬件与测试基线

- **硬件规格**：NVIDIA GB10 (20 核 ARM64, 121.6 GiB 统一内存, 273 GB/s 内存带宽, sm_121)
- **基线软件**：Ubuntu 24.04, CUDA 13.0, PyTorch 2.11+ / 2.14+, Python 3.12
- **压测协议规范**：短输入长输出、长输入短输出、高并发 C1/C2/C4/C8 阶梯压测，记录逐请求客户端 e2e 与引擎原生 decode timings。

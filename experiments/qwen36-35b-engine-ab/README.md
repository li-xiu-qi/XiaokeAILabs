# Qwen3.6-35B-A3B 双引擎 A/B：llama.cpp MTP2 对 NInfer MTP3 / DFlash7

## 要回答的问题

同一台 DGX Spark（GB10，统一内存）、同一个模型（Qwen3.6-35B-A3B）、同一套压测协议下：

1. llama.cpp 现档（UD-Q4_K_M GGUF + MTP n2，262K 上下文，单槽位）与 NInfer 官方构件（混合精度 Q4/Q5/Q6/Q8 + MTP3 或 DFlash7）的单请求 decode 吞吐差多少；
2. 官方模型卡上「MTP3 相对基线 2-3 倍」的 RTX 5090 数字，在带宽只有其约 1/6 的 GB10 上能兑现多少；
3. 两引擎在 C1/C2/C4/C8 下的扩展行为差异（llama.cpp 侧受 `-np 1` 部署约束串行化）。

## 实验对象

| 项 | llama.cpp 侧 | NInfer 侧 |
|---|---|---|
| 权重 | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`（22.66 GB，unsloth MTP-GGUF 仓）+ `mmproj-F16.gguf` | `qwen3_6_35b_a3b.ninfer`（22,790,484,480 B，HF `neroued/Qwen3.6-35B-A3B-NInfer`，混合精度，内置 Text/Vision/MTP/DFlash） |
| 投机 | MTP n2，`--spec-draft-backend-sampling` | MTP draft3 / DFlash draft7 两档，`--lm-head-draft` |
| 上下文 | 262144 | 32768（压测 prompt 短，按 27B 实验同款配置） |
| KV | q8_0 K/V | fp8 |
| 部署 | 0.0.0.0:8835，`-np 1` 常驻服务 | 127.0.0.1:19410，`--max-concurrency 8` 实验实例 |

模型卡 Limits（迁移风险评估用，非本实验测量项）：NInfer 不执行生成的工具调用；并发 1-8 固定，无抢占式批处理。

## 协议（与 27B NInfer 实验完全一致，保证跨模型可比）

C1/C2/C4/C8 各 2 轮，每轮请求数等于并发数；`max_tokens=512`，`temperature=1.0`，`top_p=0.95`，`top_k=20`，thinking 开启，`reasoning_effort=xhigh`，seed 20261025，8 条固定 prompt。记录逐请求原生 timings（decode/prefill/草稿接受）与客户端 e2e。脚本 `scripts/bench_ninfer_concurrency.py`（复用自 `ninfer-gb10-inference` 实验，未改动）。

口径提醒：客户端总吞吐含 prefill 与排队；llama.cpp 侧 `-np 1`，C2 以上请求串行，聚合吞吐不随并发上升是部署属性不是引擎缺陷。

## 环境

DGX Spark（GB10，121 GiB 统一内存，273 GB/s），llama.cpp `build-force-cublas`，NInfer GB10 fork `4c2fc2f`（`build-gb10-system-pkg`）。llama.cpp 侧压测期间 8835 为常驻服务（日活，测试期间被 benchmark 加载）；NInfer 侧为独占实验实例，启动前过 `preflight.sh` 闸门。

## 状态

- [x] llama.cpp 侧（2026-10-26 18:04 完成）：C1 82.8 / C2 84.1 / C4 82.4 / C8 54.8 t/s 聚合，原生 decode 85.0-87.6 t/s，接受率 69.9-72.5%
- [x] NInfer MTP3 侧（2026-10-26 19:20 完成）：C1 115.4 / C2 155.0 / C4 189.1 / C8 223.7 t/s 聚合，原生 decode C1 118.5 t/s，接受率 53.5-57.1%
- [x] NInfer DFlash7 侧（2026-10-26 19:22 完成）：C1 90.6 / C8 205.8 t/s 聚合，接受率 23.9-33.3%，单请求低于 MTP3
- [x] 对比报告 `reports/01-llamacpp对ninfer-35b-a3b.md`：单请求 NInfer MTP3 为 llama.cpp 的 1.37 倍；并发扩展差异最大（NInfer C8 聚合 223.7 对 llama.cpp -np 1 的 54.8）；DFlash7 低接受率下亏成本

原始数据在 `docs/evidence/`（逐请求 JSON，含 warmup 与 summary）。

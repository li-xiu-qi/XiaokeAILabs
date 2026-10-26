# SGLang：FP8 与 NVFP4 W4A4 + DFlash2 对照

## 目的

验证“FP8 + DFlash2 就是 GB10 上精度、速度与并发的最优解”能否在同一台 DGX Spark 上复现。实验控制模型 revision、DFlash2 草稿、输入提示、输出上限、KV 与 Mamba 配置，避免把量化名称当成实际速度。

## 最终协议

- 同模型仓的 `FP8/` 与 `NVFP4/W4A4/`，所有权重、视觉、MTP 和 DFlash2 文件通过 SHA256。
- SGLang 0.5.19，PyTorch 2.13.0+cu130，FlashInfer。
- 32K 单请求上下文、最多 8 并发、65,536 token KV 池、48 个 Mamba cache slots、0.50 内存配额。
- DFlash2 草稿 8、xhigh 思考、512 token 上限。
- eager 与 decode graph 两组；C1/C2/C4/C8 各 2 轮，每轮请求数等于并发数。
- 同时记录客户端总吞吐、逐请求端到端速度、原生 `spec_accept_rate`、`spec_accept_length`、完成原因与原始 timings。

## 结果

| 并发 | NVFP4 eager | FP8 eager | NVFP4 decode graph | FP8 decode graph |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 32.7 | 22.9 | 31.2 | 20.0 |
| C2 | 55.2 | 45.7 | 63.7 | 42.1 |
| C4 | 96.9 | 68.5 | 99.0 | 68.5 |
| C8 | 132.5 | 106.1 | 144.6 | 106.9 |

单位是总输出 tok/s，包含 prefill 与排队，不等于纯 decode。

eager 下 NVFP4 总吞吐分别为 FP8 的 1.42、1.21、1.41、1.25 倍；decode graph 下分别为 1.56、1.52、1.45、1.35 倍。当前协议没有 FP8 反超的并发点。

## 接受率与图优化

| 并发 | NVFP4 eager 接受率 | FP8 eager 接受率 | NVFP4 graph 接受率 | FP8 graph 接受率 |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 35.7% | 36.0% | 32.8% | 28.9% |
| C2 | 32.4% | 43.7% | 40.2% | 37.0% |
| C4 | 40.1% | 40.8% | 39.0% | 37.4% |
| C8 | 32.9% | 34.6% | 36.0% | 31.2% |

两档接受长度大多在 3.2–4.1 token。FP8 在部分档位的接受率不低于 NVFP4，但总吞吐仍更低，因此差异不能只归因于 DFlash2 接受率。尚未进行 kernel 级 profiling，不能把带宽或某一条指令写成已证实根因。

decode graph 也不是普遍加速：NVFP4 的 C2/C8 有改善，C8 约 +9%；FP8 的 C1/C2 下降，C4 基本持平，C8 仅约 +0.8%。是否开图必须按模型和并发实测。

## 质量与边界

两档的 10 道自动题均为 10/10，截断与空答均为 0；两道编程题均通过语法检查和每题 5 个功能用例。文本与图片冒烟也全部通过。小样本不替代完整公开测评，不能据此宣称两档全量精度相同。

FP8 在 0.40 和 0.45 内存配额下无法分配 KV，最终两档统一为 0.50。NVFP4 预跑数据没有与最终配置混合，本文只使用最终 eager 与 decode graph 数据。

没有测长上下文、持续并发、能耗、首 token 延迟分位数或生产故障恢复。结论仅限当前模型、SGLang、DFlash2 与本机协议。

原始结果：`docs/evidence/bench-*.json` 与 `comparison-*.md/json`。

# 27B W4A4 + DFlash2 百 TPS 截图的复现

## 为什么约 22 tok/s 不是反证

约 22 tok/s 的历史结果来自 UD-Q4_K_XL GGUF + llama.cpp + MTP2 + 关闭 thinking。截图来自 NInfer W4A4 + DFlash2 draft 7 + 最高思考。模型容器、引擎、量化、投机算法和思考配置均不同，因此不能用前一组数字否定后一组，也不能把两个速度拼成同一引擎的比较。

## 复现配置

- 固定 GB10 fork `4c2fc2f9504d2f373c84e896bf18523906a646d7`。
- V3 W4A4 + MTP + DFlash2 + Vision，SHA256 已验证。
- DFlash2 draft 7、`--lm-head-draft`、Vision、CUDA Graph。
- 32K context、262144 KV、最大 8 并发。
- 请求显式 `enable_thinking=true`、`reasoning_effort=xhigh`，最多 512 token。
- C1/C2/C4/C8 各 2 轮，保留原生 predicted_per_second、draft_n 与 draft_n_accepted。

## 结果

| 并发 | 外部截图 | 本实验 NInfer（draft 7） | 本实验 SGLang 参考（draft 8） |
| ---: | ---: | ---: | ---: |
| C1 | 48.3 | 32.8 | 31.2 |
| C2 | 76.9 | 58.4 | 63.7 |
| C4 | 97.7 | 95.2 | 99.0 |
| C8 | 146.8 | 132.1 | 144.6 |

**C8 超过 100 tok/s 已在 DGX Spark 上复现。** 截图与 SGLang 参考仅差约 1.5%，NInfer 的 C4 也处于同一量级。

本次 NInfer 的 DFlash2 接受率为 C1 33.95%、C2 36.62%、C4 36.71%、C8 32.57%；截图为 61.8%、52.8%、41.2%、44.6%。接受率差异会直接影响有效 decode，尤其 C1。截图所属硬件、提示分布和运行负载不受本实验控制，因此不追求逐点一致。

原始逐请求数据：`docs/evidence/bench-ninfer-w4a4-dflash2-draft7-xhigh.json`。SGLang 对照：`docs/evidence/bench-nvfp4-dflash2-graph.json`。

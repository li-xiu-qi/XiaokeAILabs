# Qwen3.6-35B-A3B 双引擎对比：llama.cpp MTP2 对 NInfer MTP3 / DFlash7

同一个模型、同一台 DGX Spark、同一套压测协议，换两个推理引擎跑，看速度差多少。只测速度，不测精度、工具调用和长上下文；这是小样本协议，不等于生产压测。

## 结论

**单请求上 NInfer 更快。** MTP3 跑出 118.5 t/s，llama.cpp 的 MTP2 是 86.5 t/s，差 1.37 倍。快三到四成，不是数量级的差距。

**并发上的差距更大。** NInfer 能真批量解码，C8 聚合到 223.7 t/s。llama.cpp 这套部署是 `-np 1` 单槽位，C2 往上请求就排队，聚合卡在 84 t/s 左右。

**DFlash7 在这台机器上不如 MTP3。** 接受率 23.9% 对 53.5%，单请求 92.8 对 118.5 t/s。草稿一次猜七个 token，每轮成本更高，接受率又撑不住，算下来亏。

## 结果

协议完全一致：512 token 上限，temperature 1.0，top_p 0.95，top_k 20，开 thinking，reasoning_effort 设 xhigh，C1/C2/C4/C8 各跑两轮，seed 20261025，8 条固定 prompt。单位 tok/s。

| 并发 | llama.cpp MTP2 聚合 | llama.cpp 原生 decode | NInfer MTP3 聚合 | NInfer 原生 decode | NInfer DFlash7 聚合 | NInfer 原生 decode |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 82.8 | 86.5 | **115.4** | **118.5** | 90.6 | 92.8 |
| C2 | 84.1 | 87.6 | 155.0 | 84.2 | 135.4 | 81.0 |
| C4 | 82.4 | 87.4 | 189.1 | 54.5 | 156.6 | 49.2 |
| C8 | 54.8 | 85.0 | **223.7** | 31.5 | 205.8 | 31.0 |

草稿接受率：llama.cpp 69.9 到 72.5%，NInfer MTP3 53.5 到 57.1%，NInfer DFlash7 23.9 到 33.3%。

有两点读表时容易误判，先说清楚。

llama.cpp 的 C8 聚合只有 54.8，比 C2 的 84.1 还低。这不是引擎性能塌了，是 `-np 1` 部署下八个请求挤一个槽位，排队反而增加开销。换成 `-np 8` 会怎样，我们没测，所以不能把这列数字读成 llama.cpp 的上限。

NInfer 的并发上来之后，单个请求的 decode 速度从 118.5 掉到 31.5。这是统一内存带宽被多个请求摊薄的结果，两个引擎都一样。所以比并发只能看聚合列，不能拿单请求的 decode 比。

## 为什么接受率高的反而慢

llama.cpp 的接受率更高，约 70%，NInfer 是 53.5%，但 llama.cpp 反而慢。差的地方不在草稿质量，在每轮的成本。

NInfer 的 MTP 走 `--lm-head-draft`，提议头和主模型的输出头共享，草稿前向基本免费。llama.cpp 的 MTP 头虽然也是内置的，但在本机 kernel 与批处理的效率低于 NInfer 的 fork。接受率高几个点，抵不过每轮多花的时间。

## 复现配置

llama.cpp 侧：常驻服务跑在 8835，`-np 1 -c 262144`，权重是 UD-Q4_K_M 加 MTP n2。这个是日常在跑的服务，压测期间被 benchmark 占用；空闲时不争带宽。

NInfer 侧：`qwen3_6_35b_a3b.ninfer`，模型来自 HF 的 `neroued/Qwen3.6-35B-A3B-NInfer`，SHA256 见 `docs/evidence/sha256.txt`。GB10 fork 是 `4c2fc2f`，参数 `--max-context 32768 --kv-capacity 32768 --max-concurrency 8 --kv-dtype fp8`，MTP3 与 DFlash7 各跑一遍。启动走 `run_guarded.sh`，带闸门、内存采样和审计。

加载后内存：NInfer 实例 used 78G、avail 43G，与常驻的 llama-server 占的 34G 共存。

## 边界

没测的部分：精度、工具调用、长上下文衰减、持续并发稳定性、能耗，以及 llama.cpp 的 `-np 8` 对照。

两个引擎的上下文窗口不同，llama.cpp 是 262144，NInfer 是 32768。短 prompt 下对 decode 速度影响可以忽略，但引用数字时要注明。

NInfer 的模型卡写明它不执行生成的工具调用。如果要把它当日常服务用，这是必须解决的阻塞项。

构建方公布的精度数据（AIME 2025/2026 约 90%、GPQA-Diamond 85.35%）是他们的自测结果，不是我们在本机的测量，不能当作本实验的结论引用。

## 数据

`docs/evidence/` 下三份 JSON，含逐请求的原生 timings、草稿统计、warmup 与 summary，另有 `sha256.txt` 记录构件校验值。

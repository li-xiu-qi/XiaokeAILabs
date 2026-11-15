# NInfer 在 GB10 上的移植与并发实测

> **阅读导航**：[实验入口](../../../experiments/ninfer-serving/README.md) · [DGX Spark 推理](DGX-Spark大模型推理与并发.md)

## 问题不是模型文件，而是引擎支持边界

V3 `.ninfer` 只能由 NInfer 读取，SGLang、vLLM、llama.cpp 和 Transformers 不能直接使用该容器。当前官方 `64d10fb` 在 CMake 中只接受 `CMAKE_CUDA_ARCHITECTURES=120a`，传入 GB10 的 `121a` 会实际报：

```text
NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '121a'
```

这只是第一层阻塞。源码审查还确认：

- Qwen3.5 启动规划阶段要求 compute capability 12.0；
- x86 CPUID、`xgetbv`、AVX2/AVX512 主机路径需要与 AArch64 拆分；
- 统一内存不能简单用裸 `cudaMemGetInfo` 当作可安全分配预算。

因此，"把 120a 改成 121a" 不是完整修复。与此同时，GB10 fork 与上游后续补丁证明这条移植路线有实际基础，不能反向推断 NInfer 永远不可用。

## 为什么选择固定 fork

本实验使用 `baristahaus/ninfer-gb10` commit `4c2fc2f9504d2f373c84e896bf18523906a646d7`，不跟随浮动 master。该版本允许 `120a` 和 `121a`，并补上了 Qwen3.5 的 12.1 启动门禁，也包含 V3、MTP、DFlash2 和 Vision 的测试入口。

当前公开资料中，GB10 的 Flash-Next 有真实运行记录，但这不能自动证明 Qwen3.8 27B + DFlash2 + Vision 也已通过。本实验因此重新执行构建和全部目标路径，不引用 fork 的兼容性声明代替实测。

## 构建环境与处理

构建依赖包括 CMake、GCC/G++、CUDA、Ninja、FFmpeg 开发文件和 libcurl。配置阶段有两个可复现细节：

1. 不把 libcurl 前缀全局塞进 `LD_LIBRARY_PATH`，否则系统 CMake 会误加载另一套 libcurl；只在运行 NInfer 时设置该变量。
2. 使用系统 `/usr/bin/pkg-config`，避免部分 conda 实现把系统 zlib 头目录解析为不存在的路径。

配置脚本为 [配置脚本](../../../experiments/ninfer-serving/scripts/ninfer_gb10_configure.sh)，构建使用 `-j1`。完整配置与 898/898 个目标均已完成。

## 真实运行结果

| 验证项 | 结果 |
| --- | --- |
| 构建 | 898/898 个目标编译、链接完成 |
| CTest | 初次 142/144 通过；2 个 real-artifact 环境缺项带模型后均重跑通过，跳过不冒充通过 |
| V3 host binding | 通过，架构为 `Qwen3_5ForCausalLM` |
| 最小文本生成 | 无 Graph、无投机、无视觉，16 token 完成 |
| MTP real prefix | 通过 |
| DFlash2 real | Graph 关闭、K=2、B=2、optimized=0，`accepted=15/15` |
| Vision real prefix | 通过 |
| 两个 repro | `vision_reuse_divergence` 与 `host_restore_state` 均通过 |
| DFlash2 + Vision | 红色圆形、蓝色方块和 `DGX 19410` 均识别正确 |
| 默认 CUDA Graph | 输出一致，graphs 就绪，decode 45.9 tok/s（短冒烟） |
| OpenAI 服务 | 文本答案 391 正确，图片内容全部识别正确 |

这些结果说明固定 fork 已能运行该模型包，包括 MTP、DFlash2、Vision 和 CUDA Graph。短冒烟的速度只用于确认路径，不替代性能基准。正式性能对照见下方"27B W4A4 + DFlash2 百 TPS 截图复现"一节与[实验记录](../../../experiments/ninfer-serving/evidence/)。

## 测量陷阱与结论边界

### 统一内存与 JIT

模型权重和 KV 的预算不能覆盖启动期编译内存。并行 `cicc` 可能各自占用数 GB，在统一内存机器上会瞬间挤占系统与其他推理服务。本实验曾因此触发内存门卫，随后恢复受影响服务并固定 `MAX_JOBS=1`。复现时不应默认提高编译并发，也不应只看模型静态配额。

### KV 与 Mamba cache

- DFlash2 每路需要多个 Mamba state slots。cache=16 会把实际并发压到 3，最终使用 48 以支持 C8。
- FP8 在 0.40、0.45 配额下无法分配 KV，最终两档统一为 0.50。
- 65,536 token KV 池只覆盖本实验的短负载，不代表已经验证 8×32K 同时满载。
- 125B 或其他大模型不能直接套用这些数值，必须重新做内存资格化。

### 吞吐口径

SGLang 表使用 `completion_tokens / 客户端墙钟时间`，包含 prefill 和排队，不是纯 decode。NInfer 与 llama.cpp 还分别保留原生 `predicted_per_second`、`prompt_per_second`、`draft_n` 和 `draft_n_accepted`，不能把这些不同口径混为同一列。

DFlash2 接受率对任务与文本分布非常敏感。比较引擎时必须同时查看接受率、接受长度、提示、采样与输出上限，不能只看总 TPS。

### 质量与外部截图

10 道自动题和 2 道编程题只是小样本探针，不替代 GPQA、MMLU、LCB 或长生成质量评估。很多性能请求以 512 token 截断，这是固定负载，不能用该完成率评价模型质量。

外部截图只用于检验百 TPS 是否可复现。截图所属硬件、提示内容和负载未知，因此本实验不把逐点差异解释成确定的引擎优劣。NInfer 复现使用 draft 7，SGLang 参考使用 draft 8，已在表头分开标注。

### 未覆盖范围

本实验没有验证长上下文、持续并发、视频、能耗、生产高可用、全量精度或其他 MoE 型号。GB10 fork 是第三方移植，官方直接支持状态应单独核查，不能把本实验写成官方功能承诺。

## 仍然未知的边界

- 没有验证长上下文、持续 C8、全量精度、能耗或生产故障恢复。
- fork 是第三方移植，不代表官方已承诺同等的 GB10 支持。
- 未把其他 MoE 型号的结果混入本实验；它们需要独立的模型包与数值资格化。
- 真实多模态服务还需要独立的视觉压力与视频测试，当前只证明图片路径成立。

---

## 27B W4A4 + DFlash2 百 TPS 截图的复现

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

原始逐请求数据：[NInfer](../../../experiments/ninfer-serving/evidence/bench-ninfer-w4a4-dflash2-draft7-xhigh.json) 与 [SGLang 对照](../../../experiments/ninfer-serving/evidence/bench-nvfp4-dflash2-graph.json)。

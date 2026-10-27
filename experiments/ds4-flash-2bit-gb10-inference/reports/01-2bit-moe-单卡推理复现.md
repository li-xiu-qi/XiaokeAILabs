# 2-bit 299B MoE 在单卡 GB10 上的推理实测

2026-10-27，我们在一台 DGX Spark 上把 `iggerask/DeepSeek-V4-Flash-2bit-GB10` 这个 2-bit 量化检查点跑了起来，测了非投机与 MTP K=2 两组解码速度，顺手挖出一个和版本有关的问题。这份报告记录全部过程与数字，后续引用本实验的结论以本文和 `docs/evidence/` 为准。

不覆盖的部分先说清楚：长上下文（超过 2048 token）、多次重复的方差统计、其他量化档位、其他硬件，都不在这次范围内。

## 结论

**能跑通，速度可用。** 非投机解码 20.16 tok/s，每 token 49.61 ms，权重常驻 94.69 GiB。语义冒烟通过：算术、格式指令、EOS 终止都正常。

**MTP 投机解码是负收益。** 机制能跑起来，但速度掉到 13.42 tok/s，比不开还慢 33%。原因不在模型，也不在硬件，出在 vLLM 0.23.0 对 drafter cudagraph 的一条限制上，后面细说。

## 实验对象

| 项 | 配置 |
|---|---|
| 模型 | `iggerask/DeepSeek-V4-Flash-2bit-GB10`，基于 DeepSeek-V4-Flash 299B MoE，revision `df5ebc8` |
| 量化方式 | 专家 2-bit VQ（k=1024、vdim=4、group-64 Hadamard），attn/shared/lm_head 用 NVFP4 W4A16，路由与嵌入保持 BF16，KV cache fp8 |
| 服务插件 | 同仓的 `vq2_vllm`，经 vLLM 的 general_plugins 入口自动注册，不修改 vLLM 本体 |
| 硬件 | DGX Spark，GB10，20 核 ARM64，121.6 GiB 统一内存，sm_121 |
| 软件 | Ubuntu 24.04，CUDA 13.0.88，驱动 580.173.02，torch 2.11.0+cu130，vLLM 0.23.0，Python 3.12 |
| 上下文 | 2048 token |
| 采样 | temperature 0，greedy |
| 指标 | decode tok/s（用 diff 方法扣掉 prefill）、常驻内存、finish_reason |

版本必须锁死。组件换一个版本，下面的数字就不能直接比：

| 组件 | 版本 |
|---|---|
| 模型检查点 | revision `df5ebc8fa0e70f36542ca710fcd5e2ac24fae0bb`，2026-06-24 |
| 插件 | master（含 mhc_post 的 sm_121 torch-reference 补丁） |
| vLLM | 0.23.0，原版，非 fork |
| torch / CUDA / 驱动 | 2.11.0+cu130 / 13.0（V13.0.88）/ 580.173.02 |

## 三个环境门槛

跑通的成本不在算力，在三个环境细节上。任何一个没过，实验就卡在最后一步。

### 一、JIT 编译的静默三分钟

插件的 `vq2_grouped.cu` 在首次推理前要由 torch cpp_extension 现场编译，走 nvcc，gencode 是 sm_121a，大约三分钟。这段时间进程看起来和卡死没区别：日志停在权重加载完成那行不动，GPU 在算，CPU 却很闲。

更麻烦的是编译中途被打断。`~/.cache/torch_extensions/py312_cu130/vq2_grouped_d0/lock` 会留下来，之后每个新进程都卡在这个 FileBaton 上空等。用 faulthandler 注册 SIGUSR1 打线程栈，能直接看到它停在 `file_baton.wait`。解法是删掉这个 lock 文件，编译好的 `.so` 还在缓存里，重启后 ninja 会跳过。

### 二、统一内存的预算边界

94.69 GiB 权重，加上 KV cache、cudagraph、flashinfer autotune 的临时内存，在默认 `gpu_memory_utilization=0.92` 下峰值超过 121 GiB 上限，被 OOM killer 直接 SIGKILL。进程没有 Traceback，只能从 `/proc/pressure/memory` 的停滞数据看出来是内存问题，不是崩在代码里。

把 util 降到 0.85、上下文从 4096 压到 2048 之后稳定。vLLM 自己的 KV 检查会给出准确引导值：2048 需要 0.83 GiB，实际可用 0.95 GiB。

### 三、MTP drafter 的跨流崩溃

开 `--spec-mtp 2` 之后，cudagraph capture 必然崩在 drafter 的 shared_experts 上，报 CUDA illegal memory access。

根因是这样：当 token 数不超过 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` 时，shared experts 默认走一条辅助 CUDA stream，和 router 重叠执行。capture 期间这个跨流操作在 sm_121 上是非法访存。把阈值设成 0，强制它回主流执行就好了。

这个开关不影响 target 的解码速度。不开 MTP 的两组，一组带开关（20.16），一组不带（20.39），差异在误差范围内。

## 测量数据

两组共享同一套配置：util 0.85、2048 上下文、temperature 0、单流、diff 方法计时（`time(gen=N+1) - time(gen=1)`，prefill 不计入）。唯一变量是 spec-mtp。

| 组 | decode tok/s | ms/tok | 连贯性 |
|---|---|---|---|
| 非投机（K=0） | **20.16** | 49.61 | Paris + 俳句，finish_reason=stop |
| MTP K=2 | **13.42** | 74.52 | Paris + 三行俳句，finish_reason=stop |

权重加载：45 个分片，共 94.69 GiB，耗时 41 到 101 秒，取决于 JIT 缓存状态。

## MTP 为什么是负收益

发布方公布的数字是：非投机 17.4 tok/s，MTP K=2 22.5 tok/s，也就是 MTP 带来 +29%。我们实测是反向的：20.16 掉到 13.42，−33%。差异链条完整可查。

第一，提速杠杆靠 `VQ2_DRAFTER_FULL_CG=1`，让 drafter 走完整的 cudagraph。这个补丁默认关闭，drafter 默认走 vLLM 的 PIECEWISE 模式，每步在 attention 处断图，插件注释自己写了这条路径的 draft 是 182 ms 一步的量级。

第二，我们显式打开这个开关后，vLLM 0.23.0 拒绝了。日志写得很明确：

```
CUDAGraphMode.FULL is not supported with DeepseekV4IndexerBackend backend
(support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY
```

drafter 还是落回 PIECEWISE。

第三，投机的一步等于 target 验证加 drafter 起草。74.52 ms 里，drafter 断图的开销把接受率带来的收益吃干净之后还倒贴。不开 MTP 的那组不受影响，因为 PIECEWISE 只作用在 drafter 身上。

这是版本行为差异，不是权重坏了，也不是硬件问题。发布方的报告停在 2026-06-24，当时锁的依赖组合和我们不一样：我们因为国内网络改用镜像源的同版本 wheel，还修了 lock 文件里三处内在矛盾。具体是哪个组件导致这个差异，没有定位。

我们也试过用 `enforce_eager` 绕开 capture：engine 初始化正常，但第一个请求之后就再没产出，二十分钟里 Triton 缓存静止、CPU 空闲，判定为挂死，没采用。

## 数字怎么读

20.16 和 13.42 可以直接相减。两组配置完全相同，唯一变量是 spec-mtp，涉及开关、上下文、采样方式都一致，差值成立。

发布方的 17.4 和 22.5 不能和我们这两个数混算。那是另一台机器的状态、另一套依赖组合，只作量级参照。

早期那个 20.39 也不是对照组。它缺 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`，只用来证明这个开关对 target 没有影响，不作为基线。

## 适用范围

这次实验支持三个判断。

2-bit VQ 加统一内存这条路，确实能让 299B MoE 在单卡上以 20 tok/s 交互。代价是 95 GiB 常驻把 KV 预算挤得很紧，要用长上下文就得下调 util 或者缩短窗口。

这个检查点的门槛在环境细节上，不在算力。JIT 锁、内存预算、跨流开关，这三样任意一个没处理好，实验就起不来。

MTP 在 vLLM 0.23.0 加 DS4 indexer backend 这个组合下用不了，得等上游放开 FULL cudagraph 支持，或者插件自己适配。

不支持外推到：其他量化档位、其他硬件、长上下文场景，以及发布方那套固定依赖下的 MTP 表现。

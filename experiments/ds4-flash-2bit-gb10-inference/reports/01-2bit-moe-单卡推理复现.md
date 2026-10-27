# 2-bit 299B MoE 单卡推理复现报告

**职责声明**：本报告是本实验唯一被引用作结论的文档。它记录 2026-10-27 在一台 DGX Spark 上对 iggerask/DeepSeek-V4-Flash-2bit-GB10 检查点的完整复现：环境安装、101 GB 权重加载、非投机与 MTP K=2 两组解码测量。它不覆盖：长上下文（>2,048 token）场景、多次重复的方差统计、其他量化版本或硬件。后续引用本实验数字，以本报告与 `docs/evidence/` 为准。

## 结论先行

这个 2-bit 检查点在单台 GB10 上**能跑通，且速度可用**：非投机解码 20.16 tok/s（49.61 ms/tok），权重常驻 94.69 GiB，语义冒烟通过（算术、格式指令、EOS 终止全部正常）。对作者口径的偏差只有 +16%（20.16 vs 17.4），属同量级。

MTP 投机解码是本次复现的主要负面发现：机制可运行，但在本环境是净负收益（13.42 tok/s，−33%）。根因不在模型、不在硬件，而在 vLLM 0.23.0 对 drafter cudagraph 模式的一条限制，详见下文。

## 部署：三个环境门槛

复现的真正成本不在算力，在三个环境细节。任一不过，实验停在前夜。

**torch JIT 冷启的静默编译**。vq2 插件的 `vq2_grouped.cu` 在首次推理前由 torch cpp_extension 现场编译（nvcc，sm_121a gencode，约 3 分钟）。这期间进程表现与卡死无异：日志停在权重加载完成、GPU 有计算活动、CPU 低占用。更危险的是编译未完成时中断进程，`~/.cache/torch_extensions/py312_cu130/vq2_grouped_d0/lock` 残留，后续所有新进程死等这个 FileBaton（用 faulthandler 注册 SIGUSR1 dump 堆栈可定位到 `file_baton.wait`）。解法是删除 lock；编译产物 `.so` 已在缓存，重启后 ninja 直接跳过。

**统一内存的预算边界**。94.69 GiB 权重 + KV cache + cudagraph + flashinfer autotune 的临时内存，在默认 `gpu_memory_utilization=0.92` 下峰值超过 121 GiB 系统上限，被 OOM killer 静默 SIGKILL（无 Traceback，PSI memory 停滞数据可佐证）。降到 0.85 并把上下文从 4,096 压到 2,048 后稳定；vLLM 的 KV 检查会给出精确引导值（2,048 需 0.83 GiB，可用 0.95 GiB）。

**MTP drafter 的跨流崩溃**。`--spec-mtp 2` 的 cudagraph capture 必然崩在 drafter forward 的 shared_experts：CUDA illegal memory access。根因是 shared experts 在 token 数不超过 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` 时默认走辅助 CUDA stream 与 router 重叠执行，而 capture 期间该跨流操作在 sm_121 上非法访存。设 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0` 强制主流内执行即解。该开关对 target decode 无影响：非投机组在同开关下 20.16 tok/s，与早期无开关测量 20.39 tok/s 在误差内。

## 测量数据

两组共享 0.85 util、2,048 上下文、temperature 0、单流、diff 方法计时（time(gen=N+1) − time(gen=1)，prefill 不计入），唯一变量是 spec-mtp：

| 组 | decode tok/s | ms/tok | coherence |
|---|---|---|---|
| 非投机（K=0） | 20.16 | 49.61 | Paris + 俳句，finish_reason=stop |
| MTP K=2 | 13.42 | 74.52 | Paris + 三行俳句，finish_reason=stop |

权重加载：45 个分片，94.69 GiB，41–101 秒（取决于 JIT 缓存状态）。

## MTP 负收益的根因链

作者的 MTP K=2 是 +29%（17.4 → 22.5），本复现是 −33%（20.16 → 13.42）。差异链条完整可查：

1. 作者的速度杠杆是 `VQ2_DRAFTER_FULL_CG=1`，让 MTP drafter 走 FULL cudagraph。该补丁默认关闭，drafter 默认走 vLLM 的 PIECEWISE（每步在 attention 处断裂图，插件注释自述该路径 draft 约 182ms/步量级）。
2. 本环境显式启用 `VQ2_DRAFTER_FULL_CG=1` 后，vLLM 0.23.0 拒绝该组合，日志明确记录：`CUDAGraphMode.FULL is not supported with DeepseekV4IndexerBackend backend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY`。drafter 仍落回 PIECEWISE。
3. spec 每步 = target verify + drafter draft。74.52 ms/tok 中 drafter 的 graph-break 开销吃掉了 MTP 接受率的全部收益后仍倒贴。非投机组不受影响（PIECEWISE 只作用于 drafter）。

这是 vLLM 版本行为差异，不是权重损坏或硬件问题。作者报告发布于 2026-06-24，其实验的 pinned 依赖组合与本复现不同（本复现因国内网络改用镜像源的同版本 wheel，并修正了 lock 文件三处内在矛盾）；差异具体来自哪个组件未定位。enforce_eager 绕开 capture 的路线也试过：engine 初始化正常，但首个请求后 20 分钟无产出、Triton 缓存静止、CPU 空闲，判定挂死，未采用。

## 测量陷阱单列

**两组数字可比，第三方数字不可直接相减**。20.16 与 13.42 同配置（唯一变量 spec-mtp、同 nostream 开关、同上下文与采样），差值成立。作者口径 17.4/22.5 来自不同机器状态与依赖组合，只作量级参照，不与本文数字混算差值。

**早期无开关的 20.39 不是对照组**。它缺 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`，仅用于证明该开关对 target 无影响，不作正式基线。

## 结论的适用范围

本实验支持三个判断：其一，2-bit VQ + 统一内存让 299B MoE 在单卡以约 20 tok/s 可交互，代价是 95 GiB 常驻挤占 KV 预算，长上下文需下调 util 或缩短窗口；其二，该检查点的复现门槛在环境细节（JIT 锁、内存预算、跨流开关），不在算力；其三，MTP 在 vLLM 0.23.0 + DS4 indexer backend 组合下不可用，等待上游放开 FULL cudagraph 支持或插件适配。

本实验不支持外推到：其他量化档位、其他硬件、长上下文场景、以及作者 pinned 环境上的 MTP 表现。

# 299B MoE 在单台 DGX Spark 上：两种推理配置

> **阅读导航**：[实验入口](../../../experiments/moe-2bit-fit/README.md) · [DGX Spark 推理](DGX-Spark大模型推理与并发.md)

2026-10-27，我们在 DGX Spark 上运行 `iggerask/DeepSeek-V4-Flash-2bit-GB10`，观察 2-bit VQ 检查点是否能加载，以及启用 MTP 的配置在短上下文单请求下有多快。模型以极低位宽压缩专家权重，换取将大型 MoE 放入统一内存的可能；运行时仍需为 KV cache、图捕获与其他分配留出空间。

## 测量结果

两组均使用 vLLM 0.23.0 与 vq2 插件，最大上下文 2048、KV FP8、内存配额 0.85、temperature 0、单请求，使用两次生成耗时之差估算 256 个解码步骤。

| 配置 | 解码速度 | 每 token 时间 | 日志中的模型加载分配 |
| --- | ---: | ---: | ---: |
| 非投机，FR-Spec 关闭 | 20.16 tok/s | 49.61 ms | 94.62 GiB |
| MTP K=2，FR-Spec 开启 | 13.42 tok/s | 74.52 ms | 96.86 GiB |

两次都完成一次巴黎地名与俳句提示并正常结束。该冒烟测试不能证明模型在其他任务上的准确率。加载分配来自引擎日志，不等于权重文件大小，也不是完整系统峰值。非投机摘要另记 94.69 GiB，与该组日志中的 94.62 GiB 不一致，表中采用日志值。

MTP 配置比非投机配置慢约 33%。**这比较的是两套配置**：MTP 组还启用了 FR-Spec 草稿输出头补丁，因此不能把差额单独归因于 MTP。两组各只有一次记录；没有重复测量、显著性或不同长度的稳定性测试。

## CUDA Graph 日志说明了什么

MTP 组日志报告 `CUDAGraphMode.FULL` 不受 `DeepseekV4IndexerBackend` 支持，配置降为 `FULL_DECODE_ONLY`。这能说明出现了图模式降级，但没有分项的 draft、target 验证和调度耗时，也没有成功启用完整图的对照，不能确定该降级是速度下降的根因。接受率、草稿开销和其他内核路径都可能影响最终速度。

发布方记录的速度来自另一套环境，不能与本次差值拼接为同条件 A/B。要回答图模式或 FR-Spec 的贡献，需要保留相同输入、输出长度、模型、软件版本及资源占用，在可运行配置内分别开关，记录逐阶段时间并多次重复。

## 环境与复现

模型 revision 为 `df5ebc8fa0e70f36542ca710fcd5e2ac24fae0bb`，软件为 CUDA 13.0、PyTorch 2.11.0+cu130、vLLM 0.23.0，单台 GB10。插件记录为 master 而非固定 commit，因此只能按当时记录重建，后续版本不保证一致。[实验首页](../../../experiments/moe-2bit-fit/README.md)保留安装与运行命令。

首次运行涉及 CUDA kernel JIT、模型加载与内存预留。现有记录显示，设置 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`、2048 上下文与 0.85 配额后两组完成；这不保证其他版本或负载也具备相同内存余量。若编译锁残留，应先确认没有编译进程在使用该锁，再根据实际缓存路径处理。

## 原始证据与边界

- [非投机输出](../../../experiments/moe-2bit-fit/evidence/bench-output-nospec.txt) 和 [MTP 输出](../../../experiments/moe-2bit-fit/evidence/bench-output-mtp2.txt)：配置、加载分配、冒烟文本和计时。
- [非投机摘要](../../../experiments/moe-2bit-fit/evidence/nospec-result-20261027.jsonl) 和 [MTP 摘要](../../../experiments/moe-2bit-fit/evidence/mtp2-result-20261027.jsonl)：原有实验归档；其中关于原因的字段是当时判断，应以本报告的证据边界为准。

没有测量长上下文、多并发、准确率基准、功耗、其他量化或硬件。当前结果说明两套配置在该机器上可以生成，以及该次速度和加载分配；不能据此推断生产稳定性或其他环境的速度。

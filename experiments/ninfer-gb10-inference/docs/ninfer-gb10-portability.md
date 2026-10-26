# NInfer 在 GB10 上的可移植性与实际验证

## 问题不是模型文件，而是引擎支持边界

V3 `.ninfer` 只能由 NInfer 读取，SGLang、vLLM、llama.cpp 和 Transformers 不能直接使用该容器。当前官方 `64d10fb` 在 CMake 中只接受 `CMAKE_CUDA_ARCHITECTURES=120a`，传入 GB10 的 `121a` 会实际报：

```text
NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '121a'
```

这只是第一层阻塞。源码审查还确认：

- Qwen3.5 启动规划阶段要求 compute capability 12.0；
- x86 CPUID、`xgetbv`、AVX2/AVX512 主机路径需要与 AArch64 拆分；
- 统一内存不能简单用裸 `cudaMemGetInfo` 当作可安全分配预算。

因此，“把 120a 改成 121a”不是完整修复。与此同时，GB10 fork 与上游后续补丁证明这条移植路线有实际基础，不能反向推断 NInfer 永远不可用。

## 为什么选择固定 fork

本实验使用 `baristahaus/ninfer-gb10` commit `4c2fc2f9504d2f373c84e896bf18523906a646d7`，不跟随浮动 master。该版本允许 `120a` 和 `121a`，并补上了 Qwen3.5 的 12.1 启动门禁，也包含 V3、MTP、DFlash2 和 Vision 的测试入口。

当前公开资料中，GB10 的 Flash-Next 有真实运行记录，但这不能自动证明 Qwen3.8 27B + DFlash2 + Vision 也已通过。本实验因此重新执行构建和全部目标路径，不引用 fork 的兼容性声明代替实测。

## 构建环境与处理

构建依赖包括 CMake、GCC/G++、CUDA、Ninja、FFmpeg 开发文件和 libcurl。配置阶段有两个可复现细节：

1. 不把 libcurl 前缀全局塞进 `LD_LIBRARY_PATH`，否则系统 CMake 会误加载另一套 libcurl；只在运行 NInfer 时设置该变量。
2. 使用系统 `/usr/bin/pkg-config`，避免部分 conda 实现把系统 zlib 头目录解析为不存在的路径。

配置脚本为 `scripts/ninfer_gb10_configure.sh`，构建使用 `-j1`。完整配置与 898/898 个目标均已完成。

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

这些结果说明固定 fork 已能运行该模型包，包括 MTP、DFlash2、Vision 和 CUDA Graph。短冒烟的速度只用于确认路径，不替代性能基准。正式性能对照见 `ninfer-screenshot-reproduction.md` 与 `docs/evidence/`。

## 仍然未知的边界

- 没有验证长上下文、持续 C8、全量精度、能耗或生产故障恢复。
- fork 是第三方移植，不代表官方已承诺同等的 GB10 支持。
- 未把其他 MoE 型号的结果混入本实验；它们需要独立的模型包与数值资格化。
- 真实多模态服务还需要独立的视觉压力与视频测试，当前只证明图片路径成立。

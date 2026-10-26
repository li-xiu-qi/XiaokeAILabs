# NInfer 在 DGX Spark GB10 上的部署与性能验证

本实验验证一个具体且容易被误判的问题，即 Qwen3.8 27B 的 W4A4 + DFlash2 + Vision 模型包能否在 DGX Spark 的 GB10 上真正运行，并形成可复现的高吞吐服务。验收不止于“能加载”，还包括真实生成、MTP、DFlash2、视觉、CUDA Graph、OpenAI 请求和 C1/C2/C4/C8 性能对照。

> **固定 GB10 fork 后，NInfer 已跑通全部上述路径。截图复现配置下 C8 为 132.1 tok/s；同模型 SGLang 参考为 144.6 tok/s，证明百 TPS 不是只能在其他硬件上出现。当前协议下，NVFP4 W4A4 也比 FP8 更快。**

## 为什么值得做这个实验

NInfer 官方当前源码把构建目标限定为 `sm_120a`，而 DGX Spark 是 `sm_121a`。如果只看这个错误，容易得出“模型不能在 DGX 上跑”的错误结论。真正需要分开以下四点。

1. 官方引擎当前是否直接支持 GB10；
2. 是否存在可验证的 GB10 移植路线；
3. 同一模型在 NInfer、SGLang 和 llama.cpp 下是否应采用不同的性能叙述；
4. DFlash2 接受率、量化格式、CUDA Graph 和内存配置如何影响实际速度。

这个实验的价值是给出真实构建、运行和测量证据，而不是只引用模型卡或社区数字。

## 实验对象与范围

| 项目 | 配置 |
| --- | --- |
| 模型 | Qwen3.8-27B Coder390 W4A4 + MTP + DFlash2 + Vision，V3 `.ninfer` |
| 模型 SHA256 | `3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808` |
| ModelScope 固定 revision | `196cc2ce161d7239976fcd5dc07adf06ae31a35c` |
| NInfer 移植 | `baristahaus/ninfer-gb10`，commit `4c2fc2f9504d2f373c84e896bf18523906a646d7` |
| SGLang 对照 | 同模型仓的 `NVFP4/W4A4/` 与 `FP8/` |
| 硬件 | DGX Spark，NVIDIA GB10，aarch64，统一内存 |
| 已验证软件 | CUDA 13.0、NInfer GB10 fork、SGLang 0.5.19、PyTorch 2.13.0+cu130 |
| 性能负载 | C1/C2/C4/C8 各 2 轮，512 token 上限，xhigh 思考 |

本实验**不覆盖** 35B/125B MoE、全量 GPQA/MMLU/LCB、长上下文、持续并发稳定性、能耗和生产高可用。小样本质量探针不等于完整精度测评，截图所属硬件与提示不在本实验控制范围内。

## 我们具体做了什么

- 复现官方 `64d10fb` 的 `120a` 构建拒绝，并检查 Qwen3.5 运行时门禁、x86 主机代码和统一内存约束。
- 使用固定 GB10 fork 完成 898/898 个构建目标，包括 CLI、server、测试与 benchmarks。
- 校验 V3 的 SHA256 和内部对象，确认 text、MTP、DFlash2、Vision 四个组件真实存在。
- 运行文本生成、MTP prefix、DFlash2 real、Vision prefix、两个 repro、默认 CUDA Graph 和 OpenAI 文本与图片请求。
- 用 draft 7、xhigh、512 token 复现截图中的 NInfer 配置，并保留逐请求原生 timings 与草稿接受率。
- 在 SGLang 上对同一模型做 FP8 与 NVFP4 的 eager/decode graph 对照，检查 FP8“必然最优”的说法是否在本机成立。

## 发现了什么

### 1. 官方直接支持受限，但 GB10 fork 已能跑通这个模型包

固定 fork 的构建和真实运行均通过。官方 `64d10fb` 的架构拒绝不是模型文件损坏，也不意味着 GB10 永远无法运行；它需要实际的 ARM64、运行时门禁和统一内存移植，而不是简单改一行 CMake。

完整 CTest 初次有 142/144 通过，剩余 2 项只是未提供 real-artifact 环境变量，带上 V3 后均重跑通过。其他 real tests 的跳过不冒充通过。

### 2. 百 TPS 可以复现，约 22 tok/s 是另一套实验

| 并发 | 外部截图 | 本实验 NInfer（draft 7） | 本实验 SGLang 参考（draft 8） |
| ---: | ---: | ---: | ---: |
| C1 | 48.3 | 32.8 | 31.2 |
| C2 | 76.9 | 58.4 | 63.7 |
| C4 | 97.7 | 95.2 | 99.0 |
| C8 | 146.8 | 132.1 | 144.6 |

约 22 tok/s 的历史结果来自 UD-Q4_K_XL GGUF + llama.cpp + MTP2 + 关闭 thinking，与这里的 NInfer W4A4 + DFlash2 + xhigh 不是同一个模型容器、引擎或投机路径，不能用来否定本结果。

本次 NInfer 的 DFlash2 接受率为 C1 33.95%、C2 36.62%、C4 36.71%、C8 32.57%；截图为 61.8%、52.8%、41.2%、44.6%。接受率与任务文本分布密切相关，因此不能只看总 TPS 判断引擎优劣。截图与 SGLang 的 C8 只差约 1.5%，但截图的硬件与提示不受本实验控制。

### 3. 当前 SGLang 协议下，NVFP4 W4A4 比 FP8 更快

| 并发 | NVFP4 eager | FP8 eager | NVFP4 decode graph | FP8 decode graph |
| ---: | ---: | ---: | ---: | ---: |
| C1 | 32.7 | 22.9 | 31.2 | 20.0 |
| C2 | 55.2 | 45.7 | 63.7 | 42.1 |
| C4 | 96.9 | 68.5 | 99.0 | 68.5 |
| C8 | 132.5 | 106.1 | 144.6 | 106.9 |

四个并发点、两种图配置均未出现 FP8 反超。 decode graph 对 NVFP4 的 C2/C8 有帮助，对 FP8 并非普遍加速。草稿接受率并没有解释全部差距，后续仍需 kernel 级 profiling 才能归因到具体计算或读取路径。

### 4. 小样本质量与视觉没有分出胜负

两档均通过文本算术、图片形状/颜色/文字识别、10 道确定性自动题和 2 道编程题。自动题为 10/10，编程题语法为 2/2，人工审查后每题 5 个功能用例也全部通过。该结果只说明没有在这组小探针上暴露明显质量差，不证明全量精度相同。

## 复现

### 依赖

需要 DGX Spark 或兼容的 GB10 环境，以及 CMake ≥3.28、GCC/G++、CUDA、Ninja、FFmpeg 开发文件、libcurl ≥7.85、Python 3 和 SGLang 0.5.19。Python 下载与图片依赖安装在实验自己的环境中。

```bash
cd experiments/ninfer-gb10-inference
python -m pip install -r requirements.txt
```

三类模型文件都从 ModelScope 的固定 revision `196cc2ce161d7239976fcd5dc07adf06ae31a35c` 获取，只拉取本实验需要的目录。

下载后，脚本按模型仓的 `SHA256SUMS` 逐文件校验。NVFP4 与 FP8 的校验范围同时覆盖各自的 `DFlash2-FP8` 子目录。

NInfer `SHA256SUMS` 清单里的可选 V2 包不属于本实验依赖，无需下载。

```bash
python scripts/download_models.py ninfer --model-root ./models
python scripts/download_models.py nvfp4 --model-root ./models
python scripts/download_models.py fp8 --model-root ./models
```

已有本地文件时，在对应命令后加 `--verify-only` 即可跳过下载并复核，NVFP4 的写法如下。

```bash
python scripts/download_models.py nvfp4 --model-root ./models --verify-only
```

### 构建 GB10 fork

```bash
git clone https://github.com/baristahaus/ninfer-gb10.git
cd ninfer-gb10
git checkout --detach 4c2fc2f9504d2f373c84e896bf18523906a646d7
cd ..

export NINFER_SRC=$PWD/ninfer-gb10
bash scripts/ninfer_gb10_configure.sh "$NINFER_SRC"
cmake --build "$NINFER_SRC/build-gb10-system-pkg" -j1
```

`MAX_JOBS=1` 是统一内存机器上的重要保护措施。并行 `cicc` 的瞬时内存可能远高于模型权重本身。不要把“模型显存配额够”当作“JIT 编译也安全”。

### 启动 NInfer 并做文本与图片冒烟

```bash
export NINFER_ARTIFACT=$PWD/models/NVFP4-NInfer/W4A4/Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer
bash scripts/ninfer_gb10_serve.sh start
python scripts/ninfer_openai_smoke.py --output artifacts/smoke-ninfer-server.json
```

默认服务为 32K context、262144 KV、最大 8 并发、DFlash2 draft 7、Vision 和 CUDA Graph。可用环境变量 `NINFER_MAX_CONTEXT`、`NINFER_KV_CAPACITY`、`NINFER_MAX_CONCURRENCY`、`NINFER_DRAFT_TOKENS` 调整。

### 复现截图性能

```bash
python scripts/bench_ninfer_concurrency.py \
  --base-url http://127.0.0.1:19410/v1 \
  --engine "NInfer GB10 fork 4c2fc2f" \
  --context-total 262144 \
  --enable-thinking --reasoning-effort xhigh \
  --model qwen3.8-27b-nvfp4-dflash2 \
  --levels 1,2,4,8 --rounds 2 --max-tokens 512 \
  --output artifacts/bench-ninfer-w4a4-dflash2-draft7-xhigh.json
```

运行以下命令停止服务。

```bash
bash scripts/ninfer_gb10_serve.sh stop
```

### 统一脚本完成 SGLang 四组性能对照

设置模型根目录和包含 SGLang 的 Python，再运行统一对照脚本。

```bash
export SGLANG_MODEL_ROOT=$PWD/models
export SGLANG_PYTHON=/path/to/venv/bin/python
bash scripts/run_sglang_comparison.sh
```

脚本依次运行 NVFP4 与 FP8 在 eager 和 decode graph 下的四组性能对照。

仅两个 eager 阶段额外运行文本与视觉冒烟、12 道质量题，并对其中 2 道编程题运行功能用例。

结果写入 `artifacts/`。四个 bench JSON 分别对应两档权重和两种图配置，smoke、quality 与 code-functional JSON 只在 eager 阶段生成。

汇总结果同时保存为 Markdown 与 JSON。

```text
artifacts/
├── bench-nvfp4-dflash2.json
├── bench-fp8-dflash2.json
├── bench-nvfp4-dflash2-graph.json
├── bench-fp8-dflash2-graph.json
├── smoke-nvfp4-dflash2.json
├── smoke-fp8-dflash2.json
├── quality-nvfp4-dflash2.json
├── quality-fp8-dflash2.json
├── code-functional-nvfp4-dflash2.json
├── code-functional-fp8-dflash2.json
├── comparison-eager.md
├── comparison-eager.json
├── comparison-graph.md
└── comparison-graph.json
```

### 已有服务可直接重跑质量与功能检查

下面以 NVFP4 为例。服务已启动且 `SGLANG_PYTHON` 已设置时，先重新生成原始 quality JSON，再基于它生成编程功能结果。

```bash
"$SGLANG_PYTHON" scripts/quality_probe.py \
  --base-url http://127.0.0.1:19410/v1 \
  --model qwen3.8-27b-coder390-nvfp4-w4a4-dflash2 \
  --output artifacts/quality-nvfp4-dflash2.json

"$SGLANG_PYTHON" scripts/verify_code_functional.py \
  --quality artifacts/quality-nvfp4-dflash2.json \
  --output artifacts/code-functional-nvfp4-dflash2.json
```

功能脚本只执行通过 AST 白名单和受限 builtins 检查的模型函数，并在带资源限制的子进程中运行固定用例。

## 文档与证据

- `docs/ninfer-gb10-portability.md` 记录官方阻塞、fork 选择、构建和真实运行结果。
- `docs/sglang-fp8-vs-nvfp4-results.md` 记录 SGLang 对照设计、接受率和结果解释。
- `docs/ninfer-screenshot-reproduction.md` 说明截图复现、约 22 tok/s 的口径差异与接受率。
- `docs/测量陷阱与边界.md` 说明统一内存、JIT、KV 配置、采样与不可外推范围。
- `docs/evidence/` 保存环境、模型校验、逐请求性能、质量摘要和 OpenAI 响应。公开时只移除主机标识与路径，不修改数值。

模型与引擎各自许可证仍适用。本实验是第三方 GB10 移植与性能验证，不代表 NInfer 官方已经发布同等的 GB10 支持。

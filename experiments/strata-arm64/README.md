# Strata 在 DGX Spark 上的统一内存推理实验

[English](README.en.md)

## 要回答的问题

在同一台 DGX Spark（GB10，统一内存）上，对比两套运行 Qwen3.8-Flash-Next IQ3_S 的方案：

1. Strata 的 ARM64 社区移植能否在 GB10（aarch64、sm_121）上完成构建与推理；
2. Strata 与 llama.cpp 的本次配置分别达到多少生成速度；
3. 统一内存架构下，Strata 的 expert-cache 自动分配是否可靠，需要怎样的手动调优。

**口径说明**：Strata 的 IQ 原生包强制要求 MTP（`--spec >= 2`），llama.cpp 无法加载该模型的 MTP 架构，因此对比为"Strata 启用 MTP vs llama.cpp 无 MTP"，不是同条件开关投机的公平对照。同一模型族在 llama.cpp 上的三提示词矩阵见 [llm-inference-perf](../llm-inference-perf/README.md)，量化、引擎和 MTP 开关都不同。

## 实验对象

| 项 | 配置 |
|---|---|
| Strata 来源 | PR #3622，`ferstar/Strata` 的 `gb10-022` 分支 |
| 固定提交 | `d891f006aaec64024513a1e55856afd992baaa94` |
| 模型 | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`，IQ3_S（83.62 GB，2 个 GGUF 分片） |
| 硬件 | DGX Spark，GB10，20 核 ARM64，121 GiB 统一内存，sm_121 |
| 软件 | Ubuntu 24.04，CUDA 13.0，源码构建 |
| 上下文 | 8,192 token，单请求，关闭视觉 |
| 采样 | temperature 0.7，max_tokens 按场景 20-200 |
| Strata 投机 | 既有记录为 `--spec 4 --spec-min-p 0.5 --mtp <path>`；与现存启动日志的 `spec=6` 不一致，实际基准参数待核对 |
| 对照组 | llama.cpp 0.4.0-dev（commit ccc3646），`-ngl 40 -t 8 -c 8192`，无 MTP（架构不支持该模型 MTP） |

## 版本锁定

以下版本信息用于确保实验可追溯。后续复现时若组件版本不同，结果可能不可直接比较。

| 组件 | 版本标识 |
|------|----------|
| Strata 引擎 | 0.1.47（PR #3622, commit `d891f006aaec64024513a1e55856afd992baaa94`） |
| llama.cpp | 0.4.0-dev（build 1, commit `ccc3646`），GNU 13.3.0，Linux aarch64，构建目录 `build-force-cublas` |
| 模型分片 1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`，SHA-256: `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| 模型分片 2 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`，SHA-256: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| CUDA | 13.0（`/usr/local/cuda/bin/nvcc`） |
| 实验日期 | 2026-10-26 |

## 协议

9 条请求包含预热、短输入、较长输入和重复请求，记录引擎报告的 prefill/decode tok/s、输出 token 数、完成原因及 Strata 草稿接受情况。两个引擎的部分生成长度不同，多条输出达到长度限制，不能用这些请求评价完整答案质量；最长输入实际为 131 token。

## 环境

DGX Spark（GB10，121 GiB 统一内存，273 GB/s），Strata 源码构建（aarch64，CUDA 13.0），llama.cpp `build-force-cublas`。测试前停止其他大模型服务，避免资源竞争。

## 状态

- [x] 引擎构建（2026-10-26）：aarch64 + CUDA 13.0 + sm_121，strata 0.1.47
- [x] 模型部署：IQ3_S 83.62 GB，SHA-256 校验通过
- [ ] 语义正确性：现存基准记录中多条输出被截断；17×23 的原始结果尚未找到，不能据此判定两引擎通过语义测试。
- [x] Strata 基准：生成 20.5-47.3 tok/s，prompt 49.1-166.3 tok/s，MTP 接受率 50-70%
- [x] llama.cpp 对照：生成 11.0-16.0 tok/s，prompt 21.3-83.2 tok/s
- [ ] 内存安全：既有记录称最低可用 61 GiB、无 mem-guard 事件，逐时采样证据尚待核对。
- [x] [对比报告](../../docs/zh/inference/Strata与llama.cpp在GB10上的推理对照.md)：8 条非预热请求的逐条速度比约 1.6–3.0，算术平均约 2.4；部分生成长度不同，仅反映本次请求和配置。
- [x] Qwen3.6-35B-A3B 补充测试（2026-10-26 22:22）：UD-Q4_K_XL，Strata 生成速度稳定 51.0-55.1 tok/s
- [x] Qwen3.6 跨引擎补充测试（2026-10-26 22:28）：llama.cpp + MTP 为 65.5–87.0 tok/s；这组与 Strata 的投机配置不同。

原始数据在 [evidence/](evidence/)（逐请求 JSON，含 timings 与 draft 统计；Qwen3.6 结果见 `qwen36-results.jsonl` 与 `llama-qwen36-results.jsonl`）。

## 关键发现

1. **两套配置在本机的生成速度不同**：Qwen3.8 的 8 条非预热请求中，Strata 启用 MTP 的逐条生成速度高于 llama.cpp 无 MTP 配置。由于引擎和投机解码同时变化，不能把差异单独归因于专家缓存或 MTP。
2. **Qwen3.6 的补充测试呈现相反方向**：本次 llama.cpp 启用 MTP，Strata 记录的草稿数为零；两组配置的生成速度分别为 65.5–87.0 和 51.0–55.1 tok/s。它们是不同配置的观察，不足以推导普适的引擎选型。
3. **资源与配置还需核对**：GB10 使用统一内存，不能将不同分配类别直接解释为独立显卡所需容量。现存启动日志记录 `expert_slots=10701`、`expert_cache_mib=20798`、`spec=6`，与本文所列 4096 和 `--spec 4` 不一致；尚不能确认日志与基准属于同一运行。auto 模式 OOM、CPU 瓶颈和独显可运行范围的原始证据尚待补充。

## 内存统计的边界

启动日志记录加载了 46.84 GiB，但不足以核实原先列出的两引擎资源占用数字及测量口径。不同内存分类在统一内存机器上的含义需要结合采集方式重新确认，本实验未测独立显卡。

## 复现

准备脚本面向 Linux aarch64 和 CUDA，要求 `/usr/local/cuda/bin/nvcc` 可执行，并使用 Git、Python 3.10+（含 venv 和 pip）、GNU `timeout` 和 `tee`。构建 llama.cpp 还需要 CMake、C/C++ 编译器；基准脚本需要 `curl`。历史运行环境见上表；这些配置不代表最低要求。两份模型分片约 83.6 GB，还需为构建、转换产物和运行内存预留空间，准备前检查剩余磁盘及可用内存。

从仓库根目录进入实验目录，设置工作目录。以下使用 Hugging Face Hub 下载所需分片；已有分片可以跳过下载，但仍须校验哈希：

```bash
cd experiments/strata-arm64
export STRATA_WORK_ROOT="/path/to/strata-work"
python3 -m venv "$STRATA_WORK_ROOT/.venv"
"$STRATA_WORK_ROOT/.venv/bin/python" -m pip install -r requirements.txt
"$STRATA_WORK_ROOT/.venv/bin/python" -m pip check
"$STRATA_WORK_ROOT/.venv/bin/python" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", revision="ed59f92082b1e93c0e96d60a8b11aab089b52f09", allow_patterns="IQ3_S/*.gguf", local_dir=os.path.join(os.environ["STRATA_WORK_ROOT"], "models"))'
(
  cd "$STRATA_WORK_ROOT/models/IQ3_S"
  sha256sum -c <<'EOF'
4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
EOF
)
```

模型下载工具依赖已固定在 [requirements.txt](requirements.txt)；Strata 的安装脚本管理其自身环境与依赖。下载来源见[固定版本的模型仓库](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S)。上游文件若已变化，校验会失败，应检查来源，不要将新文件当作历史模型。确认可用空间后运行准备脚本：

```bash
bash scripts/prepare_dgx.sh
```

脚本检出上表固定的 Strata 社区版本、调整构建并发，并调用上游安装脚本。准备过程可能下载依赖或权重，运行前应检查上游脚本及可用空间；模型为上表所列的两个 GGUF 分片，不进本仓。准备日志写入 `$STRATA_WORK_ROOT/artifacts/prepare.log`。脚本在新工作目录直接检出固定提交，已有检出则要求 HEAD 一致，不会自动替换。`models/IQ3_S/` 存在时，脚本通过 `--gguf-dir` 交给上游使用；目录不完整时上游会报错，不会自动补齐。上游另需准备约 5 GB 的 MTP 草稿层和转换产物，应按上游提示预留空间。

准备完成后，先检查 `$STRATA_WORK_ROOT/upstream/strata-iq3_s.json` 的 `args` 数组。准备脚本建议将 `--expert-cache` 后的值设为 `4096`，可用文本编辑器修改并保存；这是待验证的配置，不等于已确认的历史基准参数。启动前检查可用内存，并确认端口 8890 空闲。然后在另一个终端启动服务，再从实验目录运行基准：

```bash
export STRATA_WORK_ROOT="/path/to/strata-work"
cd "$STRATA_WORK_ROOT/upstream"
./run-iq3_s.sh
```

```bash
bash scripts/run_benchmark.sh
```

llama.cpp 对照可按历史版本检出源码，并使用 CUDA 构建 `llama-server`。以下构建命令依据[对应提交的上游说明](https://github.com/ggml-org/llama.cpp/blob/ccc3646c63127e32ab8e2773290a1aded6dfb69c/docs/build.md)，只固定源码和示例选项；历史构建完整选项未留存，本命令尚未重跑：

```bash
git clone https://github.com/ggml-org/llama.cpp.git "$STRATA_WORK_ROOT/llama.cpp"
git -C "$STRATA_WORK_ROOT/llama.cpp" checkout --detach ccc3646c63127e32ab8e2773290a1aded6dfb69c
cmake -S "$STRATA_WORK_ROOT/llama.cpp" -B "$STRATA_WORK_ROOT/llama.cpp/build-cuda" \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build "$STRATA_WORK_ROOT/llama.cpp/build-cuda" --target llama-server --parallel 1
"$STRATA_WORK_ROOT/llama.cpp/build-cuda/bin/llama-server" --version
```

在资源足够、Strata 服务已停止且端口 8891 空闲的情况下，使用已校验的第一分片启动服务，第二分片保留在同目录。历史基准使用 `-ngl 40 -t 8 -c 8192`，但现有材料不足以确认全部启动与构建参数：

```bash
"$STRATA_WORK_ROOT/llama.cpp/build-cuda/bin/llama-server" \
  -m "$STRATA_WORK_ROOT/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  -ngl 40 -t 8 -c 8192 --host 127.0.0.1 --port 8891
```

服务就绪后，在另一个终端从实验目录执行下列命令。两组基准均使用 Bash、curl 和 Python 标准库，不需要额外 Python 包或外部数据；提示词内置于脚本。结果按秒级时间戳写入 `$STRATA_WORK_ROOT/artifacts/benchmark/`，同秒重跑可能追加到同一文件。

```bash
export STRATA_WORK_ROOT="/path/to/strata-work"
curl -fsS http://127.0.0.1:8891/health
bash scripts/run_llama_benchmark.sh
```

准备脚本提示将 `expert-cache` 设为 4096，但现存启动日志的缓存与投机参数和历史记录不一致，见上文「关键发现」。因此这些步骤提供运行入口，尚不能保证复现表中结果；特别是无法从现存证据确认上述 llama.cpp 构建配置与历史构建完全相同。

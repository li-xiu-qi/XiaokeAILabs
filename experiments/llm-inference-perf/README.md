# llm-inference-perf：本地大模型推理性能实验
[English version](README.en.md).


本实验在 GB10 上测试投机解码，覆盖 27B 稠密、125B MoE 和 35B MoE 三个模型、三个提示词及若干草稿配置。实验结果与适用条件见 [投机解码收益与实测条件](../../docs/zh/inference/投机解码收益与实测条件.md)。同一设备上的推理引擎及 1 槽、8 槽并发对照见 [engine-slot-concurrency](../engine-slot-concurrency/README.md)。两项实验的测试协议和提示词不同，结果不能直接合并比较。

---

## 相关实验

这三个实验分别关注语言模型的生成机制、训练过程和推理性能。

- [llm-mechanics](../llm-mechanics/README.md) 介绍单步生成过程，包括分词、采样、KV 缓存和量化。讲解在 [语言模型生成机制与 KV 缓存](../../docs/zh/inference/语言模型生成机制与KV缓存.md)。
- [llm-lifecycle-fullchain](../llm-lifecycle-fullchain/README.md) 展示一个小模型从分词、预训练到对齐的训练过程。讲解在 [大语言模型从零构建与训练全链路](../../docs/zh/inference/大语言模型从零构建与训练全链路.md)。
- 本实验测试 GB10 上三个模型、三个提示词的投机解码配置。讲解在 [投机解码收益与实测条件](../../docs/zh/inference/投机解码收益与实测条件.md)。

---

[scripts](scripts/) · [configs](configs/) · [log](log/) · [tasks](tasks/)

以下命令均从本实验目录运行。

## 环境准备

- 硬件：实验报告记录的是 NVIDIA DGX Spark（GB10、128 GB 统一内存）。实际需要的内存随模型、量化、上下文长度和缓存配置变化；在其他设备上的性能不能直接套用报告中的结果。
- 系统与工具：启动和矩阵脚本使用 Bash；矩阵脚本还需要 `curl`、`grep` 和 Python 3。`bench.py` 与结果解析脚本只使用 Python 标准库，无需安装第三方 Python 包。
- 推理服务：准备 CUDA 版 `llama-server`，并确认它支持所选配置使用的参数和投机解码模式。报告记录的版本为 llama.cpp commit [`a0e6dc7`](https://github.com/ggml-org/llama.cpp/commit/a0e6dc7)；构建可参考 [llama.cpp 构建文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。
- 模型：按所测配置准备 GGUF 主模型，MTP 或 DFlash2 配置还可能需要与主模型匹配的草稿权重。默认的 Flash-Next 主模型是三分片文件，需将所有分片放在同一目录，并将第一分片传给服务。运行前确认模型许可、磁盘空间和设备内存。

以下是与脚本中部分文件名对应的公开下载入口，供准备权重时参考；原始实验未记录权重的下载来源及哈希，文件名相同不能证明是同一份权重：

- Flash-Next IQ4_XS 主模型：[三分片文件](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-IQ4_XS)。
- Flash-Next 的 Q8_0 MTP 草稿头：[MTP 文件目录](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/MTP)。
- 27B 的 Q4_K_M DFlash2 草稿模型：[模型文件目录](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main)。27B 主模型及其他配置仍需另行准备。

例如，在本实验目录安装可选的下载工具，再将 Flash-Next 三片主模型和 MTP 草稿头下载到新目录。下载体积较大，请先检查磁盘与内存；两处来源不是固定 revision，下载后应记录 SHA-256：

```bash
python -m pip install -r requirements.txt
export MODEL_ROOT=/path/to/new-models
hf download bartowski/Qwen3.8-Flash-Next-GGUF --include 'Qwen3.8-Flash-Next-IQ4_XS/*' --local-dir "$MODEL_ROOT/flash-next"
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --local-dir "$MODEL_ROOT/flash-next-mtp"
sha256sum "$MODEL_ROOT"/flash-next/Qwen3.8-Flash-Next-IQ4_XS/*.gguf "$MODEL_ROOT"/flash-next-mtp/MTP/*.gguf
```

[requirements.txt](requirements.txt) 只包含下载工具，不影响标准库基准客户端；清单未锁定版本。这两份模型是供尝试的公开来源，尚不能确定与历史权重一致，也未完成新环境兼容性验证。

可以在安装 NVIDIA 驱动、CUDA Toolkit、C/C++ 编译器、CMake 和 Git 后，参照上游构建文档尝试构建该提交的 CUDA 服务程序。在新的工作目录运行：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout --detach a0e6dc7
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release --target llama-server -j 4
./build/bin/llama-server --version
./build/bin/llama-server --help
```

本目录未记录原始构建命令、CUDA 版本、完整操作系统版本和权重哈希。上述构建及下面的脚本命令是重跑入口，尚不足以严格复现报告中的整组实验；构建后需确认服务支持配置中的投机参数，选择不同版本或权重时需重新验证兼容性和测量结果。

## 复现命令

构建完成后回到 `experiments/llm-inference-perf` 目录，将路径替换为准备好的文件，并在启动服务的终端设置：

```bash
export LLAMA_BIN=/path/to/llama-server
export MODEL_DIR="$MODEL_ROOT/flash-next/Qwen3.8-Flash-Next-IQ4_XS"
export MAIN_MODEL=Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf
export DRAFT_MODEL="$MODEL_ROOT/flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
```

`MAIN_MODEL` 是 `MODEL_DIR` 下的文件名；`DRAFT_MODEL` 是草稿文件的路径，基线模式不需要它。以上路径是占位示例，不会自动下载模型。`start-server.sh` 与 `run-matrix.sh` 的 `LLAMA_BIN` 默认路径不同，运行任一脚本时都应显式指定实际使用的服务程序。

选择一种服务模式启动，切换模式前先停止正在运行的服务：

```bash
bash scripts/start-server.sh base      # 基线，无投机解码
bash scripts/start-server.sh mtp       # MTP
bash scripts/start-server.sh dflash    # DFlash2
```

服务就绪后，在另一个终端发送测试请求：

```bash
python3 scripts/bench.py --host 127.0.0.1 --port 8835 --mode completion \
  --n-predict 400 --ignore-eos --prompt-file tasks/T1-自由生成.txt
```

这是一条提示词的一次请求，不是完整矩阵。脚本默认使用 Flash-Next 主模型和对应的 MTP 草稿头。运行 DFlash2 时，需通过 `MODEL_DIR`、`MAIN_MODEL` 和 `DRAFT_MODEL` 改为匹配的主模型与草稿文件；不能只切换启动参数。`PORT`、`CTX`、`SPEC_N_MAX` 和 `SPEC_P_MIN` 可调整端口、上下文与投机参数。启动脚本默认使用 262144 上下文并监听 `0.0.0.0`，可能占用较多内存并允许其他设备访问；请按设备资源设置 `CTX`，并在可信网络或受控防火墙环境中运行。不同配置的具体参数见 `configs/`；脚本默认值不代表完整实验矩阵。

## 批量矩阵入口

`scripts/run-matrix.sh` 使用实验目录内的 `scripts/bench.py` 和 `tasks/`，按指定配置逐项启动服务并分别运行三条提示词、每条三轮。示例（先按实际环境替换模型路径、文件名及服务程序）：

```bash
LLAMA_BIN=/path/to/llama-server \
  bash scripts/run-matrix.sh 35b /path/to/model model.gguf 32768 configs/matrix-35b.cfg
```

该脚本只停止自己启动的服务；请先确认端口未被其他服务占用。默认输出到 Git 忽略的 `.work/matrix-35b/`，也可以设置 `XL_OUT_DIR`；文件已存在时拒绝覆盖。设置 `XL_REQUIRE_EXTERNAL_ENV=1` 时必须显式指定 `XL_OUT_DIR`。`PYTHON` 可指定 Python 解释器。配置文件中的 `$HOME` 会展开为运行用户的主目录；实际模型和草稿模型路径必须事先准备。配置参数按空格拆分，草稿文件路径不要包含空格。35B 的 MTP 配置使用模型内置草稿头，与上面的外置草稿启动方式不同。2026-11-15 已在 GB10 上复跑该 35B 矩阵：五种配置、三个任务各三轮，共 45 次请求全部完成。使用 Qwen3.6-35B-A3B UD-Q4_K_M，权重 SHA-256 为 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`；CUDA 服务程序由 llama.cpp `ccc3646` 源码归档编译，CUDA Toolkit 13.0.88，构建程序未记录内部 Git 提交。三类任务的基线生成均值依次为 67.84、66.49、66.16 token/s，MTP 均值为 85.37、89.57、99.79 token/s。n-gram 的结果随任务变化，完整记录需逐项查看。历史权重哈希和构建参数不全，本次不能确认与历史二进制及权重完全一致；三轮样本也不足以证明普遍加速。

矩阵运行完成后，可查看汇总（如设置了 `XL_OUT_DIR`，请替换为相应输出路径）：

```bash
python3 scripts/parse_matrix.py .work/matrix-35b/matrix_35b.txt
```

汇总中的 `n` 是成功解析的轮次数；脚本会跳过格式错误或不完整的记录，检查结果时应确认每组均为 3，并查看原始日志。历史记录中重复请求可能复用提示词缓存，`prefill` 均值混合了不同的实际预填 token 数，不能直接作为相同输入条件下的预填性能比较；请结合原始记录中的 `prefill_tokens` 判断。

## 实验结果

实验结果及分析见[投机解码收益与实测条件](../../docs/zh/inference/投机解码收益与实测条件.md)。三轮原始记录在 `log/`，对应参数在 `configs/`。

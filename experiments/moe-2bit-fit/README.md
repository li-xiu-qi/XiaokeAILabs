# DeepSeek-V4-Flash 2-bit 在 DGX Spark 上的单卡推理实验

[English](README.en.md)

## 要回答的问题

iggerask 发布的 [DeepSeek-V4-Flash-2bit-GB10](https://huggingface.co/iggerask/DeepSeek-V4-Flash-2bit-GB10) 是一个 2-bit VQ 量化的 299B MoE 检查点，公布口径是能在单台 DGX Spark（GB10，128 GB 统一内存）上以约 22 tok/s 服务。我们在一台 GB10 上从零跑了一遍，回答三个问题：

1. 该检查点能否在一台干净的 GB10 上从零跑通（环境安装、101 GB 权重下载、加载、推理）；
2. 实际解码速度、显存占用与输出连贯性是多少，与公布数字差多少；
3. 模型自带的 MTP 投机解码在本环境是否可用，增益方向如何。

**结论先行**：能跑通。非投机组记录 20.16 tok/s、模型加载分配 94.62 GiB；一次简短问答完成。MTP K=2 加 FR-Spec 的配置记录 13.42 tok/s，比非投机组低约 33%。日志显示 CUDA Graph 配置发生降级，但现有数据没有隔离造成下降的原因，详见报告。

## 实验对象

| 项 | 配置 |
|---|---|
| 模型 | `iggerask/DeepSeek-V4-Flash-2bit-GB10`（基于 DeepSeek-V4-Flash 299B MoE；专家 2-bit VQ k=1024/vdim=4/group-64 Hadamard，attn/shared/lm_head NVFP4 W4A16，路由与嵌入 BF16，KV cache fp8） |
| 服务插件 | `iggerask/deepseek-v4-flash-2bit-gb10` 的 `vq2_vllm`（Apache-2.0，经 vLLM general_plugins 入口自动注册，非 vLLM fork） |
| 硬件 | DGX Spark，GB10，20 核 ARM64，121.6 GiB 统一内存，sm_121 |
| 软件 | Ubuntu 24.04，CUDA 13.0（V13.0.88），驱动 580.173.02，torch 2.11.0+cu130，vLLM 0.23.0，Python 3.12 |
| 上下文 | 2,048 token，单请求 |
| 采样 | temperature 0，greedy |
| 指标 | decode tok/s（diff 方法隔离 prefill）、常驻内存、finish_reason |

## 实验版本记录

后续复现时若组件版本不同，结果不可直接比较。插件仅记录了 master 分支，未记录提交号；下方下载固定模型 revision，但插件克隆未固定提交，因此不能保证取到与实验完全相同的软件版本。

| 组件 | 版本标识 |
|---|---|
| 模型检查点 | `iggerask/DeepSeek-V4-Flash-2bit-GB10`，revision `df5ebc8fa0e70f36542ca710fcd5e2ac24fae0bb`，lastModified 2026-06-24 |
| 插件 | `iggerask/deepseek-v4-flash-2bit-gb10` master（含 mhc_post sm_121 torch-reference 补丁） |
| vLLM | 0.23.0（stock，无 fork） |
| torch | 2.11.0+cu130 |
| CUDA / 驱动 | 13.0（V13.0.88）/ 580.173.02 |
| 实验日期 | 2026-10-27 |

发布方公布口径（README/ARCHITECTURE.md，2026-06-24 版）：非投机 17.4 tok/s、MTP K=2 22.5 tok/s；PPL 4.64（源 FP8+MXFP4 为 3.66）；GSM8K 95.0%、MMLU-Pro 66.4%、HumanEval 65.2%。

## 结果

两组共享内存配额、上下文和 shared-experts 参数，但非投机组关闭 FR-Spec，MTP 组启用 FR-Spec，故这是两套配置的比较：

| 组 | decode tok/s | ms/tok | coherence |
|---|---|---|---|
| 非投机（K=0） | **20.16** | 49.61 | 通过（Paris + 俳句，finish_reason=stop） |
| MTP K=2 | **13.42** | 74.52 | 通过（Paris + 三行俳句，finish_reason=stop） |

非投机组加载日志记录模型分配 94.62 GiB（45 个 safetensors 分片），MTP 组为 96.86 GiB；这不是文件大小或完整系统峰值。非投机摘要另记 94.69 GiB，来源差异尚未核清。两组速度差约 −33%；是否由图捕获、FR-Spec、接受率或其他开销造成，需进一步测量。

## 环境

单台 DGX Spark，无其他大模型服务占用。权重从 HuggingFace 镜像下载（101 GB），首次推理前 vq2 的 CUDA kernel（`vq2_grouped.cu`，sm_121a gencode）JIT 编译约 3 分钟，之后走缓存。

下列步骤依赖 ARM64 Ubuntu、GB10、可用的 CUDA 编译工具链、Python 3.12、Git、curl 和 uv。至少需要为约 101 GB 的模型文件预留磁盘空间，并另留依赖、缓存和编译产物的空间；这些步骤记录的是当时的配置，插件及外部下载源变化后需要重新核对。安装前可用 `uname -m`、`nvidia-smi`、`nvcc --version` 和 `python3.12 --version` 检查架构、驱动、CUDA 编译器与 Python；安装后在隔离环境运行 `python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"`，确认版本并检查 CUDA 是否可用。依赖校验通过不等于模型加载或性能复现通过。

## 复现

依赖由外部插件仓的 `requirements-serve.lock.txt` 和插件自身管理，因此本目录不复制一份可能失步的 Python requirements。下方修订锁文件的做法来自历史环境；上游默认分支已变化时须重新核对依赖冲突。命令会在插件仓创建环境及模型目录，并写入 `/tmp/torch-wheels` 和 `/tmp/req-rest.txt`；请先确认这些位置没有要保留的同名文件。

```bash
# 1. 插件仓（vLLM 0.23.0 + vq2 插件 + bench 脚本）
git clone https://github.com/iggerask/deepseek-v4-flash-2bit-gb10
cd deepseek-v4-flash-2bit-gb10

# 2. torch 三件套（cu130 构建；直连 download.pytorch.org 在国内网络下
#    连接建立后无数据传输，可改用 mirrors.aliyun.com/pytorch-wheels/cu130/ 镜像）
mkdir -p /tmp/torch-wheels && cd /tmp/torch-wheels
for f in torch-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
         torchvision-0.26.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
         torchaudio-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl; do
  curl -sL -O "https://mirrors.aliyun.com/pytorch-wheels/cu130/$f"
done
cd - && uv venv --python 3.12 .venv-vllm && source .venv-vllm/bin/activate
uv pip install --find-links /tmp/torch-wheels/ \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130

# 3. 其余依赖。requirements-serve.lock.txt 有三处内在矛盾需先修：
#    fsspec 2026.6.0 -> 2026.4.0（datasets 5.0.0 上限）；删除 llmcompressor
#    （推理不需要，且与 compressed-tensors/accelerate 的 pin 互斥）；
#    compressed-tensors 保持 vllm 要求的 0.17.0
grep -ivE '^(torch|torchvision|torchaudio|vq2-vllm)==' requirements-serve.lock.txt \
  | sed 's/^fsspec==2026.6.0$/fsspec==2026.4.0/; /^llmcompressor==/d' > /tmp/req-rest.txt
UV_HTTP_TIMEOUT=15 uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /tmp/req-rest.txt
uv pip install -e .

# 4. 固定历史模型 revision 下载（约 101 GB，需预留额外缓存空间）
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='iggerask/DeepSeek-V4-Flash-2bit-GB10', revision='df5ebc8fa0e70f36542ca710fcd5e2ac24fae0bb', local_dir='models/DeepSeek-V4-Flash-2bit')"

# 5. bench。两个关键环境变量：
#    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 —— 本次绕过 MTP capture 报错的设置
#    GPU_UTIL=0.85 —— 本次使用的内存配额；实际可用空间随环境变化
# 非投机组：
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 GPU_UTIL=0.85 \
  python scripts/bench.py --spec-mtp 0 --no-frspec --max-model-len 2048
# MTP K=2 组：
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 GPU_UTIL=0.85 \
  python scripts/bench.py --spec-mtp 2 --max-model-len 2048
```

下载指令固定了模型仓 revision。2026-11-13 对一份已有副本的 89 个 safetensors 文件逐一计算 SHA-256，均与该 revision 在 Hugging Face 官方 API 中的 LFS 哈希一致。原始实验权重的独立文件哈希未记录，且固定插件版本尚不可得，因此这不能证明该副本与历史运行完全相同，也不能保证严格复现。若使用其他模型目录，向上游 bench 或本仓 `min_repro.py` 显式传入 `--model /path/to/model`。上述安装和固定 revision 下载命令尚未从零重新走通。

2026-11-13 的一次最小检查在加载过程中因内存保护中止，没有生成文本。该次使用已有环境，`pip check` 报告了 `llmcompressor` 和 `fsspec` 等依赖冲突；它不能证明上述安装流程可用。后续检查该插件仓的提交号为 `abdfcce6d01269eeb2aff3d451b98e183c2de1f4`，但不能据此确定 2026-10-27 成功实验所用的插件提交，也不能将这个提交视为已完成复现的版本。显存配额不是可用内存的保证，运行前仍需检查系统余量和任务保护是否覆盖全部子进程。

2026-11-14 在独立环境再次尝试：依赖检查通过，使用较短上下文及调整过权重读取方式的插件副本，但初始化期间系统可用内存仍快速下降，任务因内存保护中止，没有生成文本。该次修改尚未证明能降低完整加载峰值，也没有验证上述从零安装流程；历史结果与本次未完成的复现应分别看待。

首次 bench 会触发 `vq2_grouped.cu` 的 nvcc 编译，本次约 3 分钟。若后续进程等待编译锁，先确认没有活跃编译进程，再检查实际缓存中的 lock。也可在插件仓根目录调用本仓脚本：将 `XL_REPO` 设为 XiaokeAILabs 克隆目录，然后执行 `bash "$XL_REPO/experiments/moe-2bit-fit/scripts/run_bench_pair.sh"`；附加参数 `nospec` 或 `mtp2` 可只运行一组。

## 已知限制

- MTP K=2 加 FR-Spec 的配置慢于本次非投机组；日志出现 FULL 图模式不受支持并降级为 FULL_DECODE_ONLY 的警告。尚无分项耗时或可用的 FULL 对照，不能据此确定性能原因，也未在发布方依赖组合下验证。
- 上下文 2,048 token（默认 4,096 在统一内存余量下不可用），长上下文场景未测。
- 数据为单机单次测量，未做多次重复与方差统计；diff 方法计时，prefill 不计入 decode 速度。

## 目录

- [单卡推理实测报告](../../docs/zh/inference/299B-MoE单机2-bit推理实测.md)：主报告，结论以此为准
- `scripts/run_bench_pair.sh`：一键跑两组 bench（环境变量与单组参数见脚本头注释）
- `scripts/min_repro.py`：最小复现，只验「能加载 + 能出一次 token」，用于隔离环境问题。默认配额为 `0.85`、上下文为 `2048`，可用 `--gpu-memory-utilization`、`--max-model-len` 修改；内存余量不足时依然可能失败。
- `scripts/launch_repro2.sh`：带诊断开关启动本仓的 min_repro.py。设置 `PLUGIN_REPO` 指向已安装环境和模型的插件仓，然后从本实验目录执行 `bash scripts/launch_repro2.sh`；日志默认写入插件仓的 `.work/`，可用 `XL_OUT_DIR` 指定。脚本输出的是超时包装进程 ID；如需用 SIGUSR1 打线程栈，应先识别实际 Python 子进程。
- `evidence/`：两组 bench 原始输出（bench-output-*.txt）与结果 JSONL

# DeepSeek-V4-Flash 2-bit 在 DGX Spark 上的单卡推理实验

## 要回答的问题

iggerask 发布的 [DeepSeek-V4-Flash-2bit-GB10](https://huggingface.co/iggerask/DeepSeek-V4-Flash-2bit-GB10) 是一个 2-bit VQ 量化的 299B MoE 检查点，公布口径是能在单台 DGX Spark（GB10，128 GB 统一内存）上以约 22 tok/s 服务。我们在一台 GB10 上从零跑了一遍，回答三个问题：

1. 该检查点能否在一台干净的 GB10 上从零跑通（环境安装、101 GB 权重下载、加载、推理）；
2. 实际解码速度、显存占用与输出连贯性是多少，与公布数字差多少；
3. 模型自带的 MTP 投机解码在本环境是否可用，增益方向如何。

**结论先行**：能跑通。非投机解码 20.16 tok/s、权重常驻 94.69 GiB、语义冒烟通过。MTP K=2 可运行但为净负收益（13.42 tok/s），根因是 vLLM 0.23.0 拒绝 drafter FULL cudagraph 导致降级，详见报告。

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

## 版本锁定

后续复现时若组件版本不同，结果不可直接比较。

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

两组共享同一配置（0.85 gpu_memory_utilization、2,048 token 上下文、`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`），唯一变量为是否启用 MTP 投机解码：

| 组 | decode tok/s | ms/tok | coherence |
|---|---|---|---|
| 非投机（K=0） | **20.16** | 49.61 | 通过（Paris + 俳句，finish_reason=stop） |
| MTP K=2 | **13.42** | 74.52 | 通过（Paris + 三行俳句，finish_reason=stop） |

权重加载后常驻 94.69 GiB（45 个 safetensors 分片）。MTP 增量 −33%，与发布方公布的 +29% 方向相反；根因与完整证据链见 `reports/01-2bit-moe-单卡推理复现.md`。

## 环境

单台 DGX Spark，无其他大模型服务占用。权重从 HuggingFace 镜像下载（101 GB），首次推理前 vq2 的 CUDA kernel（`vq2_grouped.cu`，sm_121a gencode）JIT 编译约 3 分钟，之后走缓存。

## 复现

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

# 4. 下载模型（101 GB）。HF_HUB_DISABLE_XET=1 必需，否则镜像的 xet CAS 重构返回 401
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python scripts/download_model.py

# 5. bench。两个关键环境变量：
#    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 —— MTP drafter 的 cudagraph
#      capture 在 sm_121 上会因 shared experts 的 aux stream 跨流操作抛
#      illegal memory access，禁掉重叠即解（对 target decode 无影响）
#    GPU_UTIL=0.85 —— 121 GiB 统一内存下模型 94.69 GiB 常驻，默认 0.92 会 OOM
# 非投机组：
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 GPU_UTIL=0.85 \
  python scripts/bench.py --spec-mtp 0 --no-frspec --max-model-len 2048
# MTP K=2 组：
VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0 GPU_UTIL=0.85 \
  python scripts/bench.py --spec-mtp 2 --max-model-len 2048
```

首次 bench 会触发 `vq2_grouped.cu` 的 nvcc 编译（约 3 分钟，期间进程静默、GPU 有活动，勿中断）；若编译中途进程被杀，`~/.cache/torch_extensions/py312_cu130/vq2_grouped_d0/lock` 残留会导致后续进程死等，删掉该 lock 即可。两步 bench 命令也可用 `scripts/run_bench_pair.sh` 一键执行（参数 `nospec` / `mtp2` / 无参跑两组）。

## 已知限制

- MTP K=2 增益为负，没有拿到发布方公布的 +29%；那个数字依赖 `VQ2_DRAFTER_FULL_CG=1`（drafter FULL cudagraph），vLLM 0.23.0 在 DeepseekV4IndexerBackend 上拒绝该组合并降级为 FULL_DECODE_ONLY。这是版本行为差异，未在发布方那套固定依赖组合上验证。
- 上下文 2,048 token（默认 4,096 在统一内存余量下不可用），长上下文场景未测。
- 数据为单机单次测量，未做多次重复与方差统计；diff 方法计时，prefill 不计入 decode 速度。

## 目录

- `reports/01-2bit-moe-单卡推理复现.md`：主报告，结论以此为准
- `scripts/run_bench_pair.sh`：一键跑两组 bench（环境变量与单组参数见脚本头注释）
- `scripts/min_repro.py`：最小复现，只验「能加载 + 能出一次 token」，用于隔离环境问题
- `scripts/launch_repro2.sh`：带诊断开关启动 min_repro.py，挂住时用 SIGUSR1 打线程栈
- `docs/evidence/`：两组 bench 原始输出（bench-output-*.txt）与结果 JSONL

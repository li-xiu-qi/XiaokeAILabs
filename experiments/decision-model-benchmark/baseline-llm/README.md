# Jev 系列实验：生成式基线篇

[English](README.en.md)

这组不是参赛选手，是对照组。它回答一个问题：同样做分类，一个普通生成式大模型要花多久。没有这个数字，判别式模型的优势就没有参照。

## 两个脚本在测什么

`qwen4b-latency.py` 是 Qwen3-VL-4B 的耗时基准，输入文本、任务定义、四个类别全部逐字照抄 `../laya-dual-branch/laya-latency.py`，同一台机器同一个环境，同样先预热再计时。

生成式模型的延迟要拆成两段看。prefill 是一次前向把输入读完、一个 token 都不出，等价于拿到首个 token 前的等待时间，耗时随输入长度走。decode 是之后自回归逐个生成 token 的耗时，随输出长度走。full 是两者之和。

脚本测两档 Prompt。最简档只准输出类别名，解码最短，是这个模型的最有利情形。JSON 档要输出完整 JSON 对象，工程上更常见，解码也长得多。两档都跑才能看出生成式方案在结构化输出上的真实开销。

## 环境准备与运行

脚本面向 Linux 和支持 bfloat16 的 CUDA 显卡；需要 Python、兼容设备的 CUDA 版 PyTorch，以及支持 Qwen3-VL 的 Transformers 和 Accelerate。`requirements.txt` 固定 Transformers 5.22.2、Accelerate 1.15.0，PyTorch 按设备另行安装；这不是包含全部传递依赖的锁文件，建议使用独立环境。默认模型为 `Qwen/Qwen3-VL-4B-Instruct`；基准和诊断脚本都可用 `QWEN_MODEL` 指向其他模型或本地路径，并可用 `QWEN_MODEL_REVISION` 固定版本。

例如，创建虚拟环境并按设备安装 CUDA 版 PyTorch 后，可安装其余直接依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 先按设备选择并安装兼容的 CUDA 版 PyTorch
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__); assert torch.cuda.is_available()"
```

诊断脚本强制离线读取指定模型的 tokenizer，先联网准备缓存，再从本目录执行。为固定本次基准使用的模型版本，可设置 revision：

```bash
export QWEN_MODEL=Qwen/Qwen3-VL-4B-Instruct
export QWEN_MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$QWEN_MODEL', revision='$QWEN_MODEL_REVISION')"
python qwen4b-diag.py
BENCH_OUT=qwen3-vl-result.json python qwen4b-latency.py
```

`qwen4b-diag.py` 不加载模型权重，但以离线模式加载 `QWEN_MODEL` 和 `QWEN_MODEL_REVISION` 指定的 tokenizer；需要提前准备好 tokenizer 缓存和依赖。它用于确认模板走纯文本路径。`BENCH_OUT` 可指定基准结果 JSON 的保存位置。

`scripts/clean_env_gpu_smoke.sh` 会新建独立环境，安装 CUDA PyTorch 和本目录依赖，并对缓存或可下载的 Qwen2.5-0.5B 做一次 CUDA 生成检查。从本目录运行：

```bash
bash scripts/clean_env_gpu_smoke.sh
```

默认按 Linux ARM64 示例安装 `torch==2.15.0+cu132`；其他设备应设置 `PYTORCH_SPEC` 与 `PYTORCH_INDEX_URL` 为匹配其系统和驱动的 PyTorch 构建。可用 `XL_DATA_DIR`、`XL_OUT_DIR` 指定环境和结果目录；脚本拒绝复用已存在的环境目录。模型权重需另留磁盘空间。

2026-11-15 在 DGX Spark（Linux ARM64、Python 3.12.3、NVIDIA GB10）上，从最终版 `scripts/clean_env_gpu_smoke.sh` 新建隔离环境并完整运行。PyTorch 2.15.0+cu132、Transformers 5.22.2、Accelerate 1.15.0 安装完成，`pip check` 无依赖问题；固定 revision `7ae557604adf67be50417f59c2c2f167def9a775` 的 Qwen2.5-0.5B-Instruct 在 `cuda:0` 生成 `Billing`（2 tokens），加载 8.39 秒，GPU 峰值分配 0.95 GiB。脚本默认固定这个模型 revision；更换模型时可设置 `QWEN_MODEL_ID` 和 `QWEN_MODEL_REVISION`。结果摘要见[GPU 冒烟记录](evidence/2026-11-15-clean-env-gpu-smoke.json)。这次只验证安装入口、模型加载和一次 GPU 生成，不复现默认 Qwen3-VL-4B 基准，也不评估输出质量。

### 用较小模型检查推理链路

如果想在下载默认模型前检查代码和 CUDA，可从本目录运行下面的单次前向与生成。首次运行会下载 `Qwen/Qwen2.5-0.5B-Instruct`；已有本地权重时可将 `QWEN_MODEL` 改为模型目录。该检查使用替代模型和一条英文输入，不运行完整计时循环，不能用于复现下方 Qwen3-VL-4B 指标。

```bash
QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct python - <<'PY'
import runpy

benchmark = runpy.run_path("qwen4b-latency.py", run_name="benchmark_module")
tokenizer, model, _ = benchmark["load"]()
prompt = benchmark["build_prompt"]("minimal", "en", benchmark["EN"])
result = benchmark["bench"](model, tokenizer, prompt, 12)
print("device:", next(model.parameters()).device)
print("output:", repr(result["text"]))
assert next(model.parameters()).is_cuda and result["n_new"] > 0 and result["text"].strip()
PY
```

2026-11-13 在 Linux ARM64、NVIDIA GB10、PyTorch 2.15.0+cu132、Transformers 5.22.2 和 Accelerate 1.15.0 的现有环境中，使用已缓存的上述模型执行该检查，设备为 `cuda:0`，输出为 `Billing`。这次没有验证全新环境安装、默认 4B 模型、完整基准或模型修订一致性。

`qwen4b-latency.py` 第一次跑会自动下载约 8.3 GB 权重，加载约 54 s，之后常驻 8.3 GB。跑完记得释放，别和 Nimble 那组同时驻留。

结果对应 `../evidence/` 编号 08 的结果摘录。编号 08b 是早期一次模板未生效的失败调试记录摘录，已被后续脚本修正，不作为正常复现路径。

## 为什么用 AutoTokenizer 而不是 processor

视觉语言模型的 processor 会往 Prompt 里塞图像占位 token，prefill 时间随这张虚拟图像膨胀，测出来的数不代表纯文本任务。

## 实测指标

2026-11-15 在 DGX Spark（Linux ARM64、NVIDIA GB10）上，使用隔离环境、Qwen3-VL-4B 固定 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，每种输入预热 3 次后计时 30 次。模型加载记录为 442.23 秒（包含首次约 8.3 GB 权重下载），加载后常驻 8.27 GiB，CUDA 峰值分配 8.34 GiB。四种输入均无空输出。中英文最简输出的 full 延迟 p50 分别为 98.78 ms 和 100.00 ms；JSON 输出分别为 1410.73 ms 和 1123.87 ms。完整数据见[本次 GPU 基准记录](evidence/2026-11-15-qwen3-vl-gpu-rerun.json)。

这次运行只验证了 Qwen3-VL 的完整计时流程，没有在相同运行条件下同步重跑 Laya，因此不据此计算两者的直接速度倍数。历史运行中，单次前向约 49 到 52 ms，JSON 输出约 1050 到 1450 ms；本次四种输入的 prefill p50 为 47.84 到 50.41 ms，与该范围一致。

这就是判别式与生成式的分野，不在绝对速度，在成本结构。生成式的成本跟着输出长度走，判别式模型不生成文本，输出多少字段耗时都不变。这决定了判别式模型适合高并发决策路径，不适合需要生成解释的地方。

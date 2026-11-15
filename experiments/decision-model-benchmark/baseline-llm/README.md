# Jev 系列实验：生成式基线篇

[English](README.en.md)

这组不是参赛选手，是对照组。它回答一个问题：同样做分类，一个普通生成式大模型要花多久。没有这个数字，判别式模型的优势就没有参照。

## 两个脚本在测什么

`qwen4b-latency.py` 是 Qwen3-VL-4B 的耗时基准，输入文本、任务定义、四个类别全部逐字照抄 `../laya-dual-branch/laya-latency.py`，同一台机器同一个环境，同样先预热再计时。

生成式模型的延迟要拆成两段看。prefill 是一次前向把输入读完、一个 token 都不出，等价于拿到首个 token 前的等待时间，耗时随输入长度走。decode 是之后自回归逐个生成 token 的耗时，随输出长度走。full 是两者之和。

脚本测两档 Prompt。最简档只准输出类别名，解码最短，是这个模型的最有利情形。JSON 档要输出完整 JSON 对象，工程上更常见，解码也长得多。两档都跑才能看出生成式方案在结构化输出上的真实开销。

## 怎么跑

脚本面向 Linux 和支持 bfloat16 的 CUDA 显卡；需要 Python、兼容设备的 CUDA 版 PyTorch，以及支持 Qwen3-VL 的 Transformers 和 Accelerate。仓库没有这组脚本的完整依赖锁定文件，版本与其他实验不一定兼容，建议使用独立环境。默认模型为 `Qwen/Qwen3-VL-4B-Instruct`；基准脚本可用 `QWEN_MODEL` 指向其他模型或本地路径，但诊断脚本固定读取默认模型的 tokenizer 缓存。

例如，创建虚拟环境并按设备安装 CUDA 版 PyTorch 后，可安装其余直接依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 先按设备选择并安装兼容的 CUDA 版 PyTorch
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__); assert torch.cuda.is_available()"
```

诊断脚本强制离线读取默认模型的 tokenizer，先联网准备缓存，再从本目录执行：

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')"
python qwen4b-diag.py
python qwen4b-latency.py
```

`qwen4b-diag.py` 不加载模型权重，但以离线模式加载对应 tokenizer；需要提前准备好 tokenizer 缓存和依赖。它用于确认模板走纯文本路径。

上述环境准备与 tokenizer 下载命令尚未在新建环境中端到端验证；应记录实际安装版本和所用模型 revision。

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

单次前向 49 到 52 ms，和 Laya 的 17 ms 在同一量级。但要求输出 JSON 对象后，完整一次判定从约 105 ms 涨到约 1150 至 1450 ms。

这就是判别式与生成式的分野，不在绝对速度，在成本结构。生成式的成本跟着输出长度走，判别式模型不生成文本，输出多少字段耗时都不变。这决定了判别式模型适合高并发决策路径，不适合需要生成解释的地方。

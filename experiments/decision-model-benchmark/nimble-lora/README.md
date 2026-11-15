# Jev 系列实验：Nimble 篇

[English](README.en.md)

本目录对应 `bespokelabsai/nimble` (Bespoke-Nimble-9B) 方案的实机评测与复现工程。

Nimble 的发布形态是一个 LoRA 适配器加一个版本锁定的基座模型，部署前需要先完成权重合成。在四组评测方案中，它的参数量最大，单字段延迟和显存开销也最高。这篇 README 记录它在本地单卡上的实测表现、部署步骤以及这条路线特有的工程约束。

## 文件组成与测试分工

- `nimble-deploy.py`：权重合成脚本。下载适配器与基座权重，校验 Prompt 契约，执行 PEFT 权重合并，最后把完整权重和配置文件写到用户缓存目录。
- `nimble-bench.py`：推理性能与业务准确率测试脚本。评测涵盖输入输出契约验证、单字段延迟基准、中文业务场景准确率。
- `nimble-cuda_scorer.py`：上游 CUDA 逐字段打分器副本，负责在 CUDA 上按字段顺序执行前向计算。

## 部署与复现步骤

Nimble 的部署比另外三条多一步，因为推理实现不在本仓，在独立的 Nimble 代码仓里。

从本目录执行：

需要 Linux、CUDA 显卡、Python 和 Git。先准备独立环境并按设备安装兼容 CUDA 的 PyTorch；部署脚本还依赖 `transformers`、`peft`、`huggingface_hub` 和 `accelerate`。历史脚本记录的版本为 PyTorch 2.14.0+cu130、Transformers 5.16.1、PEFT 0.20.0；仓库没有完整依赖锁文件，上游代码及适配器未锁定，重跑前应记录版本并核对兼容性。合成在 CPU 上进行，需同时容纳基座和合并权重，约各 18 GB，还要预留下载与保存的磁盘空间。

```bash
python3 -m venv .venv
source .venv/bin/activate
# 先按设备选择并安装兼容的 CUDA 版 PyTorch
python -m pip install -r requirements.txt
```

```bash
# 1. 克隆上游代码仓，bench 和 deploy 都要从这儿 import 打分器
git clone https://github.com/bespokelabsai/nimble.git ~/nimble

# 2. 合成权重
python nimble-deploy.py

# 3. 跑基准
python nimble-bench.py
```

第一步不能省。`nimble-bench.py` 和 `nimble-deploy.py` 都把 `~/nimble` 插到模块搜索路径最前面，从那里 import 上游的打分器实现。仓内的 `nimble-cuda_scorer.py` 只是留档副本，不参与运行。少了这一步，部署脚本会在校验 Prompt 契约时报文件不存在，基准脚本会直接找不到模块。

第二步合成权重。Nimble 的发布形态是一个 LoRA 适配器加一个版本锁定的基座，两者得合成完整权重才能推理。`nimble-deploy.py` 把每一步耗时都打出来，本次实测适配器 10.8 s，基座权重 488.7 s，合成 5.2 s，保存 4 s，全程约 8 分半。其中基座那 18 GB 是大头，走镜像站也要几分钟。

合成完它会把权重保存到用户缓存目录下，路径形如 `~/.cache/models/nimble-9b-<提交号>`，同时把路径写进一份配置文件。`nimble-bench.py` 读那份配置，不用你手动填。

跑完基准后可与 `../evidence/` 中编号 05 和 06 的摘录对照。延迟和显存受设备、软件版本及运行负载影响，不应要求数值完全一致；先核对模型与运行配置，再解释差异。

## 这条路线两个坑

一是合成时会看到一条 PEFT 参数告警，说适配器里有它不认识的参数。可以忽略，适配器保存时的 PEFT 版本比本地高，多出来的参数会被跳过，不影响合成结果。

二是它最占显存，17.57 GB，和其他模型一起驻留要先算好预算。

## 实测指标

在本机测试环境下，Nimble 的实测数据：

- **端到端延迟**：单次判定（两个字段）中位数 365.33 ms（英文）与 366.66 ms（中文输入），四组路线里最慢。
- **中文支持**：上游声明仅支持英文。我们只喂过一条中文输入，延迟与英文持平，字段能正常返回分数，但没有跑过中文测试集，准确率未知。
- **显存占用**：常驻显存约 17.57 GB，四组路线里最高。

## 一个关键的结构性差异

Nimble 的 CUDA 路径没有字段并行。上游 README 的区分很清楚：Mac 上 `ParallelScorer` 共享一次前向，然后并行把所有字段一起打分；CUDA scorer 是每个字段单独跑一次，每次都要带完整 Prompt。字段数增加时成本近似线性上涨。我们测出的 365.33 ms 是问两个字段的价格，字段加到十个就是另一个量级。这和 Laya 一次请求塞多道题的摊薄不是一回事，横向对比时只看单字段数字会看错。

如果你的场景字段数固定在两三个，它也能用。但相比 metask-jev-4B 的 133.7 ms 和 8.58 GB，看不到优势。

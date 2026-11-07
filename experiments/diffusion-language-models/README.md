# 扩散语言模型生成有多快？在 DGX Spark 上对照 1.5B 模型

[English](README.en.md)

如果语言模型能够一次更新多个位置，生成回答会更快吗？我们在 DGX Spark 上实测 [Fast-dLLM v2 1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B)，并与它的自回归基座 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 对照。这组数字有助于理解两种实际推理路径的开销，也展示了比较生成速度时不能忽略的输出长度与内容差异。

## 实测结果

在 BF16、单请求、最多生成 128 token 的设置下，对 3 个相同英文问题分别重复 3 次，逐次文本 token/s 的中位数如下：

| 模型 | 中位数 | 输出情况 |
|---|---:|---|
| Fast-dLLM v2 1.5B | 58.13 token/s | 三题分别返回 122、65、127 token；第二题以结束符结束 |
| Qwen2.5-1.5B-Instruct | 51.52 token/s | 三题均达到 128 token 上限，未出现结束符 |

这里的速度定义为每次生成的**可见文本重新编码后的 token 数**除以 `generate()` 耗时；计时包含提示词预填充与生成，排除分词和最终文本解码。Fast-dLLM 代码题更早结束，两个模型输出内容和长度不同，因此这些数字**不能解释为相同质量下的加速比**。3 道问题不足以判断普遍性能，也没有测试答案质量、首 token 时间、并发吞吐或不同硬件。

## 补充：两种 MoE 模型能跑多快？

为观察稀疏激活模型的实际开销，我们还运行了扩散模型 [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) 和自回归模型 [OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct)。前者约 70 亿总参数、每个 token 激活约 14 亿非嵌入参数，后者约 70 亿总参数、激活约 10 亿参数。激活参数接近并不意味着架构、计算量或输出质量相同。

两者分别在独立进程中以 BF16、单请求运行，使用上面同样的三个问题，预热一次，每题测一次；这只是冒烟测试。速度仍按可见文本重新编码 token 数除以生成耗时计算。

| 问题 | LLaDA-MoE：可见 token / 耗时 | OLMoE：可见 token / 耗时 |
|---|---:|---:|
| 数据库索引 | 126 / 18.41 秒 | 128 / 8.01 秒 |
| Python 最大值函数 | 28 / 18.62 秒 | 128 / 8.01 秒 |
| 训练与推理的区别 | 126 / 18.48 秒 | 128 / 8.01 秒 |

LLaDA-MoE 使用改写自模型卡的低置信度固定步数基线，生成长度 128、块长 32、总计 128 次前向，不使用 KV cache；它即使在较早出现结束符后也继续执行固定步数。代码题因此只产生 28 个可见文本 token，却仍耗时 18.62 秒。该实现不是模型方的 dInfer 加速实现。OLMoE 使用自回归 `generate()` 和 KV cache，三题均达到 128 token 上限而截断。LLaDA-MoE 第三题错误地称训练和推理通常使用不同参数规模，因此不能把这些输出当作质量合格的证明。

两种模型使用不同 tokenizer、训练和解码实现，输出长度及正确性也不同；一次运行不足以估计波动，不能据此推断同等质量下的速度差或扩散架构本身的优劣，也不能直接与上面重复三次的结果混成统一排名。

## 两种生成方式

![自回归与分块扩散生成路径](docs/generation-paths.svg)

自回归生成通常按顺序预测下一个 token，并借助 KV cache 避免重复计算已有上下文。Fast-dLLM v2 使用分块生成，在一个块内迭代更新多个位置。它仍需要多次模型前向计算，也需要为每个块维护状态；并行更新不意味着一次前向就完成整段回答。

这次 Fast-dLLM 设置块大小 32、小块大小 8、置信度阈值 0.9，并关闭 block cache；Qwen2.5 使用 KV cache 和贪心生成。Fast-dLLM 与对照模型还经过不同训练和生成实现，测试差异不能只归因于生成方式。

## 复现

使用 Linux、CUDA 环境和 DGX Spark。实测软件版本为 Python 3.12、PyTorch 2.14.0+cu130、Transformers 4.57.3；Fast-dLLM 的自定义模型代码与 Transformers 5.16.1 的接口不兼容。模型需要预先下载到本地，脚本只使用本地文件。运行自定义模型代码前应检查来源；模型下载和许可请分别参见上述模型卡。

```bash
python scripts/benchmark.py --model /path/to/Fast_dLLM_v2_1.5B \
  --kind diffusion --revision 25093b6f63300adfd57f72145083c8a528fe4f16 \
  --warmups 2 --repeats 3 --max-new-tokens 128 --output /tmp/diffusion.json

python scripts/benchmark.py --model /path/to/Qwen2.5-1.5B-Instruct \
  --kind autoregressive --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --warmups 2 --repeats 3 --max-new-tokens 128 --output /tmp/autoregressive.json
```

脚本每个模型使用独立进程，预热两次后对每个问题重复三次；记录输出 token ID、完整文本、前向次数、耗时和软件版本。脚本面向统一内存设备，启动及运行时检查 Linux 可用内存，低于 50 GiB 时停止。`--revision` 是记录字段，复现前需要自行确认本地权重对应的版本。

[扩散模型逐次记录](docs/diffusion-runs.json) · [自回归模型逐次记录](docs/autoregressive-runs.json) · [基准脚本](scripts/benchmark.py)

MoE 冒烟测试也可使用相同环境分别运行。先自行下载对应模型版本；`--revision` 仅记录版本，不会校验本地权重。

```bash
python scripts/benchmark-llada-moe.py --model /path/to/LLaDA-MoE-7B-A1B-Instruct \
  --revision 783d3467f108d28ac0a78d3e41af16ab05cabd8d --warmups 1 --repeats 1 \
  --steps 128 --gen-length 128 --block-length 32 --output /tmp/llada-moe.json

python scripts/benchmark.py --model /path/to/OLMoE-1B-7B-0924-Instruct \
  --kind autoregressive --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498 \
  --warmups 1 --repeats 1 --max-new-tokens 128 --output /tmp/olmoe.json
```

[LLaDA-MoE 原始记录](docs/llada-moe-smoke.json) · [OLMoE 原始记录](docs/olmoe-smoke.json) · [固定步数基准脚本](scripts/benchmark-llada-moe.py)

质量与延迟的固定任务、自动评分及运行方案见 [对照方案](docs/质量与延迟对照方案.md)；[四模型短答案实测](results/quality-latency-20261107/README.md)展示逐次回答与时间，样本仅八题。

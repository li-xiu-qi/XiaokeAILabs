# 扩散语言模型生成有多快？在 DGX Spark 上对照 1.5B 模型

如果语言模型能够一次更新多个位置，生成回答会更快吗？我们在 DGX Spark 上实测 [Fast-dLLM v2 1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B)，并与它的自回归基座 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 对照。这组数字有助于理解两种实际推理路径的开销，也展示了比较生成速度时不能忽略的输出长度与内容差异。

## 实测结果

在 BF16、单请求、最多生成 128 token 的设置下，对 3 个相同英文问题分别重复 3 次，逐次文本 token/s 的中位数如下：

| 模型 | 中位数 | 输出情况 |
|---|---:|---|
| Fast-dLLM v2 1.5B | 58.13 token/s | 三题分别返回 122、65、127 token；第二题以结束符结束 |
| Qwen2.5-1.5B-Instruct | 51.52 token/s | 三题均达到 128 token 上限，未出现结束符 |

这里的速度定义为每次生成的**可见文本重新编码后的 token 数**除以 `generate()` 耗时；计时包含提示词预填充与生成，排除分词和最终文本解码。Fast-dLLM 代码题更早结束，两个模型输出内容和长度不同，因此这些数字**不能解释为相同质量下的加速比**。3 道问题不足以判断普遍性能，也没有测试答案质量、首 token 时间、并发吞吐或不同硬件。

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

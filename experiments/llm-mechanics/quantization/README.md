# 量化语言模型：体积、困惑度与推理速度

[English](README.en.md)

模型量化通过降低部分权重的数值精度，减少模型存储和权重搬运开销。它能否加速推理，取决于硬件与实现；它对效果的影响，也必须结合应用数据评估。这里用同一 Qwen3-0.6B 权重，在同一 CPU 推理引擎中比较 BF16、Q8_0 和 Q4_K_M，展示一次可追溯记录的权衡；当前二进制源码版本未锁定，精确复现仍有缺口。

## 实测：文件更小，但速度结论受内核限制

DGX Spark 的 CPU，4 线程，llama.cpp；不使用 GPU。困惑度只计算 WikiText-2 raw 测试集的前 4 个窗口，窗口大小为 512，实际计分位置数尚未独立确认。速度是 llama-bench 的 3 次重复均值，预填充 128 token、生成 64 token 的独立测试。

| 权重格式 | GGUF 文件大小 | 困惑度（估计 ± 标准误差） | 预填充 token/s | 生成 token/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1,509,347,712 B | 19.3301 ± 2.10644 | 17.85 | 12.52 |
| Q4_K_M | 484,220,288 B | 20.8566 ± 2.33258 | 246.57 | 121.69 |
| Q8_0 | 804,753,792 B | 19.2841 ± 2.09997 | 255.64 | 112.44 |

Q4 文件约为 BF16 的 32.1%，Q8 约为 53.3%；这是文件大小，并非运行时总内存。Q4 的困惑度在本次样本上高于 BF16，但样本只有 4 个窗口；Q8 数值略低不足以说明其比 BF16 效果更好。这些数字不衡量中文能力、指令遵循或具体任务准确率。

这次 BF16 的 CPU 速度异常低，不能据此推断量化普遍有 9 至 10 倍收益。硬件报告支持 BF16，但本次程序的 CPU 特性输出未列出 BF16；是否进入了低效内核仍需进一步验证。测试未绑定 CPU 核，设备含不同类型 CPU 核，调度也可能影响速度。GPU 性能和不同引擎的结果不能由这组数据推断。

## 原理：用更少位数近似权重

一种常见的仿射量化将权重 $w$ 映射到整数 $q$，推理时用 $\widehat{w}$ 近似原值：

$$q=\operatorname{clip}\!\left(\operatorname{round}\left(\frac{w}{s}\right)+z, q_{\min},q_{\max}\right),\quad \widehat{w}=s(q-z).$$

其中 $s$ 是缩放因子、$z$ 是零点。舍入和截断会引入误差；分块、缩放和混合精度的实际编码因格式而异，上式是通用示意，并非 Q4_K_M 的逐字节定义。

理想情况下，$P$ 个参数全部以 $b$ 位存储需 $Pb/8$ 字节；实际还需要尺度等元数据，部分张量也可能采用不同精度。运行时还要计入 KV cache、计算缓冲与其他开销。权重是 4 位不意味着 KV cache 和所有运算都使用 4 位。

困惑度是被计分 token 平均负对数概率的指数：

$$\operatorname{PPL}=\exp\!\left[-\frac{1}{N}\sum_{i\in S}\log p(x_i\mid c_i)\right].$$

其中 $S$ 是实际计分的位置集合，$N=|S|$，$c_i$ 是对应位置实际可见的上下文。窗口化评估会限制上下文，可能只对窗口中的一部分位置计分；因此这里不能把 $c_i$ 简化为整篇文本的全部前文。上表的精确计分位置及窗口规则因二进制源码未锁定而无法从现有证据确认。

同一分词、数据、计分位置和上下文设置下，困惑度上升说明模型给被计分位置的实际 token 赋予的几何平均概率变低；它不是任务正确率。实际选型还要在目标领域语料、检索或问答任务上比较质量和端到端延迟。

## 方法与格式

训练后量化 PTQ 在训练完成后转换权重，某些方法需要校准数据；量化感知训练 QAT 则在训练中模拟量化误差，训练后仍需转换和对应推理内核。[GPTQ](https://arxiv.org/abs/2210.17323) 利用近似二阶信息补偿量化误差；[AWQ](https://arxiv.org/abs/2306.00978) 利用激活识别敏感通道并调整尺度。本次实验使用 llama.cpp 的 Q4_K_M 和 Q8_0 权重量化，不是 GPTQ 或 AWQ 对照。

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 是存储张量和元数据的格式，Q4_K_M、Q8_0 是相关量化类型；GGUF 本身不是量化算法。不同格式和引擎的结果，需要在相同模型、数据、硬件、批大小下重新测量。

## 尝试复现

来源模型为 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，revision `c1899de289a04d12100db370d81485cdf75e47ca`，权重 SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。数据为 [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext)，测试文件 SHA-256 为 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`。数据许可见数据集页面，文本不随本仓库分发。

设备上的 llama-bench 自报构建版本 `ccc3646`，实验时源码工作树为 `a0e6dc7384edb8f5d927155e1d26c10b4a310e6d`；两者不同，不能假定二进制由该工作树构建，也无法仅凭现有记录构建完全相同的程序；下列命令用于复做相同测试设置，不保证重现数值。以下变量分别指向模型目录、测试文本和构建好的 llama.cpp；转换使用的 Python 环境须安装转换器所需依赖。

```bash
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL" --outtype bf16 --outfile qwen-bf16.gguf
"$LLAMA_CPP/build/bin/llama-quantize" qwen-bf16.gguf qwen-q4.gguf Q4_K_M 4
"$LLAMA_CPP/build/bin/llama-quantize" qwen-bf16.gguf qwen-q8.gguf Q8_0 4
for model in qwen-bf16.gguf qwen-q4.gguf qwen-q8.gguf; do
  CUDA_VISIBLE_DEVICES='' "$LLAMA_CPP/build/bin/llama-perplexity" -m "$model" -f "$WIKI_TEST" -ngl 0 -t 4 -c 512 -b 128 --chunks 4
  CUDA_VISIBLE_DEVICES='' "$LLAMA_CPP/build/bin/llama-bench" -m "$model" -ngl 0 -t 4 -p 128 -n 64 -r 3 -o json
done
```

[原始速度 JSON 与困惑度输出](results/)保留了每次计时样本和估计误差，文件中的设备绝对路径已去除。`llama-perplexity` 默认窗口计分规则与源码版本相关；现存记录不足以确认本次二进制对应的计分规则。模型载入时出现控制 token 类型修正提示，三组一致。旧的 `quantize_qwen.py` 使用另一模型和 bitsandbytes，包含无法复核的历史输出；其中标为 FP32 的分支没有显式指定 dtype，不能确认实际精度。它不属于本次 GGUF 实测，运行时可能使用 GPU。

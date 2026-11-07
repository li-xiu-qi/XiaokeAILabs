# Quantizing language models: size, perplexity, and inference speed

[中文](README.md)

Model quantization reduces the numeric precision of some weights, lowering storage and weight-transfer requirements. Whether it accelerates inference depends on the hardware and implementation; quality impact must also be evaluated on application data. This experiment compares BF16, Q8_0, and Q4_K_M versions of the same Qwen3-0.6B weights in the same CPU inference engine. It documents one trade-off, but the binary's exact source version was not locked, leaving a gap in exact reproducibility.

## Measurements: smaller files, with an important kernel limitation

The test used a DGX Spark CPU, four threads, and llama.cpp, with no GPU layers. Perplexity was calculated on only the first four 512-token windows of the WikiText-2 raw test set; the number of scored positions has not been independently confirmed. Speed is the mean of three llama-bench runs, with separate 128-token prefill and 64-token generation tests.

| Weight format | GGUF file size | Perplexity (estimate ± reported value) | Prefill tokens/s | Generation tokens/s |
|---|---:|---:|---:|---:|
| BF16 | 1,509,347,712 B | 19.3301 ± 2.10644 | 17.85 | 12.52 |
| Q4_K_M | 484,220,288 B | 20.8566 ± 2.33258 | 246.57 | 121.69 |
| Q8_0 | 804,753,792 B | 19.2841 ± 2.09997 | 255.64 | 112.44 |

The ± values are taken from each `llama-perplexity` final output. The available record cannot verify how this binary computed them, and they are not a measure of variation across repeated runs or datasets. The formats share the same windows; assessing the stability of their quality differences requires more data and a paired evaluation.

The Q4 and Q8 files are approximately 32.1% and 53.3% of the BF16 file size. These are historically recorded file sizes, not total runtime memory; the original GGUF files and an independent file-size listing are not included, so these byte counts cannot be independently checked from the repository; the benchmark JSON `model_size` fields differ and should not be substituted for file sizes. Q4 perplexity was higher on this sample, but the sample contains only four windows; the slightly lower Q8 value does not show that Q8 is better than BF16. The measurements do not assess Chinese capability, instruction following, or accuracy on a particular task.

BF16 CPU throughput was unusually low here. The measurements do **not** show that quantization generally yields a 9–10× speedup. The hardware reports BF16 support, but the program's CPU-feature output did not list BF16; whether it used an inefficient kernel remains unverified. CPU cores were not pinned, and the device has different core types, so scheduling may also affect speed. These results cannot establish GPU performance or behavior in other engines.

## Mechanism: approximate weights with fewer bits

One common affine quantization maps a weight $w$ to integer $q$ and approximates it at inference as $\widehat{w}$:

$$q=\operatorname{clip}\!\left(\operatorname{round}\left(\frac{w}{s}\right)+z, q_{\min},q_{\max}\right),\quad \widehat{w}=s(q-z).$$

Here $s$ is the scale and $z$ the zero point. Rounding and clipping introduce error. Actual block layouts, scales, and mixed-precision encoding vary by format; this is a general illustration, not the byte-level definition of Q4_K_M.

Ideally, storing $P$ parameters at $b$ bits each takes $Pb/8$ bytes. Real formats also need metadata such as scales, and some tensors may use different precision. Runtime memory additionally includes KV cache, computation buffers, and other overhead. Four-bit weights do not imply that KV cache and all operations use four bits.

Perplexity is the exponential of mean negative log probability over scored tokens:

$$\operatorname{PPL}=\exp\!\left[-\frac{1}{N}\sum_{i\in S}\log p(x_i\mid c_i)\right].$$

Here $S$ is the set of scored positions, $N=|S|$, and $c_i$ is the context actually visible at position $i$. Windowing limits context and may score only part of each window; this binary is not tied to a verified source version, so the exact historical scoring rule is unconfirmed. With the same tokenizer, data, scored positions and context settings, increased perplexity means lower geometric-mean probability for the scored tokens. It is not task accuracy. Evaluate quality and end-to-end latency on the target domain and application, such as retrieval or question answering, before choosing a format.

## Methods and formats

Post-training quantization (PTQ) converts weights after training; some methods need calibration data. Quantization-aware training (QAT) simulates quantization error during training and still needs conversion and a suitable inference kernel. [GPTQ](https://arxiv.org/abs/2210.17323) uses approximate second-order information to compensate quantization error; [AWQ](https://arxiv.org/abs/2306.00978) uses activations to identify sensitive channels and adjust scales. This experiment uses llama.cpp Q4_K_M and Q8_0 weight quantization; it is not a GPTQ or AWQ comparison.

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) stores tensors and metadata; Q4_K_M and Q8_0 are associated quantization types. GGUF itself is not a quantization algorithm. Compare different formats or engines again under the same model, data, hardware, and batch settings.

## Attempt reproduction

The source model was [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), revision `c1899de289a04d12100db370d81485cdf75e47ca`; the weights' SHA-256 was `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. The data was [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext); the test file SHA-256 was `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. Consult the dataset page for its license; the text is not distributed in this repository.

The device's llama-bench binary reported build `ccc3646`, while the source working tree at experiment time was `a0e6dc7384edb8f5d927155e1d26c10b4a310e6d`. Because they differ, the binary cannot be assumed to have been built from that tree, and the existing record does not identify enough to reconstruct the exact program. The commands below reproduce the test settings, not necessarily the numerical results. Set the variables to the model directory, test text, and built llama.cpp tree; install the converter's Python dependencies.

```bash
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL" --outtype bf16 --outfile qwen-bf16.gguf
"$LLAMA_CPP/build/bin/llama-quantize" qwen-bf16.gguf qwen-q4.gguf Q4_K_M 4
"$LLAMA_CPP/build/bin/llama-quantize" qwen-bf16.gguf qwen-q8.gguf Q8_0 4
for model in qwen-bf16.gguf qwen-q4.gguf qwen-q8.gguf; do
  CUDA_VISIBLE_DEVICES='' "$LLAMA_CPP/build/bin/llama-perplexity" -m "$model" -f "$WIKI_TEST" -ngl 0 -t 4 -c 512 -b 128 --chunks 4
  CUDA_VISIBLE_DEVICES='' "$LLAMA_CPP/build/bin/llama-bench" -m "$model" -ngl 0 -t 4 -p 128 -n 64 -r 3 -o json
done
```

[Raw speed JSON and perplexity output](results/) retain timing samples and reported estimate errors; device-specific absolute paths have been removed. The default window-scoring rule in `llama-perplexity` depends on the source version, and the available record does not confirm the rule in this binary. A control-token type correction warning occurred consistently in all three model loads.

The older `quantize_qwen.py` uses another model and bitsandbytes and contains historical outputs that cannot be checked. Its branch labeled FP32 does not explicitly specify dtype, so its actual precision cannot be confirmed. It is not part of this GGUF measurement and may use a GPU.

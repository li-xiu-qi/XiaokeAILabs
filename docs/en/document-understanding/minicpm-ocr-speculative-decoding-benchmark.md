# Real-Hardware Inference Benchmark: Baseline vs. DSpark Speculative Decoding

> **Hardware**: NVIDIA GB10 (`aarch64`, PyTorch `2.14.0+cu130`, `bfloat16`, concurrency = 1)  
> **Target Model**: `OpenBMB/MiniCPM-OCR-1.0` (`Qwen3_5ForConditionalGeneration`, 1.3B, 24 layers: 18 `GatedDeltaNet` linear attention + 6 full attention)  
> **Draft Model**: `OpenBMB/MiniCPM-OCR-1.0-DSpark` (`Qwen3DSparkModel` / `MuseGlimmerAssistantModel`, 0.3B, 5 bidirectional block layers + low-rank Markov transition head `markov_w1`/`markov_w2` + `d2t` mapping, `block_size = 15`, `target_layer_ids = [3, 7, 11, 15, 19]`)

---

## 1. Overall Summary (4 Representative OCR Tasks, 343 Tokens)

One input is cropped from a synthetic page; the other three are official example blocks. Timing starts with the first full-input model forward pass and includes prefill and subsequent generation. Image loading and processor preprocessing are outside the timed interval. Throughput is generated tokens divided by this interval, not decode-only throughput. Token parity applies to these four greedy-output comparisons. Each case was measured once.

| Metric | Baseline (Standard Greedy Decoding) | Accelerated (+ `MiniCPM-OCR-1.0-DSpark`) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Total Model Generation Wall Time (4 tasks)** | `4.6461 s` | **`1.9676 s`** | **-57.6% latency (`2.36x` speedup)** |
| **Generated Tokens / Model Generation Time** | `73.83 tokens/s` | **`174.32 tokens/s`** | **+136.1% throughput** |
| **Average Per-Token Latency** | `13.55 ms/token` | **`5.74 ms/token`** | **-57.6% per-token time** |
| **Target Model (1.3B) Forward Passes** | `343` | **`44`** | **-87.2% target forwards** |
| **Draft Model (0.3B) Forward Passes** | `0` | **`40`** | `block_size = 15` |
| **Mean Accepted Tokens per Step ($\tau$)** | `1.00` | **`9.18 tokens/step`** | Unweighted mean of the four case averages; `11.40` on LaTeX formulas |
| **Greedy Token Parity** | Reference (`343/343`) | **`100% Identical (`True`)`** | Parity on the four tested samples |
| **PyTorch peak allocated memory (BF16)** | Not measured separately | **`4.33 GiB`** | Both models loaded during the measured cases; no separate drafter increment measured |

---

## 2. Per-Task Breakdown

| Task ID | Scenario | Tokens | Baseline Time (`tok/s`) | DSpark Time (`tok/s`) | Target Steps (`Base → DSpark`) | Mean Accept Length ($\tau$) | Speedup | Token Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `case1_cn_document` | Chinese Document Page (Text + Table + Formula) | `128` | `1.875s` (`68.25`) | **`0.935s` (`136.83`)** | `128 → 18` | `7.53` | **`2.00x`** | `True` |
| `case2_text_putnam` | Mathematical Competition Proof Text (`01_text_putnam`) | `84` | `1.076s` (`78.07`) | **`0.358s` (`234.93`)** | `84 → 9` | `10.50` | **`3.01x`** | `True` |
| `case3_formula_putnam` | Multi-line LaTeX Formula Block (`02_formula_putnam`) | `57` | `0.744s` (`76.66`) | **`0.241s` (`236.22`)** | `57 → 6` | `11.40` | **`3.08x`** | `True` |
| `case4_table_gb_standard` | Chinese National Standard Table (`03_table_gb_standard`) | `74` | `0.951s` (`77.79`) | **`0.433s` (`170.78`)** | `74 → 11` | `7.30` | **`2.20x`** | `True` |

---

## 3. Why DSpark Achieves 2.00x–3.08x Speedup on OCR Tasks

1. **Bidirectional Block Drafting (`block_size = 15`) + Low-Rank Markov Head**:
   - Unlike standard autoregressive EAGLE drafters that require 15 sequential draft forward passes to propose 15 tokens, `MiniCPM-OCR-1.0-DSpark` fuses hidden states from 5 target layers (`[3, 7, 11, 15, 19]`) and predicts all 15 draft positions in **one** 5-layer bidirectional forward pass, then refines sequential token transitions via a lightweight rank-256 Markov head (`markov_w1`, `markov_w2`) and `d2t` vocabulary mapping (`114,688 → 248,320`).
2. **High Structural Determinism in OCR Output**:
   - LaTeX formulas (`\begin{aligned}`, `\frac{...}{...}`), structured tables (`<table><tr><td>`), and standard mathematical prose exhibit strong local token predictability, allowing the drafter to achieve an average acceptance length of **7.30 to 11.40 tokens per target step** (mean **9.18 tokens/step**).
3. **Hybrid State Rollback for `Qwen3_5GatedDeltaNet`**:
   - The implementation (`patch_target_for_speculative_rollback` in [`scripts/benchmark_acceleration.py`](../../../experiments/minicpm-ocr/scripts/benchmark_acceleration.py)) records per-token recurrent states during block verification (`seq_len <= 16`) and rolls back both `conv_states` and `recurrent_states` alongside full-attention KV caches. The four tested outputs match the greedy baseline token for token; this does not establish parity for all inputs.

---

## 4. Reproduction Command

```bash
python scripts/benchmark_acceleration.py \
  --target-dir models/MiniCPM-OCR-1.0 \
  --dspark-dir models/MiniCPM-OCR-1.0-DSpark \
  --testdata-dir testdata \
  --examples-dir repo/scripts/minicpm-ocr-jev/examples \
  --output-json "$HOME/minicpm-ocr/runs/acceleration_benchmark.json"
```

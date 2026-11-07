# Real-Hardware Inference Benchmark: Baseline vs. DSpark Speculative Decoding

> **Hardware**: NVIDIA GB10 (`aarch64`, PyTorch `2.14.0+cu130`, `bfloat16`, concurrency = 1)  
> **Target Model**: `OpenBMB/MiniCPM-OCR-1.0` (`Qwen3_5ForConditionalGeneration`, 1.3B, 24 layers: 18 `GatedDeltaNet` linear attention + 6 full attention)  
> **Draft Model**: `OpenBMB/MiniCPM-OCR-1.0-DSpark` (`Qwen3DSparkModel` / `MuseGlimmerAssistantModel`, 0.3B, 5 bidirectional block layers + low-rank Markov transition head `markov_w1`/`markov_w2` + `d2t` mapping, `block_size = 15`, `target_layer_ids = [3, 7, 11, 15, 19]`)

---

## 1. Overall Summary (4 OCR Example Blocks, 343 Tokens)

The interval begins after image loading and preprocessing. It includes the first image-conditioned target-model forward pass and subsequent generation, not just decode; it excludes model loading and is not end-to-end OCR latency. The first input is a crop from a self-generated page; the other three are official example blocks, not a representative sample of real documents. Both implementations were warmed up, then the baseline ran before DSpark on each case, once. The JSON records summary numbers and equality flags, not complete token sequences, and the three official images are not included in this repository; independent reconstruction requires restoring those inputs and rerunning.

| Metric | Baseline (Standard Greedy Decoding) | Accelerated (+ `MiniCPM-OCR-1.0-DSpark`) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Total Model-Generation Wall Time (4 cases)** | `4.6461 s` | **`1.9676 s`** | **-57.6% interval time (`2.36x` ratio)** |
| **Model-Generation Throughput** | `73.83 tokens/s` | **`174.32 tokens/s`** | **+136.1% throughput** |
| **Interval Time per Generated Token** | `13.55 ms/token` | **`5.74 ms/token`** | Includes the first forward pass |
| **Target Model (1.3B) Forward Passes** | `343` | **`44`** | **-87.2% target forwards** |
| **Draft Model (0.3B) Forward Passes** | `0` | **`40`** | `block_size = 15` |
| **Mean of per-case `n_accept + 1` averages** | Not applicable | **`9.18`** | Script-defined count, not a strict accepted-token count |
| **Greedy Token Parity** | Reference | **All four summaries report `True`** | Complete token sequences were not retained for independent verification |
| **Peak PyTorch Allocation (BF16)** | Not measured separately | **`4.33 GiB`** | Both models were loaded throughout; incremental draft-model cost cannot be calculated |

---

## 2. Per-Task Breakdown

| Task ID | Scenario | Tokens | Baseline Time (`tok/s`) | DSpark Time (`tok/s`) | Target Steps (`Base → DSpark`) | Mean `n_accept + 1` | Speedup | Token Parity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `case1_cn_document` | Chinese Document Page (Text + Table + Formula) | `128` | `1.875s` (`68.25`) | **`0.935s` (`136.83`)** | `128 → 18` | `7.53` | **`2.00x`** | `True` |
| `case2_text_putnam` | Mathematical Competition Proof Text (`01_text_putnam`) | `84` | `1.076s` (`78.07`) | **`0.358s` (`234.93`)** | `84 → 9` | `10.50` | **`3.01x`** | `True` |
| `case3_formula_putnam` | Multi-line LaTeX Formula Block (`02_formula_putnam`) | `57` | `0.744s` (`76.66`) | **`0.241s` (`236.22`)** | `57 → 6` | `11.40` | **`3.08x`** | `True` |
| `case4_table_gb_standard` | Chinese National Standard Table (`03_table_gb_standard`) | `74` | `0.951s` (`77.79`) | **`0.433s` (`170.78`)** | `74 → 11` | `7.30` | **`2.20x`** | `True` |

---

## 3. How the implementation can reduce target-model passes

1. **Bidirectional Block Drafting (`block_size = 15`) + Low-Rank Markov Head**:
   - This implementation fuses hidden states from 5 target layers (`[3, 7, 11, 15, 19]`) and predicts 15 draft positions in one 5-layer bidirectional forward pass, then applies sequential token transitions via a low-rank Markov head (`markov_w1`, `markov_w2`) and `d2t` vocabulary mapping.
2. **Draft verification count**:
   - These examples report per-case means of **7.30 to 11.40** for `n_accept + 1` (unweighted mean **9.18**). The script adds one to the number of verified draft tokens each iteration; that extra token is not necessarily emitted when the last iteration ends at EOS or the length limit. It is not a strict accepted-token or emitted-token count. The examples do not isolate output structure as a cause of the measured differences.
3. **Hybrid state rollback**:
   - The implementation ([`benchmark_acceleration.py`](../../scripts/benchmark_acceleration.py)) records recurrent states during block verification (`seq_len <= 16`) and rolls back convolution and recurrent states alongside full-attention KV caches. The recorded run reports equal token IDs on these four inputs; it does not prove numerical identity of intermediate values or parity on arbitrary inputs.

---

## 4. Reproduction Command

```bash
python scripts/benchmark_acceleration.py \
  --target-dir models/MiniCPM-OCR-1.0 \
  --dspark-dir models/MiniCPM-OCR-1.0-DSpark \
  --testdata-dir testdata \
  --examples-dir repo/scripts/minicpm-ocr-jev/examples \
  --output-json /tmp/minicpm-ocr-acceleration-new.json
```

Choose a new output path so a rerun does not overwrite the historical summary. The wrapper refuses to overwrite its output and accepts `OCR_BENCHMARK_OUTPUT` for a new path.

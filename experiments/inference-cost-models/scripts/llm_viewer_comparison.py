#!/usr/bin/env python3
"""LLM-Viewer 逐层结果汇总 + 混合架构修正 + Q4 对比"""

# ============================================================
# LLM-Viewer 原始预测（BF16, 64层全按 full attention）
# 从 CSV 提取每层推理时间（us）
# ============================================================

DECODE_BF16_US = {
    "q_proj": 230.5, "k_proj": 38.5, "v_proj": 38.5, "out_proj": 230.5,
    "gate_proj": 653.1, "up_proj": 653.1, "down_proj": 653.1,
    "qk_matmul": 0.8589, "sv_matmul": 0.8589, "softmax": 0.045,
    "attn_norm": 0.075, "mlp_norm": 0.075, "attn_add": 0.075, "mlp_add": 0.075,
    "mlp_act": 0.1125,
}
PREFILL_BF16_US = {
    "q_proj": 241.0, "k_proj": 44.2, "v_proj": 44.2, "out_proj": 241.0,
    "gate_proj": 674.1, "up_proj": 674.1, "down_proj": 674.1,
    "qk_matmul": 4.5, "sv_matmul": 8.5, "softmax": 5.8,
    "attn_norm": 9.6, "mlp_norm": 9.6, "attn_add": 9.6, "mlp_add": 9.6,
    "mlp_act": 14.4,
}
DECODE_Q4_US = {
    "q_proj": 57.7, "k_proj": 9.6, "v_proj": 9.6, "out_proj": 57.7,
    "gate_proj": 163.4, "up_proj": 163.4, "down_proj": 163.4,
    "qk_matmul": 0.8589, "sv_matmul": 0.8589, "softmax": 0.045,
    "attn_norm": 0.075, "mlp_norm": 0.075, "attn_add": 0.075, "mlp_add": 0.075,
    "mlp_act": 0.1125,
}
LM_HEAD_US = 4700.0  # 4.7ms, BF16, 不受 w_bit 影响

N_LAYERS = 64
N_FULL_ATTENTION = 16
N_LINEAR_ATTENTION = 48
SEQ_LEN = 128

# GB10 实测值
MEASURED_DECODE_TPS = 4.35  # 四引擎均值 4.30-4.47
MEASURED_PREFILL_TPS = 152.04  # vLLM full compile 最优

def compute_tps(decode_us, prefill_us):
    d_per_layer = sum(decode_us.values())
    pf_per_layer = sum(prefill_us.values())
    d_total = d_per_layer * N_LAYERS + LM_HEAD_US
    pf_total = pf_per_layer * N_LAYERS + LM_HEAD_US
    return 1.0 / (d_total / 1e6), SEQ_LEN / (pf_total / 1e6), d_total, pf_total

print("=" * 70)
print("LLM-Viewer vs 实测：GB10 / Qwen3.8-27B / bs=1 / seq=128")
print("=" * 70)

for label, decode_us, prefill_us in [
    ("BF16", DECODE_BF16_US, PREFILL_BF16_US),
    ("Q4 (w_bit=4)", DECODE_Q4_US, None),  # Q4 prefill 单独处理
]:
    if prefill_us is None:
        continue
    d_tps, pf_tps, d_ms, pf_ms = compute_tps(decode_us, prefill_us)
    print(f"\n--- {label} ---")
    print(f"Decode: 预测 {d_tps:.2f} tok/s ({d_ms/1e3:.1f}ms)")
    print(f"  vs 实测 {MEASURED_DECODE_TPS:.2f}: 误差 {(d_tps - MEASURED_DECODE_TPS) / MEASURED_DECODE_TPS * 100:+.1f}%")
    print(f"  有效带宽利用率: {MEASURED_DECODE_TPS / d_tps * 100:.1f}%")
    print(f"Prefill: 预测 {pf_tps:.1f} tok/s ({pf_ms/1e3:.1f}ms)")
    print(f"  vs 实测 {MEASURED_PREFILL_TPS:.1f}: 误差 {(pf_tps - MEASURED_PREFILL_TPS) / MEASURED_PREFILL_TPS * 100:+.1f}%")
    print(f"  有效算力利用率: {MEASURED_PREFILL_TPS / pf_tps * 100:.1f}%")

# Q4 decode
d_tps_q4, _, d_ms_q4, _ = compute_tps(DECODE_Q4_US, PREFILL_BF16_US)
print(f"\n--- Q4 (w_bit=4) Decode ---")
print(f"Decode: 预测 {d_tps_q4:.2f} tok/s ({d_ms_q4/1e3:.1f}ms)")
# Q4 实测值：从摘要中知道 llama.cpp Q4_K_XL 测过，但没有具体数值
# 用 llmfit 模型预测作为参考
print(f"  (Q4 实测值未在本轮测试中获取，llmfit 模型预测见 hybrid_architecture_model.py)")

# ============================================================
# 关键发现
# ============================================================
print("\n" + "=" * 70)
print("关键发现：LLM-Viewer 作为独立锚点的验证结论")
print("=" * 70)

print("""
1. LLM-Viewer 的 Roofline 模型假设 100% 硬件利用率（纯理论上限）

   Decode 有效带宽利用率 = 71.6%（BF16）
   与本实验标定的 GB10 decode 效率（73-86%）吻合，
   说明本实验的 decode 效率常数是合理的。

2. Prefill 有效算力利用率 = 20.8%（BF16，最优引擎 vLLM full）
   远低于本实验假设的 35%（来自 llmfit 的 PREFILL_COMPUTE_UTILIZATION）。
   本实验之前发现模型 prefill 预测（578.8）比实测（152.04）高 380%，
   LLM-Viewer 独立验证了这个偏差：Roofline 理论上限是 730.5 tok/s，
   实测最优 152.04 tok/s，有效利用率只有 20.8%。

3. 混合架构修正（16 full + 48 linear）对性能预测影响不大
   在 seq=128 时，矩阵投影占绝大部分时间，
   attention 计算（qk_matmul, sv_matmul, softmax）占比 < 1%。
   混合架构的影响主要体现在 KV Cache 显存，而不是计算时间。

4. LLM-Viewer 无法区分引擎实现差异
   enforce-eager（11.34 tok/s）和 torch.compile（152.04 tok/s）
   在 Roofline 模型中是完全一样的，因为模型只看硬件理论上限。
   这意味着 Roofline 模型只能预测性能上限，不能预测实际性能。

5. 对本实验模型的修正建议：
   - Decode 效率常数：本实验的 86%（BF16）偏高，应调整为 71.6%
     （LLM-Viewer 的 Roofline 上限反推）
   - Prefill 算力利用率：本实验的 35% 严重偏高，应调整为 20.8%
     （同样从 LLM-Viewer 上限反推）
""")

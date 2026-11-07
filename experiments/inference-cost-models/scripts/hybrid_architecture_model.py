"""Mixed-architecture memory arithmetic, not a performance prediction.

Only standard full-attention KV storage is counted. Linear-layer state,
allocator overhead and workspace are caller-supplied, not measured here.
All values ending in _gb use decimal GB (10**9 bytes).
"""
import argparse
import math

KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "fp4": 0.5}


def kv_cache_gb(n_full_layers, n_kv_heads, head_dim, ctx, kv_quant="fp16", batch_size=1):
    """Dense full-attention KV bytes / 1e9 for equal-length sequences."""
    dimensions = (n_full_layers, n_kv_heads, head_dim, ctx, batch_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in dimensions):
        raise ValueError("dimensions must be non-negative integers")
    if kv_quant not in KV_BYTES:
        raise ValueError(f"unknown KV precision: {kv_quant}")
    return (2 * n_full_layers * n_kv_heads * head_dim * ctx * batch_size
            * KV_BYTES[kv_quant] / 1e9)


def estimate_memory_gb(params_b, bpp, kv_gb, overhead=0.0):
    """Sum decimal GB; caller supplies all excluded memory in overhead."""
    if any(not math.isfinite(value) or value < 0 for value in (params_b, bpp, kv_gb, overhead)):
        raise ValueError("memory components must be finite and non-negative")
    return params_b * bpp + kv_gb + overhead


def qwen3_8_27b_analysis(quant="bf16", ctx=128, batch_size=1, kv_quant="fp16"):
    """Illustrative config; verify model identity and file size before use."""
    bpp = {"bf16": 2.0, "q4_k_xl": 0.60}.get(quant)
    if bpp is None:
        raise ValueError(f"unknown example weight format: {quant}")
    kv = kv_cache_gb(16, 4, 256, ctx, kv_quant, batch_size)
    return {"weight_estimate_gb": 27.3*bpp, "kv_cache_gb": kv,
            "subtotal_gb": estimate_memory_gb(27.3, bpp, kv),
            "note": "subtotal excludes linear-layer states and runtime overhead"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16, help="full-attention layers")
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-dtype", choices=KV_BYTES, default="bf16")
    args = parser.parse_args()
    try:
        kv_gb = kv_cache_gb(args.layers, args.kv_heads, args.head_dim,
                            args.context, args.kv_dtype, args.batch)
    except ValueError as error:
        parser.error(str(error))
    print("KV payload only; excludes quantization metadata, other layer state and runtime overhead")
    print(f"KV: {kv_gb:.9f} GB = {kv_gb * 1e9 / 2**30:.9f} GiB")

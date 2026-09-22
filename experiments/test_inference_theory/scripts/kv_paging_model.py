"""
KV Cache 分页与前缀共享模型：真实服务里 KV cache 的两项修正。

前面的 KV cache 模型假设连续内存分配（正好 S 个 token 的 KV）。
真实推理服务（vLLM 的 PagedAttention）用分页管理，带来两项修正：

1. 分页内部碎片：KV cache 按固定页（block，如 16 token）分配，
   最后一个页只填一部分，平均浪费半页。
2. 前缀共享收益：多请求共享相同前缀（系统提示、few-shot 示例），
   命中时前缀的 KV 只算一次、只存一份，大幅省显存和 prefill 算力。

本模型只算这两项，前缀命中率需用真实服务数据校准（先用参数化）。

自测：分页碎片和前缀共享的公式自测，命中率收益用参数扫描展示。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import Qwen3_8_27B, kv_bytes_per_token


def paged_kv_bytes(model, ctx_len: int, page_size: int = 16) -> float:
    """分页分配的 KV cache 字节数 = ceil(S/page) * page * per_token。

    最后一个页可能只填一部分，产生内部碎片。
    """
    pages = (ctx_len + page_size - 1) // page_size
    return pages * page_size * kv_bytes_per_token(model)


def continuous_kv_bytes(model, ctx_len: int) -> float:
    """连续分配的 KV cache 字节数 = S * per_token（前面模型用的）。"""
    return ctx_len * kv_bytes_per_token(model)


def paging_overhead_ratio(model, ctx_len: int, page_size: int = 16) -> float:
    """分页碎片开销 = 分页字节 / 连续字节 - 1。

    页越大，单页浪费越多；请求越短，碎片占比越高（最后一页占比大）。
    """
    paged = paged_kv_bytes(model, ctx_len, page_size)
    cont = continuous_kv_bytes(model, ctx_len)
    return paged / cont - 1


def prefix_sharing_savings(model, num_requests: int, prefix_len: int,
                           suffix_len: int) -> dict:
    """前缀共享的显存和算力收益。

    无共享：N 个请求各存各的 KV，总量 = N * (prefix+suffix) * per_token。
    有共享：prefix 的 KV 只存一份，各请求只存自己的 suffix。
    节省的显存 = (N-1) * prefix * per_token。
    节省的 prefill 算力同理（prefix 只 prefill 一次）。
    """
    per_tok = kv_bytes_per_token(model)
    no_share = num_requests * (prefix_len + suffix_len) * per_tok
    with_share = (prefix_len + num_requests * suffix_len) * per_tok
    saved_bytes = no_share - with_share
    return {
        "no_share_gb": no_share / 1e9,
        "with_share_gb": with_share / 1e9,
        "saved_gb": saved_bytes / 1e9,
        "saving_ratio": saved_bytes / no_share,
    }


def self_test():
    print("=" * 64)
    print("KV Cache 分页与前缀共享模型自测")
    print("=" * 64)

    print("\n[1] 分页内部碎片（page_size=16, Qwen3.8-27B）")
    print("    注意：上下文若是页大小整数倍则无碎片，否则最后一页浪费")
    print(f"{'上下文':>8} {'页数':>5} {'连续GB':>8} {'分页GB':>8} {'碎片开销':>9}")
    for ctx in [100, 500, 1000, 2000, 8192, 32768]:
        pages = (ctx + 15) // 16
        cont = continuous_kv_bytes(Qwen3_8_27B, ctx) / 1e9
        paged = paged_kv_bytes(Qwen3_8_27B, ctx) / 1e9
        overhead = paging_overhead_ratio(Qwen3_8_27B, ctx)
        print(f"{ctx:>8} {pages:>5} {cont:>8.3f} {paged:>8.3f} {overhead*100:>8.1f}%")

    print("\n[2] 碎片开销：短上下文占比高，长上下文可忽略（ctx=2000）")
    print(f"{'页大小':>6} {'页数':>5} {'碎片token':>10} {'碎片开销':>9}")
    for ps in [8, 16, 32, 64]:
        pages = (2000 + ps - 1) // ps
        waste_tokens = pages * ps - 2000
        overhead = paging_overhead_ratio(Qwen3_8_27B, 2000, ps)
        print(f"{ps:>6} {pages:>5} {waste_tokens:>10} {overhead*100:>8.1f}%")

    print("\n[3] 前缀共享收益（系统提示 2000 token, 各请求后缀 500 token）")
    print(f"{'请求数':>6} {'无共享GB':>9} {'共享GB':>8} {'节省GB':>8} {'节省比':>8}")
    for n in [1, 4, 16, 64]:
        r = prefix_sharing_savings(Qwen3_8_27B, n, 2000, 500)
        print(f"{n:>6} {r['no_share_gb']:>9.2f} {r['with_share_gb']:>8.2f} "
              f"{r['saved_gb']:>8.2f} {r['saving_ratio']*100:>7.0f}%")

    print("\n[4] 多轮对话的前缀共享（每轮增长，历史全部共享）")
    print("  场景：10 轮对话，每轮新增 200 token，前缀为全部历史")
    total_no_share = 0
    total_share = 0
    per_tok = kv_bytes_per_token(Qwen3_8_27B)
    for turn in range(1, 11):
        cur_len = 200 * turn
        # 无共享：每轮从头存全部
        total_no_share += cur_len * per_tok
        # 有共享：历史已存，只存新增 200
        total_share += 200 * per_tok
    print(f"  无共享累计 KV: {total_no_share/1e9:.2f} GB")
    print(f"  前缀共享累计:  {total_share/1e9:.2f} GB")
    print(f"  节省: {(1-total_share/total_no_share)*100:.0f}%")
    print("  → 多轮对话前缀共享收益极大，这是服务化推理的核心优化")

    print("\n[5] 与前面 KV cache 连续模型的关系")
    print("  前面的 KV cache 模型给出连续分配的下限（理论最小值）")
    print("  本模型加两项修正：分页碎片（+几%开销）")
    print("                       前缀共享（多请求/多轮时 -几十%开销）")
    print("  → 真实服务的 KV cache 显存 = 连续下限经分页和共享修正")
    print("  → 分页和前缀共享的精确收益需起 llama-server 后用真实命中率校准")


if __name__ == "__main__":
    self_test()

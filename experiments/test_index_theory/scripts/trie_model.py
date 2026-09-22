# -*- coding: utf-8 -*-
"""
Trie（字典树 / 前缀树）理论性能模型

Trie 把字符串拆成字符序列，从根到节点的路径拼出一个前缀。它是前缀搜索、自动补全、
IP 路由最长前缀匹配、倒排索引词项词典的标准结构。

核心结构：
- 每个节点代表一个字符，根节点代表空前缀
- 节点之间的边标有字符，根到某节点的路径 = 一个前缀
- 标了结束标志的节点对应字典中真实存在的一个键

核心指标：
1. 节点数：无共享上界 n × L；前缀共享会大幅减少实际节点数
2. 查找时间：O(L)，与集合大小 n 无关，只取决于键长
3. 前缀搜索时间：O(L + k)，L 是前缀长度，k 是命中的结果数
4. 空间：两种实现代价差异极大
   - 数组实现：每个节点固定 σ 个槽位，σ 越大浪费越严重
   - 哈希表实现：按实际子节点数分配，但每个节点有固定对象开销

本模块全部用闭式公式给出这些指标，不依赖任何真实 Trie 实现。
"""
from dataclasses import dataclass
import math


@dataclass
class TrieSpec:
    """Trie 参数"""
    n: int = 100_000               # 键数量
    avg_len: int = 12              # 平均键长 L（主模型按定长键处理）
    sigma: int = 26                # 字符集大小（小写英文字母 = 26）
    pointer_bytes: int = 4         # 数组实现：每个子节点槽位的指针字节数
    terminator_bytes: int = 1      # 数组实现：节点上的结束标志字节数
    dict_overhead_bytes: int = 64  # 哈希表实现：每个节点 dict 对象的固定开销
    dict_entry_bytes: int = 100    # 哈希表实现：每个子节点条目的摊还开销
    array_entry_bytes: int = 8     # 数组实现（真实 Python list）：每个槽位摊还字节数


@dataclass
class TrieMetrics:
    n: int                          # 键数量
    avg_len: int                    # 键长
    sigma: int                      # 字符集大小
    node_count: int                 # 期望节点数（含前缀共享）
    node_count_upper_bound: int     # 无共享上界 n × L
    sharing_ratio: float            # node_count / 上界，越小说明前缀共享越有效
    lookup_ops: int                 # 查找操作数 = L（与 n 无关）
    prefix_search_ops: int          # 前缀搜索最小代价 = L + k，此处 k=1
    memory_array_bytes: int         # 数组实现内存
    memory_dict_bytes: int          # 哈希表实现内存
    memory_per_key_array: float     # 数组实现每键字节数
    memory_per_key_dict: float      # 哈希表实现每键字节数


def compute_node_count(n: int, length: int, sigma: int) -> float:
    """期望节点数（定长键）。

    深度 d 上的节点数 = 长度为 d 的互异前缀数。n 个独立均匀随机键在深度 d 产生
    n 个前缀，每个前缀落入 σ^d 个槽之一，故期望互异前缀数为

        E[N_d] = σ^d × (1 - (1 - σ^{-d})^n)

    总节点数 = Σ_{d=1}^{L} E[N_d]。d 较小时 σ^d 远小于 n，E[N_d] 接近 σ^d（该层被
    打满）；d 较大时 σ^d 远大于 n，E[N_d] 接近 n（几乎没有共享，每个键独占一条链）。
    拐点出现在 σ^d ≈ n，即 d ≈ log_σ(n)。

    注意计算必须走 expm1 稳定形式：σ^d 一大，1 - σ^{-d} 在 double 下会退化成 1.0，
    直接代入会让深层贡献整体归零。expm1 与 log1p 专为小量设计，不会丢精度。
    """
    if n <= 0 or length <= 0 or sigma <= 1:
        return 0.0
    total = 0.0
    for d in range(1, length + 1):
        slots = float(sigma) ** d
        # -expm1(m × log1p(-1/slots)) == 1 - (1 - 1/slots)^m，但对小量数值稳定
        total += slots * (-math.expm1(n * math.log1p(-1.0 / slots)))
    return total


def compute_node_count_varlen(n: int, len_at_least: list, sigma: int) -> float:
    """期望节点数（变长键）。

    len_at_least[d] 表示长度 ≥ d 的键有多少个。变长时只有长度够得着的键才会在深度 d
    上留下前缀，把定长公式里的 n 换成 len_at_least[d] 再逐层求和即可。
    """
    if n <= 0 or sigma <= 1:
        return 0.0
    total = 0.0
    for d in range(1, len(len_at_least)):
        m = len_at_least[d]
        if m <= 0:
            break
        slots = float(sigma) ** d
        total += slots * (-math.expm1(m * math.log1p(-1.0 / slots)))
    return total


def compute_lookup_ops(length: int) -> int:
    """查找操作数。每层一次子节点定位，共 L 次，与 n 无关。"""
    return length


def compute_prefix_search_ops(length: int, k: int) -> int:
    """前缀搜索操作数。先走 L 层定位到前缀节点，再输出 k 个命中结果。"""
    return length + k


def compute_memory_array(node_count: int, spec: TrieSpec) -> int:
    """数组实现内存。每个节点固定 σ 个槽位，无论实际有几个子节点。"""
    return node_count * (spec.sigma * spec.pointer_bytes + spec.terminator_bytes)


def compute_memory_dict(node_count: int, spec: TrieSpec) -> int:
    """哈希表实现内存。每个节点一个 dict（固定开销），每条边一个条目。"""
    edges = max(0, node_count - 1)  # 除根节点外，每个节点都是某个父节点的一条边
    return node_count * spec.dict_overhead_bytes + edges * spec.dict_entry_bytes


def compute(spec: TrieSpec) -> TrieMetrics:
    """给定参数，递推全部指标。"""
    node_count = int(round(compute_node_count(spec.n, spec.avg_len, spec.sigma)))
    upper = spec.n * spec.avg_len
    sharing = node_count / upper if upper else 0.0
    mem_array = compute_memory_array(node_count, spec)
    mem_dict = compute_memory_dict(node_count, spec)

    return TrieMetrics(
        n=spec.n,
        avg_len=spec.avg_len,
        sigma=spec.sigma,
        node_count=node_count,
        node_count_upper_bound=upper,
        sharing_ratio=sharing,
        lookup_ops=compute_lookup_ops(spec.avg_len),
        prefix_search_ops=compute_prefix_search_ops(spec.avg_len, 1),
        memory_array_bytes=mem_array,
        memory_dict_bytes=mem_dict,
        memory_per_key_array=mem_array / spec.n if spec.n else 0.0,
        memory_per_key_dict=mem_dict / spec.n if spec.n else 0.0,
    )


def _selftest():
    """公式自洽性检查"""
    spec = TrieSpec(n=100_000, avg_len=12, sigma=26)
    m = compute(spec)

    # 1. 节点数必须落在 [n, n×L] 区间内
    assert spec.n <= m.node_count <= m.node_count_upper_bound, \
        f"节点数 {m.node_count} 越界，应在 [{spec.n}, {m.node_count_upper_bound}]"

    # 2. 前缀共享必须真的省节点（键长大于 1 时）
    assert m.sharing_ratio < 1.0, "前缀共享应使节点数少于 n×L"

    # 3. 查找代价与 n 无关，只由 L 决定
    spec_big = TrieSpec(n=100_000_000, avg_len=12, sigma=26)
    assert compute(spec_big).lookup_ops == m.lookup_ops, "查找代价不应随 n 变化"

    # 4. 字符集越大，同长度下节点越多（更难被打满，也更难共享）
    spec_small_sigma = TrieSpec(n=100_000, avg_len=12, sigma=4)
    assert compute(spec_small_sigma).node_count < m.node_count, \
        "σ 越小，节点应越少（浅层更快被打满）"

    # 5. 数组实现的内存随 σ 线性增长，且 σ=256 时远大于哈希表实现
    spec_sigma256 = TrieSpec(n=100_000, avg_len=12, sigma=256)
    m256 = compute(spec_sigma256)
    assert m256.memory_array_bytes > m.memory_array_bytes
    assert m256.memory_per_key_array > m256.memory_per_key_dict, \
        "σ=256 时数组实现的每键开销应高于哈希表实现"

    # 6. 节点数随 n 单调增长
    spec_n2 = TrieSpec(n=200_000, avg_len=12, sigma=26)
    assert compute(spec_n2).node_count > m.node_count, "n 越大节点应越多"

    # 7. 前缀搜索代价 = L + k
    assert compute_prefix_search_ops(12, 50) == 62

    # 8. 变长键公式自洽：所有键长度都 ≥ 12 时，退化回定长公式
    full = [spec.n] * (spec.avg_len + 1)  # 每个深度上都有全部 n 个键的前缀
    assert round(compute_node_count_varlen(spec.n, full, 26)) == m.node_count

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  L={m.avg_len}  σ={m.sigma}")
    print(f"节点数={m.node_count:,}  （上界 n×L={m.node_count_upper_bound:,}，"
          f"共享比={m.sharing_ratio:.3f}）")
    print(f"查找={m.lookup_ops} 次操作  前缀搜索={m.prefix_search_ops} 次操作")
    print(f"数组实现={m.memory_array_bytes/1024/1024:.1f} MiB "
          f"（{m.memory_per_key_array:.0f} B/键）")
    print(f"哈希表实现={m.memory_dict_bytes/1024/1024:.1f} MiB "
          f"（{m.memory_per_key_dict:.0f} B/键）")


if __name__ == "__main__":
    _selftest()

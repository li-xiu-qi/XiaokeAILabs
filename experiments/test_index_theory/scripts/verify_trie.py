# -*- coding: utf-8 -*-
"""
Trie（字典树）实测验证

用纯 Python 自制迷你 Trie（dict 实现子节点），实测四个核心指标：
1. 节点数：插入 n 个随机字符串后，Trie 的实际节点数，对比理论期望
2. 查找延迟：O(L)，与集合大小无关
3. 前缀搜索延迟：O(L + k)，k 是命中结果数
4. 内存占用：实测 dict 实现的每键字节数，对比理论

对比不同字符集大小（σ=26/128/256）的节点内存，验证数组实现随 σ 线性膨胀的结论。

方法：生成 n 个长度 5-20 的随机字符串插入 Trie，实测上述指标。
结果写入 results/trie_<timestamp>.json
"""
import os
import sys
import json
import random
import string
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from trie_model import (
    TrieSpec, compute, compute_node_count, compute_node_count_varlen,
    compute_memory_array, compute_memory_dict,
)


class TrieNode:
    """Trie 节点。用 dict 存子节点，key 为字符，value 为子节点。"""

    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    """自制迷你 Trie。"""

    def __init__(self):
        self.root = TrieNode()
        self.node_count = 1  # 根节点
        self.comparison_count = 0  # 累计比较次数（查找用）

    def insert(self, key: str):
        """插入键。"""
        node = self.root
        for ch in key:
            child = node.children.get(ch)
            if child is None:
                child = TrieNode()
                node.children[ch] = child
                self.node_count += 1
            node = child
        node.is_end = True

    def search(self, key: str) -> bool:
        """查找键。返回是否命中。"""
        node = self.root
        for ch in key:
            self.comparison_count += 1
            node = node.children.get(ch)
            if node is None:
                return False
        self.comparison_count += 1  # 检查 is_end
        return node.is_end

    def prefix_search(self, prefix: str) -> list:
        """前缀搜索。返回所有以 prefix 开头的键。"""
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return []

        results = []
        stack = [(node, prefix)]
        while stack:
            current, path = stack.pop()
            if current.is_end:
                results.append(path)
            for ch, child in current.children.items():
                stack.append((child, path + ch))
        return results


def measure_node_count(keys: list, sigma: int) -> dict:
    """实测节点数，对比理论。"""
    trie = Trie()
    for key in keys:
        trie.insert(key)

    n = len(keys)
    # 变长键：按实际长度分布建理论曲线
    len_at_least = [0] * (max(len(k) for k in keys) + 1)
    for key in keys:
        for d in range(1, len(key) + 1):
            len_at_least[d] += 1
    theo = compute_node_count_varlen(n, len_at_least, sigma)

    # 定长近似：用平均长度代入
    avg_len = sum(len(k) for k in keys) / n
    theo_avg = compute_node_count(n, int(round(avg_len)), sigma)

    return {
        "n": n,
        "actual": trie.node_count,
        "theoretical_varlen": theo,
        "theoretical_avg_len": theo_avg,
        "upper_bound": n * max(len(k) for k in keys),
    }


def measure_lookup_latency(keys: list, num_queries: int = 1000) -> dict:
    """实测查找延迟。O(L)，与 n 无关。"""
    trie = Trie()
    for key in keys:
        trie.insert(key)

    sample = random.sample(keys, min(num_queries, len(keys)))
    total_comparisons = 0

    start = time.perf_counter()
    for key in sample:
        trie.search(key)
    elapsed = time.perf_counter() - start

    # 重新统计比较次数
    for key in sample:
        trie.search(key)
    total_comparisons = trie.comparison_count // len(sample)

    avg_len = sum(len(k) for k in sample) / len(sample)

    return {
        "n": len(keys),
        "num_queries": len(sample),
        "elapsed_ms": elapsed * 1000,
        "us_per_query": elapsed / len(sample) * 1e6,
        "avg_comparisons": total_comparisons,
        "avg_key_len": avg_len,
    }


def measure_prefix_latency(keys: list, prefix_len: int = 3, num_queries: int = 500) -> dict:
    """实测前缀搜索延迟。O(L + k)。"""
    trie = Trie()
    for key in keys:
        trie.insert(key)

    prefixes = [key[:prefix_len] for key in random.sample(keys, num_queries)]
    total_matches = 0
    total_comparisons = 0

    start = time.perf_counter()
    for prefix in prefixes:
        results = trie.prefix_search(prefix)
        total_matches += len(results)
    elapsed = time.perf_counter() - start

    return {
        "prefix_len": prefix_len,
        "num_queries": num_queries,
        "elapsed_ms": elapsed * 1000,
        "us_per_query": elapsed / num_queries * 1e6,
        "avg_matches": total_matches / num_queries,
    }


def measure_memory(keys: list, sigma: int) -> dict:
    """实测内存占用。"""
    trie = Trie()
    for key in keys:
        trie.insert(key)

    # 递归累加 sys.getsizeof
    def _node_size(node):
        total = sys.getsizeof(node)
        total += sys.getsizeof(node.children)
        for ch, child in node.children.items():
            total += sys.getsizeof(ch)
            total += _node_size(child)
        return total

    actual = _node_size(trie.root)
    theo_dict = compute_memory_dict(trie.node_count, TrieSpec(sigma=sigma))
    theo_array = compute_memory_array(trie.node_count, TrieSpec(sigma=sigma))

    return {
        "n": len(keys),
        "sigma": sigma,
        "node_count": trie.node_count,
        "actual_bytes": actual,
        "actual_bytes_per_key": actual / len(keys),
        "theoretical_dict_bytes": theo_dict,
        "theoretical_array_bytes": theo_array,
        "actual_over_dict": actual / theo_dict,
    }


def measure_sigma_comparison(keys: list) -> list:
    """对比不同 σ 下数组实现的节点内存（理论值，dict 实现与 σ 无关）。"""
    n = len(keys)
    trie = Trie()
    for key in keys:
        trie.insert(key)
    node_count = trie.node_count

    results = []
    for sigma in [26, 128, 256]:
        spec = TrieSpec(sigma=sigma)
        theo_array = compute_memory_array(node_count, spec)
        theo_dict = compute_memory_dict(node_count, spec)
        results.append({
            "sigma": sigma,
            "node_count": node_count,
            "theoretical_array_bytes": theo_array,
            "theoretical_array_bytes_per_key": theo_array / n,
            "theoretical_dict_bytes": theo_dict,
            "theoretical_dict_bytes_per_key": theo_dict / n,
            "array_over_dict": theo_array / theo_dict,
        })
    return results


def main():
    print("=== Trie 实测验证 ===\n")

    n = 100_000
    random.seed(42)
    # 生成长度 5-20 的随机字符串
    alphabet = string.ascii_lowercase
    keys = [
        "".join(random.choices(alphabet, k=random.randint(5, 20)))
        for _ in range(n)
    ]

    print(f"参数: n={n:,}, 长度 5-20, σ=26\n")

    # 1. 节点数
    print("--- 节点数 ---")
    node_result = measure_node_count(keys, sigma=26)
    print(f"实测: {node_result['actual']:,}")
    print(f"理论（变长）: {node_result['theoretical_varlen']:,.0f}")
    print(f"理论（定长近似，L={round(sum(len(k) for k in keys)/n)}）: "
          f"{node_result['theoretical_avg_len']:,.0f}")
    print(f"上界 n×L: {node_result['upper_bound']:,}")

    # 2. 查找延迟
    print("\n--- 查找延迟 ---")
    lookup_result = measure_lookup_latency(keys, num_queries=1000)
    print(f"实测: {lookup_result['us_per_query']:.2f} μs/次 "
          f"（平均键长 {lookup_result['avg_key_len']:.1f}）")

    # 3. 前缀搜索延迟
    print("\n--- 前缀搜索延迟 ---")
    prefix_result = measure_prefix_latency(keys, prefix_len=3, num_queries=500)
    print(f"实测: {prefix_result['us_per_query']:.2f} μs/次 "
          f"（平均命中 {prefix_result['avg_matches']:.1f} 个）")

    # 4. 内存
    print("\n--- 内存占用 ---")
    mem_result = measure_memory(keys, sigma=26)
    print(f"实测: {mem_result['actual_bytes']/1024/1024:.1f} MiB "
          f"（{mem_result['actual_bytes_per_key']:.0f} B/键）")
    print(f"理论（dict 实现）: {mem_result['theoretical_dict_bytes']/1024/1024:.1f} MiB")
    print(f"理论（数组实现 σ=26）: "
          f"{mem_result['theoretical_array_bytes']/1024/1024:.1f} MiB")

    # 5. σ 对比
    print("\n--- 不同 σ 的节点内存（理论）---")
    sigma_results = measure_sigma_comparison(keys)
    print(f"{'σ':>5}  {'数组实现 MiB':>15}  {'B/键':>8}  "
          f"{'dict 实现 MiB':>15}  {'数组/dict':>10}")
    for r in sigma_results:
        print(f"{r['sigma']:>5}  {r['theoretical_array_bytes']/1024/1024:>15.1f}  "
              f"{r['theoretical_array_bytes_per_key']:>8.0f}  "
              f"{r['theoretical_dict_bytes']/1024/1024:>15.1f}  "
              f"{r['array_over_dict']:>10.1f}x")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 Trie（dict 实现子节点），实测节点数、查找延迟、前缀搜索延迟、内存",
            "n": n,
            "key_len_range": [5, 20],
            "sigma": 26,
        },
        "node_count": node_result,
        "lookup_latency": lookup_result,
        "prefix_latency": prefix_result,
        "memory": mem_result,
        "sigma_comparison": sigma_results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"trie_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()

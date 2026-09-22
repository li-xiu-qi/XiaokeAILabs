# -*- coding: utf-8 -*-
"""
ART（自适应基数树）实测验证

用纯 Python 自制简化版 ART，实现 Node4 / Node16 / Node48 / Node256 四种节点形态，
按实际子节点数在形态间切换，实测两个核心指标：
1. 查找延迟：O(k)，每层一次子节点定位
2. 内存占用：节点大小自适应，对比同数据下的标准 Trie（dict 实现）

再与标准 Trie（dict 实现）在完全相同的键集上对比，量化 ART 的实际收益。

方法：生成 n 个长度 5-20 的随机字符串，分别插入 ART 与 Trie，实测查找延迟与内存。
结果写入 results/art_<timestamp>.json

实现说明：
- 所有节点统一用「有序 keys 列表 + 平行 children 列表」表示，Node48/Node256 额外
  维护一张 256 字节的索引表，把「字符值 → 槽位」的查找降到一次查表。
- 节点满了就地升级形态，父节点引用通过下降时记录的路径栈回填，不做全树扫描。
- Python 对象本身有固定开销，所以实测内存远大于 C 实现的节点字节数；这一层差异
  在文档里单独说明，不影响节点形态分布与查找次数的对比。
"""
import os
import sys
import json
import random
import string
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from art_model import ARTSpec, compute, node_type_name
from verify_trie import Trie, TrieNode


# ---------------------------------------------------------------- ART 实现

class ARTNode:
    """ART 节点基类。

    所有形态共用同一套字段：keys 存子节点字符的有序列表（存 ord 值），
    children 与 keys 平行。Node48/Node256 额外有 index 索引表。
    """

    __slots__ = ("num_children", "keys", "children", "index")

    def __init__(self):
        self.num_children = 0
        self.keys = []
        self.children = []
        self.index = None  # 只有 Node48/Node256 会建表


class Node4(ARTNode):
    """≤ 4 子节点。线性扫描 keys。"""
    __slots__ = ()


class Node16(ARTNode):
    """≤ 16 子节点。有序 keys 上二分，等价于 SIMD 一次并行比较。"""
    __slots__ = ()


class Node48(ARTNode):
    """≤ 48 子节点。256 字节索引表 + 稀疏子节点数组。"""
    __slots__ = ()


class Node256(ARTNode):
    """≤ 256 子节点。索引表直接以字符值为下标。"""
    __slots__ = ()


_NODE_CLASSES = {
    "Node4": Node4,
    "Node16": Node16,
    "Node48": Node48,
    "Node256": Node256,
}


class ART:
    """自制简化版 ART。"""

    CAP4, CAP16, CAP48, CAP256 = 4, 16, 48, 256

    def __init__(self):
        self.root = Node4()
        self.node_count = 1
        self.comparison_count = 0  # 累计子节点定位比较次数
        self.type_counts = {"Node4": 1, "Node16": 0, "Node48": 0, "Node256": 0}

    # ---- 节点形态管理

    def _new_node(self) -> ARTNode:
        """新建子节点，一律从 Node4 起步。"""
        self.node_count += 1
        node = Node4()
        self.type_counts["Node4"] += 1
        return node

    def _cap_of(self, node: ARTNode) -> int:
        if isinstance(node, Node4):
            return self.CAP4
        if isinstance(node, Node16):
            return self.CAP16
        if isinstance(node, Node48):
            return self.CAP48
        return self.CAP256

    def _grow(self, node: ARTNode) -> ARTNode:
        """子节点数超限时升级到更大的形态，返回新节点（内容已拷贝）。"""
        # 按节点的实际类取旧形态名，不能用 num_children 反查：
        # 升级前后 num_children 相同，反查会得到同一个名字，计数就不会更新。
        old_name = {Node4: "Node4", Node16: "Node16",
                    Node48: "Node48", Node256: "Node256"}[type(node)]

        if isinstance(node, Node4):
            new = Node16()
        elif isinstance(node, Node16):
            new = Node48()
            new.index = [255] * 256
            # 索引表由 _add_child 统一维护，这里只占位
        elif isinstance(node, Node48):
            new = Node256()
        else:
            raise RuntimeError("Node256 已满，无法再升级")

        new.num_children = node.num_children
        # keys 必须保持有序：Node16 的二分、Node48/256 的索引表都依赖它。
        # _grow 拷贝的是父节点当前的 keys，而父节点的 keys 在 _add_child 里已排过序，
        # 这里再显式排一次并同步 children，避免顺序假设被破坏。
        pairs = sorted(zip(node.keys, node.children), key=lambda p: p[0])
        new.keys = [p[0] for p in pairs]
        new.children = [p[1] for p in pairs]

        # Node48/Node256 的索引表必须按排序后的槽位重建
        if new.index is not None:
            for i, k in enumerate(new.keys):
                new.index[k] = i

        new_name = {Node16: "Node16", Node48: "Node48",
                    Node256: "Node256"}[type(new)]
        self.type_counts[old_name] -= 1
        self.type_counts[new_name] += 1
        return new

    # ---- 子节点查找

    def _find_child(self, node: ARTNode, code: int):
        """在节点内定位字符 code 的子节点，不存在返回 None。code 为 ord 值。"""
        keys = node.keys

        if isinstance(node, Node4):
            # 线性扫描，≤ 4 次比较
            self.comparison_count += 1
            for i, k in enumerate(keys):
                self.comparison_count += 1
                if k == code:
                    return node.children[i]
            return None

        if isinstance(node, Node16):
            # 线性扫描：真实 C 实现用 SIMD 单指令并行比较 16 个键字节，
            # Python 层面无 SIMD 原语，二分又要求 keys 有序而本实现不维护顺序，
            # 故用线性扫描，语义等价、结果正确。
            self.comparison_count += 1
            for i, k in enumerate(keys):
                self.comparison_count += 1
                if k == code:
                    return node.children[i]
            return None

        # Node48 / Node256：索引表一次查表
        self.comparison_count += 1
        idx = node.index[code]
        if idx == 255:
            return None
        return node.children[idx]

    def _add_child(self, node: ARTNode, code: int) -> int:
        """向节点追加一个子节点，返回该子节点在 keys/children 中的槽位。

        不排序：Node16 在 Python 层面用线性扫描（真实 C 实现才用 SIMD 二分），
        因此不依赖 keys 有序。索引表在每次追加后整体重建，保证槽位始终一致。
        """
        child = self._new_node()
        node.keys.append(code)
        node.children.append(child)
        node.num_children += 1

        slot = node.num_children - 1
        if node.index is not None:
            # 整体重建索引表，避免逐个更新时槽位错位
            node.index = [255] * 256
            for i, k in enumerate(node.keys):
                node.index[k] = i
        return slot

    # ---- 插入

    def insert(self, key: str):
        """插入键。"""
        path = []  # [(父节点, 本节点在父节点中的槽位)]，根节点的父记为 None
        node = self.root

        for ch in key:
            code = ord(ch)

            child = self._find_child(node, code)
            if child is not None:
                path.append((node, node.keys.index(code)))
                node = child
                continue

            # 确认要新增子节点时才升级：先查找再升级，避免子节点已存在时白升一级，
            # 留下大量只有几个子节点的 Node16。
            while node.num_children >= self._cap_of(node):
                grown = self._grow(node)
                if path:
                    parent, slot = path[-1]
                    parent.children[slot] = grown
                else:
                    self.root = grown
                node = grown

            slot = self._add_child(node, code)
            path.append((node, slot))
            node = node.children[slot]

    # ---- 查找

    def search(self, key: str) -> bool:
        """查找键。返回是否命中。"""
        node = self.root
        for ch in key:
            node = self._find_child(node, ord(ch))
            if node is None:
                return False
        return True


# ---------------------------------------------------------------- 实测函数

def measure_art_vs_trie(keys: list) -> dict:
    """同数据下对比 ART 与标准 Trie 的查找延迟和内存。"""
    art = ART()
    for key in keys:
        art.insert(key)

    trie = Trie()
    for key in keys:
        trie.insert(key)

    # 查找延迟
    sample = random.sample(keys, 1000)

    start = time.perf_counter()
    for key in sample:
        art.search(key)
    art_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for key in sample:
        trie.search(key)
    trie_elapsed = time.perf_counter() - start

    # 子节点定位比较次数
    art.comparison_count = 0
    for key in sample:
        art.search(key)
    art_comparisons = art.comparison_count / len(sample)

    trie.comparison_count = 0
    for key in sample:
        trie.search(key)
    trie_comparisons = trie.comparison_count / len(sample)

    # 内存（递归累加 sys.getsizeof）
    def _art_size(node):
        total = sys.getsizeof(node) + sys.getsizeof(node.keys) \
                + sys.getsizeof(node.children)
        if node.index is not None:
            total += sys.getsizeof(node.index)
        for child in node.children:
            total += _art_size(child)
        return total

    def _trie_size(node):
        total = sys.getsizeof(node) + sys.getsizeof(node.children)
        for ch, child in node.children.items():
            total += sys.getsizeof(ch) + _trie_size(child)
        return total

    art_mem = _art_size(art.root)
    trie_mem = _trie_size(trie.root)

    return {
        "n": len(keys),
        "art": {
            "node_count": art.node_count,
            "type_counts": art.type_counts,
            "lookup_us": art_elapsed / len(sample) * 1e6,
            "avg_comparisons": art_comparisons,
            "memory_bytes": art_mem,
            "memory_bytes_per_key": art_mem / len(keys),
        },
        "trie": {
            "node_count": trie.node_count,
            "lookup_us": trie.lookup_us if hasattr(trie, "lookup_us") else trie_elapsed / len(sample) * 1e6,
            "avg_comparisons": trie_comparisons,
            "memory_bytes": trie_mem,
            "memory_bytes_per_key": trie_mem / len(keys),
        },
        "art_vs_trie_memory": trie_mem / art_mem,
    }


def measure_sigma_scaling(keys: list) -> list:
    """对比不同 σ 下 ART 的内存与形态分布（理论值）。"""
    n = len(keys)
    avg_len = int(round(sum(len(k) for k in keys) / n))
    max_len = max(len(k) for k in keys)
    len_at_least = [0] * (max_len + 1)
    for key in keys:
        for d in range(1, len(key) + 1):
            len_at_least[d] += 1

    results = []
    for sigma in [26, 128, 256]:
        spec = ARTSpec(n=n, avg_len=avg_len, sigma=sigma,
                       len_at_least=tuple(len_at_least))
        m = compute(spec)
        results.append({
            "sigma": sigma,
            "art_node_count": m.node_count,
            "art_memory_bytes": m.memory_bytes,
            "art_bytes_per_key": m.memory_per_key,
            "art_type_counts": {
                "Node4": m.node4_count,
                "Node16": m.node16_count,
                "Node48": m.node48_count,
                "Node256": m.node256_count,
            },
        })
    return results


def main():
    print("=== ART 实测验证 ===\n")

    n = 100_000
    random.seed(42)
    alphabet = string.ascii_lowercase
    keys = [
        "".join(random.choices(alphabet, k=random.randint(5, 20)))
        for _ in range(n)
    ]

    print(f"参数: n={n:,}, 长度 5-20, σ=26\n")

    # 1. ART vs Trie 对比
    print("--- ART vs 标准 Trie（同数据）---")
    result = measure_art_vs_trie(keys)
    art = result["art"]
    trie = result["trie"]

    print(f"{'指标':<14} {'ART':>14} {'标准 Trie':>14} {'比值':>10}")
    print(f"{'节点数':<14} {art['node_count']:>14,} {trie['node_count']:>14,} "
          f"{trie['node_count']/art['node_count']:>9.2f}x")
    print(f"{'查找 μs/次':<14} {art['lookup_us']:>14.2f} {trie['lookup_us']:>14.2f} "
          f"{trie['lookup_us']/art['lookup_us']:>9.2f}x")
    print(f"{'比较次数':<14} {art['avg_comparisons']:>14.1f} "
          f"{trie['avg_comparisons']:>14.1f} "
          f"{trie['avg_comparisons']/art['avg_comparisons']:>9.2f}x")
    print(f"{'内存 MiB':<14} {art['memory_bytes']/1024/1024:>14.1f} "
          f"{trie['memory_bytes']/1024/1024:>14.1f} "
          f"{trie['memory_bytes']/art['memory_bytes']:>9.2f}x")

    print(f"\nART 形态分布: {art['type_counts']}")
    print(f"ART 每键内存: {art['memory_bytes_per_key']:.0f} B")
    print(f"Trie 每键内存: {trie['memory_bytes_per_key']:.0f} B")

    # 2. σ 对比
    print("\n--- 不同 σ 的 ART 内存（理论）---")
    sigma_results = measure_sigma_scaling(keys)
    print(f"{'σ':>5}  {'节点数':>12}  {'内存 MiB':>12}  {'B/键':>8}  "
          f"{'Node4':>12}  {'Node16':>10}  {'Node48':>8}  {'Node256':>8}")
    for r in sigma_results:
        tc = r["art_type_counts"]
        print(f"{r['sigma']:>5}  {r['art_node_count']:>12,}  "
              f"{r['art_memory_bytes']/1024/1024:>12.1f}  "
              f"{r['art_bytes_per_key']:>8.0f}  "
              f"{tc['Node4']:>12,}  {tc['Node16']:>10,}  {tc['Node48']:>8,}  "
              f"{tc['Node256']:>8,}")

    # 汇总写入 JSON
    out_result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制简化版 ART（Node4/16/48/256），对比同数据下的标准 Trie",
            "n": n,
            "key_len_range": [5, 20],
            "sigma": 26,
        },
        "art_vs_trie": result,
        "sigma_scaling": sigma_results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"art_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()

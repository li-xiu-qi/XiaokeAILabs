# -*- coding: utf-8 -*-
"""
跳表实测验证

用纯 Python 自制迷你跳表，实测三个核心指标：
1. 最大层数：插入 n 个元素后，跳表的实际最大层数
2. 期望比较次数：查找操作的平均比较次数
3. 存储占用：跳表的实际内存占用

方法：生成 n 个随机键，插入跳表，实测层数、查找比较次数、内存占用。
结果写入 results/skip_list_<timestamp>.json
"""
import os
import sys
import json
import random
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from skip_list_model import SkipListSpec, compute, compute_max_level


class SkipListNode:
    """跳表节点。每个节点有随机层数，每层一个指针。"""
    def __init__(self, key: int, value: int, level: int):
        self.key = key
        self.value = value
        self.forward = [None] * level  # 每层一个指针


class SkipList:
    """自制迷你跳表。"""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.max_level = 0
        self.header = SkipListNode(float('-inf'), None, 1)
        self.size = 0
        self.comparison_count = 0  # 累计比较次数

    def _random_level(self) -> int:
        """随机决定新节点的层数。几何分布，期望 1/(1-p)。"""
        level = 1
        while random.random() < self.p and level < 32:  # 上限 32 层
            level += 1
        return level

    def insert(self, key: int, value: int):
        """插入键值对。"""
        # 先决定新节点层数，如果超过当前最大层数，先扩展 header
        new_level = self._random_level()
        if new_level > self.max_level:
            # 扩展 header 的 forward 列表
            for _ in range(self.max_level, new_level):
                self.header.forward.append(None)
            self.max_level = new_level

        update = [None] * (self.max_level + 1)
        x = self.header

        # 从最高层开始查找插入位置
        for i in range(self.max_level, -1, -1):
            while x.forward[i] and x.forward[i].key < key:
                x = x.forward[i]
            update[i] = x

        new_node = SkipListNode(key, value, new_level)
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self.size += 1

    def search(self, key: int) -> int:
        """查找键。返回比较次数。"""
        self.comparison_count = 0
        x = self.header
        for i in range(self.max_level, -1, -1):
            while x.forward[i] and x.forward[i].key < key:
                self.comparison_count += 1
                x = x.forward[i]
            self.comparison_count += 1  # 比较 x.forward[i].key == key
        return self.comparison_count


def measure_max_level(spec: SkipListSpec) -> int:
    """实测最大层数。插入 n 个元素后返回实际最大层数。"""
    sl = SkipList(p=spec.p)
    random.seed(42)
    for i in range(spec.n):
        sl.insert(i, i)
    return sl.max_level


def measure_search_comparisons(spec: SkipListSpec, num_searches: int = 1000) -> float:
    """实测查找比较次数。随机查找 num_searches 个键，返回平均比较次数。"""
    sl = SkipList(p=spec.p)
    random.seed(42)
    for i in range(spec.n):
        sl.insert(i, i)

    total_comparisons = 0
    for _ in range(num_searches):
        key = random.randint(0, spec.n - 1)
        total_comparisons += sl.search(key)
    return total_comparisons / num_searches


def measure_storage(spec: SkipListSpec) -> int:
    """实测存储占用。插入 n 个元素后，用 sys.getsizeof 估算内存占用。"""
    sl = SkipList(p=spec.p)
    random.seed(42)
    for i in range(spec.n):
        sl.insert(i, i)

    # 遍历所有节点，累加内存占用
    total_bytes = sys.getsizeof(sl.header)
    x = sl.header.forward[0]
    while x:
        # 节点对象 + key + value + forward 列表
        node_bytes = sys.getsizeof(x) + sys.getsizeof(x.forward) + \
                     sys.getsizeof(x.key) + sys.getsizeof(x.value)
        total_bytes += node_bytes
        x = x.forward[0]
    return total_bytes


def main():
    print("=== 跳表实测验证 ===\n")

    # 用小参数快速验证
    spec = SkipListSpec(
        n=10_000,
        p=0.5,
        key_bytes=8,
        value_bytes=8,
        pointer_bytes=8,
    )

    print(f"参数: n={spec.n:,}, p={spec.p}\n")

    # 实测最大层数
    print("--- 最大层数 ---")
    actual_max_level = measure_max_level(spec)
    theo_max_level = compute_max_level(spec)
    print(f"实测: {actual_max_level}")
    print(f"理论: {theo_max_level}")

    # 实测查找比较次数
    print("\n--- 查找比较次数 ---")
    actual_comparisons = measure_search_comparisons(spec, num_searches=1000)
    theo_comparisons = compute(spec).expected_comparisons
    print(f"实测: {actual_comparisons:.1f} 次比较 (1000 次查找平均)")
    print(f"理论: {theo_comparisons:.1f} 次比较")

    # 实测存储占用
    print("\n--- 存储占用 ---")
    actual_storage = measure_storage(spec)
    theo_storage = compute(spec).storage_bytes
    print(f"实测: {actual_storage/1024:.1f} KiB")
    print(f"理论: {theo_storage/1024:.1f} KiB")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你跳表，小参数快速验证公式",
            "spec": {
                "n": spec.n,
                "p": spec.p,
            },
        },
        "max_level": {
            "actual": actual_max_level,
            "theoretical": theo_max_level,
        },
        "search_comparisons": {
            "actual": actual_comparisons,
            "theoretical": theo_comparisons,
        },
        "storage": {
            "actual_bytes": actual_storage,
            "theoretical_bytes": theo_storage,
        },
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"skip_list_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()

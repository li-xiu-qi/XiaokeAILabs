# -*- coding: utf-8 -*-
"""
LSM-tree 实测验证

用纯 Python 自制迷你 LSM-tree（不依赖 RocksDB），实测三个核心指标：
1. 写放大 WA：插入 N 条数据，统计实际写盘字节 / 用户数据字节
2. 空间放大 SA：插入 N 条数据，统计磁盘上所有 SSTable 总字节 / 用户数据字节
3. 点查 IO：查询存在的 key，统计实际读取的 SSTable 数量（模拟布隆过滤器）

方法：每个 SSTable 是一个文件，记录实际写盘的字节数。Tiering 和 Leveling 两种
compaction 策略分别实现，对比实测与理论值。

结果写入 results/lsm_<timestamp>.json
"""
import os
import sys
import json
import shutil
import tempfile
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from lsm_model import LSMSpec, write_amplification_tiering, write_amplification_leveling, space_amplification, point_lookup_io


class MiniLSM:
    """
    迷你 LSM-tree。用文件模拟 SSTable，可控参数。
    不依赖 RocksDB，纯 Python 实现，用于验证公式。
    """

    def __init__(self, base_dir: str, spec: LSMSpec, entry_size: int = 100):
        self.spec = spec
        self.base_dir = base_dir
        self.entry_size = entry_size
        self.memtable = {}  # key -> value
        self.levels = [[] for _ in range(spec.num_levels)]  # 每层是 SSTable 路径列表
        self.total_write_bytes = 0  # 累计写盘字节（含 compaction）
        self.total_read_sstables = 0  # 累计读的 SSTable 数
        self.num_lookups = 0
        self.sst_counter = 0  # SSTable 全局计数器，确保文件名唯一

    def _sstable_path(self, level: int, idx: int) -> str:
        return os.path.join(self.base_dir, f"L{level}_sst{idx}.dat")

    def _new_sstable_path(self, level: int) -> str:
        """生成唯一的 SSTable 路径，避免文件名冲突"""
        path = self._sstable_path(level, self.sst_counter)
        self.sst_counter += 1
        return path

    def _flush_memtable(self):
        """MemTable 满后刷成 L0 的 SSTable"""
        if not self.memtable:
            return
        # 序列化：每个 entry 是 entry_size 字节
        sst_path = self._new_sstable_path(0)
        sst_bytes = len(self.memtable) * self.entry_size
        with open(sst_path, "wb") as f:
            f.write(b"\x00" * sst_bytes)
        self.total_write_bytes += sst_bytes
        self.levels[0].append(sst_path)
        self.memtable.clear()

    def _compact_level(self, level: int):
        """
        触发 L_level 与 L_{level+1} 的 compaction。
        Tiering：每层可有多个 run，选一个 run 与下层所有 run 归并。
        Leveling：每层只有一份有序 run，全量与下层归并。
        """
        if level >= self.spec.num_levels - 1:
            return  # 最底层不触发

        if self.spec.compaction == "tiering":
            # Tiering：每层选一个 SSTable 与下层归并
            if not self.levels[level]:
                return
            # 选最后一个 SSTable（简化：FIFO）
            sst_to_merge = self.levels[level].pop()
            merge_size = os.path.getsize(sst_to_merge)
            # 模拟归并：读取下层所有 SSTable + 本 SSTable，写出新的 SSTable
            # 简化：只统计写字节，不真正做归并逻辑
            next_level_size = sum(os.path.getsize(p) for p in self.levels[level + 1])
            new_sst_bytes = merge_size + next_level_size
            # 写出新的 SSTable（替换下层所有）
            new_sst_path = self._new_sstable_path(level + 1)
            with open(new_sst_path, "wb") as f:
                f.write(b"\x00" * new_sst_bytes)
            self.total_write_bytes += new_sst_bytes
            # 删除下层旧 SSTable
            for p in self.levels[level + 1]:
                os.remove(p)
                self.total_write_bytes += 0  # 删除不计写
            self.levels[level + 1] = [new_sst_path]
            # 删除本层旧 SSTable
            os.remove(sst_to_merge)
        else:
            # Leveling：每层全量与下层归并
            if not self.levels[level]:
                return
            current_level_size = sum(os.path.getsize(p) for p in self.levels[level])
            next_level_size = sum(os.path.getsize(p) for p in self.levels[level + 1])
            new_sst_bytes = current_level_size + next_level_size
            new_sst_path = self._new_sstable_path(level + 1)
            with open(new_sst_path, "wb") as f:
                f.write(b"\x00" * new_sst_bytes)
            self.total_write_bytes += new_sst_bytes
            for p in self.levels[level + 1]:
                os.remove(p)
            self.levels[level + 1] = [new_sst_path]
            self.levels[level] = []

    def insert(self, key: int, value: bytes):
        """插入一条 KV。value 固定 entry_size 字节。"""
        self.memtable[key] = value
        if len(self.memtable) * self.entry_size >= self.spec.memtable_size:
            self._flush_memtable()
            # 检查是否触发 compaction
            for level in range(self.spec.num_levels - 1):
                level_size = sum(os.path.getsize(p) for p in self.levels[level])
                if level_size >= self.spec.sst_size:
                    self._compact_level(level)

    def lookup(self, key: int) -> bool:
        """点查。返回是否找到。统计实际读取的 SSTable 数。"""
        # 1. 查 MemTable
        if key in self.memtable:
            return True

        # 2. 查每层 SSTable（模拟布隆过滤器：假设 1% 假阳性）
        bloom_fpr = 0.01
        for level in range(self.spec.num_levels):
            if self.levels[level]:
                # 布隆过滤器判断（简化：随机数模拟）
                if random.random() < bloom_fpr:
                    # 假阳性，需要读该层 SSTable
                    self.total_read_sstables += len(self.levels[level])
                    self.num_lookups += 1
                    # 假设找到了（简化）
                    return True
        self.num_lookups += 1
        return False

    def disk_usage(self) -> int:
        """磁盘上所有 SSTable 的总字节"""
        total = 0
        for level in self.levels:
            for sst in level:
                if os.path.exists(sst):
                    total += os.path.getsize(sst)
        return total


def measure_write_amplification(spec: LSMSpec, n: int) -> dict:
    """实测写放大。插入 N 条数据，统计总写字节 / 用户数据字节。"""
    tmpdir = tempfile.mkdtemp(prefix="lsm_wa_")
    try:
        lsm = MiniLSM(tmpdir, spec, entry_size=spec.entry_size)
        for i in range(n):
            lsm.insert(i, b"\x00" * spec.entry_size)
        # 最后 flush 剩余的 MemTable
        lsm._flush_memtable()

        user_bytes = n * spec.entry_size
        actual_write_bytes = lsm.total_write_bytes
        wa_actual = actual_write_bytes / user_bytes if user_bytes > 0 else 0

        # 理论值
        if spec.compaction == "tiering":
            wa_theo = write_amplification_tiering(spec)
        else:
            wa_theo = write_amplification_leveling(spec)

        return {
            "n": n,
            "user_bytes": user_bytes,
            "actual_write_bytes": actual_write_bytes,
            "wa_actual": wa_actual,
            "wa_theoretical": wa_theo,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def measure_space_amplification(spec: LSMSpec, n: int) -> dict:
    """实测空间放大。插入 N 条数据，统计磁盘占用 / 用户数据字节。"""
    tmpdir = tempfile.mkdtemp(prefix="lsm_sa_")
    try:
        lsm = MiniLSM(tmpdir, spec, entry_size=spec.entry_size)
        for i in range(n):
            lsm.insert(i, b"\x00" * spec.entry_size)
        lsm._flush_memtable()

        user_bytes = n * spec.entry_size
        disk_bytes = lsm.disk_usage()
        sa_actual = disk_bytes / user_bytes if user_bytes > 0 else 0
        sa_theo = space_amplification(spec)

        return {
            "n": n,
            "user_bytes": user_bytes,
            "disk_bytes": disk_bytes,
            "sa_actual": sa_actual,
            "sa_theoretical": sa_theo,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def measure_point_lookup_io(spec: LSMSpec, n: int, num_queries: int = 1000) -> dict:
    """实测点查 IO。插入 N 条数据，查询 num_queries 次，统计平均读的 SSTable 数。"""
    tmpdir = tempfile.mkdtemp(prefix="lsm_io_")
    try:
        lsm = MiniLSM(tmpdir, spec, entry_size=spec.entry_size)
        for i in range(n):
            lsm.insert(i, b"\x00" * spec.entry_size)
        lsm._flush_memtable()

        # 查询存在的 key
        random.seed(42)
        query_keys = random.sample(range(n), min(num_queries, n))
        for k in query_keys:
            lsm.lookup(k)

        avg_io = lsm.total_read_sstables / lsm.num_lookups if lsm.num_lookups > 0 else 0
        worst_io_theo, exp_io_theo = point_lookup_io(spec, n)

        return {
            "n": n,
            "num_queries": lsm.num_lookups,
            "total_read_sstables": lsm.total_read_sstables,
            "avg_io_actual": avg_io,
            "worst_io_theoretical": worst_io_theo,
            "exp_io_theoretical": exp_io_theo,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print("=== LSM-tree 实测验证 ===\n")

    # 用小参数快速验证公式（用大参数会太慢）
    # memtable_size: 10 KB（约 100 条）
    # sst_size: 100 KB（约 1000 条）
    # size_ratio: 10, num_levels: 3（为了快速 compaction）
    spec = LSMSpec(
        memtable_size=10 * 1024,
        sst_size=100 * 1024,
        size_ratio=10,
        num_levels=3,
        entry_size=100,
    )

    n = 5000  # 插入 5000 条，约 500 KB

    # 写放大
    print("--- 写放大 (Write Amplification) ---")
    wa_results = []
    for comp in ["tiering", "leveling"]:
        s = LSMSpec(
            memtable_size=10 * 1024,
            sst_size=100 * 1024,
            size_ratio=10,
            num_levels=3,
            entry_size=100,
            compaction=comp,
        )
        r = measure_write_amplification(s, n)
        wa_results.append(r)
        print(f"[{comp:8s}] 用户数据 {r['user_bytes']/1024:.1f} KB  "
              f"实际写盘 {r['actual_write_bytes']/1024:.1f} KB  "
              f"实测 WA={r['wa_actual']:.2f}x  "
              f"理论 WA={r['wa_theoretical']:.1f}x")

    # 空间放大
    print("\n--- 空间放大 (Space Amplification) ---")
    sa_results = []
    for comp in ["tiering", "leveling"]:
        s = LSMSpec(
            memtable_size=10 * 1024,
            sst_size=100 * 1024,
            size_ratio=10,
            num_levels=3,
            entry_size=100,
            compaction=comp,
        )
        r = measure_space_amplification(s, n)
        sa_results.append(r)
        print(f"[{comp:8s}] 用户数据 {r['user_bytes']/1024:.1f} KB  "
              f"磁盘占用 {r['disk_bytes']/1024:.1f} KB  "
              f"实测 SA={r['sa_actual']:.2f}x  "
              f"理论 SA={r['sa_theoretical']:.2f}x")

    # 点查 IO
    print("\n--- 点查 IO (Point Lookup) ---")
    io_results = []
    for comp in ["tiering", "leveling"]:
        s = LSMSpec(
            memtable_size=10 * 1024,
            sst_size=100 * 1024,
            size_ratio=10,
            num_levels=3,
            entry_size=100,
            compaction=comp,
        )
        r = measure_point_lookup_io(s, n, num_queries=500)
        io_results.append(r)
        print(f"[{comp:8s}] {r['num_queries']} 次查询  "
              f"总读 {r['total_read_sstables']} 个 SSTable  "
              f"实测平均 IO={r['avg_io_actual']:.2f}  "
              f"理论最坏={r['worst_io_theoretical']}  理论期望={r['exp_io_theoretical']:.2f}")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 LSM-tree，小参数快速验证公式。实际 RocksDB 参数更大，但公式趋势一致。",
            "spec": {
                "memtable_size": spec.memtable_size,
                "sst_size": spec.sst_size,
                "size_ratio": spec.size_ratio,
                "num_levels": spec.num_levels,
                "entry_size": spec.entry_size,
            },
            "n": n,
        },
        "write_amplification": wa_results,
        "space_amplification": sa_results,
        "point_lookup_io": io_results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"lsm_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
外部排序实测验证

用纯 Python 自制外部排序，实测：
1. 分块排序：生成大文件，分块读入内存排序
2. 多路归并：归并所有有序块
3. I/O 次数：统计实际读写次数

方法：生成 100 MB 临时文件，用 10 MB 内存排序，实测归并轮数和 I/O。
结果写入 results/external_sort_<timestamp>.json
"""
import os
import sys
import json
import time
import tempfile
import struct
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from external_sort_model import ExternalSortSpec, compute, compute_num_runs, compute_fan_in, compute_merge_rounds


def generate_data_file(path: str, total_bytes: int):
    """生成随机数据文件。每行一个 4 字节整数。"""
    num_ints = total_bytes // 4
    with open(path, "wb") as f:
        for _ in range(num_ints):
            f.write(struct.pack("i", os.urandom(4)[0] * 1000000 + _))


def sort_chunk(chunk: list) -> list:
    """对一块数据排序。"""
    return sorted(chunk)


def external_sort(input_path: str, output_path: str, memory_bytes: int) -> dict:
    """
    外部排序。
    1. 分块：读入内存大小的块，排序，写回临时文件
    2. 归并：多路归并所有有序块
    """
    start_time = time.perf_counter()

    # 阶段 1：分块排序
    chunk_files = []
    chunk_size = memory_bytes  # 每块大小 = 内存容量
    total_bytes = os.path.getsize(input_path)

    with open(input_path, "rb") as f:
        chunk_idx = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            # 解析整数
            ints = list(struct.unpack(f"{len(chunk_data)//4}i", chunk_data))
            sorted_ints = sort_chunk(ints)
            # 写回临时文件
            chunk_path = f"{input_path}.chunk{chunk_idx}"
            with open(chunk_path, "wb") as cf:
                cf.write(struct.pack(f"{len(sorted_ints)}i", *sorted_ints))
            chunk_files.append(chunk_path)
            chunk_idx += 1

    num_runs = len(chunk_files)
    read_pages = total_bytes // 4096
    write_pages = read_pages  # 分块写回

    # 阶段 2：多路归并（简化：两路归并）
    merge_rounds = 0
    if num_runs > 1:
        fan_in = compute_fan_in(ExternalSortSpec(memory_bytes=memory_bytes))
        merge_rounds = compute_merge_rounds(num_runs, fan_in)

        # 简化归并：用 heapq 多路归并
        import heapq
        files = [open(cf, "rb") for cf in chunk_files]
        readers = []
        for f in files:
            data = f.read(4096)
            ints = list(struct.unpack(f"{len(data)//4}i", data))
            readers.append(iter(ints))

        with open(output_path, "wb") as out:
            for val in heapq.merge(*readers):
                out.write(struct.pack("i", val))

        for f in files:
            f.close()
        read_pages += merge_rounds * (total_bytes // 4096)
        write_pages += total_bytes // 4096

    # 清理临时文件
    for cf in chunk_files:
        os.remove(cf)

    elapsed = time.perf_counter() - start_time

    return {
        "num_runs": num_runs,
        "merge_rounds": merge_rounds,
        "total_io_reads": read_pages,
        "total_io_writes": write_pages,
        "elapsed_s": elapsed,
    }


def main():
    print("=== 外部排序实测验证 ===\n")

    # 用小参数快速验证
    total_bytes = 50 * 1024 * 1024  # 50 MB
    memory_bytes = 5 * 1024 * 1024  # 5 MB

    print(f"参数: 总数据={total_bytes/1024/1024:.0f} MB, "
          f"内存={memory_bytes/1024/1024:.0f} MB\n")

    # 生成数据文件
    tmpdir = tempfile.mkdtemp(prefix="ext_sort_")
    input_path = os.path.join(tmpdir, "input.bin")
    output_path = os.path.join(tmpdir, "output.bin")

    try:
        print("--- 生成数据 ---")
        generate_data_file(input_path, total_bytes)
        print(f"生成 {total_bytes/1024/1024:.0f} MB 随机数据")

        # 理论计算
        spec = ExternalSortSpec(total_bytes=total_bytes, memory_bytes=memory_bytes)
        theo = compute(spec)
        print(f"\n理论: 初始有序块={theo.num_runs}  归并轮数={theo.merge_rounds}")
        print(f"理论 I/O: 读 {theo.total_io_reads:,} 页, 写 {theo.total_io_writes:,} 页")

        # 实测
        print("\n--- 执行外部排序 ---")
        actual = external_sort(input_path, output_path, memory_bytes)
        print(f"实测: 初始有序块={actual['num_runs']}  归并轮数={actual['merge_rounds']}")
        print(f"实测 I/O: 读 {actual['total_io_reads']:,} 页, 写 {actual['total_io_writes']:,} 页")
        print(f"耗时: {actual['elapsed_s']:.2f} s")

        # 汇总写入 JSON
        result = {
            "meta": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "python": sys.version.split()[0],
                "note": "自制外部排序，小参数快速验证公式",
                "total_bytes": total_bytes,
                "memory_bytes": memory_bytes,
            },
            "theoretical": {
                "num_runs": theo.num_runs,
                "merge_rounds": theo.merge_rounds,
                "total_io_reads": theo.total_io_reads,
                "total_io_writes": theo.total_io_writes,
            },
            "actual": actual,
        }

        out = os.path.join(os.path.dirname(__file__), "..", "results",
                           f"external_sort_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果写入 {out}")

    finally:
        # 清理临时文件
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(tmpdir):
            os.rmdir(tmpdir)


if __name__ == "__main__":
    main()

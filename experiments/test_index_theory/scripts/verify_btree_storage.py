# -*- coding: utf-8 -*-
"""
B+树存储公式的 sqlite 实测对照

sqlite 的表组织就是 B+树（k 为 INTEGER PRIMARY KEY 即 rowid）。
本脚本扫多个 n，顺序插入与随机插入各一遍，测量：
- page_count（逻辑页数）、logical_bytes = page_count * page_size
- physical_bytes（db 文件实际大小）
与 btree_model.compute() 的理论值对比，验证两件事：
1. 存储随 n 线性增长（缩放律，与实现无关）
2. 页数比值 page_count / theo_pages 落在合理区间（标定 page_overhead 与 fill_factor）

每行存储建模：sqlite 每行 cell = rowid(varint) + record header + payload，
本模型近似 key_size=4（rowid 编码 + record header）、value_size=vlen（payload）。

结果写入 results/btree_storage_<timestamp>.json
"""
import os
import sys
import json
import time
import tempfile
import sqlite3
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from btree_model import BTreeSpec, compute

VLEN = 100          # value 定长字节数
PAGE_SIZE = 4096
SPEC = BTreeSpec(key_size=4, value_size=VLEN, page_size=PAGE_SIZE,
                 page_overhead=100, fill_factor=0.70)
N_LIST = [1_000, 10_000, 100_000, 500_000]


def build_and_measure(n, vlen, order):
    """建临时 sqlite 库，插 n 条定长记录，返回页数与字节。"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA page_size={PAGE_SIZE}")
        conn.execute("PRAGMA journal_mode=OFF")   # 避免 WAL 旁路文件干扰
        conn.execute("CREATE TABLE t(k INTEGER PRIMARY KEY, v TEXT)")
        val = "x" * vlen
        if order == "seq":
            keys = range(1, n + 1)
        else:
            keys = list(range(1, n + 1))
            random.seed(42)
            random.shuffle(keys)
        batch = []
        BATCH = 10_000
        for k in keys:
            batch.append((k, val))
            if len(batch) >= BATCH:
                conn.executemany("INSERT INTO t VALUES(?,?)", batch)
                conn.commit()
                batch = []
        if batch:
            conn.executemany("INSERT INTO t VALUES(?,?)", batch)
            conn.commit()
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.close()
        logical = page_count * PAGE_SIZE
        physical = os.path.getsize(path)
        return page_count, logical, physical
    finally:
        if os.path.exists(path):
            os.remove(path)


def run_series(order):
    out = []
    for n in N_LIST:
        t0 = time.time()
        page_count, logical, physical = build_and_measure(n, VLEN, order)
        m = compute(SPEC, n)
        ratio = page_count / m.storage_pages if m.storage_pages else float("nan")
        out.append({
            "n": n,
            "page_count": page_count,
            "logical_bytes": logical,
            "physical_bytes": physical,
            "theo_pages": m.storage_pages,
            "theo_bytes": m.storage_bytes,
            "theo_height": m.height,
            "page_ratio": ratio,            # 实测页数 / 理论页数
            "bytes_per_record": logical / n,  # 每记录存储字节（线性斜率）
        })
        print(f"[{order:6s}] n={n:>7,}  pages={page_count:>4} (theo {m.storage_pages:>4}, "
              f"ratio {ratio:.3f})  logical={logical/1024:.0f} KiB  "
              f"h={m.height}  ({time.time()-t0:.1f}s)")
    return out


def slope(series):
    """用最小二乘拟合 bytes = a + b*n，返回斜率 b（每记录字节）与截距 a。"""
    xs = [s["n"] for s in series]
    ys = [s["logical_bytes"] for s in series]
    k = len(xs)
    mx = sum(xs) / k
    my = sum(ys) / k
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    b = num / den
    a = my - b * mx
    return a, b


def main():
    print(f"page_size={PAGE_SIZE} vlen={VLEN} "
          f"spec key={SPEC.key_size} value={SPEC.value_size} "
          f"overhead={SPEC.page_overhead} fill={SPEC.fill_factor}")
    seq = run_series("seq")
    rnd = run_series("random")

    a_s, b_s = slope(seq)
    a_r, b_r = slope(rnd)
    print(f"\n线性拟合 logical_bytes = a + b*n")
    print(f"  顺序: 截距 a={a_s:.0f} B, 斜率 b={b_s:.2f} B/record")
    print(f"  随机: 截距 a={a_r:.0f} B, 斜率 b={b_r:.2f} B/record")

    # 由每记录字节反推等效填充率：fill = (key+value) / b
    net = SPEC.key_size + SPEC.value_size
    print(f"\n等效填充率反推（fill = (key+value)/b, key+value={net} B）")
    print(f"  顺序: {net/b_s:.3f}   随机: {net/b_r:.3f}")

    result = {
        "meta": {
            "page_size": PAGE_SIZE,
            "vlen": VLEN,
            "spec": {"key_size": SPEC.key_size, "value_size": SPEC.value_size,
                      "page_overhead": SPEC.page_overhead, "fill_factor": SPEC.fill_factor},
            "sqlite_version": sqlite3.sqlite_version,
            "python": sys.version.split()[0],
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "seq": seq,
        "random": rnd,
        "fit": {
            "seq": {"intercept": a_s, "slope_bytes_per_record": b_s,
                    "implied_fill": net / b_s},
            "random": {"intercept": a_r, "slope_bytes_per_record": b_r,
                       "implied_fill": net / b_r},
        },
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"btree_storage_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()

"""
xlsx -> sqlite 体积换算实测
=====================

测量同一份数据在 xlsx 与 sqlite 两种格式下的体积差，
并分解影响膨胀倍数的三个因素：数据重复度、列值长度、索引数量。

产物（生成的 xlsx / sqlite）默认写入同级 output/ 目录，
该目录已在 .gitignore 覆盖范围内，不会入库。

运行：
    python bench_size.py            # 跑全部场景
    python bench_size.py --quick    # 只跑小场景，约 1 分钟

依赖：
    openpyxl
"""

import argparse
import gc
import os
import random
import string
import sqlite3
import time
import zipfile

from openpyxl import Workbook, load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------

def human(size):
    """字节转可读字符串。"""
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return "%.2f%s" % (size, unit)
        size /= 1024.0


def xlsx_compression_ratio(path):
    """xlsx 压缩率 = 解压后 XML 总体积 / xlsx 文件体积。

    压缩率是膨胀倍数的上界来源：sqlite 体积约等于解压后的 XML 体积。
    """
    total = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.filename.endswith((".xml", ".rels")):
                total += info.file_size
    return total / float(os.path.getsize(path))


def write_xlsx(path, rows, cols, value_pool):
    """生成 xlsx，value_pool 为取值池（重复度由池大小决定）。"""
    wb = Workbook(write_only=True)
    ws = wb.create_sheet()
    ws.append(["c%d" % i for i in range(cols)])
    for _ in range(rows):
        ws.append([random.choice(value_pool) for _ in range(cols)])
    wb.save(path)
    return path


def build_sqlite(db_path, xlsx_path, cols, n_index):
    """从 xlsx 读回数据写入 sqlite，并按需建索引。

    刻意从 xlsx 读回而不是直接用内存数据，保证两边测的是同一份数据。
    """
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE d (%s)" % ", ".join("c%d TEXT" % i for i in range(cols))
    )

    wb = load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    next(it)  # 跳过表头
    placeholders = ",".join("?" * cols)
    cur.executemany("INSERT INTO d VALUES (%s)" % placeholders, it)
    wb.close()

    for i in range(min(n_index, cols)):
        cur.execute("CREATE INDEX ix%d ON d(c%d)" % (i, i))

    con.commit()
    con.close()

    # Windows 上 zipfile 与 sqlite 的句柄释放有延迟，
    # 不回收干净会导致后续 os.remove 报「句柄无效」
    gc.collect()
    return os.path.getsize(db_path)


# ----------------------------------------------------------------------
# 场景定义
# ----------------------------------------------------------------------

def scenarios(quick=False):
    """返回场景列表。

    rows / cols / vlen 控制数据规模，pooln 控制重复度（池越小重复度越高），
    idx 为建索引的列数。
    """
    if quick:
        return [
            dict(tag="常规业务数据", rows=50000, cols=15, vlen=12, pooln=2000, idx=0),
            dict(tag="枚举高重复", rows=100000, cols=20, vlen=8, pooln=5, idx=0),
            dict(tag="长文本不可去重", rows=50000, cols=2, vlen=500, pooln=50000, idx=0),
        ]
    return [
        dict(tag="常规业务数据_文本高重复", rows=50000, cols=15, vlen=12, pooln=2000, idx=0),
        dict(tag="枚举高重复_值域5", rows=200000, cols=20, vlen=8, pooln=5, idx=0),
        dict(tag="枚举高重复_值域3", rows=200000, cols=20, vlen=8, pooln=3, idx=0),
        dict(tag="长文本_不可去重", rows=100000, cols=2, vlen=500, pooln=100000, idx=0),
    ]


def index_sweep(quick=False):
    """索引数量扫描所用的基础场景。"""
    if quick:
        return dict(rows=50000, cols=20, vlen=8, pooln=5)
    return dict(rows=100000, cols=20, vlen=8, pooln=5)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑小场景")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUT, exist_ok=True)

    header = "%-26s %10s %9s %11s %9s" % (
        "场景", "xlsx", "压缩率", "sqlite", "膨胀"
    )
    print(header)
    print("-" * len(header))

    for sc in scenarios(args.quick):
        pool = [
            "".join(random.choices(string.ascii_letters, k=sc["vlen"]))
            for _ in range(sc["pooln"])
        ]
        xlsx = write_xlsx(
            os.path.join(OUT, "size_%s.xlsx" % sc["tag"]),
            sc["rows"], sc["cols"], pool,
        )
        xs = os.path.getsize(xlsx)
        ratio = xlsx_compression_ratio(xlsx)
        db_size = build_sqlite(
            os.path.join(OUT, "size_%s.sqlite" % sc["tag"]),
            xlsx, sc["cols"], sc["idx"],
        )
        ss = db_size
        print("%-26s %10s %8.1fx %11s %8.1fx" % (
            sc["tag"], human(xs), ratio, human(ss), ss / float(xs)
        ))
        try:
            os.remove(xlsx)
        except OSError:
            pass

    # ---- 索引数量扫描 ----
    base = index_sweep(args.quick)
    print("")
    print("索引数量扫描：%d 行 x %d 列，值长 %dB，取值 %d 种" % (
        base["rows"], base["cols"], base["vlen"], base["pooln"]
    ))
    pool = [
        "".join(random.choices(string.ascii_letters, k=base["vlen"]))
        for _ in range(base["pooln"])
    ]
    xlsx = write_xlsx(os.path.join(OUT, "idx_base.xlsx"),
                      base["rows"], base["cols"], pool)
    xs = os.path.getsize(xlsx)
    print("  xlsx = %s（压缩率 %.1fx）" % (human(xs), xlsx_compression_ratio(xlsx)))

    header2 = "  %-12s %11s %9s" % ("索引数", "sqlite", "膨胀")
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for n in (0, 3, 10, base["cols"]):
        ss = build_sqlite(
            os.path.join(OUT, "idx_%d.sqlite" % n), xlsx, base["cols"], n
        )
        print("  %-12d %11s %8.1fx" % (n, human(ss), ss / float(xs)))

    try:
        os.remove(xlsx)
    except OSError:
        pass


if __name__ == "__main__":
    t0 = time.time()
    run()
    print("\n耗时 %.1f 秒" % (time.time() - t0))

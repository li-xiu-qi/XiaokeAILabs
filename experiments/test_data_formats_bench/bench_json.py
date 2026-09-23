"""
json 与 xlsx 的体积与读写耗时实测
===============================

同一份数据（行数 x 列数 x 取值池相同）写成 json 与 xlsx，
对比落盘体积与读写耗时。

测量的三个量：
  1. 体积：json 的字节数 / xlsx 的字节数
  2. 写耗时：数据 -> 落盘
  3. 读耗时：落盘 -> 内存对象

产物写入同级 output/ 目录，该目录已被根 .gitignore 的 **/output*/ 覆盖。

运行：
    python bench_json.py            # 全部场景
    python bench_json.py --quick    # 只跑小场景

依赖：
    openpyxl

JSON 侧用三种形态：
  - 全量：每行写成独立对象（[{"c0": .., "c1": ..}, ...]），最常用
  - 列式：每列一个数组（{"c0": [..], "c1": [..]}），紧凑，重复值只存一次
  - jsonl：每行一个对象独立成行（{"c0": ..}\n{"c1": ..}\n），无外层数组

jsonl 与 json 全量形态的体积几乎相同（只差外层数组括号与行间逗号），
所以体积维度上它不值得单独测。要测的是 json 做不到的三件事：
边读边筛只取前 N 行、末尾追加一行、以及流式读取时的内存峰值。
"""

import argparse
import gc
import json
import os
import random
import string
import time
import tracemalloc
import zipfile

from openpyxl import Workbook, load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def human(size):
    """字节转可读字符串。"""
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return "%.2f%s" % (size, unit)
        size /= 1024.0


def build_rows(rows, cols, value_pool):
    """在内存里生成一份数据，保证 json 与 xlsx 测的是同一份。"""
    return [[random.choice(value_pool) for _ in range(cols)]
            for _ in range(rows)]


# ----------------------------------------------------------------------
# 写入
# ----------------------------------------------------------------------

def write_json_rows(path, data, cols):
    """全量形态：行数组，每行一个对象。"""
    keys = ["c%d" % i for i in range(cols)]
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        json.dump([dict(zip(keys, r)) for r in data], f, ensure_ascii=False)
    return time.time() - t0


def write_json_cols(path, data, cols):
    """列式形态：列数组，重复值只存一次。"""
    t0 = time.time()
    obj = {"c%d" % i: [r[i] for r in data] for i in range(cols)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return time.time() - t0


def write_jsonl(path, data, cols):
    """jsonl 形态：每行一个独立 JSON 对象，行尾换行。"""
    keys = ["c%d" % i for i in range(cols)]
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(dict(zip(keys, r)), ensure_ascii=False))
            f.write("\n")
    return time.time() - t0


def append_jsonl_line(path, cols):
    """jsonl 追加一行：只打开文件末尾写一行，不触碰已有内容。"""
    line = json.dumps({"c%d" % i: "v%d" % i for i in range(cols)},
                      ensure_ascii=False) + "\n"
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    return time.time() - t0


def append_json_rows_line(path, cols):
    """json 全量形态追加一行：必须读回全量、追加、整文件重写。"""
    keys = ["c%d" % i for i in range(cols)]
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    arr.append(dict(zip(keys, ["v%d" % i for i in range(cols)])))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False)
    return time.time() - t0


def write_xlsx(path, data):
    """xlsx 侧：write_only 工作簿，共享字符串表由 openpyxl 自动维护。"""
    wb = Workbook(write_only=True)
    ws = wb.create_sheet()
    ws.append(["c%d" % i for i in range(len(data[0]))])
    for r in data:
        ws.append(r)
    t0 = time.time()
    wb.save(path)
    return time.time() - t0


# ----------------------------------------------------------------------
# 读回
# ----------------------------------------------------------------------

def decompose_xlsx_read(path):
    """把 xlsx 读取拆成解压与 XML 解析两段，分别计时。

    xlsx 读取慢在解析不在解压，这个拆分是结论的依据，单测总耗时看不出占比。
    """
    def unzip_only():
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                z.read(info)

    def full_read():
        wb = load_workbook(path, read_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        next(it)
        rows = list(it)
        wb.close()

    t0 = time.time()
    unzip_only()
    t_unzip = time.time() - t0

    t0 = time.time()
    full_read()
    t_full = time.time() - t0
    return t_unzip, t_full - t_unzip


def read_json_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        t0 = time.time()
        data = json.load(f)
    return data, time.time() - t0


def read_json_cols(path):
    with open(path, "r", encoding="utf-8") as f:
        t0 = time.time()
        obj = json.load(f)
    return obj, time.time() - t0


def read_jsonl_full(path):
    """jsonl 全量读：逐行 json.loads，内存形态等价于 json 全量。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        t0 = time.time()
        for line in f:
            rows.append(json.loads(line))
    return rows, time.time() - t0


def read_jsonl_stream(path, limit=1000):
    """jsonl 流式读：只取前 limit 行，读满即停，不解析剩余部分。

    这是 jsonl 相对 json 的核心差异，json 必须先解析完整文件才拿得到第一行。
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        t0 = time.time()
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows, time.time() - t0


def read_xlsx(path):
    wb = load_workbook(path, read_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    next(it)
    rows = list(it)
    wb.close()
    return rows


def timeit(fn, repeat=3):
    """重复执行取平均耗时（秒）。"""
    t0 = time.time()
    for _ in range(repeat):
        fn()
    return (time.time() - t0) / repeat


def peak_mb(fn):
    """执行 fn 并返回（结果, 峰值内存 MB），峰值由 tracemalloc 计量。"""
    gc.collect()
    tracemalloc.start()
    try:
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak / (1024.0 * 1024.0)


# ----------------------------------------------------------------------
# 场景
# ----------------------------------------------------------------------

def scenarios(quick=False):
    if quick:
        return [
            dict(tag="常规业务数据", rows=50000, cols=15, vlen=12, pooln=2000),
            dict(tag="枚举高重复", rows=100000, cols=20, vlen=8, pooln=5),
        ]
    return [
        dict(tag="常规业务数据_文本高重复", rows=50000, cols=15, vlen=12, pooln=2000),
        dict(tag="枚举高重复_值域5", rows=200000, cols=20, vlen=8, pooln=5),
        dict(tag="枚举高重复_值域3", rows=200000, cols=20, vlen=8, pooln=3),
        dict(tag="长文本_不可去重", rows=100000, cols=2, vlen=500, pooln=100000),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑小场景")
    parser.add_argument("--decompose", action="store_true",
                        help="额外拆分 xlsx 读取的解压与解析耗时")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUT, exist_ok=True)
    results = []

    header = "%-26s %10s %10s %10s %10s %10s" % (
        "场景", "xlsx", "json行", "json列", "jsonl", "jsonl/xlsx"
    )
    print(header)
    print("-" * len(header))

    for sc in scenarios(args.quick):
        pool = [
            "".join(random.choices(string.ascii_letters, k=sc["vlen"]))
            for _ in range(sc["pooln"])
        ]
        data = build_rows(sc["rows"], sc["cols"], pool)

        xp = os.path.join(OUT, "json_%s.xlsx" % sc["tag"])
        jr = os.path.join(OUT, "json_%s_rows.json" % sc["tag"])
        jc = os.path.join(OUT, "json_%s_cols.json" % sc["tag"])
        jl = os.path.join(OUT, "json_%s.jsonl" % sc["tag"])

        # 写
        tw = write_xlsx(xp, data)
        tj = write_json_rows(jr, data, sc["cols"])
        tk = write_json_cols(jc, data, sc["cols"])
        tl = write_jsonl(jl, data, sc["cols"])

        xs = os.path.getsize(xp)
        js = os.path.getsize(jr)
        ks = os.path.getsize(jc)
        ls = os.path.getsize(jl)

        # 读：全量（各 3 次取均值）
        tr = timeit(lambda: read_json_rows(jr), repeat=3)
        tx = timeit(lambda: read_xlsx(xp), repeat=1)
        tc = timeit(lambda: read_json_cols(jc), repeat=3)
        tl_full = timeit(lambda: read_jsonl_full(jl), repeat=3)

        # 读：jsonl 流式，只取前 1000 行（不触碰文件其余部分）
        _, tl_stream = read_jsonl_stream(jl, limit=1000)

        # 追加一行：jsonl 只写末尾，json 全量形态必须读回并整文件重写
        tl_app = timeit(lambda: append_jsonl_line(jl, sc["cols"]), repeat=3)
        tj_app = timeit(lambda: append_json_rows_line(jr, sc["cols"]), repeat=1)

        # 内存峰值：全量读（json 行式 / jsonl 全量 / jsonl 流式 1000 行）
        out_j, mj = peak_mb(lambda: read_json_rows(jr))
        del out_j
        out_l, ml = peak_mb(lambda: read_jsonl_full(jl))
        del out_l
        out_s, ms = peak_mb(lambda: read_jsonl_stream(jl, limit=1000))
        del out_s
        gc.collect()

        print("%-26s %10s %10s %10s %10s %9.1fx" % (
            sc["tag"], human(xs), human(js), human(ks), human(ls), ls / float(xs)
        ))
        print("%-26s %10s %10s %10s %10s" % (
            "  写耗时", "%.2fs" % tw, "%.2fs" % tj, "%.2fs" % tk, "%.2fs" % tl
        ))
        print("%-26s %10s %10s %10s %10s" % (
            "  全量读", "%.2fs" % tx, "%.2fs" % tr, "%.2fs" % tc, "%.2fs" % tl_full
        ))
        print("%-26s %10s %10s %10s %10s" % (
            "  追加一行", "-", "%.2fs" % tj_app, "-", "%.4fs" % tl_app
        ))
        print("%-26s %10s %10s %10s %10s" % (
            "  流式读前1000行", "-", "-", "-", "%.4fs" % tl_stream
        ))
        print("%-26s %10s %10s %10s %10s" % (
            "  峰值内存(MB)", "-", "%.1f" % mj, "-", "%.1f/%.1f" % (ml, ms)
        ))

        results.append({
            "tag": sc["tag"],
            "rows": sc["rows"], "cols": sc["cols"],
            "vlen": sc["vlen"], "pooln": sc["pooln"],
            "xlsx_bytes": xs,
            "json_rows_bytes": js,
            "json_cols_bytes": ks,
            "jsonl_bytes": ls,
            "xlsx_write_s": tw,
            "json_rows_write_s": tj,
            "json_cols_write_s": tk,
            "jsonl_write_s": tl,
            "xlsx_read_s": tx,
            "json_rows_read_s": tr,
            "json_cols_read_s": tc,
            "jsonl_read_s": tl_full,
            "jsonl_stream1000_s": tl_stream,
            "json_rows_append_s": tj_app,
            "jsonl_append_s": tl_app,
            "json_rows_peak_mb": mj,
            "jsonl_peak_mb": ml,
            "jsonl_stream_peak_mb": ms,
        })

        del data
        gc.collect()
        for p in (xp, jr, jc, jl):
            try:
                os.remove(p)
            except OSError:
                pass

    if args.decompose:
        # 用最后一个场景（单元格最少、值最长）拆解，最能看出解析占主导
        sc = scenarios(args.quick)[-1]
        pool = [
            "".join(random.choices(string.ascii_letters, k=sc["vlen"]))
            for _ in range(sc["pooln"])
        ]
        data = build_rows(sc["rows"], sc["cols"], pool)
        xp = os.path.join(OUT, "decompose.xlsx")
        write_xlsx(xp, data)
        tu, tp = decompose_xlsx_read(xp)
        print("\nxlsx 读取拆解（%s，%d 个单元格）：" % (
            sc["tag"], sc["rows"] * sc["cols"]))
        print("  解压 zip   %.2f s" % tu)
        print("  XML 解析  %.2f s（占 %.0f%%）" % (tp, tp / (tu + tp) * 100))
        try:
            os.remove(xp)
        except OSError:
            pass

    with open(os.path.join(OUT, "bench_json_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("\n耗时 %.1f 秒" % (time.time() - t0))

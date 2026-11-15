#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_parquet.py — 查看 parquet 语料结构：行数、字段、字符量抽样。"""

import sys

import pyarrow.parquet as pq


def main() -> int:
    path = sys.argv[1]
    pf = pq.ParquetFile(path)
    print("file:", path)
    print("num_rows:", pf.metadata.num_rows)
    print("num_row_groups:", pf.metadata.num_row_groups)
    print("schema:", pf.schema_arrow)
    total_chars = 0
    n_docs = 0
    for i in range(min(pf.num_row_groups, 4)):
        rg = pf.read_row_group(i)
        texts = rg.column("text").to_pylist()
        n_docs += len(texts)
        total_chars += sum(len(t) for t in texts)
    print(f"sampled {n_docs} docs from first 4 row groups, {total_chars:,} chars")
    print("avg chars/doc:", total_chars // max(n_docs, 1))
    sample = pf.read_row_group(0).column("text").to_pylist()[0]
    print("first doc head:", sample[:300].replace("\n", "\\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

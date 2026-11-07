# SciFact 共享候选池

[English](README.en.md) | 中文

Embedding、重排序和 ColBERT 蒸馏共用这一份候选池生成工具。它按证据标注筛选训练正例、过滤跨划分近似查询，再用 BM25 生成候选。不同实验如何抽样和使用候选，以各实验说明为准。

需要 Python 3.11 或更高版本，只用标准库，无需 GPU。下载 [BEIR SciFact](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) 并解压，取得其中的 `corpus.jsonl`、`queries.jsonl` 和 `qrels/`；从 [SciFact 原始发布包](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz) 解压取得 `claims_train.jsonl`。来源及使用许可见 [BEIR](https://github.com/beir-cellar/beir) 和 [SciFact](https://github.com/allenai/scifact/blob/master/LICENSE.md)。数据不在仓库内。

在仓库根目录运行，替换输入路径：

```bash
python utils/scifact-pools/prepare_pools.py --data /path/to/scifact \
  --original-claims /path/to/claims_train.jsonl --out /path/to/prepared
```

输出 `pools.json` 和 `data-audit.json`。后者记录四个 BEIR 输入、原始 claims、配置及输出的 SHA-256。对照 [source-hashes.json](source-hashes.json)，四个输入和原始 claims 应匹配；Linux 上复算的历史候选池 SHA-256 为 `e7cac48e7083bdd948e2773136c1df9672b4ca8f787ee5b774697b32d689c5f8`。Windows CPU 重建的查询划分、训练候选及开发和测试候选顺序与 Linux 一致，但部分 BM25 浮点分数末位不同，因此文件哈希不同；要逐字节复现历史文件，请使用 Linux 并核对哈希。清单的 `original_pool_config.json` 是历史原始配置哈希，不是当前公开配置文件的哈希；两者无需相等。脚本不会自动将哈希与清单比较，使用者需自行核对。不同输入或配置不应直接与历史指标比较。

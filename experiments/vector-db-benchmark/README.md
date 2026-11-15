# 六种向量检索方案：索引、召回与存储对比

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/retrieval/vector-search/README.md)。

[scripts](scripts/) · [config](config/) · [figures](figures/) · [环境搭建](环境搭建.md)

## 环境准备与复现：从一组配置开始

下面以 Windows、PowerShell 和 sqlite-vec 为例，避免一次启动所有服务。命令从本实验目录运行；也可以先用[合成数据做安装检查](环境搭建.md#1-安装与冒烟检查)。

本目录的 `requirements.txt` 汇总了各 Python 数据库适配器的依赖，并不代表所有数据库服务和客户端已在同一环境中验证兼容。建议按目标适配器和下方步骤选择性安装；Docker 服务、SurrealDB 可执行文件及历史探针各有额外准备要求，详见[环境搭建说明](环境搭建.md)。

2026-11-14 在 Linux ARM64、Python 3.12.3 的新隔离环境中安装 NumPy 2.5.2 和 sqlite-vec 0.1.9，SQLite 版本为 3.45.1，`pip check` 通过。下列命令对 30,000 条 768 维合成向量及 50 条查询完成 sqlite-vec 基准，结果 JSON 的配置项没有 `error`。这是 CPU 合成数据检查，不代表真实语料或其他数据库已复现；本次原始语料下载地址连接超时。

```bash
python3 -m venv /path/to/venvs/vector-db
/path/to/venvs/vector-db/bin/python -m pip install numpy==2.5.2 sqlite-vec==0.1.9
/path/to/venvs/vector-db/bin/python -m pip check
/path/to/venvs/vector-db/bin/python scripts/index_bench.py --only sqlite --synthetic --rounds 2
```

查看 `results/indexbench-synthetic-*.json` 的各项结果是否包含 `error`；该命令会生成本次测试用的独立数据库文件并尝试清理，建议在独立实验副本中执行。

2026-11-14 又在 Linux ARM64 的独立复制环境中，以 scikit-learn 使用的 20 Newsgroups 下载包及 SHA-256 校验语料，使用固定版本 `sentence-transformers/all-MiniLM-L6-v2`（revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`）在 CPU 编码了 11,293 条训练文本和 600 条测试文本，生成 384 维向量。sqlite-vec 的真实向量索引基准使用其中 40 条查询，文本存储 A/B 也运行完成，结果均无配置错误。环境为 Python 3.12.3、NumPy 2.5.4、sentence-transformers 6.0.1、sqlite-vec 0.1.9、SQLite 3.45.1，`pip check` 通过。该环境由已有环境复制调整，非从零安装；设备并行运行其他任务，本次耗时不用于性能对比，也未覆盖其他数据库。

### 1. 准备 Python、数据和模型

sqlite-vec 的合成数据安装检查记录使用 Windows、Python 3.13.0、NumPy 2.5.2 和 sqlite-vec 0.1.9。下面的 Windows 文本向量生成及完整基准尚未重新端到端验证；`sentence-transformers` 没有锁定版本，重跑时请记录其版本和模型 revision。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.5.2 sqlite-vec==0.1.9 sentence-transformers
.\.venv\Scripts\python.exe -c "import sqlite3, sqlite_vec, numpy, sentence_transformers; c=sqlite3.connect(':memory:'); c.enable_load_extension(True); sqlite_vec.load(c); print('sqlite-vec', c.execute('select vec_version()').fetchone()[0], 'numpy', numpy.__version__, 'sentence-transformers', sentence_transformers.__version__)"
& {
    foreach ($path in @('data/20news', 'data/20news-bydate.tar.gz', 'models/all-MiniLM-L6-v2')) {
        if (Test-Path -LiteralPath $path) { throw "目标已存在，请使用新的实验副本：$path" }
    }
    New-Item -ItemType Directory -Force data/20news, models | Out-Null
    Invoke-WebRequest 'https://ndownloader.figshare.com/files/5975967' -OutFile data/20news-bydate.tar.gz
    if ((Get-FileHash data/20news-bydate.tar.gz -Algorithm SHA256).Hash -ne '8F1B2514CA22A5ADE8FBB9CFA5727DF95FA587F4C87B786E15C759FA66D95610') { throw '20 Newsgroups archive SHA-256 mismatch' }
    tar -xzf data/20news-bydate.tar.gz -C data/20news
}
```

上述检查会在内存中加载 SQLite 扩展并打印版本；若 Python 的 SQLite 不支持加载扩展，需要更换支持扩展加载的 Python 构建。将 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 的完整模型文件下载到 `models/all-MiniLM-L6-v2/`，包括模型权重、配置和 tokenizer 文件。以下命令固定本次使用的 revision；下载前请预留空间。`gen_emb.py` 强制离线读取这个目录，不会自动下载。

```powershell
& {
    if (Test-Path -LiteralPath 'models/all-MiniLM-L6-v2') { throw '模型目录已存在，请使用新的实验副本' }
    .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='models/all-MiniLM-L6-v2')"
}
```

下载和下列文本向量生成步骤尚未在新建环境中完整验证：

```powershell
.\.venv\Scripts\python.exe scripts/gen_emb.py --texts-output data/20news_texts.npz
```

脚本在同一次读取语料时生成向量 `data/20news_emb.npz` 和同序文本 `data/20news_texts.npz`。正文去除邮件头、合并空白并截取前 1200 字符；测试划分按每个类别的文件名排序，先取前 30 个再过滤短文本，因此不是随机样本。向量文件和文本文件分别记录有序文本内容的 SHA-256，文本存储 A/B 与混合检索会重新计算并校验摘要和行数。摘要用于检查两个文件声明的文本内容与顺序，并不验证向量数组是否被改动。语料及其使用条件应按实际下载版本核对；本仓不分发原始语料、模型或生成的数据。

脚本不会覆盖已有输出；如需保留多份数据，可通过 `--data-dir`、`--model`、`--output` 和 `--texts-output` 指定输入及输出位置。运行中断可能留下不完整文件，请检查后选择新的输出路径；基准脚本目前固定读取 `data/` 下的默认文件名，使用其他路径前需在独立工作副本中准备好这两个文件。历史向量没有文本摘要，无法确认与现有语料的正文一致；不能把补生成的文本与历史向量直接配对。新生成的数据及结果也不能当作历史数值的原始证据。其他数据库和 Windows 环境下的真实语料流程尚未重新端到端验证。

### 2. 运行并检查结果

```powershell
.\.venv\Scripts\python.exe scripts/index_bench.py --only sqlite
```

脚本按实际数据生成精确 top-10，再执行指定数据库的索引测试，写入 `results/indexbench-*.json`。脚本会捕获单个配置的异常，查看 JSON 是否包含 `error`，不能仅凭进程退出码判断成功。保留该 JSON、完整环境与配置，才能追溯自己得到的结果。要测试其他库，先按照[环境与测试方法](环境搭建.md)准备对应服务，再调整 `--only`；例如 `--only milvus qdrant`。索引基准为本次运行生成独立的集合或本地路径，并尝试清理本次创建的对象，部分目录或异常时的对象可能残留；sqlite-vec 路径已在独立测试目录实测；其他数据库的隔离改动仍需在各自独立测试实例和数据目录中实测。其他历史基准及探针脚本尚未完成同样的隔离处理，不要对含已有数据的实例运行。SurrealDB 测试还需安装 `psutil`，当前启动路径使用 Windows 可执行文件，Linux 复跑前需适配并核对。

按上述命令同时生成两份文件并通过一致性检查后，可运行 `scripts/index_bench.py --only sqlite --text-storage`。历史探针脚本不一定检查同序关系，例如 `scripts/probe_vikingdb.py` 尚无此检查；运行前需核对各自的输入和隔离要求。图表脚本包含历史表格的硬编码数据，其默认图不应视为新结果；使用新 JSON 绘图前，需检查脚本对参数档位和输入文件的假设。新图不能补作历史数据的原始证据。

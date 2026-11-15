# 用 SciFact 微调重排序模型：让搜索结果更靠前

[English](README.en.md)

[完整实验讲解](../../docs/zh/retrieval/reranking/重排序监督微调.md)说明问题、原理和结果；以下是代码与复现入口。请在工作副本中运行，数据与输出不要写入公开仓。

## 离线复算

只需 Python 标准库。脚本按已保存的 9,000 条候选记录重排，核对每题和全体的 nDCG@10、MRR@10，以及候选的 Hit@30 和 Recall@30。BM25 用保存的候选顺序，模型分数并列时沿用该顺序。

```bash
python scripts/verify_ranking.py
```

## 动手训练：准备数据、检查流程并运行

实测环境为 Linux aarch64、Python 3.12.3、CUDA 13.0、PyTorch 2.14.0+cu130、Transformers 5.17.0、NumPy 2.5.3。数据准备使用 `hashlib.file_digest`，需要 Python 3.11 或更新版本。训练脚本要求支持原生 BF16 的 CUDA GPU，并检查系统可用内存至少 24 GiB；不同硬件的显存需要量取决于配置。数据准备只用 Python 标准库。以下在本实验目录执行：先创建独立环境，并按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)选择匹配系统架构和 GPU 的 CUDA 构建；模型首次运行时按配置中的名称与 revision 下载，也可传 `--model-path` 指向已准备的本地模型。下载和训练需要网络及仓外磁盘空间。

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装与系统、架构和 GPU 匹配的 PyTorch
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), '训练需要可用的 CUDA GPU'"
python -c "import torch; assert torch.cuda.is_bf16_supported(including_emulation=False), 'GPU 不支持原生 BF16'"
```

下面依次下载并校验数据、构造训练样本、运行冒烟检查及正式训练。BEIR 提供检索语料及相关性标签，原始 SciFact 提供句级证据，用于筛选训练正例。`WORK` 请指定到仓库外；该目录可供重排序蒸馏实验复用。

```bash
WORK=/path/to/new-working-directory
mkdir -p "$WORK"
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o "$WORK/scifact.zip"
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o "$WORK/scifact-original.tar.gz"
(cd "$WORK" && sha256sum -c <<'EOF'
536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  scifact.zip
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be  scifact-original.tar.gz
EOF
)
unzip -q "$WORK/scifact.zip" -d "$WORK"
tar -xOzf "$WORK/scifact-original.tar.gz" data/claims_train.jsonl > "$WORK/claims_train.jsonl"

python scripts/prepare_scifact.py --data "$WORK/scifact" \
  --original-claims "$WORK/claims_train.jsonl" --config config/config.json --out "$WORK/prepared"

python scripts/train_scifact_reranker.py --data "$WORK/scifact" --prepared "$WORK/prepared" \
  --config config/config.json --out "$WORK/smoke" --smoke

python scripts/train_scifact_reranker.py --data "$WORK/scifact" --prepared "$WORK/prepared" \
  --config config/config.json --out "$WORK/formal" --save-model
```

[依赖清单](requirements.txt)固定了历史运行中的 NumPy 和 Transformers，PyTorch 及传递依赖未锁定，因此不是完整环境锁文件，也未在全新环境中端到端验证。数据准备命令另外需要 `curl`、`unzip`、`tar` 和 `sha256sum`；训练脚本与配置位于本目录。模型版本固定在 [配置](config/config.json)。

2026-11-14 在 Linux ARM64、NVIDIA GB10 上，用固定版本 RoBERTa-base 和已核对哈希的 SciFact 数据与候选池完成 180 步 CUDA 训练、保存及 300 条测试查询评估。BM25 与微调模型的 nDCG@10 分别为 0.6617 和 0.6606，差值的逐查询配对 bootstrap 95% 区间为 [-0.0342, 0.0317]，不能据此认定微调稳定优于 BM25。测试查询与训练查询分离，但部分相关文档出现在训练中；候选池覆盖与单次随机种子也限制结论。运行环境为 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.17.0、NumPy 2.5.3，独立复制并调整的环境通过 `pip check`，不是从零安装。

复画结果图额外需要 `python -m pip install matplotlib`，可执行 `python scripts/plot_ranking.py evidence/summary.json "$WORK/ranking.png"`，避免覆盖仓库中的示例图。本实验保留脚本、配置和基于文档 ID 的评估证据。BEIR 数据卡标注 CC BY-SA 4.0。原始 SciFact 的论断与证据标注采用 CC BY 4.0，摘要语料采用 ODC-By 1.0。数据和权重从各自来源获取。 [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) · [AllenAI SciFact](https://github.com/allenai/scifact) · [原始许可](https://github.com/allenai/scifact/blob/master/LICENSE.md) · [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base)

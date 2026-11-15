# scaling — 数据量级扩展对比实验（机器无关部分）
[English version](README.en.md).


回答一个问题：**同一模型架构、同一留出验证集下，预训练数据量级递增时，
loss / BPB / 吞吐 / 显存怎么变。** 当前保留的实测记录是单卡结果，双机 DDP 对比尚未完成；下文保留复现对比的操作思路。

本目录只保留与机器无关的部分：

| 文件 | 作用 |
|---|---|
| `collect_scaling.py` | 汇总本地结果目录（`--results-dir`，缺省 `$XL_OUT_DIR/scaling`）为 JSON / 对比表 / 三张图。不做任何远程调用 |
| `profile_vram_batch.py` | 显存、数据规模与批次吞吐基准：模型尺寸走 `--d-model/--n-layers/--n-heads/--d-ff` 参数，缺省即扩展实验口径（d_model=320, n_layers=6, n_heads=8, d_ff=1280） |
| `inspect_parquet.py` | 查看 parquet 语料结构（行数、字段、字符量抽样） |

## 多机编排与跨设备运行说明

本实验需要对比单卡（`single`）与双机双卡（`dual`，涉及两台独立设备）的吞吐及收敛差异。由于跨设备拉起命令高度依赖各自的网络拓扑、认证方式、主机地址和网卡命名，可移植性低，因此本仓不内置绑定特定集群环境的远程拉起脚本，仅保留通用的本地/单机训练、指标汇总、显存基准与语料检查工具：

- 跨设备执行前，确保两台设备代码版本、PyTorch/CUDA 环境一致，且 `$XL_DATA_DIR` 下均可访问相同的 parquet 语料与固定验证切片。
- 两台设备之间需互通 `MASTER_ADDR:MASTER_PORT`，多网卡环境需显式导出 `NCCL_SOCKET_IFNAME` 与 `GLOO_SOCKET_IFNAME`（完整 `torchrun --nnodes=2` 原理、网卡指定与产物落点见实验顶层 [`README.md`](../../README.md) 的「GPU 运行」一节）。
- 只有主节点（`node-rank=0`）负责写入 `{pretrain,sft}_metrics.json`；实验结束后只需把各档指标文件汇总到同一目录即可离线出表画图。

## 环境与数据准备

命令从实验根目录 `llm-lifecycle-fullchain/` 运行。先按[顶层环境说明](../../README.md#环境要求)安装依赖和 CUDA 版 PyTorch，再安装读取 Parquet 和下载文件的工具：

```bash
python -m pip install pyarrow 'huggingface_hub>=0.34,<1.0'
python -c "import torch, pyarrow; print('CUDA:', torch.cuda.is_available(), 'pyarrow:', pyarrow.__version__); assert torch.cuda.is_available()"
```

历史记录使用 [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) 的 `shard_00000.parquet` 作训练语料、`shard_06542.parquet` 作固定验证语料，以及 [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) 的 `data/train-00000-of-00004.parquet` 作 SFT 数据。文件需要网络及磁盘空间；以下只下载所需分片，避免下载整个数据集：

```bash
hf download karpathy/climbmix-400b-shuffle shard_00000.parquet shard_06542.parquet --repo-type dataset --local-dir datasets/climbmix
hf download HuggingFaceTB/smol-smoltalk data/train-00000-of-00004.parquet --repo-type dataset --local-dir datasets/smol-smoltalk
python scripts/scaling/inspect_parquet.py datasets/climbmix/shard_00000.parquet
python scripts/scaling/inspect_parquet.py datasets/climbmix/shard_06542.parquet
```

历史结果没有记录数据集 revision 或文件哈希，因此上述下载不能保证与历史文件逐字一致。重跑时应记录所用 revision 和哈希，并遵守上游许可。ClimbMix 的训练与验证分片分开，SFT 则由脚本将加载的指令对按种子切分训练与验证；这不保证跨数据源无内容重复。此处下载命令尚未端到端验证。

## 复现步骤

### 1. 全链路入口和显存检查

```bash
# 在单机或外部 GPU 服务器上按常规流程运行（可按需设置 XL_OUT_DIR / XL_DATA_DIR / XL_RUN_ID）
bash run.sh --config configs/full.json

# 显存与吞吐基准（在目标 GPU 上执行）
python scripts/scaling/profile_vram_batch.py --device cuda --amp bf16 --out "${XL_OUT_DIR:-.work/out/local}/scaling"
```

`run.sh --config configs/full.json` 是全链路入口，不会自动执行下面的多数据量级对照，也不会自动使用下载的 Parquet。若缺少 `corpus.txt`，它会生成玩具数据，行为见顶层 README。

### 2. 单机数据量级对照

下面展示 100K 字符一档的实际调用。它使用训练分片的前 3000 万字符训练共享分词器，再使用前 10 万字符预训练，并从该 checkpoint 接续 SFT。其他档使用同一个分词器，将 `--max-chars` 分别改为 `2000000` 或 `20000000`、输出改到新目录。训练分片各档嵌套，不是独立抽样。

```bash
export XL_RUN_ID=scaling-single-t1
export XL_DATA_DIR="$PWD/datasets"
export XL_OUT_DIR="$PWD/.work/scaling/single/t1_micro_100k"
TOKENIZER="$PWD/.work/scaling/tokenizer"
TRAIN="$PWD/datasets/climbmix/shard_00000.parquet"
VAL="$PWD/datasets/climbmix/shard_06542.parquet"
SFT="$PWD/datasets/smol-smoltalk/data/train-00000-of-00004.parquet"

python scripts/01_train_tokenizer.py --corpus-path "$TRAIN" --max-chars 30000000 --vocab-size 8192 --output-dir "$TOKENIZER"
python scripts/03_pretrain.py --device cuda --amp bf16 --tokenizer "$TOKENIZER/tokenizer.json" \
  --train-source "$TRAIN" --val-source "$VAL" --max-chars 100000 --val-max-chars 2000000 \
  --d-model 320 --n-layers 6 --n-heads 8 --d-ff 1280 --block-size 256 \
  --steps 600 --batch-size 16 --eval-every 100 --eval-windows 64 --no-baseline
python scripts/05_sft.py --device cuda --amp bf16 --tokenizer "$TOKENIZER/tokenizer.json" \
  --init "$XL_OUT_DIR/checkpoints/pretrain.pt" --sft-path "$SFT" --test-source "$VAL" \
  --steps 200 --batch-size 16 --eval-every 50 --hit-eval-rows 24 --val-eval-rows 256
```

历史的 20M 档使用 1800 步预训练，其他两档各 600 步。示例中的 `--test-source` 复用预训练验证分片，因此 SFT 后的 BPB/PPL 是同一分片上的追踪指标，并非独立测试集指标。结果及权重写入 `XL_OUT_DIR`；各次运行应选择新目录，避免覆盖。上述命令按当前脚本参数整理，尚未完整复跑历史配置，不能承诺重现历史数字。

### 3. 双机扩展对照（`dual`）与汇总

如果用两台设备复现 `single` 与 `dual` 的量级扩展对比：

1. **组织输出目录**：为每个数据量级档 `<tier>` 约定独立输出目录 `<results-dir>/<mode>/<tier>`（其中 `<mode>` 为 `single` 或 `dual`）。
2. **分别跑完各档**：
   - `single`：在单机上以 `XL_OUT_DIR=<results-dir>/single/<tier>` 对各量级档先运行 `03_pretrain.py`（通过 `--max-chars` 与 `--train-source` / `--val-source` 控制数据规模与留出验证集），再运行 `05_sft.py` 接续微调（通过 `--init` 指定预训练权重，并通过 `--sft-path` / `--test-source` 指定 SFT 数据与外部测试流）。
   - `dual`：在设备 A（`node-rank=0`，设置 `XL_OUT_DIR=<results-dir>/dual/<tier>`）与设备 B（`node-rank=1`）上同时用 `torchrun --nproc_per_node=1 --nnodes=2 --master_addr=<NODE0_IP> --master_port=29500` 运行相同量级档；注意按对照口径决定是否将每卡 `--batch-size` 减半以保持全局总 batch 一致。
3. **汇总出图**：把 `<results-dir>/single/` 与 `<results-dir>/dual/` 下的各档指标放到同一目录（仅需 `{pretrain,sft}_metrics.json`），在任意机器上运行：

```bash
python scripts/scaling/collect_scaling.py --results-dir "<results-dir>" --modes single,dual
```

## 产物

`scaling_summary.json`、`comparison_table.md`、
`fig_data_scaling_{single,dual}.png`、`fig_speedup_single_vs_dual.png`、
`fig_sft_downstream_{single,dual}.png`，以及各量级原始指标与 loss 曲线。
脱敏副本见实验目录 `evidence/`。

## 硬件需求档

- 冒烟口径（smoke.json）：纯 CPU 可跑，几分钟级
- 完整口径（full.json）：使用支持 bf16 的单卡 GPU；实际显存取决于模型、批次和设备，最低需求未验证，可参考 `profile_vram_batch.py` 的具体扫描结果
- 多卡 DDP 的 CUDA 通信取决于 PyTorch 构建及后端；Linux 可使用 NCCL，其他环境的 gloo/CUDA 路径需单独验证

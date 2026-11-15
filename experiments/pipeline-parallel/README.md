# 两台电脑一起运行大模型：跨机通信探索

[English](README.en.md)

如果一台电脑放不下或处理不了目标模型，能不能让两台电脑各负责一部分计算？本目录保留为回答这个问题所写的通信测试和模型分段原型。相关原理与本次 Windows、Linux 混合环境的思路见[双卡流水线并行推理实现原理](../../docs/zh/inference/双卡流水线并行推理.md)。

**当前状态**：双机 CPU/Gloo 和 GPU/NCCL 的 256 元素往返通信检查均通过。2026 年 11 月 14 日，先将同一权重文件传到两台设备并核对 SHA-256 一致，再用 Gloo 和 NCCL 完成 Qwen2.5-0.5B-Instruct 双机 GPU 生成；一条提示词的 8 个输出 token 均与单机基线一致。此前两份接收副本的哈希不一致，具体来源仍未查明。时间线核对进一步将异常范围缩小到 11 月 13 日哈希仍一致之后、11 月 14 日首次发现差异之前；这只能确定发现区间，不能证明差异是在传输、接收写盘还是更早的源文件状态中产生。后续一次受校验的传输中，发送端和接收端流式及落盘哈希均一致，说明该传输路径可以正确复制文件，但不能据此还原旧副本为何不同。本次是单条输入的功能检查，不代表其他输入、模型或性能加速结论。

逐块字节比较显示，两份旧文件与校验通过的新文件相比，分别有一段和两段 32 字节跨度的差异，其中实际不同的字节为 27、27 和 29 个。这排除了整文件截断，但当时还不能确定差异产生于哪个环节。

2026 年 11 月 15 日再次只读核查：当前发送端缓存中的 `model.safetensors` 为 988,097,824 字节，SHA-256 为 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`，与 [Hugging Face 固定 revision 的官方文件校验值](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors)一致。接收端保留的两份旧文件大小也都是 988,097,824 字节，但 SHA-256 分别为 `71ef72df5184a32e01792df388b187a035ffa75c1f1df5a57699829ab7b5d1e3`（`model.safetensors`）和 `a591bf149b8e53f19184af06e24f80e7bef86822f31f6218ee55036fcceb4605`（`model-check.safetensors`）；它们都与官方文件不同。逐字节比较确认第一份有 27 个不同字节，位置范围为 673,989,287–673,989,318；第二份有 27 个不同字节，范围为 220,871,871–220,871,902，另有 29 个不同字节，范围为 583,115,647–583,115,678。位置按 `cmp -l` 的 1 起始字节编号记录。当前可观察到的不一致集中在接收端保留的这两份旧文件；但原传输没有保留发送端与接收端的即时哈希和逐步日志，因此无法确认差异发生于传输、接收写盘，还是更早的源文件状态。后来的受控传输在发送、接收流和落盘后均得到相同哈希，证明该次传输完整，不足以还原旧文件的成因。当前用于生成的两端模型文件、配置、分词器和词表文件均逐项匹配；完整目录清单仍有差异，较完整的一侧另含 `.gitattributes`、`LICENSE`、`README.md` 和 `configuration.json`，所以只确认了推理所需文件一致，并非完整快照完全相同。本次仅做哈希和字节核查，没有重跑生成。

## 目标、成功条件与预算

按优先级分两级，判定标准事先写死，不从事后结果挑口径：

**第一级：两机逐元素张量通信**（优先）。两端各一个进程，探针张量经网络原样往返后逐元素比对。成功条件：`probe_element_mismatches == 0` 且 `probe_max_abs_diff == 0`，一个完整运行内无通信错误。失败时保留记录并排查问题；运行前应设置外部总时限，避免通信异常时长时间等待。

**第二级：单机与双机确定性生成对照**。在两端各自已有可用的本地小模型前提下，同一 prompt、同一 seed、同一生成参数（贪心解码）下，对照单机完整前向与双机分段前向生成的 token 序列。成功条件：两端 token 序列逐位一致。终止条件：一致即停；不一致记录首个分歧位置即停，不转去调参。预算：1 个模型、最多 3 条 prompt、1 次完整对照。

两级边界：通信通过只证明张量传输正确，不证明模型跑通；模型跑通只证明这条链路成立，不外推性能结论。投机解码不在本实验范围内，不做测试。

## 这些代码尝试做什么

- `scripts/pipeline_stage.py` 和 `scripts/run_smoke.sh`：两台设备分别启动一个进程，发送随机张量并进行模拟计算，用于尝试检查跨机通信；不加载模型，也不生成文字。
- `scripts/real_pp2_inference.py`：尝试把模型层分到两台设备，逐次重新计算生成所需的内容。
- `scripts/real_pp2_kv_cache.py`：尝试在分段计算中保存并复用注意力缓存。
- `scripts/real_pp2_speculative.py`：多候选生成的尝试。它没有实现候选接受校验和回滚，不能作为正确的投机解码实现；为避免误用，脚本直接执行时会停止。

这些模型原型的设计都是先在每台设备加载完整模型再裁剪层，因此不能证明它们可以运行单台设备放不下的模型；一次小模型对照也不能证明兼容所有模型。

## 尝试通信测试

下面先运行已经通过的 CPU/Gloo 通信检查。两台设备需要能互相连接，并安装包含 Gloo 的 PyTorch 和 NumPy。启动器使用 Bash。Windows 用户可在 Linux 环境中运行；模型测试还需要各自可用的 GPU 和本地模型文件。

先分别在两台设备准备环境，并参考 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装适合本机的版本：

```bash
cd experiments/pipeline-parallel
python3 -m venv .venv
source .venv/bin/activate
# 根据目标设备安装带 CUDA 和 NCCL 的 PyTorch
python -m pip install -r requirements.txt
python -c "import torch, numpy, transformers; print('torch:', torch.__version__, 'numpy:', numpy.__version__, 'transformers:', transformers.__version__, 'CUDA available:', torch.cuda.is_available(), 'NCCL available:', torch.distributed.is_nccl_available())"
```

`requirements.txt` 包含通信脚本使用的 NumPy、模型原型使用的 Transformers，以及下载模型所需的 Hugging Face Hub 客户端；PyTorch 根据目标系统单独选择，避免默认包装成不支持目标 CUDA/NCCL 的版本。清单未锁版本，安装结果可能与下方验证环境不同。

运行模型原型前，在两台设备分别下载同一版本的 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，并检查模型卡和许可。以下示例固定了一个公开的模型提交；历史对照未记录完整模型 revision，不能断言它与当时使用的文件完全相同。下载需要网络和磁盘空间，请为 `MODEL_PATH` 选择新目录：

```bash
export MODEL_PATH="$HOME/models/Qwen2.5-0.5B-Instruct"
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
if [ -e "$MODEL_PATH" ]; then
  echo 'Choose a new MODEL_PATH' >&2
else
  python -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", revision=os.environ["MODEL_REVISION"], local_dir=os.environ["MODEL_PATH"], allow_patterns=["*.json", "*.safetensors", "*.txt"])'
fi
```

两端路径可以不同，但所用模型文件和 revision 必须一致。**双机生成前须在两端生成并比较整个快照的 SHA-256 清单**；只比较 `model.safetensors` 不够，因为配置、分词器和分片文件也会影响加载。模型目录应是专用快照目录；以下命令排除 Hugging Face `local_dir` 生成的 `.cache/` 元数据，并需在两台设备各自执行，再将第一台清单传给第二台比较：

```bash
cd "$MODEL_PATH"
find . -type f ! -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > "$HOME/qwen-model-sha256.txt"
```

把第一台生成的 `qwen-model-sha256.txt` 复制到第二台后，在第二台模型目录执行：

```bash
cd "$MODEL_PATH"
find . -type f ! -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > "$HOME/qwen-model-sha256.local.txt"
diff -u "$HOME/qwen-model-sha256.txt" "$HOME/qwen-model-sha256.local.txt"
```

`diff` 无输出且退出码为 0 才表示文件名集合与每个文件的内容哈希均一致；否则停止模型对照并重新下载或传输。通信测试无需下载模型。

将仓库放到两台设备后，在各自的 `experiments/pipeline-parallel/` 目录执行。把 `STAGE1_ADDRESS` 替换为两台设备实际可以访问的第一台设备地址，先启动第一台，再启动第二台：

```bash
# 第一台设备，IFACE1 替换为第一台的实际网卡名
MASTER_ADDR=STAGE1_ADDRESS GLOO_SOCKET_IFNAME=IFACE1 \
  bash scripts/run_smoke.sh stage1 --backend gloo --device cpu --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --warmup 1 --iters 2 --timeout-min 1 \
  --out .work/smoke-pipeline-metrics.json
```

```bash
# 第二台设备，IFACE2 替换为第二台的实际网卡名
MASTER_ADDR=STAGE1_ADDRESS GLOO_SOCKET_IFNAME=IFACE2 \
  bash scripts/run_smoke.sh stage2 --backend gloo --device cpu --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --warmup 1 --iters 2 --timeout-min 1
```

两端应使用相同的主节点地址和端口，并允许必要的网络连接。默认端口为 29500，可通过 `MASTER_PORT` 修改。需要分别指定能让两台设备互相访问的网卡；本次环境中主机名指向回环地址，未指定网卡时初始化超时。测试只运行了两次正式迭代，计时不足以比较性能。

第一台设备会打印 JSON 并写入上面的 `.work/` 文件，包含状态、配置和计时等信息。指标结构（schema `xiaokeailabs/smoke-pipeline-metrics/v2`）：

- `nodes.stage1` 与 `nodes.stage2` 分别记录本节点设备、环境、收发与计算计时和显存峰值。
- `results.numerical` 是 stage 1 对回传探针的逐元素比对，stage 2 接收这一判定。
- 报告中的 `status=passed` 表示探针一致且第一阶段已收到第二阶段的指标。还须检查两端进程退出码以确认第二阶段收到汇总确认；这不证明模型跑通。
- 有效吞吐按 GB/s 记录（1 GB = 1e9 字节）。往返耗时包含两端计算与等待，不能当作网络带宽。
- Gloo 阶段同步使用 `--barrier-timeout`；NCCL 同步使用进程组超时 `--timeout-min`。通信异常时的退出行为仍需实测。

`.work/` 是本地运行输出，不随仓库提供，也不是已保存的历史结果。指定的输出文件若已存在，脚本会报错，请使用独立的输出路径。

## GPU/NCCL 通信测试

两端均需可用的 NVIDIA GPU、支持 CUDA 和 NCCL 的 PyTorch，以及相互可达的网络。2026 年 11 月 14 日的小张量测试在 DGX Spark Linux 与 RTX Spark N1X Windows/WSL2 上通过；两端均为 Ubuntu 24.04 ARM64、Python 3.12.3、PyTorch 2.15.0+cu132 和 NCCL 2.30.7。这是一组实测组合，不是最低版本要求。

在两台设备分别激活上述独立环境，先运行第一端，再尽快运行第二端。将地址和网卡名替换为实际值，确保第一端的端口可从第二端访问。此例禁用 InfiniBand，使用指定网卡；两端参数必须一致。为避免通信异常时任务持续占用资源，应由能够限制总运行时间并清理所有子进程的任务管理器启动两端。

```bash
# 第一台设备
MASTER_ADDR=STAGE1_ADDRESS NCCL_SOCKET_IFNAME=IFACE1 NCCL_IB_DISABLE=1 \
  bash scripts/run_smoke.sh stage1 --backend nccl --device cuda --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --local-layers 1 --warmup 1 --iters 2 --timeout-min 1 \
  --out .work/nccl-metrics.json
```

```bash
# 第二台设备
MASTER_ADDR=STAGE1_ADDRESS NCCL_SOCKET_IFNAME=IFACE2 NCCL_IB_DISABLE=1 \
  bash scripts/run_smoke.sh stage2 --backend nccl --device cuda --dtype fp32 \
  --batch 1 --seq 8 --hidden 32 --local-layers 1 --warmup 1 --iters 2 --timeout-min 1
```

检查两端退出码和第一端 JSON：本次 `status` 为 `passed`，256 个探针元素不一致数为 0，最大绝对误差为 0。换一组设备或网络后仍需重新验证；此结果不代表模型分段推理已经通过 NCCL。

## 轻量自测

`scripts/test_pipeline_stage_gloo.py` 是不依赖 GPU 的协议自测：本机起两个 Gloo CPU 进程跑一轮正式通信，并检查纯函数口径与上述指标结构。任何装有 torch（CPU 版即可）的机器都可以执行：

```bash
python scripts/test_pipeline_stage_gloo.py
```

自测旨在检查逐元素校验、双端指标和吞吐单位。一次 Windows AMD64、Python 3.13.0、torch 2.15.0+cpu、NumPy 2.5.4 的本机两进程自测通过；这不替代双机验证。

## 模型原型的边界

模型原型还依赖 `transformers` 和两端均可读取的本地模型文件，需要提供角色、模型路径和第一台设备地址。本次在 Linux 和 Windows 的 Linux 环境之间完成了一个小模型对照：模型以 BF16 加载，两台设备各计算 24 层中的 12 层，激活值转为 CPU 上的 FP32 张量，通过 Gloo 传输。不同模型架构和依赖版本能否运行尚未确认。

`real_pp2_inference.py` 提供 `baseline` 单机基线和 `stage1`、`stage2` 两机入口。对照时两端均应显式设置 `--temperature 0`、相同的 `--seed`、提示词和生成步数；`--out` 可将基线或第一阶段的输入及生成 token ID 保存为 JSON，已有文件不会被覆盖。检查两端退出码，并逐项比较 token ID。跨设备和框架版本的数值差异仍可能使其他输入产生不同结果。

例如，在两台设备上按上述步骤准备相同的 Qwen2.5-0.5B-Instruct 模型，并把地址和网卡替换为实际值。先在第一台设备运行基线，确认结束后再依次启动两个阶段：

```bash
mkdir -p .work
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/real_pp2_inference.py --role baseline --model-path "$MODEL_PATH" \
  --prompt 'The capital of France is' --max-new-tokens 8 --temperature 0 --seed 0 \
  --out .work/baseline.json

# 第一台设备
GLOO_SOCKET_IFNAME=IFACE1 python scripts/real_pp2_inference.py \
  --role stage1 --model-path "$MODEL_PATH" --master-addr STAGE1_ADDRESS \
  --prompt 'The capital of France is' --max-new-tokens 8 --temperature 0 --seed 0 \
  --out .work/stage1.json
```

```bash
# 第二台设备：MODEL_PATH 指向该设备上的同一模型
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
GLOO_SOCKET_IFNAME=IFACE2 python scripts/real_pp2_inference.py \
  --role stage2 --model-path "$MODEL_PATH" --master-addr STAGE1_ADDRESS \
  --prompt 'The capital of France is' --max-new-tokens 8 --temperature 0 --seed 0
```

首次使用前需创建 `.work/`。两个阶段的 `--master-port` 默认均为 29505；如需修改，两边应保持一致。脚本在 CUDA 不可用时会退回 CPU，因此应先检查两台设备的 CUDA 状态。

也可以使用 NCCL 传输 GPU 张量。先确认两端 CUDA 和 PyTorch NCCL 均可用；将上述两端命令的网卡环境变量分别改为 `NCCL_SOCKET_IFNAME=IFACE1` 和 `NCCL_SOCKET_IFNAME=IFACE2`，并在两端都加上 `--backend nccl`。NCCL 不支持此脚本的 CPU 回退；若不指定 `--backend`，默认使用 Gloo。需要禁用 InfiniBand 时，可在两端设置 `NCCL_IB_DISABLE=1`。模型对照前须按上方步骤逐文件核验两端 SHA-256。

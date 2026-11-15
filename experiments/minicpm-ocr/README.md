# MiniCPM-OCR 双模型部署：1.0 解析 + Jev 质检打分

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/document-understanding/MiniCPM文档解析与质检.md)。

[scripts](scripts/) · [assets](assets/) · [evidence](evidence/)

历史运行的软件环境、模型和部署替代路径见[运行条件与部署记录](#历史运行条件与部署)。

以下命令均从本实验目录运行。

## 环境准备

历史部署使用 aarch64 Linux、Python 3.12 和支持 BF16 的 CUDA GPU；需要 Bash、Git、curl、tar 及足够存储模型、源码、浏览器和输出的磁盘空间。以下是按仓内脚本整理的准备路径；历史 OCR、评分及渲染三个环境的完整依赖未锁定，干净环境尚未端到端验证。

### OCR 服务

先准备隔离环境，按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)在 OCR 虚拟环境中安装适配设备的 CUDA 版 torch 与相匹配的 torchvision，再执行：

```bash
mkdir -p "$HOME/minicpm-ocr"
python3.12 -m venv "$HOME/minicpm-ocr/.venv-ocr"
OCR_PY="$HOME/minicpm-ocr/.venv-ocr/bin/python"
# 在 OCR_PY 对应环境安装适配 CUDA 的 torch 和匹配的 torchvision 后，安装 Transformers 固定源码及其依赖
"$OCR_PY" -m pip install -r requirements-ocr.txt
bash scripts/01_download.sh
bash scripts/02b_fetch_sources.sh
bash scripts/04_opencv.sh
"$OCR_PY" -m pip check
"$OCR_PY" -c 'import torch, torchvision, transformers, cv2, minicpmvocr; print("CUDA:", torch.cuda.is_available(), "Torch:", torch.__version__, "torchvision:", torchvision.__version__, "Transformers:", transformers.__version__)'
```

[requirements-ocr.txt](requirements-ocr.txt) 是 OCR 依赖候选清单，固定 Transformers 源码版本；torch、torchvision 和 SDK 源码需按上面单独安装，未固定的版本会随时间变化。SDK 安装脚本使用 `--no-deps`；`pip check` 如报告缺项，需对照下载源码内 SDK 的依赖清单在该环境补齐并再次检查，不能把导入成功当作完整运行验证。`01_download.sh` 默认使用 `$HOME/minicpm-ocr`，可设置 `MINICPM_OCR_HOME` 指向新的隔离目录；脚本拒绝修改已有目标文件，并在识别模型权重大小或 SHA-256 与历史记录不符时退出。模型有数 GB；它不会下载 Jev 或 DSpark。该校验只覆盖主模型权重，不固定其他文件版本。

解析或 WebUI 需要配置文件。下面仅创建尚不存在的配置，替换示例中的两个路径占位符；使用前还需按下载到的 SDK 检查配置格式：

```bash
mkdir -p "$HOME/minicpm-ocr/configs"
"$OCR_PY" -c 'from pathlib import Path; import os; p=Path.home()/"minicpm-ocr/configs/spark.yaml"; s=Path("scripts/spark.yaml").read_text(); p.open("x").write(s.replace("${MINICPM_OCR_HOME:-$MINICPM_OCR_HOME}", str(Path.home()/"minicpm-ocr")))'
```

### Jev 评分与渲染

从 [MiniCPM-OCR-Jev 模型页](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)获取完整模型快照到 `$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev`，包括 `model.safetensors`、配置和处理器文件；记录实际 revision 与哈希。评分代码直接从该文件读取 `score.weight`。评分和渲染分别使用独立环境，以下是安装起点：

```bash
python3.12 -m venv "$HOME/minicpm-ocr/jev/venv-score"
python3.12 -m venv "$HOME/minicpm-ocr/jev/venv-render"
export JEV_PYTHON="$HOME/minicpm-ocr/jev/venv-score/bin/python"
export RM_RENDER_PYTHON="$HOME/minicpm-ocr/jev/venv-render/bin/python"
export JEV_REPO="$HOME/minicpm-ocr/repo/scripts/minicpm-ocr-jev"
# 按 PyTorch 官方指南在 JEV_PYTHON 对应环境安装 CUDA torch 和匹配的 torchvision
"$JEV_PYTHON" -m pip install -r requirements-jev-score.txt
"$RM_RENDER_PYTHON" -m pip install -r requirements-jev-render.txt
"$RM_RENDER_PYTHON" -m playwright install chromium
"$JEV_PYTHON" -m pip check
"$JEV_PYTHON" -c 'import torch, torchvision, transformers; print("Torch:", torch.__version__, "torchvision:", torchvision.__version__, "Transformers:", transformers.__version__)'
"$RM_RENDER_PYTHON" -m pip check
node --version  # 官方渲染预检查需要 Node.js 18+
test -f "$JEV_REPO/rm_render.py" && test -f "$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev/model.safetensors"
"$JEV_PYTHON" -c 'import os, sys; sys.path.insert(0, os.environ["JEV_REPO"]); import rm_render; rm_render.check_assets(); print("渲染资源与 Node 可用")'
```

评分端使用 [requirements-jev-score.txt](requirements-jev-score.txt)，渲染端使用 [requirements-jev-render.txt](requirements-jev-render.txt)，两者不应合并；评分侧还须单独安装适合 GPU 的 torch 和匹配的 torchvision，渲染侧还须安装 Chromium、Node.js 和系统库。评分端直接导入官方 `rm_render.py`，渲染进程也调用其功能；应核对当前 SDK 中该文件及其依赖，并根据实际报错补装。浏览器还可能需要操作系统库与字体；Pillow 和 Playwright 版本来自历史记录，不保证与当前上游源码兼容。导入和 `pip check` 不代表渲染、评分或 GPU 推理已经通过；启动后需检查健康接口及实际样例。

2026-11-15 在 Linux ARM64、NVIDIA GB10 上，使用 Jev 固定版本 `a0ed278eb6aa1f333a711665ab610055a4a816ab` 对官方三个带参考渲染图的样例完成 GPU 打分。文本、公式、表格样例的分数分别为 0.00253、0.96544、0.00098；样例记录的参考值分别为 0.00702、0.96552、0.00116。每个样例只评估一条候选，这不是准确率评测，也没有运行生成渲染图、评分服务或识别到排序的闭环。使用 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.23.0，复制的独立环境通过 `pip check`，不是从零安装。模型加载时报告缺少 `lm_head.weight` 和额外的 `score.weight`；本评分脚本直接加载后者并使用主干输出，不使用生成头。运行记录与权重哈希见[评分摘要](evidence/jev-score-2026-11-15.json)。

下载该固定版本并运行官方样例（模型约 4.4 GB；`JEV_REPO` 按上文指向已准备的官方源码）：

```bash
"$JEV_PYTHON" -c 'from huggingface_hub import snapshot_download; import os; snapshot_download("openbmb/MiniCPM-OCR-Jev", revision="a0ed278eb6aa1f333a711665ab610055a4a816ab", local_dir=os.path.expanduser("~/minicpm-ocr/models/MiniCPM-OCR-Jev"))'
"$JEV_PYTHON" scripts/jev_score.py --model "$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev" --examples "$JEV_REPO/examples" --json-out jev-examples.json
```

该命令会写入指定的 JSON 文件，重复运行会覆盖；请按需选择新文件。官方源码下载自当时的 main，未固定提交，样例可能变化。本次权重经镜像下载，未验证上述官方站点的直接下载。后续的渲染和评分服务验证见下文。

2026-11-15 评分服务通过健康检查，并使用新建的隔离渲染环境重新渲染官方文本样例的两条候选：人工原文得分 0.99508，OCR 候选得分 0.00723，原文排第一；样例记录的参考值分别为 0.99501 和 0.00702。渲染环境从空虚拟环境安装，Playwright 1.62.0、Pillow 12.3.0、Markdown 3.10.3，依赖检查和 Chromium 启动检查通过；评分环境由已有环境复制。单一样例不能代表整体准确率；这一步没有调用 OCR 服务，后续单块识别到排序的验证见下文。样例源码未固定提交，详见[服务验证记录](evidence/jev-service-2026-11-15.json)。

准备好上面的模型及两个环境后，在前台启动服务（按 Ctrl+C 停止）：

```bash
"$JEV_PYTHON" scripts/jev_server.py --host 127.0.0.1 --port 18501 --model "$HOME/minicpm-ocr/models/MiniCPM-OCR-Jev"
```

在另一个已设置 `JEV_PYTHON` 和 `JEV_REPO` 的终端检查并发送官方文本样例：

```bash
curl --fail http://127.0.0.1:18501/healthz
"$JEV_PYTHON" scripts/jev_probe.py --url http://127.0.0.1:18501 --case "$JEV_REPO/examples/01_text_putnam"
```

测试前确认端口空闲；服务终端也需设置 `JEV_REPO` 和 `RM_RENDER_PYTHON`。缺少 Chromium 系统库或字体时需先补齐，渲染结果可能随依赖和字体变化。

同日还把识别、渲染与评分服务接起来，处理同一个官方文本块。OCR 新输出与样例参考文本逐字不同；参考文本、OCR 输出、删除参考文本末行的候选得分依次为 0.995084、0.995028、0.00000489，三项均完成评分，参考文本排第一。这里使用的是上游样例提供的参考文本，未另做人工核验；单一样例不能说明识别准确率或排序稳定性。两个服务和渲染端使用前述环境，完整记录见[单块流程摘要](evidence/jev-loop-2026-11-15.json)。

复跑该单块流程时，先按上面的步骤准备模型和 Jev 服务，确认 18499、18501 端口空闲。另开终端启动 OCR 服务，以下命令使用前述默认模型位置：

```bash
export OCR_MODEL_DIR="$HOME/minicpm-ocr/models/MiniCPM-OCR-1.0"
"$HOME/minicpm-ocr/.venv-ocr/bin/transformers" serve "$OCR_MODEL_DIR" \
  --host 127.0.0.1 --port 18499 --device cuda --dtype bfloat16 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

在设置了 `OCR_MODEL_DIR`、`JEV_REPO`、`JEV_PYTHON` 的第三个终端检查服务、提取上游参考文本并运行。输出目录须为本次实验单独选定，脚本会覆盖目录内同名结果：

```bash
curl --fail http://127.0.0.1:18499/health
curl --fail http://127.0.0.1:18501/healthz
export LOOP_OUT="${LOOP_OUT:?请设置独立输出目录}"
mkdir -p "$LOOP_OUT"
"$JEV_PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["JEV_REPO"])/"examples/01_text_putnam/case.json"; c=json.loads(p.read_text(encoding="utf-8")); t=next(x["text"] for x in c["candidates"] if x.get("role")=="ground_truth"); pathlib.Path(os.environ["LOOP_OUT"],"reference.txt").write_text(t,encoding="utf-8")'
"$JEV_PYTHON" scripts/jev_loop_block.py --block "$JEV_REPO/examples/01_text_putnam/block.png" --truth "$LOOP_OUT/reference.txt" --max-tokens 512 --out-dir "$LOOP_OUT"
```

完成后分别在两个服务终端按 Ctrl+C。上游样例、系统字体及未锁定的依赖变化都可能影响结果。

### 单图 GPU 推理冒烟

安装 OCR 依赖并下载识别模型后，可从实验目录执行：

```bash
"$OCR_PY" scripts/smoke_gpu.py \
  --model-dir "$HOME/minicpm-ocr/models/MiniCPM-OCR-1.0" \
  --image evidence/testpage_cn.jpg
```

本次在 Linux aarch64、NVIDIA GB10 上用 Python 3.12、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、Transformers 5.22.2 和 Pillow 12.0.0 执行过单图推理，输出包含「关于人工智能文档识别技术的测试报告」。当时复用了已有 PyTorch 环境，并在隔离目录补装图像依赖；这不是从空环境安装验证，也不代表 SDK 服务、布局分析、Jev 评分或加速对照已运行。不同 CUDA 组合应使用设备支持的 PyTorch 与 torchvision，不能仅凭这一次冒烟推断普遍兼容。

2026-11-14 在另一份独立环境中重跑同一张测试图，GPU 输出包含相同标题，进程退出码为 0。环境使用 Python 3.12.3、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、Transformers 5.23.0、Pillow 12.0.0 和 accelerate 1.15.0；安装后 `pip check` 通过。环境复制自已有组合并补装依赖，并非从空环境安装；本次没有运行 SDK 服务、布局分析和 Jev 评分。

同日又在独立的服务环境中，按 `02b_fetch_sources.sh` 准备 SDK 和固定的 Transformers 源码，并补装 `jiter`、`sniffio`、`opentelemetry-api`、`requests` 后通过 `pip check`。环境使用 PyTorch 2.15.0+cu132、Transformers 源码提交 `549e5b410733fb50e8efbbe4aa585cb89f6478ea` 和 MiniCPM-OCR SDK 1.0.5；SDK 源码下载自当时的 main，未固定提交。`transformers serve` 在 CUDA 上启动后，健康检查成功，向 `/v1/chat/completions` 发送测试图片获得 HTTP 200 和识别文本；响应受 128 token 上限截断。测试后服务已关闭。这只验证推理服务，不代表 SDK 布局解析、WebUI 或 Jev 评分已跑通；服务环境基于已有组合复制，并非从零安装。

## 历史操作入口与复现缺口

服务脚本目前无法验证进程归属，`minicpm-ocr.sh stop`、`jev_server.sh stop` 和 `webui.sh stop` 会拒绝操作；请先检查进程身份，再手动停止自己启动的服务。服务已有响应或端口被检测为占用时，脚本不会将其认作本实验的服务；启动前仍应自行检查端口占用。

本实验尚未提供从空环境完成部署的完整流程。下面是历史操作入口，不可直接从头依次执行：

- `02b_fetch_sources.sh` 需要事先创建 `$HOME/minicpm-ocr/.venv-ocr` 并安装兼容 CUDA 的 PyTorch 及其他依赖；仓内未提供经过端到端验证的完整依赖清单。
- `01_download.sh` 下载 OCR 和布局模型，不下载 Jev 或 DSpark 权重；使用 Jev 或加速对照前需自行按来源准备对应权重，具体 revision 未全部锁定。
- `minicpm-ocr.sh` 和 `webui.sh` 使用 `$HOME/minicpm-ocr/configs/spark.yaml`，脚本不会创建该文件。仓内 `scripts/spark.yaml` 是历史配置示例，使用前需按安装的 SDK 版本检查配置格式与模型路径，不能假定其中环境变量会被展开。
- Jev 服务需要打分环境、官方渲染源码及独立渲染环境；`04_opencv.sh` 只安装 OCR 布局所需的 OpenCV，不会建立 Jev 渲染环境。打分解释器可通过 `JEV_PYTHON` 指定；渲染器通过 `JEV_REPO` 和 `RM_RENDER_PYTHON` 指定。仓内没有已验证的一键准备命令。

历史环境为 aarch64 Linux 与 CUDA；其他架构和从零安装尚未端到端验证。模型、依赖和加速运行需要额外磁盘空间。

```bash
# 1. 拉 SDK 与权重（走 ModelScope）
bash scripts/01_download.sh
bash scripts/02b_fetch_sources.sh

# 2. 起 1.0 推理服务（OpenAI 兼容，18499）
bash scripts/minicpm-ocr.sh start

# 3. 起 1.0 自带 WebUI（8950）
bash scripts/webui.sh start

# 4. 安装 OCR 布局依赖；Jev 所需环境仍须按上面的说明自行准备
bash scripts/04_opencv.sh
bash scripts/jev_server.sh start     # 前置环境准备好后启动打分服务，18501

# 5. 冒烟：单块打分
"${JEV_PYTHON:-$HOME/xinfer-env/bin/python}" scripts/jev_score.py --examples "$HOME/minicpm-ocr/repo/scripts/minicpm-ocr-jev/examples"

# 6. 闭环：识别一个块并排序三候选
python scripts/jev_loop_block.py --block <块图> --truth <人工确认原文>

# 7. 推理加速对比评测：基准对比 DSpark 投机采样与特征压缩提速比
bash scripts/run_acceleration_benchmark.sh
```

`02b_fetch_sources.sh` 在已有 `repo` 或 `sources/transformers` 目录时会中止，避免覆盖现有代码。识别模型下载固定到 ModelScope commit `a165c0943964e167b46e32d67bbcb9deab26a8d4`，并核验历史主权重哈希；布局模型仍从 `master` 下载，SDK 从 `main` 下载，后两者版本可能漂移，因此整套环境仍不能严格复现。

多数脚本默认使用 `$HOME/minicpm-ocr`；各脚本支持的路径变量请以脚本为准。`01_download.sh` 和 `02b_fetch_sources.sh` 支持 `MINICPM_OCR_HOME`；其他脚本可能仍使用固定的默认目录。

加速对照的新运行结果默认写入 `$HOME/minicpm-ocr/runs/acceleration_benchmark.json`，可用 `OCR_BENCHMARK_OUTPUT` 指定其他位置；仓库中已有的实测证据不会被该命令覆盖。

历史运行将打分与渲染分成两个 venv；渲染侧使用 `playwright 1.62.0` 与 `Pillow 12.3.0`，以免渲染依赖变化影响分数。仓内脚本目前不会自动创建并安装该环境。

### 其他脚本前置条件

`scripts/06_parse.sh` 额外调用未随仓库提供的 `$HOME/preflight.sh`，不能在新环境直接运行；可用 `bash scripts/minicpm-ocr.sh parse /path/to/input.pdf` 调用 SDK 解析入口。`scripts/make_testpage.py` 使用 Pillow，但字体路径写死为 Windows 挂载路径且输出到当前目录，新环境需自行调整字体路径并确认输出文件不存在。`scripts/plot_acceleration.py` 另外需要 `python -m pip install matplotlib`，读取和输出路径以脚本为准。

加速基准还需 DSpark 模型及测试图片，路径可通过 `DSPARK_MODEL_DIR`、`OCR_TESTDATA_DIR` 和 `JEV_EXAMPLES_DIR` 设置；历史使用的 DSpark revision 未锁定，仓内没有已验证的完整获取命令。运行结果会写到 `OCR_BENCHMARK_OUTPUT`，指定新文件以避免覆盖。

## 目录

```
README.md                    本文件
scripts/                     部署与运行脚本，按执行顺序编号
  jev_score.py               Jev 打分器（transformers 后端，含 prompt md5 断言）
  jev_server.py              Jev 打分服务（18501），/v1/rm/score_best 与 score_best_batch
  jev_server.sh              打分服务启停
  jev_ui.html                打分网页，服务端每次请求现读，改版不用重启
  jev_probe.py               官方样例「自带渲染图 vs 本机重渲染」对比
  jev_loop_block.py          闭环驱动：1.0 识别一块 → Jev 排三候选
  benchmark_acceleration.py  推理加速对比评测工具（DSpark/投机采样 vs 基线）
  run_acceleration_benchmark.sh 一键运行推理加速对比评测
  make_testpage.py           生成测试页
evidence/                    测试页、解析结果样例与加速评测数据
  acceleration_benchmark.json 推理加速指标原始数据 (JSON)
```

打分网页调 `/ui/score` 与 `/ui/ocr`，后者转调 1.0 服务按内容类型分发到官方 prompt map（`Text Recognition:` / `Formula Recognition:` / `Table Recognition:`）。1.0 服务不在时，打分本身不受影响，只是文本要手动贴。

## 历史运行条件与部署

本文记录这次实验使用的模型、环境与替代部署路径，帮助理解复现条件；实验现象及其适用范围见[文档解析与质检讲解](../../docs/zh/document-understanding/MiniCPM文档解析与质检.md)。

| 项 | 配置 |
|---|---|
| 识别模型 | `openbmb/MiniCPM-OCR-1.0`，Apache-2.0，权重 SHA-256 `83f4e433...` |
| 判别模型 | `openbmb/MiniCPM-OCR-Jev`，Apache-2.0，4.13 GiB |
| 布局模型 | `PaddlePaddle/PP-DocLayoutV3_safetensors`，127 MB，torch provider |
| SDK | `github.com/OpenBMB/MiniCPM-OCR`，含 `minicpmvocr` 与自带 WebUI |
| Transformers | 上游 commit `549e5b4`（当时 PyPI 5.21.0 尚无 `minicpm_ocr` 架构） |
| 设备 | GB10，aarch64，sm_121，121 GiB 统一内存，273 GB/s 带宽 |
| 软件环境 | Ubuntu 24.04、CUDA 13.0.88、PyTorch 2.14.0+cu130、Python 3.12 |
| 下载通道 | ModelScope |

本次尝试的 vLLM、SGLang 与 Paddle 安装路线没有跑通；这不代表它们在所有 aarch64 环境下不可用。

| 依赖 | 本次遇到的问题 | 本次采用的路径 |
|---|---|---|
| vLLM / SGLang | 安装脚本有 x86_64 检查，Docker Hub 不通 | `transformers serve`，OpenAI 兼容接口 |
| Paddle 布局检测 | PyPI 无 linux aarch64 轮子 | SDK 自带的纯 torch 布局 provider |
| Jev 官方 serve | 依赖 vLLM 的 RM 接口 | 用 Transformers 获取 Qwen3_5 主干末 token hidden state，乘 `score.weight` 计算 logits；prompt 加 MD5 断言以检查漂移 |

历史运行的其他结果和命令见[实验入口](README.md)与[加速对照报告](../../docs/en/document-understanding/minicpm-ocr-speculative-decoding-benchmark.md)。模型和 SDK 下载脚本未全部固定版本，严格复现仍需核对来源与哈希。

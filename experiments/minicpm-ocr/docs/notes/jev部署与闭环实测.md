# MiniCPM-OCR-Jev 质检模型的部署与闭环实测（DGX Spark）

2026-10-28。同一台 aarch64 GB10 机器，接在 1.0 的解析链路后面。

## 它是什么，和 1.0 什么关系

两个模型不是同一任务的两种版本，是流水线的两端：

| | MiniCPM-OCR 1.0 | MiniCPM-OCR-Jev |
|---|---|---|
| 任务 | 文档图像 → Markdown/LaTeX/HTML | 判「待判文本 + 其渲染图」与原图是否一致 |
| 结构 | 1.3B 生成式（SigLIP + Qwen3.5） | 2B 判别式（Qwen3.5-VL 2B + 两分类 value head） |
| 输入 | 一张图 | 原图 + 文本渲染图 + 文本，三段交错成一个 prompt |
| 输出 | 文本 | 两个 logits，`prob_yes = softmax(logits)[1]`，1 = 一致 |

Jev 的输出不是「改写后的文本」，是**质量分**。它按块工作：一个版面块（一段文字、一个公式、一张表）配它的 OCR 结果，打一个 0 到 1 的分。

**不要把本模型与 `XiaokeAILabs/experiments/jev-open-source` 混为一谈。** 那边测的是 TypeSafe Jev 的社区复刻（Laya / metask-jev / Nimble / Jev-Omni），做通用路由与决策打分；这里的 MiniCPM-OCR-Jev 是 OpenBMB 的 OCR 质检模型，只判「OCR 文本 + 其渲染图」是否忠实还原原图。两者只共享名字里的 Jev。

1.0 的 SDK 里原本就留了 RM 接口（`pipeline/rm_client.py` + test-time scaling），Jev 就是设计成插在那个位置上的模型。官方仓的 `scripts/minicpm-ocr-jev/` 里同时给了打分器、渲染器、HTTP 服务和 SDK 适配层。

## 目的、门槛与终止条件

**目的**：把这台机器上没有 vLLM 的约束下能不能跑起 Jev 判清楚，并测出它判得准不准、快不快、值不值得接进 1.0 的链路。不评估它在全部文档类型上的准确率，不外推到其他硬件。

**达成门槛**（事先写死，2026-10-28 逐条核对）：

| # | 门槛 | 状态 |
|---|---|---|
| 1 | 权重完整，`score.weight` 在位且形状对 | 已达成（4,426,567,144 字节，`[2,2048]` BF16） |
| 2 | 复现官方公布分数，偏差在同一量级 | 已达成（三样块，最大偏差 0.0045） |
| 3 | 本机渲染器可用，重渲染不改变结论 | 已达成（`check_renderer` 2 close 1 drifted，分数差 ≤0.006） |
| 4 | 闭环跑通：1.0 输出经 Jev 排序，破坏版被压到 0 | 已达成 |
| 5 | 接进 1.0 SDK 的 TTS 路径 | **未测**（SDK 全页/块解析在本机卡在布局阶段，见「未解的问题」） |

**终止条件**：门槛 1 至 4 任一失败且两轮内修不好即停；不转去编译 vLLM；不为提高分数换渲染版本。

## 关键结论

**vLLM 不是必需品。** 官方只给了 `vllm serve --runner pooling --convert classify` 这条路，本机装不上（`install.sh` 有 `x86_64` 硬校验）。替代做法是拿 transformers 的 Qwen3_5 主干取最后一个 token 的 hidden state，乘权重里的 `score.weight`，自己算两分类 logits。prompt 用官方 `infer_vllm.py` 的 `build_prompt()` 原文，导入时 md5 断言（`9aab68e8ba8ab4da17ba9752bddb199c`）校验一致，所以分数尺度没有漂。

**分数复现官方值。** 用样例自带的参考渲染图打分：

| 样块 | 候选 | 本机 | 官方 | 偏差 |
|---|---|---|---|---|
| text_putnam_2009 | ocr_output | 0.00248 | 0.00702 | -0.0045 |
| formula_putnam_1999 | ocr_output | 0.96470 | 0.96552 | -0.0008 |
| table_gb_standard | ocr_output | 0.00098 | 0.00116 | -0.0002 |

官方 README 注明分数不可逐位复现（vLLM 批处理与 kernel 差异），这个量级属正常。

**本机重渲染不改变结论。** 让本机渲染器重出图再打分，与官方值的差仍在 0.006 以内：

| 样块 | 候选 | 本机重渲染 | 官方 |
|---|---|---|---|
| formula_putnam_1999 | ground_truth | 0.97028 | 0.96813 |
| formula_putnam_1999 | ocr_output | 0.96518 | 0.96552 |
| table_gb_standard | ground_truth | 0.04674 | 0.04075 |
| text_putnam_2009 | ground_truth | 0.99522 | 0.99501 |

注意 `table_gb_standard` 的 ground_truth 本身只有 0.047，官方值也是 0.041。这不是部署问题：样例自己的注释说这条的表头渲染与原图版式不同，模型判「不一致」是它的真实判断。

**闭环有效。** 拿官方文字块做端到端：1.0 识别 → Jev 判三个候选。

```
原文      0.99522   ← best
1.0输出   0.99505
破坏版    0.00000
```

1.0 识别 1.3 秒，输出 166 字符，内容逐字正确（只把 `$...$` 写成 `\(...\)` 的行内公式记法），Jev 给它 0.995；人为删掉末行，分数掉到 0。Jev 端到端 2.5 秒（渲染 1.188s + 前向 1.298s）。

## 性能与代价

| 指标 | 实测 |
|---|---|
| 权重 | 4.13 GiB（单文件 safetensors） |
| 加载耗时 | 32.5 s（冷启动，含 vision tower） |
| 显存峰值 | 5.18 GiB（bf16） |
| 单候选前向 | 0.13 s（短块）到 1.16 s（长块，含首调用开销） |
| 渲染 | 0.95 到 1.19 s/块（Chromium 冷启） |
| 端到端一次判 3 个候选 | 2.5 s |

对照 1.0：识别一页 A4 是 11.2 秒、3 GiB。Jev 属于可以常驻、按页附加的代价。

## 部署结构

- 打分环境 `~/xinfer-env`：transformers 5.16.1（原生支持 `qwen3_5`）、torch 2.14.0+cu130。与 1.0 复用同一套，没有新建环境。
- 渲染环境 `$MINICPM_OCR_HOME/jev/venv-render`：playwright 1.62.0 + Pillow 12.3.0 + Markdown 3.10.3，版本按官方 `requirements.txt` 钉死，Chromium headless shell 151.0.7922.34。
- **两个环境刻意分开**：vLLM 只要求 `pillow>=12.3.0`，渲染器要求精确的 `12.3.0`，放同一个环境里一次 `pip install -U` 就会把 Pillow 推走并静默改变每个分数。
- 服务端口 18501，接口 `/v1/rm/score_best` 与 `/v1/rm/score_best_batch`，字段与官方 `serve/app.py` 同构，`RMClient` 可直接对接。

## 未解的问题

**1.0 的 SDK 解析小块会卡住。** 全页 A4 能正常解析（11.2 秒），但把官方文字块当输入喂给 `minicpmvocr parse` 会卡在布局阶段，超过 4 分钟无输出，只能杀进程。闭环因此改走直连 chat completions 的路径（`jev_loop_block.py` 就是这么实现的）。这是 1.0 侧的问题，不是 Jev 的，但它挡住了 SDK 自带的 TTS 集成。

**CJK 字体没钉死。** 官方要求 `apt-get install fonts-noto-cjk` 并卸载 wqy/unifont/ipafont，本机无 sudo。`check_renderer.py` 的结果是 2 close / 1 drifted（表格块 2.54% 像素差，平均 delta 110/255）。实测分数影响 ≤0.006，可用于本机自比，但对外公布绝对值前要先解决字体。

**SDK 直连 Jev 没实测。** HTTP 契约对齐了，但没让 1.0 的 `RMClient` 真的调过来跑一轮 TTS。

## 复现命令

```bash
# 服务启停（首次启动约 35 秒加载权重）
bash $MINICPM_OCR_HOME/jev/jev_server.sh start
bash $MINICPM_OCR_HOME/jev/jev_server.sh status

# 官方三样块打分（自带渲染图 vs 本机重渲染）
~/xinfer-env/bin/python $MINICPM_OCR_HOME/jev/jev_probe.py \
    --case $MINICPM_OCR_HOME/repo/scripts/minicpm-ocr-jev/examples/01_text_putnam

# 闭环：1.0 识别块 → Jev 排序
~/xinfer-env/bin/python $MINICPM_OCR_HOME/jev/jev_loop_block.py \
    --block $MINICPM_OCR_HOME/repo/scripts/minicpm-ocr-jev/examples/01_text_putnam/block.png \
    --truth $MINICPM_OCR_HOME/runs/jevloop_block/truth.txt

# 渲染器保真度自检
cd $MINICPM_OCR_HOME/repo/scripts/minicpm-ocr-jev && \
    $MINICPM_OCR_HOME/jev/venv-render/bin/python renderer/check_renderer.py
```

## 踩坑记录

- **`--with-deps` 装不了，但 Chromium 能下**：`playwright install chromium`（不带 `--with-deps`）在无 sudo 下可用，缺的共享库本机已有。cdn.playwright.dev 可达，约 111 MiB。
- **渲染器和打分器必须是两个解释器**：`rm_render.render_candidates()` 用 `sys.executable` 起渲染子进程，服务跑在 xinfer-env 里就会 `ModuleNotFoundError: playwright`。解法是显式传 `python_bin`，服务里读 `RM_RENDER_PYTHON` 环境变量。
- **`pkill -f` 会杀掉自己的 ssh 会话**：远端 `bash -c` 的命令行包含匹配串时，`pkill -f <关键词>` 会连发起方一起杀，表现为 ssh 退出码 255 且无输出。需要按端口取 pid（`ss -ltnp | grep :端口`）或把脚本经 stdin 送过去执行。
- **`lm_head.weight` MISSING 是无害的**：权重里没存 lm_head（只存了 `score.weight`），transformers 会随机初始化它并报 MISSING。我们只用 `model.model` 取 hidden state，不做生成，那 1 GiB 参数从不用到。
- **`/v1/models` 列出五个模型名不代表服务错了**：`transformers serve` 会把本地缓存的模型一起列出来（Qwen3-Embedding-8B、gemma-4-12B-it 等）。请求里的 `model` 仍必须填模型目录绝对路径。

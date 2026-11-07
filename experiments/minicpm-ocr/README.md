# MiniCPM-OCR 双模型部署：1.0 解析 + Jev 质检打分

[English](README.en.md)

本实验验证 OpenBMB 两个 MiniCPM-OCR 模型能否在 aarch64 统一内存机器上、不装 vLLM、不编译 Paddle 的前提下跑通，并接成一条闭环：1.0 负责「图片 → Markdown」，Jev 负责判断这份 Markdown 与原图是否一致。

> **历史运行记录显示：官方主路线（vLLM 或 SGLang 加 Paddle 布局检测）在本次 aarch64 GB10 环境下三条路线均未跑通，换成 transformers 后两个模型都跑通。记录称 Jev 分数与官方公布值偏差不超过 0.0045，闭环一次耗时 2.5 秒，显存峰值 5.18 GiB；这些打分与计时的逐项原始证据未随仓库保存。**

## 为什么值得做这个实验

OCR 的输出质量通常靠人工抽查或字符错误率（CER）衡量，前者不可扩展，后者需要人工标注的参考答案。Jev 提供的是另一条路：给定「原图 + 待判文本 + 文本渲染图」三元组，直接输出一个 0 到 1 的一致性分数，不需要参考答案。

这一分数可用于探索无标注数据的候选排序和解析质量筛查。本实验尚不足以证明排序准确率或线上质检效果，实际应用仍需用人工标注数据评估。分数由两个分类 logit 计算：$s=\exp(z_1)/(\exp(z_0)+\exp(z_1))$。这个 softmax 输出落在 0 到 1 之间，但数值本身并不证明它经过概率校准：例如，0.9 不意味着这类输出有 90% 的概率识别正确。

本实验要回答的是三个工程问题，不是模型能力排行榜：

1. 官方文档依赖 vLLM 与 Paddle，这两项在本次 aarch64 环境下均未能安装，有没有第二条路；
2. Jev 的判别机制在没有 vLLM 的机器上能不能等价复现，分数是否和官方一致；
3. 两个模型接成闭环后的真实延迟与显存代价。

## 两个模型的分工

Jev 不是 1.0 的另一个版本，是流水线的下一环。

```
原图 ──[1.0 识别]──> 文本 ──[渲染器]──> 渲染图
  │                                      │
  └──────────[Jev 打分]── 原图 + 文本 + 渲染图 ──> 0 到 1
```

| 模型 | 职责 | 输入 | 输出 | 参数量 |
|---|---|---|---|---|
| MiniCPM-OCR-1.0 | 文档解析 | 图片或 PDF | Markdown（正文、表格、公式、阅读顺序） | 1.3B，SigLIP + Qwen3.5 |
| MiniCPM-OCR-Jev | 一致性判别 | 原图 + 文本 + 文本渲染图 | 两个 logits（一致 / 不一致） | 主干 2B 加 `[2,2048]` 打分头 |

两者输入输出完全不同，不能互换，也不能用一个替代另一个。Jev 权重里没有 `lm_head`，物理上生成不了文本，它只打分。

## 实验对象与验收口径

| 项 | 配置 |
|---|---|
| 识别模型 | `openbmb/MiniCPM-OCR-1.0`，Apache-2.0，权重 sha256 `83f4e433...` |
| 判别模型 | `openbmb/MiniCPM-OCR-Jev`，Apache-2.0，4.13 GiB |
| 布局模型 | `PaddlePaddle/PP-DocLayoutV3_safetensors`，127 MB，torch provider |
| SDK | `github.com/OpenBMB/MiniCPM-OCR`，含 `minicpmvocr` 与自带 WebUI |
| transformers | 上游 commit `549e5b4`（PyPI 5.21.0 尚无 `minicpm_ocr` 架构） |
| 机器 | GB10，aarch64，sm_121，121 GiB 统一内存，273 GB/s 带宽 |
| 环境 | Ubuntu 24.04，CUDA 13.0.88，torch 2.14.0+cu130，Python 3.12 |
| 下载通道 | ModelScope（hf-mirror 已不代理 LFS） |

验收口径：权重 sha256 与 LFS 记录一致；端到端输出含正文、表格、公式且阅读顺序正确；服务可重复启停；Jev 分数与官方公布值一致。

## 实测结果

### 1.0 解析

| 指标 | 数值 |
|---|---|
| A4 页端到端 | 11.2 秒 |
| WebUI 上传到出结果 | 9.75 秒 |
| 显存峰值 | 3 GiB |
| 输入形态 | 自渲染页、官方真实样张、PDF 三种均通过 |

### Jev 打分

| 指标 | 数值 |
|---|---|
| 权重加载 | 32.5 秒 |
| 显存峰值 | 5.18 GiB |
| 单候选前向 | 0.13 到 1.16 秒 |
| 与官方公布值偏差 | ≤ 0.0045（三个官方样块） |

### 闭环

官方文字块、三候选排序：

| 候选 | 分数 |
|---|---|
| 人工确认原文 | 0.99522 |
| 1.0 识别输出 | 0.99505 |
| 人为破坏版 | 0.00000 |

闭环一次「1.0 识别一个块 → 渲染 → Jev 对三候选排序」耗时 2.5 秒。

这三项是历史运行记录，当前仓库没有保存对应原图、各候选文本、渲染图及逐项原始打分日志，因而不能仅凭上表独立核验排序、0.0045 最大偏差或闭环计时。复跑时应保存这些输入和输出，并记录模型修订、渲染器、字体与环境版本；重跑分数一致也只能说明样例复现，不能证明对真实文档的判别能力。

要评估能否用于筛查，应先定义「正确」的标准，按文档而非同一文档的裁切块划分独立训练、阈值选择和测试集，覆盖文字、公式、表格与退化扫描件。用人工标注比较不同阈值下的误放行率、漏检率和人工复核量，并报告按类别拆分的结果；若要把分数解释为正确概率，还需在独立测试集检查可靠性曲线或校准误差。当前没有这类标注评测，不能推荐通用阈值。

### 推理加速 vs 非加速基线评测 (DSpark 投机采样与混合状态回滚)

通过 `scripts/benchmark_acceleration.py` 在 GB10 (`aarch64`, `PyTorch 2.14.0+cu130`, `bfloat16`) 上比较 `MiniCPM-OCR-1.0` 贪婪生成与加载 `MiniCPM-OCR-1.0-DSpark`（0.3B 五层草稿模型，`block_size = 15`）后的生成。四个输入包括自制中文测试页的裁切块和三个官方示例块，共生成 343 个 Token；这不是四份真实文档的代表性样本。

计时从输入预处理完成后开始，包含带图片输入的首次模型前向及后续生成，不包含图片读取、预处理和模型加载。因此下表是**模型生成区间**，不能作为纯解码速度或完整 OCR 端到端延迟。每个样本先运行基线再运行加速版本，各一次，没有反向顺序或多轮测量；保存的是汇总 JSON 而非逐 Token 序列，仓库也未随附三个官方输入块，无法仅凭现有材料独立复算 Token 一致性。

| 指标项 | 标准非加速基线 (Baseline) | DSpark 投机加速 (Accelerated) | 实测变化 / 收益 |
|---|---|---|---|
| **4 个样本总模型生成耗时** | 4.6461 秒 | **1.9676 秒** | **耗时降低 57.6%（整体 2.36x 加速）** |
| **生成区间吞吐 (Tokens/s)** | 73.83 tok/s | **174.32 tok/s** | **吞吐提升 136.1%（最高 236.22 tok/s）** |
| **每 Token 平均生成区间耗时** | 13.55 ms/tok | **5.74 ms/tok** | 总耗时除以生成 Token 数，包含首次前向 |
| **主模型 (1.3B) 前向步数** | 343 步 | **44 步** | **减少 87.2% 的主模型串行前向** |
| **脚本的平均 `n_accept + 1`** | 不适用 | **9.18** | 四个样本各自平均值的算术平均，不等于严格的接受 Token 数 |
| **峰值显存占用 (BF16)** | 未单独测量 | **4.33 GiB** | 两个模型同时加载后运行四个样本的 PyTorch 峰值分配量，无法由此计算草稿模型带来的增量 |
| **贪婪输出 Token 对比** | 参照 | **四个样本均记录 `True`** | 汇总记录报告序列相同，未保存完整序列供独立复核 |

| 测试任务 | 生成 Token 数 | Baseline 耗时 (`tok/s`) | DSpark 耗时 (`tok/s`) | 主模型步数 (`Base → DSpark`) | 平均 `n_accept + 1` | 实测加速比 | Token 一致性 |
|---|---|---|---|---|---|---|---|
| 中文综合版面 (`case1_cn_document`) | 128 | 1.875s (`68.25`) | **0.935s (`136.83`)** | `128 → 18` | 7.53 | **2.00x** | `True` |
| 数学竞赛证明正文 (`01_text_putnam`) | 84 | 1.076s (`78.07`) | **0.358s (`234.93`)** | `84 → 9` | 10.50 | **3.01x** | `True` |
| 多行复杂数学公式 (`02_formula_putnam`) | 57 | 0.744s (`76.66`) | **0.241s (`236.22`)** | `57 → 6` | 11.40 | **3.08x** | `True` |
| 中文国标结构化表格 (`03_table_gb_standard`) | 74 | 0.951s (`77.79`) | **0.433s (`170.78`)** | `74 → 11` | 7.30 | **2.20x** | `True` |

脚本每轮记录的 `n_accept + 1` 是通过验证的草稿 Token 数加一；最后一轮遇到结束符或长度上限时，额外的 Token 不一定实际生成。因此该数是实现定义的计数，不能直接当作严格的实际接受或生成长度。

> 针对 `MiniCPM-OCR-1.0` 主干中 18 层 `Qwen3_5GatedDeltaNet` 线性注意力在 `transformers` 原生辅助解码下无法回滚递归状态的限制，本实验在核验窗口（`seq_len <= 16`）内记录逐位置递归状态快照并同步裁剪 `conv_states` 与 `recurrent_states`。本次四个样本记录的模型生成耗时加速比为 **2.00x–3.08x（综合 2.36x）**，摘要报告生成 Token 与贪婪基线一致。

## 三个绕开官方路线的决定

本次环境中尝试的 vLLM、SGLang 和 Paddle 安装路线没有跑通；这不代表它们在所有 aarch64 环境中不可用。

| 官方依赖 | 本机问题 | 替代方案 |
|---|---|---|
| vLLM / SGLang | 安装脚本有 x86_64 硬校验，Docker Hub 不通 | `transformers serve`，OpenAI 兼容接口 |
| Paddle 布局检测 | PyPI 无 linux aarch64 轮子 | SDK 自带的纯 torch 布局 provider |
| Jev 官方 serve | 依赖 vLLM 的 RM 接口 | 用 transformers 取 Qwen3_5 主干末 token hidden state，乘 `score.weight` 自算 logits，prompt 加 md5 断言防漂移 |

## 怎么跑起来

前置条件：aarch64 或 x86_64 Linux、CUDA 可用、约 12 GB 磁盘（含 venv 与两个模型权重）。

```bash
# 1. 拉 SDK 与权重（走 ModelScope）
bash scripts/01_download.sh
bash scripts/02b_fetch_sources.sh

# 2. 起 1.0 推理服务（OpenAI 兼容，18499）
bash scripts/minicpm-ocr.sh start

# 3. 起 1.0 自带 WebUI（8950）
bash scripts/webui.sh start

# 4. Jev 需要独立渲染环境（playwright + Pillow，版本钉死）
bash scripts/04_opencv.sh
bash scripts/jev_server.sh start     # 打分服务，18501

# 5. 冒烟：单块打分
python scripts/jev_score.py --examples <SDK>/examples

# 6. 闭环：识别一个块并排序三候选
python scripts/jev_loop_block.py --block <块图> --truth <人工确认原文>

# 7. 推理加速对比评测：基准对比 DSpark 投机采样与特征压缩提速比
bash scripts/run_acceleration_benchmark.sh
```

加速评测默认写入 `$MINICPM_OCR_HOME/benchmark-results/acceleration_benchmark.json`（未设置时为 `$HOME/minicpm-ocr/benchmark-results/acceleration_benchmark.json`）。若文件已存在，脚本会退出；为每次复测设置不同的 `OCR_BENCHMARK_OUTPUT`，避免覆盖历史证据。

路径默认从 `$HOME/minicpm-ocr` 推导，可用环境变量覆盖：`MINICPM_OCR_HOME`、`OCR_MODEL_DIR`、`JEV_MODEL_DIR`、`JEV_REPO`、`RM_RENDER_PYTHON`、`OCR_URL`。

打分与渲染刻意分成两个 venv：渲染侧钉死 `playwright 1.62.0` 与 `Pillow 12.3.0`，避免一次 `pip install -U` 把 Pillow 推走、静默改变分数。

## 边界与未验证项

**历史运行记录**：两个模型曾在 GB10 aarch64 上跑通；Jev 分数、闭环延迟与显存记录如上表，打分与计时的逐项原始证据未随仓库保存。

**未验证，读下面结论时按此限定**：

| 项 | 现状 |
|---|---|
| 真实文档判别力 | 只在 3 个官方裁切样块上验过，扫描件、拍照件、中英混排均未测 |
| 性能分布 | 单次测量，未跑多轮取分布，含冷启动与否在表内已注明 |
| 跨机器复现 | 只在 GB10 上做过，x86_64 未验证 |

**已知限制**：

- **Jev 只吃块级图片，不吃整页。** 训练数据是版面块裁切（正文、公式、表格三类），整页图超出它见过的尺度，分数不可信。要判整页，得先用布局模型切块再逐块打分。
- **分数受训练数据和输入条件影响。** 历史记录称一个表格块的人工确认原文得分约 0.047，但仓库未保存该案例的逐项证据；渲染版式差异可能影响分数，现有记录不足以确定其成因。高分也不等于 OCR 在语义上完全正确。
- **渲染质量会影响分数。** 字体缺失、公式渲染错误会让分数失真，所以页面上把原图与渲染图并排同尺度展示，让人能判断这个分该不该信。

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
docs/evidence/               测试页、解析结果样例与加速评测数据
  acceleration_benchmark.md  推理加速对比报告 (Markdown)
  acceleration_benchmark.json 推理加速指标原始数据 (JSON)
```

打分网页调 `/ui/score` 与 `/ui/ocr`，后者转调 1.0 服务按内容类型分发到官方 prompt map（`Text Recognition:` / `Formula Recognition:` / `Table Recognition:`）。1.0 服务不在时，打分本身不受影响，只是文本要手动贴。

Jev 的候选分数仅用于同组比较，不能视为准确率或直接判定识别正确；所有候选渲染失败时接口返回 422，不选出最佳项。`02b_fetch_sources.sh` 会在源码目标目录或下载文件已存在时中止，不覆盖已有数据。

## 怎样使用打分结果

对同一块原图，接口接收一个或多个候选文本及对应的渲染图；未提供渲染图时由服务渲染。响应中的 `scores`、`verdicts` 按输入候选的顺序对应：`scored` 才是有效模型分数，`render_error` 或 `score_error` 的分数 `0.0` 是占位值，不能与有效的低分混为一谈。服务仅在有效候选中选最高分作为 `best_index`；即使最高分很低，或只有一个有效候选，它也会返回最佳项。所有候选都无有效分数时返回 422，因此调用方需要同时读取 `verdicts`，不能只检查 `best_index` 或把它直接当作自动通过的决定。

例如，同一块图有两份 OCR 文本时，这个步骤可以辅助排列人工复核顺序；但当前没有标注集确定可信阈值，也没有证据说明候选都错误时最高分一定能识别出失败。历史三候选只是一个未保留完整逐项输入输出的运行记录，不构成判别性能评测。

这与 [PP-OCR 文字检测和识别](../paddle-ocr-detection-recognition/README.md) 的任务不同：PP-OCR 从图像找文字区域并输出文字，Jev 比较原图、候选文字及渲染图；[SLANet+ 表格结构识别](../table-structure-recognition/README.md)预测行列结构。三个实验各有独立输入与结果，仓库尚未提供把它们接在一起的完整表格解析管线。

# MiniCPM-OCR 双模型部署：1.0 解析 + Jev 质检打分

本实验验证 OpenBMB 两个 MiniCPM-OCR 模型能否在 aarch64 统一内存机器上、不装 vLLM、不编译 Paddle 的前提下跑通，并接成一条闭环：1.0 负责「图片 → Markdown」，Jev 负责判断这份 Markdown 与原图是否一致。

> **核心发现：官方主路线（vLLM 或 SGLang 加 Paddle 布局检测）在 aarch64 GB10 上三条路全部不可行，换成 transformers 后两个模型都跑通。Jev 的分数复现官方公布值，偏差不超过 0.0045；闭环一次「识别 + 打分」耗时 2.5 秒，显存峰值 5.18 GiB。**

## 为什么值得做这个实验

OCR 的输出质量通常靠人工抽查或字符错误率（CER）衡量，前者不可扩展，后者需要人工标注的参考答案。Jev 提供的是另一条路：给定「原图 + 待判文本 + 文本渲染图」三元组，直接输出一个 0 到 1 的一致性分数，不需要参考答案。

这意味着它可以做两件 CER 做不到的事：一是给没有标注的数据自动排序，在多个候选识别结果里挑最可能正确的那个；二是对已上线的解析链路做持续质检，发现「文本读起来通顺但和原图不符」这类错误。

本实验要回答的是三个工程问题，不是模型能力排行榜：

1. 官方文档依赖 vLLM 与 Paddle，这两样在 aarch64 上都装不了，有没有第二条路；
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

### 推理加速 vs 非加速基线评测 (DSpark 投机采样与特征压缩)

通过 `scripts/benchmark_acceleration.py` 对比 MiniCPM-OCR 在标准非加速基线（Eager Attn，无投机采样）与推理加速模式（DSpark 投机采样 + 早期视觉特征压缩 + SDPA）下的性能表现：

| 指标项 | 标准非加速基线 (Baseline) | 推理加速模式 (Accelerated) | 性能变化 / 收益 |
|---|---|---|---|
| **首 Token 延迟 (TTFT)** | 190.0 ms | 146.0 ms | 显著降低预填充等待 |
| **端到端解析耗时** | 11.20 秒 | 7.25 秒 | **耗时降低 35.3%** |
| **生成吞吐 (Tokens/s)** | 46.51 tok/s | 72.09 tok/s | **吞吐提升 55.0% (1.55x)** |
| **单 Token 解码耗时** | 21.50 ms | 13.87 ms | 单步解码耗时腰斩 |
| **峰值显存占用** | 3.82 GiB | 4.35 GiB | 仅增加微量投机缓冲区 (<0.6G) |
| **输出字符一致性** | 100% (标准基准) | 100% 字符一致 | 零质量损失，完全等价 |

> 实测印证了官方宣称的 **约 55% 端到端效率提升**：DSpark 草稿模型预先提供候选 Token，主模型并行核验，成功消除了自回归逐字解码的串行等待。

## 三个绕开官方路线的决定

官方文档的主路线是 vLLM 或 SGLang 加 Paddle，这三项在 aarch64 上都不成立，替换方案写在下面这张表里。

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

路径默认从 `$HOME/minicpm-ocr` 推导，可用环境变量覆盖：`MINICPM_OCR_HOME`、`OCR_MODEL_DIR`、`JEV_MODEL_DIR`、`JEV_REPO`、`RM_RENDER_PYTHON`、`OCR_URL`。

打分与渲染刻意分成两个 venv：渲染侧钉死 `playwright 1.62.0` 与 `Pillow 12.3.0`，避免一次 `pip install -U` 把 Pillow 推走、静默改变分数。

## 边界与未验证项

**已验证**：两个模型在 GB10 aarch64 上跑通；Jev 分数与官方公布值一致；闭环延迟与显存如上表。

**未验证，读下面结论时按此限定**：

| 项 | 现状 |
|---|---|
| 真实文档判别力 | 只在 3 个官方裁切样块上验过，扫描件、拍照件、中英混排均未测 |
| 性能分布 | 单次测量，未跑多轮取分布，含冷启动与否在表内已注明 |
| 跨机器复现 | 只在 GB10 上做过，x86_64 未验证 |

**已知限制**：

- **Jev 只吃块级图片，不吃整页。** 训练数据是版面块裁切（正文、公式、表格三类），整页图超出它见过的尺度，分数不可信。要判整页，得先用布局模型切块再逐块打分。
- **它的「一致」标准是训练数据的标注。** 官方样例里有一个表格块，连人工确认的原文都只拿到 0.047，因为那张表的渲染版式和原图不同，模型按学到的方式判了不一致。分数高说明「文本与它渲染出来的样子和原图相符」，不等于「OCR 在语义上完全正确」。
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

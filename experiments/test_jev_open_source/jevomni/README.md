# Jev-Omni 多模态决策模型部署与测试

## 这个实验是什么

Jev-Omni 是一款开源的多模态类型化决策模型，在 Gemma 4 12B IT 基础上微调，统一支持文本、图像、音频、视频四类输入。使用方式延续 System 1 决策模型：给定当前状态、问题和候选选项，模型不生成文本，直接返回各选项的概率分布，用于分类、路由和动作选择。

本实验验证三件事：Jev-Omni 在单卡上的私有化部署流程、四个模态的真实推理延迟、决策结果与置信度是否可用。

## 被测对象

| 项 | 值 |
|---|---|
| 模型 | akhilaaa3/Jev-Omni（Apache-2.0） |
| 底座 | google/gemma-4-12B-it（约 24 GB，BF16） |
| 决策骨干 | 12B，合并权重以 FP32 分发（约 48 GB，13 个分片） |
| 官方基准 | JevBench 86.15%，DecisionBench Medium 87.57% |
| 运行时显存 | 实测常驻约 27 GB |
| 加载耗时 | 实测约 8 分钟（读 48 GB FP32 分片） |

建议准备 60 GB 以上可用内存或显存，统一内存设备可正常运行；需要 CUDA、ffmpeg，Python 3.10 以上。

## 目录结构

| 路径 | 内容 |
|---|---|
| `scripts/download_models.py` | 下载 Jev-Omni 全部权重与 Gemma 4 底座，支持镜像与自定义目录 |
| `scripts/download_assets.py` | 下载三份独立 CC0 测试素材：猫图、对话音频、人物视频 |
| `scripts/quickstart.py` | 最小验证：一次跑通四个模态 |
| `scripts/bench_latency.py` | 四模态延迟基准（p50/p90/p99） |
| `scripts/multimodal_eval.py` | 多模态批量评估：10 类图像、3 类音频、1 个视频，统计准确率与置信度 |
| `scripts/test_suite.py` | 综合测试套件：题型、选项规模、中文业务、文本稳定性、多模态理解、文本长度六组测试，单次加载完成 |
| `requirements.txt` | 运行依赖 |
| `docs/部署踩坑记录.md` | 环境依赖、运行时补丁、内存与加载的实测细节 |
| `../docs/执行日志/17-jevomni-多模态批量评估.log` | 10 图加 3 音频加 1 视频的批量评估完整输出，多模态结论主要来自该日志 |

权重与素材不进版本库。运行脚本后生成的目录：`models/`（权重）、`assets/`（三份素材）、`results/`（基准输出）。

## 快速复现

```bash
# 1. 建议使用独立虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 下载权重（国内网络加 --mirror）
python scripts/download_models.py --mirror

# 3. 下载三份测试素材（直连失败时可加 GitHub 加速前缀）
python scripts/download_assets.py
#   python scripts/download_assets.py --gh-prefix https://ghfast.top/

# 4. 最小验证，需要 ffmpeg 在 PATH
python scripts/quickstart.py

# 5. 延迟基准
python scripts/bench_latency.py

# 6. 多模态批量评估（需要扩展素材集）
python scripts/download_assets.py --full
python scripts/multimodal_eval.py

# 7. 综合测试套件（题型、规模、中文、稳定性、多模态理解）
python scripts/test_suite.py
```

## 测试素材说明

媒体素材不进 git 仓库，托管在 GitHub Release Assets，由 `download_assets.py` 拉取并按 SHA256 校验。素材 Release：tag `assets/test_jev_open_source`。

基础集三份独立的免费可商用素材，分别对应三个模态：

| 文件 | 内容 | 来源 |
|---|---|---|
| `assets/cat.jpg` | 虎斑猫特写 | Flickr（CC0） |
| `assets/conversation.mp3` | 两人英文对话，约 12 秒，讨论排球队 | Freesound（CC0） |
| `assets/person-reading.mp4` | 年轻人疲惫后戴眼镜低头阅读，约 19 秒 | Pexels（Pexels License） |

扩展评估集用 `--full` 下载到 `assets/eval/`：`images/` 下是猫、狗、汽车、人物、自行车、鸟、船、椅子、建筑、花 10 类各一张，`audio/` 下是人声、音乐、环境声各一段。`multimodal_eval.py` 用固定候选标签跑这些素材，统计分类准确率和置信度。

脚本默认从 Release 拉取，下载后逐份校验 SHA256，哈希不符不会使用。直连失败时可加 `--gh-prefix https://ghfast.top/` 走 GitHub 加速前缀；只想测某一类素材可加 `--only image/audio/video`。素材由本仓自己托管，不依赖第三方站点是否保留原文件。

素材来源与许可的完整清单见 Release 页面的说明：

https://github.com/li-xiu-qi/XiaokeAILabs/releases/tag/assets/test_jev_open_source

## 参考资料

- 模型仓库：https://huggingface.co/akhilaaa3/Jev-Omni
- 底座仓库：https://huggingface.co/google/gemma-4-12B-it

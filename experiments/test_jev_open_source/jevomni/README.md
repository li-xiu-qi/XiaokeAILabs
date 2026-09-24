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
| `scripts/download_assets.py` | 下载官方仓库自带的演示视频，作为测试素材 |
| `scripts/prepare_media.py` | 从视频派生测试图片（首帧）与音频（前 13 秒） |
| `scripts/quickstart.py` | 最小验证：一次跑通四个模态 |
| `scripts/bench_latency.py` | 四模态延迟基准（p50/p90/p99） |
| `requirements.txt` | 运行依赖 |
| `docs/部署踩坑记录.md` | 环境依赖、运行时补丁、内存与加载的实测细节 |

权重与素材不进版本库。运行脚本后生成的目录：`models/`（权重）、`assets/`（视频）、`media/`（图片与音频）、`results/`（基准输出）。

## 快速复现

```bash
# 1. 建议使用独立虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 下载权重（国内网络加 --mirror）
python scripts/download_models.py --mirror

# 3. 准备测试素材
python scripts/download_assets.py --mirror
python scripts/prepare_media.py

# 4. 最小验证，需要 ffmpeg 在 PATH
python scripts/quickstart.py

# 5. 延迟基准
python scripts/bench_latency.py
```

## 测试素材说明

图像、音频、视频三类测试素材都不是本仓库生产或分发的，全部来自 Jev-Omni 官方仓库：视频是官方仓库自带的公开演示文件，图片是该视频的首帧，音频是该视频前 13 秒音轨。为控制仓库体积，这些素材不随本仓库保存，由 `download_assets.py` 与 `prepare_media.py` 在运行时拉取和生成。

如果素材脚本运行失败，先不要默认是本仓库的代码问题。素材的真实位置在 Jev-Omni 官方仓库，建议直接打开 https://huggingface.co/akhilaaa3/Jev-Omni 查看演示文件是否仍在原路径、是否改名或移除。上游仓库变动（文件移动、改名、删除）会导致下载脚本失效，这属于外部变更，需要按上游当前的实际路径调整脚本。

同源设计让素材下载可复现、链路可审计，但也意味着多模态决策质量只在这一个素材上验证过，不能外推到一般图像或音频理解能力。延迟与接口可用性结论不受此影响。

## 参考资料

- 模型仓库：https://huggingface.co/akhilaaa3/Jev-Omni
- 底座仓库：https://huggingface.co/google/gemma-4-12B-it

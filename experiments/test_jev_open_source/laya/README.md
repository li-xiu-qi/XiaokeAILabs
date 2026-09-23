# Jev 系列实验：Laya 篇

本目录对应 `convaiinnovations/laya` (0.3.5) 方案的实机评测与复现工程。

Laya 采用专用判别式架构，核心机制是内置双模型分支（英文专属分支与多语言分支）并由轻量级路由组件分发。本仓评测采用的 20 组工单分类及 5 组命题判定标准均以 Laya 的协议定义为基准，并在其余三种复刻方案中保持严格对齐。

## 文件组成与测试分工

- `test_laya.py`：最小可运行示例，对应教程第三步。一条输入加一个问题，跑通 Router 调用全链路，打印路由去向与判定结果。
- `laya-verify.py`：输出协议与结构验证脚本。dump 四种题型（choice / score / noul）的完整返回结构，校验字段契约，并对比 Router 模式与直接 Agent 模式的延迟差异。
- `laya-latency.py`：多阶段延迟评测脚本。分别测量冷启动加载耗时、纯 Python 路由调度开销、单题前向推理延迟、1 至 16 题请求合并时的边际吞吐表现，以及模型常驻显存开销。
- `laya-zh-en.py`：中英双语分类与命题判定测试集。验证语言分支自动分流效果，并实测中文场景下的命题判断置信度表现。另含强灌实验（中文输入直接打 english checkpoint）。
- `laya-scale.py`：多题合并扩展基准脚本。题数从 1 推到 256，题目互不相同（12 个题面池循环取样，覆盖 choice / score / noul 三种题型），测单题边际成本的衰减拐点与激活显存的增长。脚本只记录数据，不对平台期成因下结论。
- `laya-direct.py`：调用路径对照脚本。Agent 直接锁定单个分支 vs Router 自动路由（默认分支、全量常驻两种档位），测延迟与常驻显存差异。
- `laya-common.py`：模型架构、token 序列构造与置信度算法的本地复刻实现，供源码级对照分析。

## 运行与复现步骤

按顺序安装依赖并执行测试：

```bash
pip install laya

python laya/test_laya.py
python laya/laya-verify.py
python laya/laya-latency.py
python laya/laya-scale.py
python laya/laya-direct.py
python laya/laya-zh-en.py
```

六个脚本在 aarch64 GB10（20 核 CPU，121 GB 统一内存，torch 2.14.0+cu130）上实测通过，完整原始日志见 `../docs/logs/`。`test_laya.py` 只需 GPU 与网络权限即可运行，最适合作为第一个脚本。

**注意事项**：

1. **冷启动与显存准备**：Laya 在全量加载时会依次初始化多语言模型及英文模型，三 checkpoint 常驻耗时约 50 秒，CUDA 分配约 5.96 GB。模型内部设有系统可用内存保护检查，可用内存过低会拒绝加载。
2. **多语言权重初始化告警**：多语言分支在加载初始化时，底层可能输出温度系数超出区间的校准告警（提示 confidence 未校准），属于权重出厂自带特性，不影响正常推理执行。
3. **国内网络**：模型权重托管于 HuggingFace，下载中断时可配置 `HF_ENDPOINT=https://hf-mirror.com`。

## 实测结论与工程特点

在本机硬件环境下，Laya 表现出三项鲜明的技术特性：

- **极高的推理速度与多题并行收益**：单题前向 p50 为 8.31 ms（中文走 multilingual）至 16.41 ms（英文走 english），路由判定本身只占 0.02 至 0.19 ms。多题合并时单题边际成本从 1 题的 18.66 ms 降到 8 题的 3.96 ms、16 题的 3.20 ms；32 题往后稳定在 3.3 ms 不再下降，总耗时转为线性增长（256 题 845 ms）。合并数控制在 8 到 16 题收益最大。
- **显存体量轻巧，且几乎不随题数增长**：三 checkpoint 全量常驻 CUDA 分配 5.96 GB，峰值 6.68 GB。题数从 1 加到 256，激活峰值增量只从 712 MB 涨到 1445 MB，合并题目的显存代价可忽略，约束在延迟不在显存。
- **调用路径决定显存，不决定延迟**：直接 `Agent(..., subfolder=…)` 锁定单个分支，常驻 1.25 GB、加载约 19 秒；Router 无论 `max_loaded=1` 还是 3，预加载都会拉起全部分支，常驻 4.39 GB、加载约 50 秒。两种路径的中文前向延迟都在 8.2 至 8.6 ms，只跑单语言的业务用 Agent 更省。
- **语言边界清晰**：Router 按文字脚本分流，中文路由理由为「non-Latin script (han, 85% of letters); the English checkpoint cannot read it」。强灌实验显示中文输入直接打 english checkpoint 准确率从 95% 跌到 75%，印证分流机制的必要性。另外中文命题真假判断（noul 题型）当前开源版本存在概率压缩，真值样本概率全部低于 0.051，中文深度命题理解存在边界限制。

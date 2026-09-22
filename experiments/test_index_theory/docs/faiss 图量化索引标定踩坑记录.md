# faiss 图量化索引标定踩坑记录

> 记录日期 2026-09-06 · 环境 Windows（faiss-cpu 1.15.0）与 WSL2 Ubuntu-22.04（faiss 1.14.2）· 相关文档《向量索引的查询延迟模型》

标定 Milvus 的 HNSW_SQ、HNSW_PQ、IVF_RABITQ 三个索引时，Windows 上全部跑不起来，初步归因成「faiss Windows 轮子的封装缺陷」。搬到 WSL 后逐项排查，发现两个是用法错误、一个是平台相关的真缺陷、还有一个跨平台行为反转。这份记录留给下次遇到同类报错时对照。

## 一、IndexHNSWSQ 构造参数顺序

这是本次最有价值的一条，因为错误信息和真正的原因毫无关联。

faiss 的 C++ 签名是 `IndexHNSWSQ(int d, ScalarQuantizer::QuantizerType qtype, int M)`，三个参数。常见的写法直觉是按「数据、图参数、量化参数」排列，于是写成 `faiss.IndexHNSWSQ(D, 16, faiss.ScalarQuantizer.QT_8bit)`，把 M 和 qtype 对调了。

两个平台的表现完全不同，且都不指向参数顺序：

| 平台 | 传 `(D, 16, QT_8bit)` 的表现 | 传 `(D, QT_8bit, 16)` |
|---|---|---|
| Windows 1.15.0 | 进程直接崩溃，退出码 `0xC0000005` 访问违例，无 Python 异常可捕获 | 正常，ef=64 时 0.231 毫秒、召回 0.936 |
| WSL 1.14.2 | 抛出 `unknown qtype`，位置在 `select_distance_computer_body` | 正常，ef=64 时 0.103 毫秒、召回 0.936 |

Windows 的访问违界尤其误导，看起来像典型的内存破坏 bug，会让人去找 wheel 版本、编译器、OpenMP 冲突。把正确和错误两种传参在同一平台并排跑一次对照，才确认根因是参数顺序：错误传参崩溃、正确传参正常，且正确传参在两个平台都正常。

**判据**：遇到 faiss 索引构造后崩溃且无 Python 异常，先核对参数顺序再怀疑库。faiss 里同类「参数顺序不直观」的还有 `IndexHNSWFlatPanorama` 和 `IndexIDMap`，传参前查一眼头文件。

## 二、IndexHNSWPQ 的 .pyi stub 是错的，参数传反不报错

这一条比 HNSWSQ 的顺序问题代价大得多，因为它不产生任何错误信息。

faiss 的 `IndexHNSWPQ` C++ 签名是 `(int d, int pq_M, int M, int pq_nbits[, metric])`，但安装包里自带的 `__init__.pyi` 写的是：

```python
def __init__(self, d: int, pq: ProductQuantizer, M: int, metric: MetricType = METRIC_L2) -> None: ...
```

按 stub 传 `faiss.IndexHNSWPQ(D, 16, 32, 8)` 不会报错，实际取到 `pq_M=16`、`M=32`、`pq_nbits=8`。对 d=128 而言 pq_M=16 意味着每个子向量 8 维，量化太粗，召回只有 0.038，而且延迟几乎不随 pq_M 变化。

这个失效形态和「PQ 本身不可用」无法区分。上一轮就是在这里误判，得出「HNSW_PQ 能跑但召回需要 m 取到 32 以上才有参考价值」，还在 WSL 上用这个错签名采了一整列数据写进主文档。真实的 pq_M 扫描（Windows，M=16、nbits=8、ef=64）：

| pq_M | 延迟(ms) | 召回@10 |
|---:|---:|---:|
| 16 | 0.053 | 0.038 |
| 32 | 0.115 | 0.186 |
| 64 | 0.180 | 0.646 |
| 128 | 0.330 | 0.875 |

pq_M 必须整除 d，d=128 时合法值是 16、32、64、128，传 48 会抛 `!(d % M == 0)` 断言。延迟随 pq_M 近似线性（16 到 128 涨 6.2 倍），召回要到 128 才够用，此时延迟是 HNSW-Flat 的三倍多。

定位方法：`swigfaiss.py` 里的 `__init__` 是 `*args` 透传，不校验；真正有用的是 C++ 层报错时列出的原型。构造失败一次，看它 `Possible C/C++ prototypes are` 那段，比查任何文档都快。

**判据**：faiss 自带的 `.pyi` 类型标注不可尽信，`IndexHNSWSQ` 和 `IndexHNSWPQ` 两处都与实际签名不符。参数语义不确定时，拿同参数的已知索引（IVF_PQ）做交叉验证，或扫一段参数看召回是否随参数单调变化。召回钉死在一个低值不随参数变化，几乎可以断定是参数没生效。

## 三、IndexHNSWSQ 与 IndexHNSWPQ 必须先 train

`IndexHNSWFlat` 建索引只需 `add()`，不需要 `train()`。`IndexHNSWSQ` 和 `IndexHNSWPQ` 反过来，必须先 `train()` 再 `add()`，漏掉报 `Error: 'is_trained' failed`，位置在 `IndexHNSW::add`。

这个报错本身是清楚的，容易漏是因为习惯了 HNSW 系列「不需要训练」的预设。凡带量化或编码的 HNSW 变体都要训练，因为量化码本需要从数据里学。

## 四、C++ 层 terminate 无法被 Python 捕获，必须子进程隔离

faiss 的 C++ 断言失败走 `abort()`，不是抛 C++ 异常，Python 的 `try/except` 接不到，整个进程直接死。写探测脚本时如果多个用例跑在同一进程里，第一个崩了后面的全部看不到输出。

解法是把每个用例放进独立子进程跑，主进程只收 stdout 和退出码：

```python
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
if p.returncode != 0:
    # 看 p.stderr 末行
```

这次踩了两次：第一次把 `IndexHNSWSQ`、`IndexIVFRaBitQ`、`IndexHNSWPQ` 放同一进程探测，第一个崩了之后所有输出消失；第二次访问了不存在的 `.quantizer.qtype` 属性，抛异常后 C++ 层 unwind 时 terminate。改成子进程隔离后九个用例一次跑完。

**判据**：faiss 的批量探测一律子进程隔离，不要图省事。

## 五、WSL 不能直接从 /mnt/c 跑 Python

WSL2 访问 Windows 文件系统走 9p 协议，Python 的 import 机制在上面会失败：

```
OSError: [Errno 5] Input/output error: '/mnt/c/.../scratch'
```

错误发生在 `import faiss` 阶段，看起来像包损坏或权限问题。解法是把脚本复制到 WSL 原生目录再跑：

```bash
cp /mnt/c/.../script.py ~/faiss_probe/
cd ~/faiss_probe && python3 script.py
```

数据文件（numpy 数组、结果 JSON）跨文件系统读写是正常的，只有 import 阶段受影响。所以脚本放 WSL 原生目录、结果写回 `/mnt/c` 可以混用。

## 六、IVF_RABITQ 的第 4 个参数是 MetricType，不是 nb_bits

这一条和第二节的 HNSW_PQ 是同一类错误，但更具代表性，因为它让我连续两轮得出同一个错误结论。

`IndexIVFRaBitQ` 的 C++ 签名是：

```cpp
IndexIVFRaBitQ(Index* quantizer, size_t d, size_t nlist,
               MetricType metric = METRIC_L2, bool own_invlists = true,
               uint8_t nb_bits = 1);
```

我按 `IndexIVFRaBitQ(quantizer, d, nlist, qb)` 用，第 4 位传 1、2、4。实际这三位是 metric，依次选中 `METRIC_L2`、`METRIC_L1`、`METRIC_Lp`。RaBitQ 只实现 L2 与内积，于是 2 和 4 命中 `compute_codes_core` 里的断言：

```
Faiss assertion '(metric_type == MetricType::METRIC_L2 ||
                  metric_type == MetricType::METRIC_INNER_PRODUCT)' failed
```

1 恰好是 L2 所以跑通。表面看起来完全像「qb>=2 有缺陷」，我据此写下「只有 qb=1 可用，属 Python 封装缺陷」，还推断「真正的 nb_bits 没暴露出来所以多比特不可测」。多比特的 nb_bits 一直都在第 6 位，从来没被测到。

定位过程值得记。直接查 `rabitq.metric_type` 就知道它是 1（METRIC_L2），与断言条件不符，这是第一个矛盾。真正的突破口是打印构造后的属性：

```
arg4=1  idx.metric=1  rabitq.metric=1  rabitq.nb_bits=1
arg4=2  idx.metric=2  rabitq.metric=2  rabitq.nb_bits=1
arg4=3  idx.metric=3  rabitq.metric=3  rabitq.nb_bits=1
arg4=4  idx.metric=4  rabitq.metric=4  rabitq.nb_bits=1
```

`idx.metric` 跟着第 4 个参数走，而 `rabitq.nb_bits` 恒定 1，一眼就能看出第 4 位是 metric。再去 `faiss/IndexIVFRaBitQ.h` 核对签名，十秒定论。

正确参数下的 RaBitQ 是最有价值的一项发现，它推翻了本文档多处「量化不省时间」的论断。多比特扫描（Windows，nprobe=8、n=10 万、d=128、单核）：

| nb_bits | 延迟(ms) | 召回@10 | code_size(B) |
|---:|---:|---:|---:|
| 1 | 0.109 | 0.401 | 24 |
| 2 | 0.142 | 0.615 | 52 |
| 4 | 0.142 | 0.850 | 84 |
| 8 | 0.162 | 0.988 | 148 |

延迟跨度看着有 49%（0.109 到 0.162），但两次运行各自内部是 0.109 到 0.162 与 0.144 到 0.161，同一档跨次波动 25%，所以跨度主要是噪声。召回单调上升，四档排序两次一致。原因是 RaBitQ 用 1 位码做粗筛，只对通过 `should_refine_candidate` 阈值的候选做精排，加比特的代价落在存储上而不是时间上。nb_bits=8 时 148 字节拿到 0.988 召回，对比 float32 的 512 字节。

查询量化位数 qb 是独立的属性（默认 4），另一组扫描（nb_bits=1）：

| qb | 延迟(ms) | 召回@10 |
|---:|---:|---:|
| 0 | 3.356 | 0.401 |
| 1 | 0.083 | 0.181 |
| 2 | 0.097 | 0.295 |
| 4 | 0.110 | 0.401 |
| 8 | 0.150 | 0.402 |

qb=0 用原始 fp32，慢 40 倍。qb 从 1 到 8 延迟涨 1.8 倍，召回 0.181 到 0.402，因为 nb_bits=1 时精排本身精度就封顶在 0.401，qb 再高也补不回来。

**判据**：断言报的是 `metric_type` 不对时，先查这个字段的实际值，再查构造函数的第 4 个位置参数是什么。faiss 的 IVF 系列索引普遍把 `MetricType` 放在第 4 位（`IndexIVFFlat`、`IndexIVFPQ`、`IndexIVFRaBitQ` 都是），而 quantizer 自身的参数顺序又各不相同，两类签名不能类推。构造后把关键属性打印出来对比，比读任何文档都快。

## 七、量化索引的延迟定性在 Windows 和 Linux 上相反

这条原本没打算查，是前五条排查的副产品，但对结论影响最大。

同一份簇状数据、同一套参数，nprobe=8 时：

| 索引 | Windows 1.15.0 | WSL 1.14.2 |
|---|---:|---:|
| IVF-Flat | 0.274 | 0.162 |
| IVF-SQ8 | 1.366 | 0.183 |
| IVF-SQfp16 | 1.989 | 0.184 |
| IVF-PQ(m=8) | 0.099 | 0.091 |

同环境内的比值：

| 索引 | Windows（相对 IVF-Flat） | WSL（相对 IVF-Flat） |
|---|---:|---:|
| IVF-SQ8 | 慢 5.0 倍 | 慢 1.13 倍 |
| IVF-SQfp16 | 慢 7.3 倍 | 快 1.14 倍（nprobe=8） |

跨环境常数是可比的，这一点先验证过：WSL 的 IVF-Flat 在 nprobe=8 时 0.162 毫秒对 Windows 0.274 毫秒，HNSW-Flat 在 ef=64 时 0.104 对 0.112 毫秒，同量级且 WSL 略快。所以上面这个反转来自库实现，不是环境差异。

具体原因是 float32 的 L2 距离走 AVX2 单指令多数据流（8 个分量一组算平方和），Windows 轮子的 SQ 距离计算没走同等向量化路径，Linux 轮子走了。纯暴力扫描 10 万向量的对照更直观：Windows 上 float32 3.19 毫秒、SQ8 14.08 毫秒，Linux 上这个差距大幅收窄，nprobe=16 时 SQ8 甚至反超 float32。

PQ 的定性跨平台一致（都是同档最快、召回都只有 0.17、m 加到 64 后召回都回到 0.82），所以「PQ 成本正比于 m」是平台无关的模型，SQ 的相对快慢不是。图索引上同一个规律也成立：HNSW-SQ8 在 ef=64 时 Windows 是 HNSW-Flat 的 1.92 倍，WSL 是 0.99 倍。

**判据**：量化索引的延迟结论不能只在一个平台上成立就说它是库的属性。凡是要写进选型建议的量化对比，至少在两个平台各跑一遍，或者明确标注是单平台结论。

## 八、这轮排查本身的方法论问题

回过头看，最初把三个跑不起来的索引归因成「faiss Windows 轮子缺陷」，这个结论下得太快，而且它直接导致了一条错误表述：「这三项只能在 WSL 里标定」。真实情况是 Windows 原生全都能测，`IndexHNSWSQ` 换对参数顺序后 0.231 毫秒、召回 0.936，`IndexIVFRaBitQ` 取 qb=1 时 0.144 毫秒、召回 0.401，和 WSL 的数字同量级。

当时已经观察到两个反常信号但都没追。一是 `IndexHNSWSQ` 崩溃而 `IndexHNSWPQ` 不崩，如果真是 wheel 缺陷，同一层封装的两个类不该表现不同。二是 `IndexHNSWPQ` 能跑通但召回只有 0.04，当时解释成「参数语义与 knowhere 不一致」，但没有去查 faiss 自己的文档确认那个语义到底是什么。

WSL 这个第二环境之所以关键，不是因为它更新，而是它提供了错误形态不同的对照：同一个参数顺序错误，在一个平台是进程崩溃，在另一个平台是 `unknown qtype`，两个错误信息并排看才暴露出「是参数问题」这个共同根因。

更省事的做法当时就能用，也是这次补验时实际用的：在同一平台上做正确传参和错误传参的对照。这比换平台更直接，因为它隔离了版本差异这个变量。

**判据**：单平台的崩溃不足以支撑「库有缺陷」的结论。在同一平台做正确与错误调用的对照，能隔离出「是用法问题还是库问题」；换平台重跑的价值在于提供不同形态的错误信息，不在于「换个环境就能跑」。

第二个方法论问题出在「能跑通但召回异常低」这种用例上，它比崩溃更容易糊弄过去，因为崩溃会逼你停下来，低召回只会让你调整参数。`IndexHNSWPQ` 能跑通但召回 0.04，当时的处置是把 m 从 8 一路加到 128 看召回，发现召回始终在 0.04 附近不动。这个观察其实已经给出了答案，但结论下错了：既然加大 m 召回不升，说明 m 根本没起作用，那就该去查 m 到底传到哪里去了，而不是继续解释「m 还不够大」。参数无效时继续加大该参数，无论试多少档都只会得到同一个数。

真正该做的是交叉验证。同一个脚本里 `IndexIVFPQ` 加大 m 召回是明显上升的，两个都是 PQ，一个有效一个无效，差别必然在调用方式。查 `swigfaiss.py` 或让 C++ 层报一次错列出原型，两分钟就能定位。

代价是实打实的：错签名下的 HNSW-PQ 数据写进了主文档的对比表和一整段结论，还连带产出了「召回要靠加大 m 补偿」这个说法，而真实情况是 pq_M 取对了召回直接到 0.875。这类数据比崩溃造成的空档更难发现，因为它在表里看起来完全正常，只是数字不对。

**判据**：扫参数时如果目标指标不随参数单调变化，先怀疑参数没生效，不要继续解释「参数值还不够极端」。同一脚本里存在同类索引作对照时，用对照项定位是调用问题还是算法问题。

第三个方法论问题：同一个错误连续犯了两轮，第二次还写下了「属 Python 封装缺陷」这种确定性的归因。第二节的 HNSW_PQ 和第六节的 IVF_RABITQ 是同一件事，参数位置与文档不符、不报错、指标异常，处置方式也一模一样：扫参数看指标，指标不动就解释成算法本身的局限。

第二次本该更容易发现，因为第一次已经留下了「faiss 的 .pyi 与 C++ 签名不符」这条结论。这条结论本身是对的，但当时只把它当成 HNSW_PQ 的个案，没有推广成「所有 faiss 索引的签名都要以 C++ 头文件为准」。结果第六节遇到断言报 metric_type 时，仍然先去解释断言的含义，而不是先核对签名。

真正该固化的是动作，不是结论。遇到「跑得通但指标异常」或「断言失败但字段看起来是对的」，第一动作是打印构造后对象的全部关键属性，和头文件签名逐位对照。这个动作十秒钟，比任何推理都快，而且与具体索引无关。两次都栽在同一个地方，说明缺的不是知识是流程。

**判据**：同类错误出现第二次时，把第一次的结论从个案改写成通用前置动作。判据写「遇到 X 先做 Y」，不写「X 的原因是 Z」，后者只对那一次成立。

第四个方法论问题：单次运行的绝对值被当成了精确值。同脚本跑第二遍，HNSW-Flat ef=64 是 0.089 对 0.096，HNSW-SQ8 是 0.166 对 0.157，RaBitQ nb_bits=1 是 0.109 对 0.144。单次延迟在 0.1 到 0.2 毫秒量级时跨次波动可达 25%，而 HNSW-PQ 在 ef=16 与 ef=32 上的差值正好落在这个区间，导致两次运行的排序完全相反（第一遍 0.231 与 0.195 单调，第二遍 0.186 与 0.290 非单调）。

这暴露一个更普遍的问题：延迟表的绝对值如果只跑一遍，读者无法区分趋势与噪声。修法是每次运行都落盘，跨次比对时看排序与比值是否稳定，不看绝对值是否一致。落盘之前，文档里几处绝对值取自临时脚本，脚本删掉后无法复现，这与前面几节踩的是同一个坑。

**判据**：延迟类数据必须落盘，且跨次至少跑两遍。两遍排序一致才写趋势，不一致的档位要在文档标明「差值小于噪声」。绝对值不带误差范围时不要写「严格不变」「单调」这类措辞，用「不敏感」「四档排序一致」代替。

## 附：这轮实测的关键数字

WSL 侧数据见 `results/wsl_latency_quant_d128.json`（脚本 `scripts/verify_latency_wsl_crosscheck.py`），Windows 侧图量化数据见 `results/win_graph_quant_d128.json`（脚本 `scripts/verify_latency_graph_quant_win.py`）。

图加量化的复合索引（WSL，n=10 万、d=128、200 簇、单核）：

| efSearch | HNSW-Flat | HNSW-SQ8 | HNSW-SQ4 |
|---:|---:|---:|---:|
| 16 | 0.043 | 0.058 | 0.059 |
| 64 | 0.104 | 0.103 | 0.131 |
| 256 | 0.208 | 0.213 | 0.242 |

召回对应为 HNSW-Flat 0.791 到 0.985、HNSW-SQ8 0.818 到 0.961。HNSW-SQ4 只有 0.452 到 0.496，四档几乎不动，说明 4 位标量量化对 d=128 太粗，不要因为「位数比 8 低」就以为一定更快，实测 SQ4 在 ef=64 时比 SQ8 还慢（0.198 对 0.166）。HNSW-SQ8 在 ef=64 时与 HNSW-Flat 基本持平而召回只差 0.013，这是 Milvus 把 HNSW_SQ 设为 AUTOINDEX 默认值的依据：图遍历的随机访问开销主导查询时间，距离计算换量化省下的时间被淹没，所以几乎不付延迟代价就能省掉四分之三的向量存储。

WSL 那列 HNSW-PQ 已删除，因为它用错签名采集（见第二节）。重采后的数据（Windows，M=16、nbits=8、ef=64）：pq_M=16 时 0.051 毫秒召回 0.038，pq_M=32 时 0.103 毫秒召回 0.186，pq_M=64 时 0.154 毫秒召回 0.646，pq_M=128 时 0.268 毫秒召回 0.875。延迟随 pq_M 近似线性（16 到 128 涨 5.2 倍），但 pq_M 取到 128 时延迟是 HNSW-Flat 的三倍，性价比不如 HNSW_SQ。两次运行绝对值略有出入（0.053/0.092/0.174/0.277），比值与单调性一致。这四档现已固化进 `scripts/verify_latency_graph_quant_win.py`，落盘 `win_graph_quant_d128.json`。

IVF_RaBitQ（Windows，nprobe=8）。多比特 nb_bits 扫描：1 位时 0.109 毫秒召回 0.401，2 位时 0.142 毫秒召回 0.615，4 位时 0.142 毫秒召回 0.850，8 位时 0.162 毫秒召回 0.988。延迟跨度主要是噪声（两次运行同档差 25%），召回单调上升，这是 RaBitQ 与其他量化索引最本质的差别。查询位数 qb 扫描（nb_bits=1）：qb=0 用原始 fp32 时 3.356 毫秒，qb=1 时 0.083 毫秒召回 0.181，qb=4 时 0.110 毫秒召回 0.401，qb=8 时 0.150 毫秒召回 0.402。

第 4 位误传 metric=2（METRIC_L1）时进程 abort，退出码 3221226505，断言信息为 `(metric_type == MetricType::METRIC_L2 || metric_type == MetricType::METRIC_INNER_PRODUCT)`，见第六节。

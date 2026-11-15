# 全双工实时多模态交互：MiniCPM-o 4.5

[English](README.en.md)

## 背景与边界

2026 年 11 月，Datawhale 公众号文章[《李沐：我带着新模型回来了！》](https://mp.weixin.qq.com/s/UM2fEPDewjzz-CFuXCGicw)介绍了 BosonAI 李沐团队的全双工实时多模态模型 Higgs Realtime Duplex：45B MoE 前端主模型，把时间切成 160 毫秒窗口，边听边说，思考 token 与工具调用也拆进时间窗，支持异步工具调用与前台/后台模型协同。文章同时说明：该 Duplex 模型的权重未公开，官方仅开放候补名单；已开放 API 的 Higgs Realtime 是另一个端到端模型/服务，与 Duplex 不是同一模型。

本实验不访问候补名单、私有服务或付费 API，也**不声称复现 Higgs**。这里使用 Apache-2.0 许可的 [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5)（9B 端到端音视频模型，官方 [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 提供实时全双工演示与双工接口）作为开源近似，在本设备上验证同类链路的实际表现。MiniCPM-o 4.5 和 Higgs Realtime Duplex 都支持连续音视频输入及并行语音输出，但架构、规模、时间片和后台工具能力不同；结果不可互推。本实验新增了一次客户端按 160 毫秒音频块发送的探针，只验证输入节奏和双工链路；不能据此断言模型内部也按 160 毫秒调度。

## 文章关键能力与本实验范围

下表对应 [Datawhale 文章](https://mp.weixin.qq.com/s/UM2fEPDewjzz-CFuXCGicw)对 Higgs Realtime Duplex 的关键能力描述。「已验证」仅指固定录制材料上的链路观察；「未验证」项均未测试，不能据本实验数据推断 Higgs 的表现，也不构成本实验对相应能力的支持。

| 文章关键能力 | 本实验对应情况 |
|---|---|
| 端到端架构：输入侧保留语音/视觉编码器，输出侧保留音频/视频解码器，取消中间转文字环节 | 使用对象本身即端到端音视频模型；本实验不验证架构实现，只验证链路行为 |
| 边听边说：按 160 毫秒窗口持续接收输入并输出语音，可输出静音、附和或在用户停顿时接话 | 部分验证：MiniCPM-o 4.5 的客户端以 160 毫秒节奏发送录制音视频期间，收到语音输出；不是对 Higgs 能力或模型内部时间片的验证 |
| 用户中途插话时模型调整并停止发声 | 插话探针记录到 `listen` 事件，但最后一段服务端语音 delta 在插话后 8.42 秒出现，超过预设的 2 秒停止门槛；真人插话未测 |
| 160 毫秒时间窗处理输入、输出与后台任务 | 仅验证客户端分块：探针按 160 毫秒发送音频；模型内部时间窗、思考 token 与后台任务调度未验证 |
| 端到端延迟约 0.7–0.9 秒、真人约 200 毫秒内接话 | 未验证：未测真人感受的端到端延迟；视频 run 单块 wall clock 中位数 868 ms 不是对话延迟 |
| 分段思考：思考 token 拆进时间窗 | 未验证 |
| 异步工具调用与分段回复 | 未验证：MiniCPM-o 4.5 不覆盖该架构能力 |
| 前端交互模型 + 后台大模型协同 | 未验证：同上 |
| 数字人（实时人脸视频）输出 | 未验证：本实验输出只有语音 |
| 零样本换声换脸 | 未验证 |
| Token 经济学：音视频流约 2.6K tokens/分钟，仅保留最近 5 分钟完整上下文 | 未验证：未测量 token 开销 |
| Big Bench Audio 98.9 等评测成绩 | 未验证：未复现任何 Higgs 评测 |

## 当前状态

### 本次验证（2026-11-15）

在 NVIDIA GB10 上使用官方 Demo 三服务栈，分别为视频和音频模式启动独立服务进程，避免同一 worker 连续处理第二个会话时返回 403。两次均使用模型仓库固定版本 `fb09628e98b1f55d8d5b863e0eccfcdd11bff499` 和录制材料 `assets/omni_duplex1.mp4`（35.3 秒）。模型使用 SDPA，加载约 7 秒；未安装 flash-attn。视频和音频 run 的可公开指标见 [`evidence/2026-11-15-gpu-duplex-results.json`](evidence/2026-11-15-gpu-duplex-results.json)。

| 指标 | 视频双工 | 纯音频双工 |
|---|---:|---:|
| 输入 | 100 个 1 秒音频块、100 帧 JPEG（源视频抽出 36 帧，末帧按节奏重复） | 70 个 1 秒音频块、0 帧 |
| 模型响应 | 126 个响应 delta 带视觉指标 | 0 个响应 delta 带视觉指标 |
| 输出语音 | 28 块、27.6 秒，RMS 0.126 | 25 块、23.92 秒，RMS 0.103 |
| 首段语音 | 6.74 秒 | 6.36 秒 |
| 输入输出重叠 | 93.7 秒 | 64.1 秒 |

两种模式都收到非静音的 24 kHz 语音，且在客户端继续发送输入时已有语音输出。视频模式返回了电梯楼层等画面描述。结果支持“录制流持续输入期间，模型可以输出语音并处理画面”这一有限结论；不是对 Higgs 的复现，也没有测得真人感受到的端到端延迟。

仍未验证真人麦克风和扬声器交互、真人听辨及真人中途插话。纯音频运行单独启动了新服务栈；当前公开脚本也按每次一种模式运行。

### 全新隔离环境 GPU 探针（2026-11-15）

随后在 NVIDIA GB10 上新建隔离 Python 环境，使用官方 Demo 固定提交 `47709a9210dfd71afa76c058e017fc8c4db5c8d2`，并从设备已有缓存复用模型权重。环境的基础包由现有可用环境复制，补装实时后端缺失的依赖后 `pip check` 通过；这不是从空环境在线安装全部依赖，也没有重新下载模型。实际版本和指标见 [`evidence/2026-11-15-clean-env-video.json`](evidence/2026-11-15-clean-env-video.json)。

| 指标 | 视频双工探针（30 秒上限） |
|---|---:|
| 输入 | 30 个 1 秒音频块、30 帧视频 |
| 含视觉指标的响应 | 38 个 delta |
| 输出语音 | 17 块、17 秒，24 kHz，RMS 0.117 |
| 首段文字 / 语音 | 7.08 秒 / 7.09 秒 |
| 输入输出重叠 | 23.63 秒 |

探针达到 30 秒会话上限并正常清理服务与端口。它验证的是固定录制材料、无头客户端下的 GPU 链路，不是真人交互或 Higgs 复现；本次也未测试真人插话或人感延迟。此前的 60 秒受控复跑使用另一套新隔离环境与 FFmpeg 6 运行库，结果仍单独保留，不与本次指标合并。

### 160 毫秒客户端分块探针（2026-11-15）

为对应文章提到的 160 毫秒时间片，本次在 NVIDIA GB10 上把探针客户端音频块从 1 秒改为 160 毫秒，并按音频块时间戳选取对应视频帧；视频源为 Demo 随附的 20 秒录制片段。模型权重复用设备缓存，使用独立 Python 环境与 FFmpeg 6.1.1 运行库；TorchCodec 加载与视频解码预检通过。Demo 源码缓存没有 Git 元数据，因此本次无法独立核对其精确提交。指标见 [`evidence/2026-11-15-160ms-video.json`](evidence/2026-11-15-160ms-video.json)。

| 指标 | 视频双工探针（20 秒会话上限） |
|---|---:|
| 客户端输入 | 112 个 160 毫秒音频块、112 次视频帧发送；源视频 20 帧按时间戳选取并重复使用 |
| 含视觉指标的响应 | 34 个 delta |
| 倾听事件 / 发言后回到倾听 | 30 / 11 次 |
| 首段文字 / 语音 | 13.14 / 13.15 秒 |
| 输出语音 | 2 块、2 秒，24 kHz，RMS 0.0479 |
| 输入输出重叠 | 7.01 秒；输入仍在发送时有语音输出 |

运行在 20 秒上限结束，服务端口均已释放。这个结果说明客户端可以按 160 毫秒发送录制输入，且 MiniCPM-o 4.5 在该链路中处理画面并于输入持续期间输出语音；它不证明模型内部采用同样时间窗。首次语音约 13.15 秒、语音块数较少，不能解释为低延迟真人交互。真人插话、麦克风/扬声器和 Higgs 架构均未验证。

### 录制语音插话探针（新增验证项）

文章强调的不只是边听边说，还包括用户插话时模型能否停止后续发声并转回倾听。现有连续录制探针不能验证这个行为。本探针在已有 MiniCPM-o-Demo 服务中先播放一段录制输入，等收到第一段模型语音后，再把第二段录制语音以 160 毫秒分块送入；`force_listen` 保持关闭，并持续发送画面帧。它测量 Demo 协议事件，不接麦克风、不播放到扬声器，也不能代表真人听到的停止延迟。

本次目的：检查固定录制语音在模型发言期间到达时，MiniCPM-o 4.5 是否记录到倾听状态，以及语音输出 delta 在插话后持续多久。成功判据预先定为：同一会话中初始输入和插话输入均已发送，插话前已有模型语音；插话后出现至少一个 `listen` 事件，且最后一个语音输出 delta 不晚于插话输入开始后 2 秒。缺少任一信号即记为未通过或证据不足，不通过调整提示词或增加重跑改变判定。单次运行上限 45 秒，插话后观察 8 秒；只运行一次，依赖、模型加载或会话建立失败即停止。

在已按前文启动 Demo 栈的环境中，选择两段带音轨和画面的录制视频。可使用模型仓库中的 `assets/omni_duplex1.mp4` 作为初始输入、官方 Demo 的 `assets/samples/compile.mp4` 作为插话输入：

```bash
XL_OUT_DIR=/path/to/out/barge-in \
PROBE_KIND=barge-in PROBE_MODE=video PROBE_MAX_SESSION_S=45 \
INITIAL_VIDEO=/path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
INTERRUPT_VIDEO=/path/to/MiniCPM-o-Demo/assets/samples/compile.mp4 \
PROBE_CHUNK_MS=160 PROBE_INITIAL_SECONDS=4 PROBE_INTERRUPT_SECONDS=4 \
PROBE_OBSERVE_AFTER_S=8 \
DEMO_DIR=/path/to/MiniCPM-o-Demo MODEL_DIR=/path/to/MiniCPM-o-4_5 \
PYTHON=/path/to/demo-venv/bin/python \
VIDEO=/path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
FFMPEG_LIB_DIR=/path/to/ffmpeg-libs/aarch64-linux-gnu \
  bash scripts/run_demo_verify.sh
```

脚本写出 `probe_barge_in_metrics.json`，包含输入阶段、首段模型语音时间、插话起点、插话后的语音 delta 数及倾听事件时间。delta 是服务端发出的语音片段事件，不代表扬声器播放完成；即使达到判据，也只说明录制素材下的协议链路出现停止/倾听信号，不能据此声称完成真人插话测试或复现 Higgs。

### 首次 GPU 结果（2026-11-15）

在 NVIDIA GB10 上使用新建隔离 Python 环境完成一次探针。空环境依赖检查、`pip check`、TorchCodec/FFmpeg 解码预检和 CUDA 检查均通过（Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0）；安装使用设备已有缓存，冷缓存安装未验证。会话发送了 25 个初始块和 25 个插话块，客户端分块为 160 毫秒；插话前收到一段模型语音，插话后也记录到 `listen` 事件。但最后一段服务端语音 delta 在插话开始后 8.42 秒才出现，超过预先设定的 2 秒门槛，因此**本次未通过插话停止判据**。会话与依赖流程正常结束，不等于插话行为通过；没有为改变结果而重跑。

语音 delta 只表示服务端发出音频数据，不能证明这些数据何时播放给听者，也不能直接换算成人的感知延迟。结果只适用于这次固定录制材料、MiniCPM-o 4.5 与 Demo 协议；不代表真人实时插话表现，也不构成 Higgs Realtime Duplex 的复现。脱敏指标见 [`evidence/2026-11-15-barge-in-probe.json`](evidence/2026-11-15-barge-in-probe.json)。

### 后续复跑检查（2026-11-15）

当时使用当前仓库脚本重新启动服务，模型加载成功，但探针未建立会话或发送输入。后端因找不到 FFmpeg 共享库而关闭会话；脚本随后停止服务并确认端口释放。这次失败没有产生新的模型能力数据。之后定位并修复了共享库准备问题，步骤见下文。

追加检查发现系统没有 FFmpeg 共享库。最初误判 OpenCV wheel 附带的 FFmpeg 5 `libavutil.so.57` 和 FFmpeg 6 的 `libavutil.so.58` 都不满足 `.so.60` 要求，因此当时没有启动 GPU 探针。后续确认 torchcodec 0.15 自带 FFmpeg 4–8 对应后端，FFmpeg 6.1.1 的 `.so.58` 可以加载 core6；真正缺的是若干传递依赖及 blas/lapack 子目录。下方复跑记录给出修复和实测结果。

### 受控复跑：新隔离环境修复 FFmpeg 依赖（2026-11-15）

上一次复跑被 torchcodec 的 FFmpeg 共享库挡住后，按受控复跑门槛在执行设备上准备新环境：不改旧隔离环境、不写系统目录，为本次运行新建隔离环境，并把 FFmpeg 6.1.1 运行库及传递依赖解包到实验专用目录下的独立前缀。预检实际通过：当前 Python 导入了 torchcodec 并加载其 FFmpeg 后端，AudioDecoder 能打开录制视频，GPU 可用，服务依赖导入完整。随后只运行一次视频模式探针，会话上限 60 秒，指标见 [`evidence/2026-11-15-rerun-newenv-video.json`](evidence/2026-11-15-rerun-newenv-video.json)：

| 指标 | 本次受控复跑（video，60 秒上限） |
|---|---:|
| 会话 | 成功建立，达到 60 秒上限结束 |
| 输入 | 59 个 1 秒音频块、59 帧 JPEG |
| 模型响应 | 80 个响应 delta 带视觉指标 |
| 输出语音 | 29 块、28.72 秒，RMS 0.135 |
| 首段语音 | 7.25 秒 |
| 输入输出重叠 | 52.76 秒 |
| 发言转倾听 | 19 次 |

本次运行使用新隔离环境与新准备的 FFmpeg 前缀，与前面两次成功运行不是同一环境；它证明该链路可以在执行设备上从已缓存软件包重新准备并再次得到非静音语音，不推翻也不替代此前的视频/音频 run。纯音频模式本次未再运行。

### 空 venv 实时环境安装自检（2026-11-15）

为把「从空环境安装实时 Python 依赖」变成可复用、可复核的一步，新增脚本 [`scripts/validate_realtime_environment.sh`](scripts/validate_realtime_environment.sh)：创建空 venv（默认 `python3.12`，venv 默认放 `.work/venv-realtime`，不进入 Git；可用 `PYTHON_BIN`、`VENV_DIR` 覆盖），按固定顺序安装 `pip==24.0`、`torch==2.11.0` 与 `torchaudio==2.11.0`、`torchcodec==0.15.0`（PyTorch cu130 index）和 `requirements-realtime.txt`，随后依次执行 `pip check`、import 检查（av、torch、torchaudio、torchcodec、transformers、fastapi、websockets、stepaudio2_minicpmo；发行包模块名不同时如实记录实际 import 的模块）、CUDA 可用性检查，最后用 Python 标准库生成 1 秒 WAV 并用 TorchCodec `AudioDecoder` 解码、要求采样数大于 0，以验证 FFmpeg backend。任何一步失败立即停止，不改版本重试；脚本产出可机器读的 `summary.json`（有 `XL_OUT_DIR` 时写其中，否则写 venv 内的 `validation-summary.json`），只含公开信息：系统架构/OS、Python 与包版本、pip check、CUDA、解码结果与失败阶段/错误。

2026-11-15 的第一次远端运行在安装 torch 时失败：当时的代理设置无法连接 PyTorch cu130 软件源。失败记录见 [`evidence/2026-11-15-proxy-install-failure.json`](evidence/2026-11-15-proxy-install-failure.json)。

第二次在新建空 venv 中运行，只为该次运行清除了代理变量。Python 包安装和 `pip check` 均通过；随后 TorchCodec 因找不到 FFmpeg 共享库而无法导入，错误包含缺少 `libavutil.so.56`。CUDA 和 WAV 解码检查未到达。结果见 [`evidence/2026-11-15-clean-install-env.json`](evidence/2026-11-15-clean-install-env.json)。

第四次空 venv 运行使用无代理连接安装 PyTorch，Python 依赖安装和 `pip check` 均通过；脚本也找到 PyAV wheel 的 `av.libs` 目录。但 TorchCodec 仍无法加载兼容的 FFmpeg 后端，CUDA 检查和 WAV 解码检查没有执行。记录见[空环境 FFmpeg ABI 检查](evidence/2026-11-15-empty-venv-ffmpeg-abi.json)。因此，空环境中的 TorchCodec 动态库加载、CUDA 检查与解码自检仍未闭环；先前独立环境中的 GPU 流式运行结果不受影响。

此后又进行了两次远端空 venv 自检，均未改变上述状态。第一次仍在安装 torch 时因设备侧代理拒绝连接 PyTorch cu130 软件源而失败，未安装 PyTorch wheel，记录见 [`evidence/2026-11-15-proxy-blocked-rerun.json`](evidence/2026-11-15-proxy-blocked-rerun.json)。第二次在新建 venv 中安装依赖并通过 `pip check`；wheel 文件来自设备已有的软件包缓存，因此这不是冷缓存下载验证。脚本定位到 PyAV wheel 自带的 FFmpeg 运行库并加入 `LD_LIBRARY_PATH`，但 TorchCodec 仍无法从该目录加载后端（加载器报告缺少 `libavutil.so.56`），CUDA 与解码检查未到达，记录见 [`evidence/2026-11-15-pyav-bundled-ffmpeg-load-failure.json`](evidence/2026-11-15-pyav-bundled-ffmpeg-load-failure.json)。

### FFmpeg ABI 有界修复运行（2026-11-15）

针对空 venv 自检反复卡在 TorchCodec 加载 FFmpeg 后端的问题，本次做一次有界修复，并在同一次远端运行内完成环境校验与探针。根因：torchcodec 0.15 按 FFmpeg 主版本各带一个后端（`libtorchcodec_core4`..`core8`），分别按 soname 加载 `libavutil.so.56`..`libavutil.so.60`；PyAV wheel 内置运行库目录 `av.libs` 里只有哈希改名的文件（形如 `libavutil-<hash>.so.60.8.100`），动态链接器永远不会从中解析出这些 soname，因此把 `LD_LIBRARY_PATH` 指向 PyAV 库目录从一开始就不可能满足任何 core 后端。修复：把 FFmpeg 6.1.1 的 Ubuntu arm64 运行库及传递依赖解包到本次运行专用的私有前缀，`libavutil.so.58` 满足 core6 后端。

新增两个脚本：[`scripts/prepare_ffmpeg_prefix.sh`](scripts/prepare_ffmpeg_prefix.sh)（解包 `.deb`、校验所需 soname 与 `blas/`、`lapack/` 子目录、以 torchcodec 导入加 `AudioDecoder` 打开录制视频作门禁）和 [`scripts/repair_ffmpeg_and_verify.sh`](scripts/repair_ffmpeg_and_verify.sh)（按「前缀准备 → 空 venv 校验 → 单次 60 秒视频探针」执行；任一阶段失败即停止并写失败证据，不重试、不追加探针）。设备侧路径（Demo 源码、模型快照、Demo 环境、`.deb` 目录）由调用方经环境变量注入，脚本内不写死。

本次远端运行三阶段全部通过（`.deb` 与 wheel 来自设备已有缓存；pip 索引元数据缓存过期后的一次索引获取，按运行范围清除了代理变量，与 2026-11-15 早前一次运行的做法相同）：

| 阶段 | 结果 |
|---|---|
| FFmpeg 前缀 | 80 个 `.deb` 解包；所需 soname 与 `blas/`、`lapack/` 子目录齐备；torchcodec 导入与录制视频解码门禁通过 |
| 空 venv 校验 | 安装与 `pip check` 通过；av、torch、torchaudio、torchcodec、transformers、fastapi、websockets、stepaudio2 全部导入；CUDA 可用（GB10，CUDA 13.0）；1 秒 WAV 解码通过 |
| 视频 GPU 探针 | 会话建立，达到 60 秒上限，服务与端口正常清理 |

探针指标（video，60 秒上限，模型版本 `fb09628e98b1f55d8d5b863e0eccfcdd11bff499`，素材 `assets/omni_duplex1.mp4`）：

| 指标 | 本次修复运行 |
|---|---:|
| 输入 | 59 个 1 秒音频块、59 帧 JPEG |
| 模型响应 | 81 个响应 delta 带视觉指标 |
| 输出语音 | 23 块、23.2 秒，24 kHz，RMS 0.0988 |
| 首段文字 / 语音 | 7.48 秒 / 7.49 秒 |
| 输入输出重叠 | 52.75 秒 |
| 倾听事件 / 发言后回到倾听 | 35 / 31 次 |

模型在输入持续期间输出语音并描述电梯画面（楼层从 21 楼到 24 楼）。完整记录见 [`evidence/2026-11-15-ffmpeg-abi-repair.json`](evidence/2026-11-15-ffmpeg-abi-repair.json)。这闭合了「空 venv + TorchCodec 可加载 FFmpeg ABI」的校验缺口，但它仍只使用设备已有缓存：冷缓存下载、在线获取 FFmpeg 包、其他平台与其他 FFmpeg 主版本均未验证；真人麦克风/扬声器、真人插话、人感延迟与 Higgs 复现同样未验证。本次运行不替代此前的视频、纯音频、30 秒与 160 毫秒结果。

### 较早的探索记录（非上表结果）

本目录中更早的 JSON 记录属于探索运行：离线按块运行官方 `as_duplex()` 接口处理录制视频，一次 35 块（26 块倾听、9 块发言），另一次 35 块（24 块倾听、11 块发言）并保存 34.2 秒 24 kHz 音频、经独立 Whisper 模型部分核对；另有 4 块冒烟运行（均为倾听响应），以及一次 4 块官方端到端测试（均为倾听响应、无音频输出）。03:29 的实时探针记录还误在“audio”模式发送了视频帧，不能作为纯音频数据。以上都不替代当前视频和音频 run 的指标。

## 环境准备

官方模型卡的离线示例要求 Linux、NVIDIA GPU、Python 3.10、`transformers==4.51.0`、`torch>=2.3.0,<=2.8.0`、匹配的 torchaudio、`minicpmo-utils[all]>=1.0.5,<=1.0.6` 和 FFmpeg；官方 Web 演示建议 GPU 显存大于 28 GB。模型权重约 18.7 GB，另需语音解码文件、Demo 源码与输出空间。

依赖覆盖判断（按脚本实际 import 与官方 Demo 固定 commit 的 `requirements.txt` 核对，2026-11-15）：离线运行使用本目录的 `requirements.txt`；实时服务与探针使用新增的 [`requirements-realtime.txt`](requirements-realtime.txt)。两者都不包含硬件相关的 torch、torchaudio、torchcodec，需按 CUDA、系统架构和 PyTorch 官方安装渠道单独安装。实时依赖文件固定了本次验证环境中的 Python 包版本，不能替代适配目标设备的 PyTorch 轮子，也不能据此推定其他平台已验证。

以下实时环境步骤按本次 Linux ARM64 GPU 运行记录整理。PyTorch、torchaudio 和 torchcodec 必须先选择与设备 CUDA/架构匹配的构建。示例采用本次验证版本；PyTorch 2.11 超出模型卡记录的 `<=2.8.0` 范围，因此先在目标设备确认可用性再用于其他机器。

```bash
python3.12 -m venv .venv-realtime
. .venv-realtime/bin/activate
python -m pip install pip==24.0
# 下列 CUDA 13.0 构建对应本次验证设备；其他设备请按 PyTorch 官方安装选择器调整。
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install torchcodec==0.15.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-realtime.txt
python -m pip check
python -c "import torch, torchaudio, torchcodec, av, numpy, PIL, websockets; print(torch.__version__, torch.cuda.is_available())"
python -c "from torchcodec.decoders import AudioDecoder; print('torchcodec FFmpeg backend loaded')"
```

上述步骤已固化为 [`scripts/validate_realtime_environment.sh`](scripts/validate_realtime_environment.sh)（按 Linux ARM64 + CUDA 13.0 整理；其他平台需按 PyTorch 官方安装选择器选择匹配轮子）。系统未提供 FFmpeg 共享库时，先按下方 FFmpeg 步骤准备独立前缀，再用 `FFMPEG_LIB_DIR` 指向其 `usr/lib/aarch64-linux-gnu` 目录；脚本会把该目录和可选的 `blas/`、`lapack/` 子目录加入 `LD_LIBRARY_PATH`。

```bash
PYTHON_BIN=python3.12 bash scripts/validate_realtime_environment.sh
# 覆盖 venv 位置（默认 <实验目录>/.work/venv-realtime，Git 忽略）：
VENV_DIR=/path/to/venv bash scripts/validate_realtime_environment.sh
# 如系统没有 FFmpeg 共享库：
FFMPEG_LIB_DIR=/path/to/ffmpeg-root/usr/lib/aarch64-linux-gnu bash scripts/validate_realtime_environment.sh
```

脚本产出 `summary.json`（只含公开信息，不含主机名、用户名、地址与私有路径）。本次实跑证明空环境依赖安装与 `pip check` 可通过；未显式提供 `FFMPEG_LIB_DIR` 时，脚本会自动使用 PyAV wheel 自带的运行库，但 TorchCodec 仍无法从中加载兼容后端，后续导入和解码检查失败。其他设备需提供 TorchCodec 实际可加载的 FFmpeg 共享库（例如按上述步骤准备的独立前缀），或显式传入 `FFMPEG_LIB_DIR`。2026-11-15 的修复运行闭合了这一缺口：显式传入指向 FFmpeg 6.1.1 前缀的 `FFMPEG_LIB_DIR` 后，同一脚本在空 venv 中完成安装、`pip check`、导入、CUDA 与解码检查，全部通过（见上一节）。

为避免跟随默认分支变化，可固定官方 Demo 的公开版本。以下 commit 于 2026-11-15 从官方 GitHub API 核实，其 `requirements.txt` 内容也于同日读取并用于上面的依赖覆盖判断；它适合作为当前安装示例，但没有证据证明它与历史成功 run 的私有源码归档完全一致：

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git "$DEMO_DIR"
git -C "$DEMO_DIR" checkout 47709a9210dfd71afa76c058e017fc8c4db5c8d2
python -m pip install -r "$DEMO_DIR/requirements.txt"
python -m pip check
# 探针自身 import 检查：Demo 依赖未显式包含 PyAV 与 Pillow，缺失时在同一环境补装
python -c "import av, numpy, PIL, websockets; print('probe imports OK')"
# 例如：python -m pip install av pillow
```

不要混装两个环境的依赖：历史记录显示 `minicpmo-utils==1.0.6` 与 Demo 的 `librosa>=0.10.2` 存在约束冲突，把本实验 `requirements.txt` 装进 Demo 环境可能无法通过依赖解析；实时链路以 Demo 依赖为准，缺什么补什么。

若 `torchcodec` 无法加载 FFmpeg 共享库，先安装与当前 torchcodec 兼容的 FFmpeg 运行库（包括 `libavutil`、`libavcodec`、`libavformat`、`libswscale`、`libswresample` 及它们的系统依赖）。安装后，在启动模型服务前，用当前虚拟环境确认 torchcodec 能实际加载：

```bash
ffmpeg -version
# 确认当前 Python 环境的 torchcodec 能加载其 FFmpeg 后端
python -c "from torchcodec.decoders import AudioDecoder; print('torchcodec FFmpeg backend loaded')"
```

`ffmpeg -version` 只能确认命令行程序存在，不能证明 Python 扩展依赖的共享库已正确加载；以上 Python 导入检查才是启动本实验的门槛。若使用非系统目录的运行库，应先将完整依赖链加入 `LD_LIBRARY_PATH`，然后通过同一导入检查。不可用不同 FFmpeg 主版本的库改名或软链接替代。TorchCodec 官方兼容表说明 0.15 与 PyTorch 2.11 兼容，但这不替代 FFmpeg 动态库加载检查（[TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)）。

### FFmpeg 共享库：本次已验证的准备步骤（2026-11-15 受控复跑）

执行设备（Ubuntu 24.04 arm64）无系统 FFmpeg，torchcodec 0.15 需要 FFmpeg 4-8 之一的共享库。2026-11-15 受控复跑实际使用并验证的步骤，全部在实验专用目录内完成，不写系统目录、不改旧环境：

1. 取得 FFmpeg 6.1.1 运行库及传递依赖的 Ubuntu arm64 软件包：`libavutil58`、`libavcodec60`、`libavformat60`、`libavdevice60`、`libavfilter9`、`libswscale7`、`libswresample4`、`libpostproc57`，以及 `libzvbi0t64`、`libcjson1`、`libmbedcrypto7t64`、`libblas3`、`liblapack3`、`libva2`、`libva-drm2`、`libplacebo338`、`libdav1d7`、`libass9`、`libx264-164`、`libx265-199`、`libsdl2-2.0-0` 等依赖。本次从已有软件包缓存取得 `.deb` 并成功解包；在线下载步骤未在本次运行中验证。
2. 用 `dpkg-deb -x` 把上述 `.deb` 解包到独立前缀（例如 `ffmpeg-root/`）。
3. `LD_LIBRARY_PATH` 指向该前缀的 `usr/lib/aarch64-linux-gnu`，并追加同级的 `blas/` 与 `lapack/` 子目录：blas/lapack 的真实库位于子目录，顶层符号链接由安装脚本创建，`dpkg-deb -x` 不会生成。
4. 门禁：先运行上面的 Python 导入检查（本次还追加了用 `AudioDecoder` 打开录制视频的冒烟），通过后才启动模型服务。

要点：torchcodec 0.15 为 FFmpeg 4-8 各带一个后端，FFmpeg 6.1.1（`libavutil.so.58`）即可满足其 core6 后端；此前README 记录中“需要 `.so.60`”的判断有误。真正容易漏的是 `libzvbi`、`libcjson`、`libmbedcrypto` 三个传递依赖和 blas/lapack 子目录。隔离环境方面，本次新建环境由既有可用环境复制而来（uv 离线安装因缓存缺少 `absl-py` 未完成），旧环境未改动；从包缓存全新安装整套 Python 依赖仍未验证。

此前的探索环境（对应上表视频/音频 run）为 Python 3.12.3、torch 2.11.0+cu130、torchaudio 2.11.0+cu130、torchcodec 0.15.0、transformers 4.51.0、accelerate 1.12.0、fastapi 0.162.7、uvicorn 0.54.0、websockets 17.4、PyAV 16.0.1；该旧环境 `pip check` 有未解决项。本次新增探针使用单独隔离环境，版本与状态见证据 JSON，不能把旧环境的问题套到新环境，也不能把新环境说成空白安装验证。

设备无系统 FFmpeg 时，Demo 后端的 torchcodec 需要 ABI 兼容的 FFmpeg 共享库。2026-11-15 的修复运行确认了 PyAV 内置库为何不可用：PyAV 16.0.1 wheel 的 `av.libs` 目录里只有哈希改名的库文件（形如 `libavutil-<hash>.so.60.8.100`），而 torchcodec 的 core4..core8 后端按 soname 加载 `libavutil.so.56`..`libavutil.so.60`，动态链接器不会把二者匹配起来；这与空 venv 运行中「从 av.libs 加载失败并报告缺少 `libavutil.so.56`」的现象一致。可用的修复是按上一节步骤把 FFmpeg 6.1.1 运行库及传递依赖解包成独立前缀，再用 `FFMPEG_LIB_DIR` 指向其 `usr/lib/aarch64-linux-gnu`（含 `blas/`、`lapack/` 子目录）。据此，本节早前「30 秒探针环境使用 PyAV 内附 FFmpeg 8 运行库」的归因与上述机制不符，该次运行的实际库来源已无法从现存记录中核实，其指标本身不受影响；旧的 60 秒复跑则确实使用另行解包的 FFmpeg 6.1.1 运行库。探针用 PyAV 抽取音轨与帧，不需要 `ffmpeg` 命令行工具。2026-11-15 修复运行已从设备缓存完成「空 venv 安装 FFmpeg 运行库、加载 TorchCodec、CUDA 与解码检查」的闭环；冷缓存下载与在线获取 FFmpeg 包仍未验证。

上述准备与校验已固化为脚本：

```bash
# 仅准备前缀并过 torchcodec 门禁（.deb 来源目录与门禁解释器必填）
FFMPEG_DEB_DIRS="/path/to/deb-dir1 /path/to/deb-dir2" \
GATE_PYTHON=/path/to/venv/bin/python \
GATE_VIDEO=/path/to/recorded.mp4 \
  bash scripts/prepare_ffmpeg_prefix.sh
# 一键完成「前缀准备 → 空 venv 校验 → 单次 60 秒视频探针」
# 设备侧路径由调用方注入；任一阶段失败即停止并写失败证据，不重试
XL_OUT_DIR=/path/to/out XL_DATA_DIR=/path/to/data XL_RUN_ID=demo-run \
DEMO_DIR=/path/to/MiniCPM-o-Demo MODEL_DIR=/path/to/MiniCPM-o-4_5 \
DEMO_PYTHON=/path/to/demo-venv/bin/python VIDEO=/path/to/recorded.mp4 \
FFMPEG_DEB_DIRS="/path/to/deb-dir1 /path/to/deb-dir2" \
  bash scripts/repair_ffmpeg_and_verify.sh
```

注意：`repair_ffmpeg_and_verify.sh` 固定只跑视频模式、单次探针，会话上限不得超过 60 秒。空 venv 阶段只校验依赖安装、`pip check`、CUDA、TorchCodec 导入与 WAV 解码；MiniCPM-o Demo 整栈的 GPU 探针则在 `DEMO_PYTHON` 指定的现有 Demo 环境中运行，并使用同一 FFmpeg 前缀。因此，本次未在空 venv 中端到端运行 Demo 服务。

## 模型、数据与运行

模型版本 `fb09628e98b1f55d8d5b863e0eccfcdd11bff499`；录制视频与参考音频使用该版本模型仓库的 `assets/omni_duplex1.mp4` 和 `assets/HT_ref_audio.wav`，使用时遵循原仓库相关许可。下载需包含模型权重以及 `assets/token2wav/` 语音解码文件。模型和数据均不应提交到本仓库。

离线按块运行（官方 `as_duplex()` 接口，非实时双工）：

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5 \
  --revision fb09628e98b1f55d8d5b863e0eccfcdd11bff499 \
  --local-dir "$MODEL_DIR"
python scripts/offline_duplex.py \
  --model-dir "$MODEL_DIR" \
  --video "$MODEL_DIR/assets/omni_duplex1.mp4" \
  --reference-audio "$MODEL_DIR/assets/HT_ref_audio.wav" \
  --output "$OUTPUT_DIR/chunks.json" --audio-output "$OUTPUT_DIR/response.wav" --max-chunks 35
```

实时双工链路验证（官方 Demo 三服务栈 + 流式探针）。先取得 [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 源码，然后：

本次受控复跑门槛：仅在 torchcodec 成功加载 ABI 匹配的 FFmpeg 库后启动模型；最多 1 次视频探针、最长 60 秒。成功标准是会话建立、至少发送 1 个音频块并收到非静音语音。环境或模型启动失败、或达到 60 秒上限即停止，不调参或追加运行。

```bash
DEMO_DIR=/path/to/MiniCPM-o-Demo   # 含 gateway.py、worker.py、py_backend/ 的源码树
MODEL_DIR=/path/to/MiniCPM-o-4_5   # 本地模型快照目录
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git "$DEMO_DIR"
python3 -m venv .venv-demo
. .venv-demo/bin/activate
# 先按 PyTorch 官方说明安装与 CUDA/设备架构匹配的 torch、torchaudio 和 torchcodec
python -m pip install -r "$DEMO_DIR/requirements.txt"
python -m pip check
# 探针自身 import 检查：缺失时在同一环境补装（例如 python -m pip install av pillow）
python -c "import av, numpy, PIL, websockets; print('probe imports OK')"

VIDEO=$MODEL_DIR/assets/omni_duplex1.mp4
FFMPEG_LIB_DIR=/path/to/ffmpeg-libs/aarch64-linux-gnu   # 设备无系统 FFmpeg 时；可含 blas/、lapack/ 子目录
XL_OUT_DIR=/path/to/out/video PROBE_MODE=video \
DEMO_DIR=$DEMO_DIR MODEL_DIR=$MODEL_DIR PYTHON=$(command -v python) VIDEO=$VIDEO \
FFMPEG_LIB_DIR=$FFMPEG_LIB_DIR \
  bash scripts/run_demo_verify.sh
# 160 毫秒客户端分块探针（用 Demo 自带的 20 秒录制视频）
VIDEO=$DEMO_DIR/assets/samples/compile.mp4
XL_OUT_DIR=/path/to/out/video-160ms PROBE_MODE=video \
PROBE_CHUNK_MS=160 PROBE_MAX_SESSION_S=20 \
DEMO_DIR=$DEMO_DIR MODEL_DIR=$MODEL_DIR PYTHON=$(command -v python) VIDEO=$VIDEO \
FFMPEG_LIB_DIR=$FFMPEG_LIB_DIR \
  bash scripts/run_demo_verify.sh
# 音频模式另启一套服务，XL_OUT_DIR 必须使用另一个新目录
XL_OUT_DIR=/path/to/out/audio PROBE_MODE=audio \
DEMO_DIR=$DEMO_DIR MODEL_DIR=$MODEL_DIR PYTHON=$(command -v python) VIDEO=$VIDEO \
FFMPEG_LIB_DIR=$FFMPEG_LIB_DIR \
  bash scripts/run_demo_verify.sh
```

每次调用脚本只运行一种模式，并启动后端（默认 22500 端口，`--model-path` 指向本地模型）、工作进程（22400，`--backend-server-url` 指向后端）与网关（8006，HTTP 模式）。模式可设 `PROBE_MODE=video` 或 `PROBE_MODE=audio`；视频模式每个音频块附带一帧，音频模式不发送视频帧。分别调用两次可以避免同一 worker 连续处理会话时返回 403。会话上限用 `PROBE_MAX_SESSION_S`（视频默认 100 秒）、`PROBE_AUDIO_MAX_SESSION_S`（音频默认 70 秒）控制；2026-11-15 受控复跑设 `PROBE_MAX_SESSION_S=60`。探针 `scripts/demo_realtime_probe.py` 适配自官方 `examples/realtime/video_probe.py`：WebSocket 协议、按真实节奏发送与指标口径不变，仅媒体抽取改用 PyAV（设备无 `ffmpeg` 命令行）。每次运行产出对应的 `probe_<mode>_metrics.json`、`summary.json`、`report.md`、服务日志和 WAV 音频。结束后脚本会停止本次启动的服务并核对端口释放。

单独运行探针（服务已就绪时）：

```bash
python scripts/demo_realtime_probe.py --url http://127.0.0.1:8006 --mode video \
  --video "$VIDEO" --max-session-s 100 \
  --output out/probe_video_metrics.json --audio-output out/probe_video_audio.wav
```

## 已知限制

- 全部结论基于固定录制材料与无头客户端；没有真人麦克风、扬声器、听辨或真实插话。
- 常规探针默认按 1 秒块发送；另有一次 160 毫秒客户端分块试跑，尚未验证真人交互或模型内部调度。视频 run 的单块 wall clock 中位数为 868 ms，它不是人感受到的端到端延迟。
- 文章还介绍了数字人视频输出、异步工具调用和前后台模型协同；本实验使用 MiniCPM-o 4.5，只验证录制音视频持续输入时模型能处理画面并输出语音，不覆盖这些架构能力，也不据此推断 Higgs 的延迟或成本。
- 插话探针虽在插话后收到 `listen` 事件，但最后一段服务端语音 delta 出现在插话后 8.42 秒，未达到 2 秒停止门槛；这不是通过的插话停止测试。录制输入与协议事件不等同于真人插话、听到停止或模型已停止播放。
- 依赖安装需要适配设备 CUDA 的 torch 栈和 TorchCodec 可加载的 FFmpeg 动态库；实时 Python 包固定在 [`requirements-realtime.txt`](requirements-realtime.txt)。2026-11-15 修复运行已闭合空 venv 缺口：显式 `FFMPEG_LIB_DIR` 指向解包的 FFmpeg 6.1.1 前缀时，安装、`pip check`、导入、CUDA 与解码全部通过；PyAV wheel 的 `av.libs` 只有哈希改名的库文件，动态链接器按 soname 无法匹配，不能作为修复路径。该运行仍只使用设备已有缓存：冷缓存下载、在线获取 FFmpeg 包与其他平台未验证。较早探索环境里的 `minicpmo-utils`、`librosa` 与 `decord` 问题不应套用到这套新环境。
- 较早的视频和音频运行使用设备上已有、`pip check` 有未解决项的隔离环境。后续 30 秒探针使用另一套隔离环境，版本和 `pip check` 通过状态见 [`evidence/2026-11-15-clean-env-video.json`](evidence/2026-11-15-clean-env-video.json)。torch 2.11 高于模型卡注明的 2.8.0 上限。
- 2026-11-15 受控复跑使用为该运行新建的隔离环境（由既有可用环境复制，uv 离线安装因缓存缺 `absl-py` 未完成；旧环境未改动）与按上文步骤准备的 FFmpeg 6.1.1 前缀，因此共享库依赖已能重新准备并验证；从零安装整套环境及在线获取 FFmpeg 软件包仍未验证，本次使用了已有的 `.deb` 缓存。同日修复运行同样只用设备缓存（pip 索引元数据缓存过期后的一次索引获取按运行范围清除了代理变量），并使用新的空 venv 与新前缀。
- 成功结果包括单台设备上的较早视频与纯音频运行、一次 60 秒受控复跑、一次独立 30 秒视频探针和一次 60 秒 FFmpeg ABI 修复运行（新空 venv + 新前缀）；没有真人麦克风/扬声器、真人听辨或真人插话测试。各次运行使用不同环境和会话上限，不能合并当作重复试验。

## 证据

| 文件 | 内容 |
|---|---|
| `evidence/2026-11-15-gpu-duplex-results.json` | 两次成功 GPU 流式运行的公开指标汇总，以及一次因 FFmpeg 共享库缺失而未进入探针的历史复跑 |
| `evidence/2026-11-15-rerun-newenv-video.json` | 2026-11-15 受控复跑：新隔离环境与 FFmpeg 6.1.1 前缀的准备、torchcodec 加载预检，以及一次 60 秒视频探针的指标 |
| `evidence/2026-11-15-ffmpeg-abi-repair.json` | 2026-11-15 有界修复运行：PyAV 内置库哈希改名导致 soname 不匹配的根因、FFmpeg 6.1.1 前缀准备与 torchcodec 门禁、空 venv 校验（版本/pip check/CUDA/解码）与一次 60 秒视频 GPU 探针指标 |
| `evidence/2026-11-15-clean-env-video.json` | 另一套新隔离环境中的 30 秒 GPU 视频探针、软件版本与准备边界 |
| `evidence/2026-11-15-proxy-install-failure.json` | 第一次空 venv 安装尝试：代理连接失败，未安装 torch |
| `evidence/2026-11-15-clean-install-env.json` | 一次空 venv 安装尝试：Python 依赖安装与 `pip check` 通过，TorchCodec 因 FFmpeg 共享库缺失而未能导入；CUDA 与解码检查未到达 |
| `evidence/2026-11-15-empty-venv-ffmpeg-abi.json` | 后续空 venv 检查：自动找到 PyAV 库目录，但 TorchCodec 未能加载兼容 FFmpeg 后端；CUDA 与解码检查未到达 |
| `evidence/2026-11-15-proxy-blocked-rerun.json` | 2026-11-15 后续空 venv 复跑：安装 torch 时代理拒绝连接 PyTorch cu130 软件源，未安装 PyTorch wheel，后续检查未到达 |
| `evidence/2026-11-15-pyav-bundled-ffmpeg-load-failure.json` | 2026-11-15 后续空 venv 复跑：依赖安装与 `pip check` 通过，脚本使用 PyAV wheel 自带 FFmpeg 运行库，但 TorchCodec 仍无法加载后端；CUDA 与解码检查未到达 |
| `evidence/2026-11-15-160ms-video.json` | 160 毫秒客户端音频分块视频探针：输入/输出重叠和视觉响应指标；不代表模型内部调度 |
| `evidence/2026-11-15-barge-in-probe.json` | 单次 GB10 插话探针：新隔离环境与 CUDA/解码预检通过；记录到倾听事件，但最后语音 delta 晚于预设门槛，结论为未通过 |
| `evidence/2026-11-15-auto-ffmpeg-detect-install-blocked.json` | 空 venv 自检在 PyTorch 安装阶段被代理阻断，尚未运行 FFmpeg 自动发现 |
| `evidence/2026-11-15-demo-verify-032901.json` | 较早的探索记录；其中 audio 模式也发送了视频帧，不用于上方纯音频结果 |
| `evidence/2026-11-15-recorded-smoke.json`、`evidence/2026-11-15-recorded-full.json`、`evidence/2026-11-15-recorded-audio.json` | 历史记录：离线按块运行（4 块冒烟、35 块全文、35 块含音频） |
| `evidence/2026-11-15-demo-recorded.json` | 历史记录：官方端到端测试 4 块录制材料，均为倾听响应 |

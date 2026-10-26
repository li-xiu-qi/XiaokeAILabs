# 公开证据文件说明

本目录保存已经完成的 NInfer 与 SGLang 实验结果。公开时只移除主机标识和内部路径，不把路径改为虚构来源；数值、完成原因、草稿统计与逐请求 timings 均未修改。

## 环境与模型

- `environment.json`：GB10、CUDA、SGLang、Ninja、fork、模型 revision 与逐文件校验结果，不含主机名。
- `ninfer-artifact-summary.json`：V3 版本、组件、对象数与 SHA256；模型路径统一写为 `model.ninfer`。
- `ninfer-runtime-summary.json`：构建、CTest、MTP/DFlash2/Vision/CUDA Graph 和 OpenAI 结果，以及截图复现汇总。

## 性能原始记录

- `bench-ninfer-w4a4-dflash2-draft7-xhigh.json`：截图配置的 NInfer 逐请求结果。
- `bench-nvfp4-dflash2.json`、`bench-fp8-dflash2.json`：SGLang eager 原始记录。
- `bench-nvfp4-dflash2-graph.json`、`bench-fp8-dflash2-graph.json`：decode graph 原始记录。
- `comparison-eager.md/json`、`comparison-graph.md/json`：由原始结果生成；eager 汇总含质量与冒烟，graph 汇总只含性能。

SGLang 的总吞吐包含 prefill 和排队；NInfer 还保留原生 predicted_per_second、draft_n 与 draft_n_accepted。各文件中的口径不能直接混为同一纯 decode 指标。

## 质量与 OpenAI 冒烟

- `quality-nvfp4-dflash2-summary.json`、`quality-fp8-dflash2-summary.json`：自动题、截断、空答与失败 case id。
- `code-functional-nvfp4-dflash2.json`、`code-functional-fp8-dflash2.json`：人工确认无 import/I/O 后执行的编程功能用例。
- `smoke-ninfer-server.json`：NInfer OpenAI 文本与图片真实响应，包含原生 timings。
- `public-script-checks.json`：公开脚本的 NInfer/SGLang 冒烟、停止与重启、固定 revision 下载校验和质量验证。

大型构建日志不整包公开，关键结果以结构化 JSON 和专题文档呈现。没有将跳过测试写成通过，也没有将 35B/125B 的未完成内容混入这些证据。

# Agent 能力部件：Function Calling、MCP 与记忆

一个 Agent 要能干活，需要三样东西：调用工具（function calling / tool use）、获取外部工具（MCP 协议）、记住说过什么（记忆管理）。这个目录把三样分开实测，从裸 API 调用到完整 ReAct 智能体逐步演化。

## 子目录与阅读顺序

| 目录 | 部件 | 内容 |
|---|---|---|
| `function_calling/` | 工具调用的最小形态 | Function Call 规范定义、DeepSeek V3 实测、本地 tool use 实现、OpenAI 兼容兜底客户端 |
| `test_mcp/` | 工具获取的协议层 | JSON-RPC 与 stdio 传输原理、MCP server/client 实战、ReAct 智能体 v2 到 v6 的完整演化 |
| `test_memory/` | 上下文管理 | token 计数（本地模型实测长文档成本）、本地模型对话配置、短期记忆机制 |

## function_calling：从 API 规范到自己实现

`test_function_call.ipynb` 走完整链路：先测 API 连通，再按传统 Function Call 格式定义函数规范（名称、参数、描述），让模型输出结构化调用意图，本地执行后把结果回传。`local_tool_use_implementation.ipynb` 是不依赖商业 API 的本地 tool use 实现。`fallback_openai_client.py` 是 OpenAI 兼容客户端的兜底封装，换模型服务方时复用。

## test_mcp：从 stdio 原理到 ReAct 智能体

分两层。`mcp_theory/` 是协议原理：JSON-RPC 消息格式、stdin/stdout 与 stderr 的传输演示，理解 MCP 为什么这么设计。`mcp_practice/` 是可运行的 server 和 client（含 README 与依赖）。

`react_agent*.py` 六个版本是同一条演化线，按版本号读就是一部「ReAct 智能体怎么变聪明」的简史：

| 版本 | 改进 |
|---|---|
| `react_agent.py` | 基础版：思考、行动、观察循环 |
| v2 | 基于 MCP 工具的反应式推理 |
| v3 到 v4 | 中间迭代 |
| v5 | 智能工具选择：按问题类型决定是否调工具 |
| v6 | 用 AI 判断替代关键词匹配，选工具本身也交给模型 |

v5 到 v6 的差别值得注意：关键词匹配规则写死就脆，换成模型判断后泛化变强，代价是多一次调用。`smart_agent.py` 与 `test_react.py` 是配套入口。

## test_memory：长文档的成本账

`calculate_tokens.py` 用 ModelScope 上的 Qwen3 tokenizer 实测一段 Markdown 的 token 数，回答的是「把整篇文档塞进上下文要花多少预算」这个在做 Agent 时的真实问题。`chat_with_local_model.py` 是本地模型（vLLM 一类）的调用配置，`short_memory.py` 是短期记忆的裁剪逻辑。

三部分合起来是给 Agent 开发者的从零件到整体的路径：先会调工具，再把工具接到 MCP，最后管好上下文。

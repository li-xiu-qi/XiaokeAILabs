# MCP 理论代码实现

这个目录包含了基于架构文章的 MCP 协议代码实现。

## 文件说明

### 核心实现文件

1. **`jsonrpc_handler.py`** - JSON-RPC 2.0 处理器
   - 实现 JSON-RPC 2.0 协议的消息创建、验证和处理
   - 包含错误处理和批处理支持
   - 提供完整的协议演示

2. **`simple_mcp_server.py`** - 简单 MCP 服务器
   - 基础的 MCP 服务器实现
   - 支持工具调用和提示系统
   - 演示基本的 Stdio 传输

3. **`stdio_mcp_client.py`** - Stdio 客户端
   - MCP 客户端实现，使用 Stdio 传输
   - 完整的初始化流程
   - 自动化测试和演示

4. **`advanced_mcp_server.py`** - 高级 MCP 服务器
   - 功能更完整的服务器实现
   - 支持资源管理、日志记录
   - 包含更多工具和提示示例

5. **`mcp_protocol_demo.py`** - 协议演示
   - 展示 MCP 协议的各个方面
   - 包含初始化流程、消息交换、错误处理
   - 传输方式对比说明

### 测试和运行

6. **`run_tests.py`** - 测试运行器
   - 自动化测试脚本
   - 检查文件完整性
   - 提供手动测试指导

## 快速开始

### 1. 运行协议演示
```bash
python mcp_protocol_demo.py
```

### 2. 测试 JSON-RPC 处理器
```bash
python jsonrpc_handler.py
```

### 3. 运行完整客户端-服务器测试
```bash
python stdio_mcp_client.py
```

### 4. 手动测试服务器

启动简单服务器：
```bash
python simple_mcp_server.py
```

启动高级服务器：
```bash
python advanced_mcp_server.py
```

然后在另一个终端中发送 JSON-RPC 消息进行测试。

### 5. 运行所有测试
```bash
python run_tests.py
```

## 示例 JSON-RPC 消息

### 初始化流程
```json
// 1. 初始化请求
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"TestClient","version":"1.0.0"}}}

// 2. 初始化完成通知
{"jsonrpc":"2.0","method":"notifications/initialized"}
```

### 工具操作
```json
// 获取工具列表
{"jsonrpc":"2.0","id":2,"method":"tools/list"}

// 调用工具
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"Hello MCP!"}}}
```

### 提示操作
```json
// 获取提示列表
{"jsonrpc":"2.0","id":4,"method":"prompts/list"}

// 获取特定提示
{"jsonrpc":"2.0","id":5,"method":"prompts/get","params":{"name":"greeting","arguments":{"name":"筱可"}}}
```

## 架构特点

### Stdio 传输
- 使用标准输入输出进行进程间通信
- 零网络开销，高效本地通信
- 自动进程生命周期管理

### JSON-RPC 2.0 协议
- 标准化的消息格式
- 支持请求-响应和通知模式
- 完整的错误处理机制

### MCP 功能
- **工具系统**: 可调用的功能接口
- **提示系统**: 模板化的对话启动器
- **资源管理**: 文件和数据访问
- **日志记录**: 调试和监控支持

## 学习路径

1. **理解基础** - 先运行 `mcp_protocol_demo.py` 了解协议概念
2. **学习 JSON-RPC** - 运行 `jsonrpc_handler.py` 掌握消息格式
3. **实践通信** - 运行 `stdio_mcp_client.py` 体验客户端-服务器交互
4. **深入功能** - 探索 `advanced_mcp_server.py` 的高级特性
5. **自定义开发** - 基于示例代码创建自己的 MCP 应用

## 技术参考

- **MCP 官方规范**: https://modelcontextprotocol.io/specification/2025-03-26
- **JSON-RPC 2.0**: https://www.jsonrpc.org/specification
- **架构文档**: architecture_article.md

---

这些代码示例完全基于 MCP 官方规范实现，可以作为学习和开发 MCP 应用的起点。

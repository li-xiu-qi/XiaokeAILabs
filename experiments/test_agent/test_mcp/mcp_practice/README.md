# FastMCP 工具服务器与客户端示例

本项目演示了如何基于 FastMCP 框架实现一个本地工具服务器（`mcp_server.py`）以及对应的 Python 客户端（`client_demo.py`），用于测试和调用服务器提供的工具、提示和资源。

## 目录结构

```
client_demo.py          # 客户端示例（STDIO 连接），测试服务器所有功能
client_http_demo.py     # 客户端示例（HTTP 连接），测试服务器所有功能
mcp_server.py           # MCP 工具服务器，提供工具/提示/资源
requirements.txt        # 依赖包列表
```

## 快速开始

### 1. 安装依赖

请先确保已安装 Python 3.7 及以上版本。

```bash
pip install -r requirements.txt
```

### 2. 启动 MCP 工具服务器

#### STDIO 模式（用于 client_demo.py）:
```bash
python mcp_server.py
```

#### HTTP 模式（用于 client_http_demo.py）:
服务器已配置为 HTTP 模式，运行后会在 `http://127.0.0.1:9000/mcp/` 启动
```bash
python mcp_server.py
```

### 3. 运行客户端测试

#### STDIO 客户端:
```bash
python client_demo.py
```

#### HTTP 客户端:
```bash
python client_http_demo.py
```

客户端会自动连接本地服务器，依次测试：

- 工具调用（如数学表达式计算）
- 提示生成（如主题解释、代码生成请求）
- 资源读取（如问候语、配置信息）

## 主要功能说明

### mcp_server.py

- 提供数学表达式计算工具（`calculate`）
- 提供主题解释和代码生成的 prompt
- 提供简单问候语和配置信息的资源

### client_demo.py

- 自动连接 MCP 服务器（STDIO 模式）
- 全面测试所有工具、提示和资源的调用与读取
- 控制台输出详细测试结果

### client_http_demo.py

- 通过 HTTP 连接 MCP 服务器
- 全面测试所有工具、提示和资源的调用与读取
- 控制台输出详细测试结果

## 依赖说明

- fastmcp：FastMCP 框架
- httpx：HTTP 客户端库
- mcp：MCP 协议库

请根据 `requirements.txt` 安装所有依赖。

## 适用场景

- AI 工具链开发
- Prompt 工具服务器演示
- Python 异步客户端开发

---

如有问题欢迎反馈！

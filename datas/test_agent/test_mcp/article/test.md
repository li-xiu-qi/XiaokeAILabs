好的，以下是翻译后的网页内容：

---

**根节点 - 模型上下文协议**

**根节点** 是模型上下文协议（MCP）中的一个概念，用于定义服务器可以操作的边界。它们为客户端提供了一种方式，用于告知服务器相关资源及其位置。

## 根节点是什么？

根节点是一个 URI（统一资源标识符），客户端建议服务器关注的焦点。当客户端连接到服务器时，它会声明服务器应该处理哪些根节点。虽然根节点主要用于文件系统路径，但它们可以是任何有效的 URI，包括 HTTP URL。

## 为什么使用根节点？

根节点有以下几个重要的作用：

1. **引导**：它们告知服务器相关资源和位置。
2. **明确性**：根节点清晰地表明哪些资源属于您的工作空间。
3. **组织性**：多个根节点可以让您同时处理不同的资源。

## 根节点的工作原理

当客户端支持根节点时，它会：

1. 在连接时声明 `roots` 功能。
2. 向服务器提供一组建议的根节点。
3. 如果支持，通知服务器根节点的变化。

虽然根节点是信息性的，并非严格强制执行，但服务器应该：

1. 尊重提供的根节点。
2. 使用根节点 URI 来定位和访问资源。
3. 优先在根节点边界内执行操作。

## 常见用例

根节点通常用于定义：

- 项目目录
- 仓库位置
- API 端点
- 配置位置
- 资源边界

## 最佳实践

在使用根节点时：

1. 只建议必要的资源。
2. 使用清晰、描述性的名称来命名根节点。
3. 监控根节点的可访问性。
4. 平稳处理根节点的变化。

## 示例

以下是一个典型的 MCP 客户端如何暴露根节点的示例：

这种配置建议服务器同时关注本地仓库和 API 端点，同时保持它们逻辑上的分离。

![img](https://mintlify.s3.us-west-1.amazonaws.com/mcp/logo/light.svg)![img](https://mintlify.s3.us-west-1.amazonaws.com/mcp/logo/dark.svg)

---
以下是网页内容提取的核心信息：

### Sampling（采样）
- **功能**：Model Context Protocol（MCP）的采样功能允许服务器通过客户端请求LLM（大型语言模型）完成任务，实现复杂的代理行为，同时保持安全性和隐私性。
- **工作流程**：
  1. 服务器向客户端发送`sampling/createMessage`请求。
  2. 客户端审查请求并可修改。
  3. 客户端从LLM采样。
  4. 客户端审查生成的内容。
  5. 客户端将结果返回给服务器。
- **设计特点**：采用“人在回路”的设计，确保用户对LLM看到和生成的内容保持控制。

### 消息格式
- **请求参数**：
  - **Messages**：包含会话历史的数组，每条消息有`role`（用户或助手）和`content`（文本或图像内容）。
  - **Model preferences**：服务器指定模型选择偏好，包括模型名称建议（`hints`）、成本优先级（`costPriority`）、速度优先级（`speedPriority`）和智能优先级（`intelligencePriority`）。
  - **System prompt**：可选字段，服务器可请求特定系统提示，客户端可修改或忽略。
  - **Context inclusion**：指定包含的MCP上下文，选项为“none”（无额外上下文）、“thisServer”（请求服务器的上下文）或“allServers”（所有连接的MCP服务器的上下文）。
  - **Sampling parameters**：调整LLM采样的参数，如`temperature`（控制随机性）、`maxTokens`（最大生成token数）、`stopSequences`（停止生成的序列）和`metadata`（提供商特定参数）。

### 响应格式
客户端返回完成结果。

### 最佳实践
- 提供清晰、结构良好的提示。
- 适当处理文本和图像内容。
- 设置合理的token限制。
- 通过`includeContext`包含相关上下文。
- 在使用响应前验证。
- 优雅地处理错误。
- 考虑对采样请求进行速率限制。
- 文档化预期的采样行为。
- 使用各种模型参数进行测试。
- 监控采样成本。

### 人类监督控制
- 对于提示：
  - 客户端应向用户展示提议的提示。
  - 用户应能够修改或拒绝提示。
  - 系统提示可以被过滤或修改。
  - 客户端控制上下文的包含。
- 对于生成内容：
  - 客户端应向用户展示生成的内容。
  - 用户应能够修改或拒绝生成内容。
  - 客户端可以过滤或修改生成内容。
  - 用户控制使用哪个模型。

### 安全注意事项
- 验证所有消息内容。
- 清理敏感信息。
- 实施适当的速率限制。
- 监控采样使用情况。
- 加密传输中的数据。
- 处理用户数据隐私。
- 审计采样请求。
- 控制成本暴露。
- 实施超时机制。
- 优雅地处理模型错误。

### 常见模式
- **代理工作流**：采样支持阅读和分析资源、基于上下文做出决策、生成结构化数据、处理多步任务和提供交互式协助等代理模式。
- **上下文管理**：最佳实践包括请求最小必要的上下文、清晰地结构化上下文、处理上下文大小限制、按需更新上下文和清理过时的上下文。
- **错误处理**：应捕获采样失败、处理超时错误、管理速率限制、验证响应、提供备用行为和适当记录错误。

### 限制
- 采样依赖于客户端的能力。
- 用户控制采样行为。
- 上下文大小有限制。
- 可能存在速率限制。
- 需要考虑成本。
- 模型可用性可能不同。
- 响应时间可能不同。
- 不支持所有内容类型。

---
以下是网页内容的提取信息：

### 网页标题
Tools - Model Context Protocol

### 网页内容概述
该网页介绍了 Model Context Protocol (MCP) 中的工具（Tools），它允许服务器向客户端暴露可执行功能，使 LLM（大型语言模型）能够与外部系统交互、执行计算以及在现实世界中采取行动。

### 主要内容

#### 工具概述
- **发现**：客户端可以通过 `tools/list` 接口列出可用工具。
- **调用**：工具通过 `tools/call` 接口被调用，服务器执行请求的操作并返回结果。
- **灵活性**：工具可以是从简单计算到复杂 API 交互的任何功能。

工具通过唯一名称标识，并可包含描述以指导使用。与资源不同，工具代表动态操作，可以修改状态或与外部系统交互。

#### 工具定义结构
文中给出了工具定义的结构示例，但未详细列出结构内容。

#### 工具类型示例
- **系统操作**：与本地系统交互的工具。
- **API 集成**：封装外部 API 的工具。
- **数据处理**：转换或分析数据的工具。

#### 最佳实践
- 提供清晰、描述性的名称和描述。
- 使用详细的 JSON Schema 定义参数。
- 在工具描述中包含示例，展示模型应如何使用它们。
- 实现适当的错误处理和验证。
- 对于长时间操作，使用进度报告。
- 保持工具操作的专注和原子性。
- 文档化预期的返回值结构。
- 实现适当的超时机制。
- 考虑对资源密集型操作进行速率限制。
- 记录工具使用情况，以便调试和监控。

#### 安全考虑
- **输入验证**：验证所有参数，防止命令注入等安全问题。
- **访问控制**：实现认证和授权检查，审计工具使用情况，限制请求速率，监控滥用行为。
- **错误处理**：不向客户端暴露内部错误，记录安全相关错误，适当处理超时，清理错误后的资源，验证返回值。

#### 动态工具发现
- 客户端可以随时列出可用工具。
- 服务器可以通过 `notifications/tools/list_changed` 通知客户端工具的变更。
- 工具可以在运行时添加或删除。
- 工具定义可以更新（但需谨慎操作）。

#### 错误处理
工具错误应在结果对象中报告，而不是作为 MCP 协议级别的错误。当工具遇到错误时：
1. 在结果中将 `isError` 设置为 `true`。
2. 在 `content` 数组中包含错误详细信息。

#### 工具注解
工具注解提供有关工具行为的额外元数据，帮助客户端了解如何呈现和管理工具。注解是提示，不应作为安全决策的依据。

- **目的**：提供与模型上下文无关的 UX 特定信息，帮助客户端分类和适当呈现工具，传达工具的潜在副作用，协助开发直观的工具审批界面。
- **可用注解**：
    - `title`：工具的人类可读标题，适用于 UI 显示。
    - `readOnlyHint`：如果为 `true`，表示工具不修改其环境，默认为 `false`。
    - `destructiveHint`：如果为 `true`，工具可能执行破坏性更新（仅当 `readOnlyHint` 为 `false` 时才有意义），默认为 `true`。
    - `idempotentHint`：如果为 `true`，用相同参数重复调用工具没有额外效果（仅当 `readOnlyHint` 为 `false` 时才有意义），默认为 `false`。
    - `openWorldHint`：如果为 `true`，工具可能与“开放世界”中的外部实体交互，默认为 `true`。
- **最佳实践**：
    - 准确描述副作用。
    - 使用描述性标题。
    - 正确标记幂等性。
    - 设置适当的开放/封闭世界提示。
    - 记住注解只是提示。

#### 测试策略
- **功能测试**：验证工具在有效输入下正确执行，并适当处理无效输入。
- **集成测试**：使用真实和模拟的依赖项测试工具与外部系统的交互。
- **安全测试**：验证认证、授权、输入清理和速率限制。
- **性能测试**：检查在负载下的行为、超时处理和资源清理。
- **错误处理**：确保工具通过 MCP 协议正确报告错误并清理资源。



-----


以下是网页内容的提取：

### 标题
**Prompts - Model Context Protocol**

### 内容概述
Prompts（提示）是Model Context Protocol（模型上下文协议，简称MCP）中的一种功能，允许服务器定义可复用的提示模板和工作流，客户端可以轻松地将其展示给用户和大型语言模型（LLMs）。它提供了一种强大的方式来标准化和共享常见的LLM交互。

### Prompt结构
每个提示都通过以下方式定义：
```json
{
  "name": "string",              // 提示的唯一标识符
  "description": "string",      // 人类可读的描述（可选）
  "arguments": [                // 参数列表（可选）
    {
      "name": "string",          // 参数标识符
      "description": "string",  // 参数描述（可选）
      "required": "boolean"     // 参数是否必需
    }
  ]
}
```

### 发现Prompts
客户端可以通过`prompts/list`端点发现可用的prompts：
```json
// 请求
{
  "method": "prompts/list"
}

// 响应
{
  "prompts": [
    {
      "name": "analyze-code",
      "description": "Analyze code for potential improvements",
      "arguments": [
        {
          "name": "language",
          "description": "Programming language",
          "required": true
        }
      ]
    }
  ]
}
```

### 使用Prompts
要使用一个prompt，客户端需要发送一个`prompts/get`请求：
```json
// 请求
{
  "method": "prompts/get",
  "params": {
    "name": "analyze-code",
    "arguments": {
      "language": "python"
    }
  }
}

// 响应
{
  "description": "Analyze Python code for potential improvements",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Please analyze the following Python code for potential improvements:\n\n```python\ndef calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total = total + num\n    return total\n\nresult = calculate_sum([1, 2, 3, 4, 5])\nprint(result)\n```"
      }
    }
  ]
}
```

### 动态Prompts
Prompts可以是动态的，包含以下内容：

#### 嵌入资源上下文
```json
{
  "name": "analyze-project",
  "description": "Analyze project logs and code",
  "arguments": [
    {
      "name": "timeframe",
      "description": "Time period to analyze logs",
      "required": true
    },
    {
      "name": "fileUri",
      "description": "URI of code file to review",
      "required": true
    }
  ]
}
```
在处理`prompts/get`请求时：
```json
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Analyze these system logs and the code file for any issues:"
      }
    },
    {
      "role": "user",
      "content": {
        "type": "resource",
        "resource": {
          "uri": "logs://recent?timeframe=1h",
          "text": "[2024-03-14 15:32:11] ERROR: Connection timeout in network.py:127\n[2024-03-14 15:32:15] WARN: Retrying connection (attempt 2/3)\n[2024-03-14 15:32:20] ERROR: Max retries exceeded",
          "mimeType": "text/plain"
        }
      }
    },
    {
      "role": "user",
      "content": {
        "type": "resource",
        "resource": {
          "uri": "file:///path/to/code.py",
          "text": "def connect_to_service(timeout=30):\n    retries = 3\n    for attempt in range(retries):\n        try:\n            return establish_connection(timeout)\n        except TimeoutError:\n            if attempt == retries - 1:\n                raise\n            time.sleep(5)\n\ndef establish_connection(timeout):\n    # Connection implementation\n    pass",
          "mimeType": "text/x-python"
        }
      }
    }
  ]
}
```

#### 多步骤工作流
```javascript
const debugWorkflow = {
  name: "debug-error",
  async getMessages(error) {
    return [
      {
        role: "user",
        content: {
          type: "text",
          text: `Here's an error I'm seeing: ${error}`
        }
      },
      {
        role: "assistant",
        content: {
          type: "text",
          text: "I'll help analyze this error. What have you tried so far?"
        }
      },
      {
        role: "user",
        content: {
          type: "text",
          text: "I've tried restarting the service, but the error persists."
        }
      }
    ];
  }
};
```

### 示例实现
以下是一个完整的在MCP服务器中实现prompts的示例：
```javascript
import { Server } from "@modelcontextprotocol/sdk/server";
import {
  ListPromptsRequestSchema,
  GetPromptRequestSchema
} from "@modelcontextprotocol/sdk/types";

const PROMPTS = {
  "git-commit": {
    name: "git-commit",
    description: "Generate a Git commit message",
    arguments: [
      {
        name: "changes",
        description: "Git diff or description of changes",
        required: true
      }
    ]
  },
  "explain-code": {
    name: "explain-code",
    description: "Explain how code works",
    arguments: [
      {
        name: "code",
        description: "Code to explain",
        required: true
      },
      {
        name: "language",
        description: "Programming language",
        required: false
      }
    ]
  }
};

const server = new Server({
  name: "example-prompts-server",
  version: "1.0.0"
}, {
  capabilities: {
    prompts: {}
  }
});

// 列出可用的prompts
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: Object.values(PROMPTS)
  };
});

// 获取特定的prompt
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const prompt = PROMPTS[request.params.name];
  if (!prompt) {
    throw new Error(`Prompt not found: ${request.params.name}`);
  }

  if (request.params.name === "git-commit") {
    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Generate a concise but descriptive commit message for these changes:\n\n${request.params.arguments?.changes}`
          }
        }
      ]
    };
  }

  if (request.params.name === "explain-code") {
    const language = request.params.arguments?.language || "Unknown";
    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Explain how this ${language} code works:\n\n${request.params.arguments?.code}`
          }
        }
      ]
    };
  }

  throw new Error("Prompt implementation not found");
});
```

### 最佳实践
在实现prompts时，建议遵循以下最佳实践：
1. 使用清晰、描述性的prompt名称。
2. 为prompts和参数提供详细的描述。
3. 验证所有必需的参数。
4. 优雅地处理缺失的参数。
5. 考虑为prompt模板进行版本控制。
6. 在适当的情况下缓存动态内容。
7. 实现错误处理。
8. 文档化预期的参数格式。
9. 考虑prompt的组合性。
10. 使用各种输入测试prompts。

### UI集成
Prompts可以在客户端UI中以以下形式展示：
- 斜杠命令（Slash commands）
- 快速操作（Quick actions）
- 上下文菜单项（Context menu items）
- 命令面板条目（Command palette entries）
- 引导工作流（Guided workflows）
- 交互表单（Interactive forms）

### 更新和变更
服务器可以通过以下方式通知客户端关于prompt的变更：
1. 服务器能力：`prompts.listChanged`
2. 通知：`notifications/prompts/list_changed`
3. 客户端重新获取prompt列表

### 安全考虑
在实现prompts时，需要注意以下安全问题：
- 验证所有参数。
- 清理用户输入。
- 考虑速率限制。
- 实现访问控制。
- 审计prompt的使用情况。
- 适当地处理敏感数据。
- 验证生成的内容。
- 实现超时机制。
- 考虑prompt注入风险。
- 文档化安全要求

-----
以下是网页内容的中文翻译：

---

### 网页标题
资源 - 模型上下文协议

### 网页内容
资源是模型上下文协议（MCP）的核心组成部分，允许服务器向客户端公开数据和内容，这些数据和内容可以被客户端读取，并作为大型语言模型（LLM）交互的上下文。

#### 概述
资源是任何一种数据，MCP 服务器希望将其提供给客户端。这可以包括：
- 文件内容
- 数据库记录
- API 响应
- 实时系统数据
- 截图和图像
- 日志文件
- 以及更多

每个资源通过一个唯一的 URI 进行标识，并且可以包含文本或二进制数据。

#### 资源 URI
资源通过 URI 进行标识，其格式如下：
```
[协议]://[主机]/[路径]
```
例如：
- `file:///home/user/documents/report.pdf`
- `postgres://database/customers/schema`
- `screen://localhost/display1`

协议和路径结构由 MCP 服务器实现定义。服务器可以定义自己的自定义 URI 方案。

#### 资源类型
资源可以包含两种类型的内容：

- **文本资源**
  - 文本资源包含 UTF-8 编码的文本数据，适用于：
    - 源代码
    - 配置文件
    - 日志文件
    - JSON/XML 数据
    - 纯文本

- **二进制资源**
  - 二进制资源包含以 base64 编码的原始二进制数据，适用于：
    - 图像
    - PDF 文件
    - 音频文件
    - 视频文件
    - 其他非文本格式

#### 资源发现
客户端可以通过两种主要方法发现可用资源：

- **直接资源**
  - 服务器通过 `resources/list` 端点公开具体的资源列表。每个资源包含以下内容：
    ```json
    {
      "uri": "string",           // 资源的唯一标识符
      "name": "string",          // 人类可读的名称
      "description": "string?",  // 可选的描述
      "mimeType": "string?",     // 可选的 MIME 类型
      "size": "number?"          // 可选的大小（以字节为单位）
    }
    ```

- **资源模板**
  - 对于动态资源，服务器可以公开 URI 模板，客户端可以使用这些模板构建有效的资源 URI：
    ```json
    {
      "uriTemplate": "string",   // 遵循 RFC 6570 的 URI 模板
      "name": "string",          // 这种类型的资源的可读名称
      "description": "string?",  // 可选的描述
      "mimeType": "string?"      // 所有匹配资源的 MIME 类型
    }
    ```

#### 读取资源
客户端通过带有资源 URI 的 `resources/read` 请求来读取资源。

服务器会返回资源内容列表：
```json
{
  "contents": [
    {
      "uri": "string",          // 资源的 URI
      "mimeType": "string?",    // 可选的 MIME 类型

      // 以下二者之一：
      "text": "string",         // 文本资源
      "blob": "string"          // 二进制资源（base64 编码）
    }
  ]
}
```

#### 资源更新
MCP 通过两种机制支持资源的实时更新：

- **列表变更**
  - 服务器可以通过 `notifications/resources/list_changed` 通知客户端其可用资源列表发生变化。

- **内容变更**
  - 客户端可以订阅特定资源的更新：
    1. 客户端发送带有资源 URI 的 `resources/subscribe` 请求。
    2. 资源发生变化时，服务器发送 `notifications/resources/updated` 通知。
    3. 客户端可以通过 `resources/read` 获取最新内容。
    4. 客户端可以通过 `resources/unsubscribe` 取消订阅。

#### 示例实现
以下是一个在 MCP 服务器中实现资源支持的简单示例：
```javascript
const server = new Server({
  name: "example-server",
  version: "1.0.0"
}, {
  capabilities: {
    resources: {}
  }
});

// 列出可用资源
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "file:///logs/app.log",
        name: "Application Logs",
        mimeType: "text/plain"
      }
    ]
  };
});

// 读取资源内容
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;

  if (uri === "file:///logs/app.log") {
    const logContents = await readLogFile();
    return {
      contents: [
        {
          uri,
          mimeType: "text/plain",
          text: logContents
        }
      ]
    };
  }

  throw new Error("Resource not found");
});
```

#### 最佳实践
在实现资源支持时：
1. 使用清晰、描述性的资源名称和 URI。
2. 提供有助于 LLM 理解的描述。
3. 在已知的情况下设置适当的 MIME 类型。
4. 为动态内容实现资源模板。
5. 为频繁变更的资源使用订阅。
6. 以清晰的错误消息优雅地处理错误。
7. 对于大型资源列表，考虑分页。
8. 适当缓存资源内容。
9. 在处理之前验证 URI。
10. 文档化你的自定义 URI 方案。

#### 安全注意事项
在公开资源时：
- 验证所有资源 URI。
- 实现适当的访问控制。
- 清理文件路径以防止目录遍历攻击。
- 小心处理二进制数据。
- 考虑对资源读取进行速率限制。
- 审计资源访问。
- 在传输过程中加密敏感数据。
- 验证 MIME 类型。
- 为长时间运行的读取操作设置超时。
- 适当处理资源清理。

---




# 模型上下文协议（MCP）实战指南：从入门到精通

## 引言

模型上下文协议（Model Context Protocol，简称 MCP）是一个革命性的开放标准，它定义了如何在客户端（如 Claude Desktop、IDE 或其他 AI 应用）和服务器之间建立安全、可控的连接。通过 MCP，我们可以为大型语言模型提供结构化的上下文信息，实现更强大的 AI 代理功能。

本文将通过实际案例，带您深入了解 MCP 的核心概念和实际应用，帮助您快速掌握这一强大工具。

## MCP 核心概念速览

### 1. 根节点（Roots）

根节点定义了服务器可以操作的边界，为客户端提供了一种方式来告知服务器相关资源及其位置。

### 2. 资源（Resources）

资源是 MCP 的核心组成部分，允许服务器向客户端公开数据和内容，包括文件、数据库记录、API 响应等。

### 3. 工具（Tools）

工具允许服务器向客户端暴露可执行功能，使 LLM 能够与外部系统交互、执行计算和采取行动。

### 4. 提示（Prompts）

提示是可复用的模板和工作流，客户端可以轻松地将其展示给用户和 LLM。

### 5. 采样（Sampling）

采样功能允许服务器通过客户端请求 LLM 完成任务，实现复杂的代理行为。

## 实战案例一：构建文件管理 MCP 服务器

让我们从一个简单的文件管理服务器开始，逐步了解 MCP 的实现。

### 项目初始化

```bash
```bash
# 创建 Conda 环境 (例如，环境名为 mcp_env，使用 Python 3.9)
conda create -n mcp_env python=3.9 -y

# 激活 Conda 环境
conda activate mcp_env

# 安装依赖
pip install mcp
```

```

```python
# requirements.txt
mcp==0.4.0
aiofiles==23.2.1
aiosqlite==0.19.0
httpx==0.25.2
```

### 基础服务器设置

```python
# server.py
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    Prompt,
    TextContent,
    ImageContent,
    EmbeddedResource
)
import aiofiles
import mimetypes

# 创建服务器实例
server = Server("file-manager-server")

# 定义工作目录
WORK_DIR = Path.cwd()
```

### 实现资源管理

```python
# 工具函数
def get_mime_type(filename: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type
    
    # 自定义映射
    ext_map = {
        '.py': 'text/x-python',
        '.js': 'text/javascript',
        '.json': 'application/json',
        '.md': 'text/markdown',
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.css': 'text/css'
    }
    
    ext = Path(filename).suffix.lower()
    return ext_map.get(ext, 'text/plain')

@server.list_resources()
async def list_resources() -> List[Resource]:
    """列出所有可用资源"""
    try:
        resources = []
        
        for file_path in WORK_DIR.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                resources.append(Resource(
                    uri=f"file://{file_path.absolute()}",
                    name=file_path.name,
                    description=f"文件: {file_path.name}",
                    mimeType=get_mime_type(file_path.name),
                    size=stat.st_size
                ))
        
        return resources
    except Exception as e:
        raise RuntimeError(f"无法列出资源: {str(e)}")

@server.read_resource()
async def read_resource(uri: str) -> str:
    """读取资源内容"""
    try:
        # 移除 file:// 前缀
        file_path = Path(uri.replace('file://', ''))
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return content
    except Exception as e:
        raise RuntimeError(f"无法读取文件 {file_path}: {str(e)}")
```

### 实现工具功能

```python
# 定义工具
@server.list_tools()
async def list_tools() -> List[Tool]:
    """返回可用工具列表"""
    return [
        Tool(
            name="create_file",
            description="创建新文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "要创建的文件名"
                    },
                    "content": {
                        "type": "string",
                        "description": "文件内容"
                    }
                },
                "required": ["filename", "content"]
            }
        ),
        Tool(
            name="delete_file",
            description="删除文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "要删除的文件名"
                    }
                },
                "required": ["filename"]
            }
        ),
        Tool(
            name="search_files",
            description="在文件中搜索文本",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "搜索模式"
                    },
                    "extension": {
                        "type": "string",
                        "description": "文件扩展名过滤（可选）"
                    }
                },
                "required": ["pattern"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """调用工具"""
    try:
        if name == "create_file":
            return await create_file(arguments["filename"], arguments["content"])
        elif name == "delete_file":
            return await delete_file(arguments["filename"])
        elif name == "search_files":
            extension = arguments.get("extension")
            return await search_files(arguments["pattern"], extension)
        else:
            raise ValueError(f"未知工具: {name}")
    except Exception as e:
        return [TextContent(type="text", text=f"错误: {str(e)}")]

# 工具实现函数
async def create_file(filename: str, content: str) -> List[TextContent]:
    """创建文件"""
    file_path = WORK_DIR / filename
    
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(content)
    
    return [TextContent(type="text", text=f"成功创建文件: {filename}")]

async def delete_file(filename: str) -> List[TextContent]:
    """删除文件"""
    file_path = WORK_DIR / filename
    
    if file_path.exists():
        file_path.unlink()
        return [TextContent(type="text", text=f"成功删除文件: {filename}")]
    else:
        raise FileNotFoundError(f"文件不存在: {filename}")

async def search_files(pattern: str, extension: Optional[str] = None) -> List[TextContent]:
    """在文件中搜索文本"""
    results = []
    
    for file_path in WORK_DIR.iterdir():
        if not file_path.is_file():
            continue
            
        if extension and not file_path.name.endswith(extension):
            continue
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if pattern in content:
                lines = content.split('\n')
                matching_lines = [
                    {"line": line, "number": i + 1}
                    for i, line in enumerate(lines)
                    if pattern in line
                ]
                
                results.append({
                    "file": file_path.name,
                    "matches": matching_lines
                })
        except Exception:
            # 忽略无法读取的文件
            continue
    
    return [TextContent(type="text", text=json.dumps(results, indent=2, ensure_ascii=False))]
```

### 实现提示功能

```python
@server.list_prompts()
async def list_prompts() -> List[Prompt]:
    """返回可用提示列表"""
    return [
        Prompt(
            name="analyze_code",
            description="分析代码文件并提供改进建议",
            arguments=[
                {
                    "name": "filename",
                    "description": "要分析的文件名",
                    "required": True
                }
            ]
        ),
        Prompt(
            name="generate_docs",
            description="为代码生成文档",
            arguments=[
                {
                    "name": "filename", 
                    "description": "要生成文档的文件名",
                    "required": True
                }
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """获取提示内容"""
    if name == "analyze_code":
        filename = arguments["filename"]
        file_path = WORK_DIR / filename
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return {
            "description": f"分析文件 {filename} 的代码质量",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"请分析以下代码文件并提供改进建议:\n\n文件名: {filename}\n\n代码内容:\n```\n{content}\n```"
                    }
                }
            ]
        }
    
    elif name == "generate_docs":
        filename = arguments["filename"]
        file_path = WORK_DIR / filename
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return {
            "description": f"为文件 {filename} 生成文档",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"请为以下代码生成详细的文档:\n\n文件名: {filename}\n\n代码内容:\n```\n{content}\n```"
                    }
                }
            ]
        }
    
    else:
        raise ValueError(f"未知提示: {name}")
```

### 启动服务器

```python
async def main():
    """启动 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    print("文件管理 MCP 服务器已启动", file=sys.stderr)
    asyncio.run(main())
```

## 实战案例二：数据库集成 MCP 服务器

现在让我们构建一个更复杂的例子，集成数据库功能。

### 数据库配置

```python
# database.py
import aiosqlite
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        
    async def connect(self):
        """连接数据库并创建表"""
        self.db = await aiosqlite.connect(self.db_path)
        
        # 创建示例表
        await self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        
        await self.db.commit()
    
    async def get_users(self) -> List[Dict[str, Any]]:
        """获取所有用户"""
        async with self.db.execute("SELECT * FROM users ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def get_posts(self) -> List[Dict[str, Any]]:
        """获取所有文章"""
        query = """
            SELECT p.*, u.name as user_name 
            FROM posts p 
            JOIN users u ON p.user_id = u.id 
            ORDER BY p.created_at DESC
        """
        async with self.db.execute(query) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def create_user(self, name: str, email: str) -> int:
        """创建新用户"""
        cursor = await self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        await self.db.commit()
        return cursor.lastrowid
    
    async def create_post(self, user_id: int, title: str, content: str) -> int:
        """创建新文章"""
        cursor = await self.db.execute(
            "INSERT INTO posts (user_id, title, content) VALUES (?, ?, ?)",
            (user_id, title, content)
        )
        await self.db.commit()
        return cursor.lastrowid
    
    async def search_posts(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索文章"""
        query = """
            SELECT p.*, u.name as user_name 
            FROM posts p 
            JOIN users u ON p.user_id = u.id 
            WHERE p.title LIKE ? OR p.content LIKE ?
            ORDER BY p.created_at DESC
        """
        async with self.db.execute(query, (f"%{keyword}%", f"%{keyword}%")) as cursor:
            rows = await cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    async def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'db'):
            await self.db.close()
```

### 数据库 MCP 服务器

```python
# db_server.py
import asyncio
import json
import sys
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from database import DatabaseManager

# 创建服务器和数据库管理器
server = Server("database-mcp-server")
db_manager = DatabaseManager('./blog.db')

@server.list_tools()
async def list_tools() -> List[Tool]:
    """返回数据库操作工具列表"""
    return [
        Tool(
            name="list_users",
            description="获取所有用户列表",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="create_user",
            description="创建新用户",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "用户名"},
                    "email": {"type": "string", "description": "邮箱地址"}
                },
                "required": ["name", "email"]
            }
        ),
        Tool(
            name="list_posts",
            description="获取所有文章列表",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="create_post",
            description="创建新文章",
            inputSchema={
                "type": "object",
                "properties": {
                    "userId": {"type": "integer", "description": "作者用户ID"},
                    "title": {"type": "string", "description": "文章标题"},
                    "content": {"type": "string", "description": "文章内容"}
                },
                "required": ["userId", "title", "content"]
            }
        ),
        Tool(
            name="search_posts",
            description="搜索文章",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["keyword"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """调用数据库工具"""
    try:
        if name == "list_users":
            users = await db_manager.get_users()
            return [TextContent(
                type="text", 
                text=json.dumps(users, indent=2, ensure_ascii=False)
            )]
        
        elif name == "create_user":
            user_id = await db_manager.create_user(
                arguments["name"], 
                arguments["email"]
            )
            return [TextContent(
                type="text", 
                text=f"成功创建用户，ID: {user_id}"
            )]
        
        elif name == "list_posts":
            posts = await db_manager.get_posts()
            return [TextContent(
                type="text", 
                text=json.dumps(posts, indent=2, ensure_ascii=False)
            )]
        
        elif name == "create_post":
            post_id = await db_manager.create_post(
                arguments["userId"],
                arguments["title"],
                arguments["content"]
            )
            return [TextContent(
                type="text", 
                text=f"成功创建文章，ID: {post_id}"
            )]
        
        elif name == "search_posts":
            results = await db_manager.search_posts(arguments["keyword"])
            return [TextContent(
                type="text", 
                text=json.dumps(results, indent=2, ensure_ascii=False)
            )]
        
        else:
            raise ValueError(f"未知工具: {name}")
            
    except Exception as e:
        return [TextContent(type="text", text=f"错误: {str(e)}")]

async def main():
    """启动数据库 MCP 服务器"""
    # 初始化数据库
    await db_manager.connect()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    finally:
        await db_manager.close()

if __name__ == "__main__":
    print("数据库 MCP 服务器已启动", file=sys.stderr)
    asyncio.run(main())
```

## 实战案例三：API 集成与采样功能

让我们构建一个集成外部 API 并支持采样功能的高级 MCP 服务器。

### API 集成服务器

```python
# api_server.py
import asyncio
import json
import random
import sys
from typing import Any, Dict, List
from datetime import datetime, timedelta

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("api-integration-server")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """返回 API 工具列表"""
    return [
        Tool(
            name="fetch_weather",
            description="获取天气信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "days": {"type": "integer", "description": "预报天数", "default": 1}
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="translate_text",
            description="翻译文本",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要翻译的文本"},
                    "from_lang": {"type": "string", "description": "源语言"},
                    "to_lang": {"type": "string", "description": "目标语言"}
                },
                "required": ["text", "from_lang", "to_lang"]
            }
        ),
        Tool(
            name="analyze_sentiment",
            description="情感分析",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要分析的文本"}
                },
                "required": ["text"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """调用 API 工具"""
    try:
        if name == "fetch_weather":
            return await fetch_weather(
                arguments["city"], 
                arguments.get("days", 1)
            )
        elif name == "translate_text":
            return await translate_text(
                arguments["text"],
                arguments["from_lang"],
                arguments["to_lang"]
            )
        elif name == "analyze_sentiment":
            return await analyze_sentiment(arguments["text"])
        else:
            raise ValueError(f"未知工具: {name}")
    except Exception as e:
        return [TextContent(type="text", text=f"错误: {str(e)}")]

# API 实现函数
async def fetch_weather(city: str, days: int) -> List[TextContent]:
    """获取天气信息（模拟数据）"""
    # 在实际应用中，这里应该调用真实的天气 API
    conditions = ["晴", "多云", "雨", "雪"]
    
    forecast = []
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "temperature": random.randint(10, 35),
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 90),
            "wind_speed": random.randint(5, 25)
        })
    
    weather_data = {
        "city": city,
        "days": days,
        "forecast": forecast
    }
    
    return [TextContent(
        type="text", 
        text=json.dumps(weather_data, indent=2, ensure_ascii=False)
    )]

async def translate_text(text: str, from_lang: str, to_lang: str) -> List[TextContent]:
    """翻译文本（模拟翻译）"""
    # 在实际应用中，这里应该调用真实的翻译 API
    
    # 简单的模拟翻译逻辑
    if from_lang.lower() == "en" and to_lang.lower() == "zh":
        # 英译中的一些简单映射
        translations = {
            "hello": "你好",
            "world": "世界",
            "how are you": "你好吗",
            "thank you": "谢谢"
        }
        translated = translations.get(text.lower(), f"[翻译] {text}")
    elif from_lang.lower() == "zh" and to_lang.lower() == "en":
        # 中译英的一些简单映射
        translations = {
            "你好": "hello",
            "世界": "world",
            "谢谢": "thank you"
        }
        translated = translations.get(text, f"[Translated] {text}")
    else:
        translated = f"[{from_lang}->{to_lang}] {text}"
    
    translation_result = {
        "original": text,
        "translated": translated,
        "from_language": from_lang,
        "to_language": to_lang,
        "confidence": round(random.uniform(0.8, 1.0), 2)
    }
    
    return [TextContent(
        type="text", 
        text=json.dumps(translation_result, indent=2, ensure_ascii=False)
    )]

async def analyze_sentiment(text: str) -> List[TextContent]:
    """情感分析（模拟分析）"""
    # 在实际应用中，这里应该调用真实的情感分析 API
    
    # 简单的关键词情感分析
    positive_words = ["好", "棒", "优秀", "开心", "高兴", "喜欢", "love", "good", "great", "excellent"]
    negative_words = ["坏", "差", "糟糕", "生气", "难过", "讨厌", "bad", "terrible", "hate", "awful"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.9, 0.6 + positive_count * 0.1)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.9, 0.6 + negative_count * 0.1)
    else:
        sentiment = "neutral"
        confidence = 0.7
    
    analysis = {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "details": {
            "positive": round(confidence if sentiment == "positive" else random.uniform(0.1, 0.4), 2),
            "negative": round(confidence if sentiment == "negative" else random.uniform(0.1, 0.4), 2),
            "neutral": round(confidence if sentiment == "neutral" else random.uniform(0.1, 0.4), 2)
        },
        "keywords": {
            "positive_found": positive_count,
            "negative_found": negative_count
        }
    }
    
    return [TextContent(
        type="text", 
        text=json.dumps(analysis, indent=2, ensure_ascii=False)
    )]

async def main():
    """启动 API 集成 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    print("API 集成 MCP 服务器已启动", file=sys.stderr)
    asyncio.run(main())
```

## 客户端配置

要使用我们创建的 MCP 服务器，需要在客户端（如 Claude Desktop）中进行配置。

### Claude Desktop 配置

创建或编辑配置文件 `%APPDATA%\Claude\claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "file-manager": {
      "command": "python",
      "args": ["C:/path/to/your/server.py"],
      "env": {
        "PYTHONPATH": "C:/path/to/your/project",
        "PYTHONUNBUFFERED": "1"
      }
    },
    "database": {
      "command": "python", 
      "args": ["C:/path/to/your/db_server.py"],
      "env": {
        "PYTHONPATH": "C:/path/to/your/project",
        "PYTHONUNBUFFERED": "1"
      }
    },
    "api-integration": {
      "command": "python",
      "args": ["C:/path/to/your/api_server.py"],
      "env": {
        "PYTHONPATH": "C:/path/to/your/project",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### 启动脚本

为了便于管理，可以创建启动脚本：

```python
# start_servers.py
import subprocess
import sys
from pathlib import Path

def start_server(server_file: str, server_name: str):
    """启动 MCP 服务器"""
    try:
        print(f"正在启动 {server_name} 服务器...")
        
        # 激活虚拟环境并启动服务器
        cmd = [
            sys.executable,  # 当前 Python 解释器
            str(Path(__file__).parent / server_file)
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"{server_name} 服务器已启动，PID: {process.pid}")
        return process
        
    except Exception as e:
        print(f"启动 {server_name} 服务器失败: {e}")
        return None

if __name__ == "__main__":
    servers = [
        ("server.py", "文件管理"),
        ("db_server.py", "数据库"),
        ("api_server.py", "API集成")
    ]
    
    processes = []
    for server_file, server_name in servers:
        process = start_server(server_file, server_name)
        if process:
            processes.append((process, server_name))
    
    try:
        # 等待所有服务器
        for process, name in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\n正在停止所有服务器...")
        for process, name in processes:
            process.terminate()
            print(f"{name} 服务器已停止")
```

## 最佳实践与注意事项

### 1. 安全性

- 始终验证输入参数
- 实现适当的访问控制
- 对敏感操作进行审计
- 使用安全的数据传输方式

### 2. 性能优化

- 实现合适的缓存机制
- 对长时间运行的操作设置超时
- 考虑实现分页功能
- 监控资源使用情况

### 3. 错误处理

- 提供清晰的错误信息
- 实现优雅的降级机制
- 记录详细的错误日志
- 避免暴露内部实现细节

### 4. 文档化

- 为每个工具提供详细描述
- 包含参数示例
- 说明预期的返回值格式
- 提供使用指南

### 5. 测试策略

- 编写单元测试
- 进行集成测试
- 模拟各种边界情况
- 测试错误处理逻辑

## 进阶技巧

### 动态资源发现

```python
# 实现动态资源模板
@server.list_resource_templates()
async def list_resource_templates():
    """返回资源模板列表"""
    return [
        {
            "uriTemplate": "file://{path}",
            "name": "文件系统",
            "description": "访问文件系统中的任意文件",
            "mimeType": "text/plain"
        },
        {
            "uriTemplate": "db://users/{id}",
            "name": "用户记录", 
            "description": "按 ID 访问用户记录",
            "mimeType": "application/json"
        }
    ]
```

### 实时更新通知

```python
# 实现资源变更通知
class ResourceWatcher:
    def __init__(self, server):
        self.server = server
        self.subscribers = set()
    
    def subscribe(self, uri: str):
        """订阅资源变更"""
        self.subscribers.add(uri)
    
    async def notify_change(self, uri: str):
        """通知资源变更"""
        if uri in self.subscribers:
            await self.server.request_notification(
                "notifications/resources/updated",
                {"uri": uri}
            )
```

### 高级采样功能

```python
# 实现复杂的采样逻辑
@server.create_message()
async def create_message(messages, system_prompt=None, include_context=None):
    """处理采样请求"""
    
    # 添加上下文信息
    if include_context == "thisServer":
        context = await gather_server_context()
        system_message = {
            "role": "system",
            "content": {
                "type": "text",
                "text": f"服务器上下文: {json.dumps(context, ensure_ascii=False)}"
            }
        }
        messages.insert(0, system_message)
    
    # 返回处理后的消息
    return {
        "model": "claude-3-sonnet",
        "messages": messages,
        "systemPrompt": system_prompt or "您是一个有用的 AI 助手"
    }

async def gather_server_context():
    """收集服务器上下文信息"""
    return {
        "available_tools": ["create_file", "delete_file", "search_files"],
        "working_directory": str(WORK_DIR),
        "server_status": "running",
        "timestamp": datetime.now().isoformat()
    }
```

## Python 特有的实现技巧

### 异步编程最佳实践

```python
# 使用 asyncio 的最佳实践
import asyncio
from asyncio import Lock, Queue
from contextlib import asynccontextmanager

class AsyncResourceManager:
    def __init__(self):
        self._lock = Lock()
        self._cache = {}
    
    @asynccontextmanager
    async def get_resource(self, uri: str):
        """异步资源管理上下文管理器"""
        async with self._lock:
            if uri not in self._cache:
                self._cache[uri] = await self._load_resource(uri)
            resource = self._cache[uri]
        
        try:
            yield resource
        finally:
            # 清理资源
            pass
    
    async def _load_resource(self, uri: str):
        """异步加载资源"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        return f"Resource for {uri}"
```

### 类型提示和数据验证

```python
# 使用 Pydantic 进行数据验证
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
from enum import Enum

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"

class WeatherCondition(str, Enum):
    SUNNY = "晴"
    CLOUDY = "多云"
    RAINY = "雨"
    SNOWY = "雪"

class WeatherForecast(BaseModel):
    date: str = Field(..., description="日期，格式：YYYY-MM-DD")
    temperature: int = Field(..., ge=-50, le=60, description="温度，摄氏度")
    condition: WeatherCondition = Field(..., description="天气状况")
    humidity: int = Field(..., ge=0, le=100, description="湿度百分比")
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('日期格式必须为 YYYY-MM-DD')

class SentimentAnalysis(BaseModel):
    text: str = Field(..., description="要分析的文本")
    sentiment: SentimentType = Field(..., description="情感类型")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    
    class Config:
        use_enum_values = True
```

### 错误处理和日志记录

```python
import logging
from functools import wraps
from typing import Callable, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_tool_calls(func: Callable) -> Callable:
    """工具调用日志装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = kwargs.get('name', 'unknown')
        logger.info(f"调用工具: {tool_name}")
        
        try:
            result = await func(*args, **kwargs)
            logger.info(f"工具 {tool_name} 执行成功")
            return result
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {str(e)}")
            raise
    
    return wrapper

class MCPError(Exception):
    """MCP 自定义异常基类"""
    pass

class ResourceNotFoundError(MCPError):
    """资源未找到异常"""
    pass

class ToolExecutionError(MCPError):
    """工具执行异常"""
    pass
```

### 配置管理

```python
# config.py
import os
from pathlib import Path
from pydantic import BaseSettings, Field

class MCPServerConfig(BaseSettings):
    """MCP 服务器配置"""
    
    # 基础配置
    server_name: str = Field(default="mcp-server", env="MCP_SERVER_NAME")
    server_version: str = Field(default="1.0.0", env="MCP_SERVER_VERSION")
    
    # 文件系统配置
    work_directory: Path = Field(default_factory=Path.cwd, env="MCP_WORK_DIR")
    max_file_size: int = Field(default=10*1024*1024, env="MCP_MAX_FILE_SIZE")  # 10MB
    
    # 数据库配置
    database_url: str = Field(default="sqlite:///./mcp.db", env="MCP_DATABASE_URL")
    
    # API 配置
    api_timeout: int = Field(default=30, env="MCP_API_TIMEOUT")
    max_concurrent_requests: int = Field(default=10, env="MCP_MAX_CONCURRENT")
    
    # 日志配置
    log_level: str = Field(default="INFO", env="MCP_LOG_LEVEL")
    log_file: str = Field(default="mcp_server.log", env="MCP_LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 使用配置
config = MCPServerConfig()
```

### 测试策略

```python
# test_mcp_server.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from mcp.types import TextContent

from server import server, create_file, delete_file, search_files

@pytest.fixture
def event_loop():
    """为异步测试提供事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_server():
    """模拟 MCP 服务器"""
    return server

class TestFileOperations:
    """文件操作测试"""
    
    @pytest.mark.asyncio
    async def test_create_file_success(self, tmp_path):
        """测试成功创建文件"""
        # 设置临时目录
        with patch('server.WORK_DIR', tmp_path):
            result = await create_file("test.txt", "Hello, World!")
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "成功创建文件" in result[0].text
            
            # 验证文件是否实际创建
            test_file = tmp_path / "test.txt"
            assert test_file.exists()
            assert test_file.read_text(encoding='utf-8') == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self, tmp_path):
        """测试成功删除文件"""
        # 创建测试文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content", encoding='utf-8')
        
        with patch('server.WORK_DIR', tmp_path):
            result = await delete_file("test.txt")
            
            assert len(result) == 1
            assert "成功删除文件" in result[0].text
            assert not test_file.exists()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, tmp_path):
        """测试删除不存在的文件"""
        with patch('server.WORK_DIR', tmp_path):
            with pytest.raises(FileNotFoundError):
                await delete_file("nonexistent.txt")
    
    @pytest.mark.asyncio
    async def test_search_files(self, tmp_path):
        """测试文件搜索功能"""
        # 创建测试文件
        file1 = tmp_path / "file1.py"
        file1.write_text("def hello():\n    print('Hello, World!')", encoding='utf-8')
        
        file2 = tmp_path / "file2.py"  
        file2.write_text("def goodbye():\n    print('Goodbye!')", encoding='utf-8')
        
        with patch('server.WORK_DIR', tmp_path):
            result = await search_files("Hello", ".py")
            
            assert len(result) == 1
            content = json.loads(result[0].text)
            assert len(content) == 1
            assert content[0]["file"] == "file1.py"

class TestDatabaseOperations:
    """数据库操作测试"""
    
    @pytest.fixture
    async def db_manager(self):
        """数据库管理器夹具"""
        from database import DatabaseManager
        
        # 使用内存数据库进行测试
        manager = DatabaseManager(":memory:")
        await manager.connect()
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_create_user(self, db_manager):
        """测试创建用户"""
        user_id = await db_manager.create_user("测试用户", "test@example.com")
        assert user_id is not None
        
        users = await db_manager.get_users()
        assert len(users) == 1
        assert users[0]["name"] == "测试用户"
    
    @pytest.mark.asyncio
    async def test_create_post(self, db_manager):
        """测试创建文章"""
        # 先创建用户
        user_id = await db_manager.create_user("作者", "author@example.com")
        
        # 创建文章
        post_id = await db_manager.create_post(
            user_id, 
            "测试标题", 
            "测试内容"
        )
        assert post_id is not None
        
        posts = await db_manager.get_posts()
        assert len(posts) == 1
        assert posts[0]["title"] == "测试标题"

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 性能优化和监控

```python
# performance.py
import time
import asyncio
from functools import wraps
from typing import Dict, List
from collections import defaultdict
import psutil

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
    
    def timing_decorator(self, func_name: str = None):
        """计时装饰器"""
        def decorator(func):
            name = func_name or func.__name__
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.metrics[name].append(duration)
                    self.call_counts[name] += 1
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.metrics[name].append(duration)
                    self.call_counts[name] += 1
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取性能统计"""
        stats = {}
        for name, times in self.metrics.items():
            if times:
                stats[name] = {
                    "count": self.call_counts[name],
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        return stats
    
    def get_system_stats(self) -> Dict[str, float]:
        """获取系统统计"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_memory": psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }

# 使用示例
monitor = PerformanceMonitor()

@monitor.timing_decorator("file_creation")
async def monitored_create_file(filename: str, content: str):
    """带监控的文件创建"""
    return await create_file(filename, content)
```

### 部署和生产环境配置

```python
# deployment.py
import os
import signal
import sys
from pathlib import Path

class GracefulShutdown:
    """优雅关闭处理器"""
    
    def __init__(self):
        self.shutdown = False
        self.cleanup_tasks = []
    
    def add_cleanup_task(self, task):
        """添加清理任务"""
        self.cleanup_tasks.append(task)
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在优雅关闭...")
        self.shutdown = True
        
        # 执行清理任务
        for task in self.cleanup_tasks:
            try:
                task()
            except Exception as e:
                print(f"清理任务执行失败: {e}")
        
        print("服务器已安全关闭")
        sys.exit(0)

# 配置信号处理
shutdown_handler = GracefulShutdown()
signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)

# 环境配置
class Config:
    """配置管理"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.database_path = os.getenv("DATABASE_PATH", "mcp_server.db")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
        self.allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", ".txt,.md,.py,.json").split(",")
        
        # API配置
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.translate_api_key = os.getenv("TRANSLATE_API_KEY")
        
        # 服务器配置
        self.server_host = os.getenv("SERVER_HOST", "localhost")
        self.server_port = int(os.getenv("SERVER_PORT", "8000"))
        
    def validate(self):
        """验证配置"""
        if not self.weather_api_key:
            print("警告：未设置天气API密钥")
        if not self.translate_api_key:
            print("警告：未设置翻译API密钥")

config = Config()
config.validate()
```

### 高级功能示例

#### 1. 文件版本控制工具

```python
import hashlib
import json
from datetime import datetime
from typing import Dict, List

class FileVersionControl:
    """文件版本控制"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / ".versions"
        self.versions_path.mkdir(exist_ok=True)
        
    def get_file_hash(self, file_path: Path) -> str:
        """获取文件哈希"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    async def save_version(self, file_path: str, comment: str = ""):
        """保存文件版本"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_hash = self.get_file_hash(path)
        version_info = {
            "timestamp": datetime.now().isoformat(),
            "hash": file_hash,
            "comment": comment,
            "size": path.stat().st_size
        }
        
        # 保存版本信息
        versions_file = self.versions_path / f"{path.name}.versions.json"
        versions = []
        if versions_file.exists():
            with open(versions_file, 'r', encoding='utf-8') as f:
                versions = json.load(f)
        
        versions.append(version_info)
        with open(versions_file, 'w', encoding='utf-8') as f:
            json.dump(versions, f, ensure_ascii=False, indent=2)
        
        # 保存文件副本
        version_file = self.versions_path / f"{path.name}.{file_hash[:8]}"
        with open(path, 'rb') as src, open(version_file, 'wb') as dst:
            dst.write(src.read())
        
        return version_info
    
    def get_versions(self, file_path: str) -> List[Dict]:
        """获取文件版本历史"""
        path = Path(file_path)
        versions_file = self.versions_path / f"{path.name}.versions.json"
        
        if not versions_file.exists():
            return []
        
        with open(versions_file, 'r', encoding='utf-8') as f:
            return json.load(f)

# 添加版本控制工具
version_control = FileVersionControl("./managed_files")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出可用工具，包括版本控制"""
    tools = [
        types.Tool(
            name="save_file_version",
            description="保存文件版本快照",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "comment": {
                        "type": "string",
                        "description": "版本注释"
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="get_file_versions",
            description="获取文件版本历史",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    }
                },
                "required": ["file_path"]
            }
        )
    ]
    return tools

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """处理工具调用，包括版本控制"""
    try:
        if name == "save_file_version":
            version_info = await version_control.save_version(
                arguments["file_path"],
                arguments.get("comment", "")
            )
            return [types.TextContent(
                type="text",
                text=f"版本已保存: {json.dumps(version_info, ensure_ascii=False, indent=2)}"
            )]
        
        elif name == "get_file_versions":
            versions = version_control.get_versions(arguments["file_path"])
            return [types.TextContent(
                type="text",
                text=f"版本历史:\n{json.dumps(versions, ensure_ascii=False, indent=2)}"
            )]
        
        else:
            raise ValueError(f"未知工具: {name}")
    
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"错误: {str(e)}"
        )]
```

#### 2. 智能代码分析工具

```python
import ast
import re
from typing import Dict, List, Any

class CodeAnalyzer:
    """代码分析器"""
    
    def analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """分析Python文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            analysis = {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity": 0,
                "lines_of_code": len(content.splitlines()),
                "docstring_coverage": 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "has_docstring": ast.get_docstring(node) is not None
                    })
                
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        "has_docstring": ast.get_docstring(node) is not None
                    })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            analysis["imports"].append(f"{module}.{alias.name}")
            
            # 计算文档字符串覆盖率
            total_items = len(analysis["functions"]) + len(analysis["classes"])
            documented_items = sum(1 for f in analysis["functions"] if f["has_docstring"])
            documented_items += sum(1 for c in analysis["classes"] if c["has_docstring"])
            
            if total_items > 0:
                analysis["docstring_coverage"] = documented_items / total_items * 100
            
            return analysis
        
        except SyntaxError as e:
            return {"error": f"语法错误: {str(e)}"}
    
    def check_code_quality(self, file_path: str) -> Dict[str, List[str]]:
        """检查代码质量"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = {
            "style": [],
            "complexity": [],
            "security": []
        }
        
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            # 检查行长度
            if len(line) > 120:
                issues["style"].append(f"第{i}行：行过长 ({len(line)} 字符)")
            
            # 检查硬编码密码
            if re.search(r'password\s*=\s*["\'][^"\']*["\']', line, re.IGNORECASE):
                issues["security"].append(f"第{i}行：可能包含硬编码密码")
            
            # 检查SQL注入风险
            if re.search(r'execute\s*\(\s*["\'].*%.*["\']', line):
                issues["security"].append(f"第{i}行：可能存在SQL注入风险")
        
        return issues

# 添加代码分析工具
code_analyzer = CodeAnalyzer()

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出可用工具，包括代码分析"""
    tools.extend([
        types.Tool(
            name="analyze_python_code",
            description="分析Python代码文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Python文件路径"
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="check_code_quality",
            description="检查代码质量问题",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "代码文件路径"
                    }
                },
                "required": ["file_path"]
            }
        )
    ])
    return tools
```

#### 3. 实时监控和报告

```python
import asyncio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

class MonitoringService:
    """监控服务"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.running = False
    
    async def start_monitoring(self):
        """开始监控"""
        self.running = True
        while self.running:
            # 收集指标
            metrics = {
                "timestamp": datetime.now(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_connections": len(psutil.net_connections()),
                "process_count": len(psutil.pids())
            }
            
            self.metrics_history.append(metrics)
            
            # 检查告警
            await self.check_alerts(metrics)
            
            # 保持最近24小时的数据
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m["timestamp"] > cutoff_time
            ]
            
            await asyncio.sleep(60)  # 每分钟收集一次
    
    async def check_alerts(self, metrics: Dict):
        """检查告警条件"""
        alerts = []
        
        if metrics["cpu_percent"] > 80:
            alerts.append({
                "level": "WARNING",
                "message": f"CPU使用率过高: {metrics['cpu_percent']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        if metrics["memory_percent"] > 85:
            alerts.append({
                "level": "WARNING",
                "message": f"内存使用率过高: {metrics['memory_percent']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        if metrics["disk_usage"] > 90:
            alerts.append({
                "level": "CRITICAL",
                "message": f"磁盘使用率危险: {metrics['disk_usage']:.1f}%",
                "timestamp": metrics["timestamp"]
            })
        
        self.alerts.extend(alerts)
        
        # 记录告警
        for alert in alerts:
            logger.warning(f"告警: {alert['message']}")
    
    def generate_report(self) -> str:
        """生成监控报告"""
        if not self.metrics_history:
            return "暂无监控数据"
        
        latest = self.metrics_history[-1]
        
        # 计算平均值
        avg_cpu = sum(m["cpu_percent"] for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m["memory_percent"] for m in self.metrics_history) / len(self.metrics_history)
        
        report = f"""
# 系统监控报告

## 当前状态 ({latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})
- CPU使用率: {latest['cpu_percent']:.1f}%
- 内存使用率: {latest['memory_percent']:.1f}%
- 磁盘使用率: {latest['disk_usage']:.1f}%
- 活跃连接数: {latest['active_connections']}
- 进程数量: {latest['process_count']}

## 平均性能 (24小时)
- 平均CPU使用率: {avg_cpu:.1f}%
- 平均内存使用率: {avg_memory:.1f}%
- 数据点数量: {len(self.metrics_history)}

## 最近告警
"""
        
        # 添加最近的告警
        recent_alerts = sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:10]
        for alert in recent_alerts:
            report += f"- [{alert['level']}] {alert['message']} ({alert['timestamp'].strftime('%H:%M:%S')})\n"
        
        return report
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False

# 启动监控服务
monitoring_service = MonitoringService()

# 在服务器启动时开始监控
async def start_server_with_monitoring():
    """带监控的服务器启动"""
    # 启动监控任务
    monitoring_task = asyncio.create_task(monitoring_service.start_monitoring())
    
    # 添加清理任务
    shutdown_handler.add_cleanup_task(monitoring_service.stop_monitoring)
    
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    finally:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
```

## 总结

通过本实战指南，我们全面介绍了如何使用Python开发MCP（Model Context Protocol）服务器。从基础概念到高级功能，从简单示例到生产部署，涵盖了MCP开发的各个方面：

### 关键知识点

1. **MCP协议理解**：掌握了资源、工具和提示的核心概念
2. **Python异步编程**：学会了使用asyncio、aiofiles等异步库
3. **服务器架构设计**：了解了如何构建可扩展的MCP服务器
4. **错误处理和日志**：实现了完善的错误处理和监控机制
5. **性能优化**：学会了缓存、性能分析和资源监控
6. **部署运维**：掌握了容器化部署和生产环境配置

### 实用功能

- 文件管理和版本控制
- 数据库集成和数据操作
- API集成和外部服务调用
- 代码分析和质量检查
- 实时监控和性能分析
- 告警管理和通知系统

### 最佳实践

1. **安全性**：输入验证、权限控制、敏感数据保护
2. **可靠性**：错误恢复、优雅关闭、健康检查
3. **性能**：异步编程、缓存策略、资源优化
4. **可维护性**：模块化设计、测试覆盖、文档完善
5. **可扩展性**：插件机制、配置管理、容器化部署

MCP为AI应用与外部系统的集成提供了标准化的解决方案。通过Python的强大生态系统，我们可以快速构建功能丰富、性能优异的MCP服务器，为AI应用提供强大的上下文支持和工具能力。

希望这个实战指南能帮助你快速上手MCP开发，构建出符合实际需求的高质量MCP服务器！

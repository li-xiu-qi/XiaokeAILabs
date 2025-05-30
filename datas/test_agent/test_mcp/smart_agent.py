#!/usr/bin/env python3
"""
智能体 - 使用火山引擎 API + MCP 工具服务器
"""

import os
import json
import requests
import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载环境变量
load_dotenv()

class LocalAgent:
    def __init__(self):
        self.api_key = os.getenv("VOLCES_API_KEY")
        self.base_url = os.getenv("VOLCES_BASE_URL")
        self.model = os.getenv("VOLCES_TEXT_MODEL")
        self.mcp_session = None
        self.available_tools = {}
    
    async def start_mcp_connection(self):
        """启动 MCP 连接"""
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_tools_server.py"],
        )
        
        self.server_connection = stdio_client(server_params)
        self.read, self.write = await self.server_connection.__aenter__()
        self.mcp_session = ClientSession(self.read, self.write)
        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()
        
        # 获取可用工具
        tools = await self.mcp_session.list_tools()
        for tool in tools.tools:
            self.available_tools[tool.name] = tool.description
            
        print(f"✅ MCP 工具服务器已连接，可用工具: {list(self.available_tools.keys())}")
    
    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """调用 MCP 工具"""
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)
            return result.content[0].text
        except Exception as e:
            return f"工具调用失败: {str(e)}"
    async def detect_and_use_tools(self, user_message: str) -> str:
        """检测用户意图并使用 MCP 工具"""
        message_lower = user_message.lower()
        results = []
        
        print(f"🔍 检测用户意图: {user_message}")
        
        # 计算
        if "计算" in message_lower or any(op in user_message for op in ['+', '-', '*', '/', '=']):
            import re
            expression_match = re.search(r'[\d\+\-\*/\.\(\)\s]+', user_message)
            if expression_match:
                expression = expression_match.group().strip()
                print(f"🧮 调用计算工具: {expression}")
                result = await self.call_mcp_tool("calculate", {"expression": expression})
                results.append(result)
        
        # 时间
        if "时间" in message_lower or "几点" in message_lower:
            print("🕐 调用时间工具")
            result = await self.call_mcp_tool("get_time", {})
            results.append(result)
        
        # 文件列表 - 改进检测逻辑
        if "文件" in message_lower and ("列表" in message_lower or "列出" in message_lower or "有什么" in message_lower):
            directory = "."  # 默认当前目录
            
            # 检测目录
            if "当前目录" in message_lower or "这里" in message_lower:
                directory = "."
            elif "上一级" in message_lower or "上级" in message_lower or ".." in user_message:
                directory = ".."
            elif "根目录" in message_lower:
                directory = "/"
            
            print(f"📁 调用文件列表工具: {directory}")
            result = await self.call_mcp_tool("list_files", {"directory": directory})
            results.append(result)
        
        # 回显
        if "回显" in message_lower:
            text = user_message.replace("回显", "").strip()
            if text:
                print(f"📢 调用回显工具: {text}")
                result = await self.call_mcp_tool("echo", {"text": text})
                results.append(result)
        
        # 读取文件
        if "读取文件" in message_lower or "打开文件" in message_lower:
            words = user_message.split()
            for word in words:
                if "." in word and len(word) > 3:  # 简单的文件名检测
                    print(f"📖 调用读取文件工具: {word}")
                    result = await self.call_mcp_tool("read_file", {"filename": word})
                    results.append(result)
                    break
        
        if results:
            print(f"✅ 工具执行完成，返回 {len(results)} 个结果")
        else:
            print("ℹ️ 未匹配到需要执行的工具")
            
        return "\n".join(results) if results else ""
    def call_ai(self, user_message: str, tool_results: str = "") -> str:
        """调用火山引擎 API"""
        if tool_results:
            # 如果有工具结果，让AI基于结果回答
            system_prompt = f"""你是一个智能助手。用户刚才的请求已经通过工具执行完成，工具返回的结果如下：

{tool_results}

请基于这个工具执行结果，给用户一个简洁、友好的回答。不要重复工具的原始输出，而是要解释和总结结果。"""
        else:
            # 没有工具结果，正常对话
            system_prompt = f"""你是一个智能助手，可以通过工具执行以下功能：
- 数学计算
- 获取当前时间  
- 列出目录文件
- 文件读写操作
- 回显文本
- 读取环境变量

请回答用户的问题。如果用户的请求需要使用工具，请告诉用户我会帮他们执行。"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"⚠️ API 响应状态: {response.status_code}")
                print(f"⚠️ API 响应内容: {response.text}")
                return f"API 调用失败: {response.status_code}"
        except Exception as e:
            print(f"⚠️ API 调用异常: {str(e)}")
            return f"API 错误: {str(e)}"
    async def chat(self, user_message: str) -> str:
        """处理用户消息"""
        # 1. 检测并执行 MCP 工具
        tool_results = await self.detect_and_use_tools(user_message)
        
        # 2. 调用 AI 生成回答
        ai_response = self.call_ai(user_message, tool_results)
        
        # 3. 返回结果 - 简化格式
        if tool_results:
            return f"🤖 {ai_response}"
        else:
            return f"🤖 {ai_response}"
    
    async def close(self):
        """关闭 MCP 连接"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if hasattr(self, 'server_connection'):
            await self.server_connection.__aexit__(None, None, None)

async def main():
    agent = LocalAgent()
    
    try:
        # 启动 MCP 连接
        await agent.start_mcp_connection()
        
        print("\n🤖 本地智能体已启动！输入 'quit' 退出。")
        print("💡 我可以通过 MCP 工具计算、查看时间、列出文件等。\n")
        
        while True:
            try:
                user_input = input("👤 您: ")
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                response = await agent.chat(user_input)
                print(f"{response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {str(e)}\n")
    
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
简单的 MCP 服务器示例
基于架构文章中的代码实现
"""

import sys
import json
import asyncio
from typing import Dict, Any, Optional


class SimpleMCPServer:
    """简单的 MCP 服务器实现"""
    
    def __init__(self):
        self.running = True
        self.capabilities = {
            "tools": {"listChanged": True},
            "prompts": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "logging": {}
        }
        self.server_info = {
            "name": "SimpleMCPServer",
            "version": "1.0.0"
        }

    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理初始化请求"""
        return {
            "protocolVersion": "2025-03-26",
            "capabilities": self.capabilities,
            "serverInfo": self.server_info,
            "instructions": "这是一个简单的 MCP 服务器示例"
        }

    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具列表请求"""
        return {
            "tools": [
                {
                    "name": "echo",
                    "description": "回显输入的文本",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "要回显的文本"
                            }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "calculate",
                    "description": "简单的数学计算",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，如 '2 + 3'"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            ]
        }

    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具调用请求"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "echo":
            text = arguments.get("text", "")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Echo: {text}"
                    }
                ]
            }
        
        elif name == "calculate":
            expression = arguments.get("expression", "")
            try:
                # 简单计算（生产环境需要更安全的实现）
                result = eval(expression)
                return {
                    "content": [
                        {
                            "type": "text", 
                            "text": f"{expression} = {result}"
                        }
                    ]
                }
            except Exception as e:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"计算错误: {str(e)}"
                        }
                    ],
                    "isError": True
                }
        
        else:
            raise ValueError(f"未知工具: {name}")

    async def handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理提示列表请求"""
        return {
            "prompts": [
                {
                    "name": "greeting",
                    "description": "生成问候语",
                    "arguments": [
                        {
                            "name": "name",
                            "description": "要问候的人的名字",
                            "required": True
                        }
                    ]
                }
            ]
        }

    async def handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取提示请求"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "greeting":
            user_name = arguments.get("name", "朋友")
            return {
                "description": f"为 {user_name} 生成的问候语",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请为 {user_name} 生成一个友好的问候语"
                        }
                    }
                ]
            }
        else:
            raise ValueError(f"未知提示: {name}")

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理接收到的消息"""
        method = message.get("method")
        params = message.get("params", {})
        request_id = message.get("id")
        
        try:
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list(params)
            elif method == "tools/call":
                result = await self.handle_tools_call(params)
            elif method == "prompts/list":
                result = await self.handle_prompts_list(params)
            elif method == "prompts/get":
                result = await self.handle_prompts_get(params)
            elif method == "notifications/initialized":
                # 初始化完成通知，无需响应
                return None
            else:
                raise ValueError(f"未知方法: {method}")
                
            # 返回成功响应
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            # 返回错误响应
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "内部错误",
                    "data": str(e)
                }
            }

    async def run(self):
        """运行服务器主循环"""
        print("SimpleMCPServer 启动，等待消息...", file=sys.stderr)
        
        while self.running:
            try:
                # 从 stdin 读取消息
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # 解析 JSON-RPC 消息
                message = json.loads(line)
                
                # 处理消息
                response = await self.handle_message(message)
                
                # 发送响应
                if response:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                # JSON 解析错误
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "解析错误",
                        "data": str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)
                
            except Exception as e:
                # 其他错误
                print(f"服务器错误: {e}", file=sys.stderr)


if __name__ == "__main__":
    server = SimpleMCPServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("服务器停止", file=sys.stderr)

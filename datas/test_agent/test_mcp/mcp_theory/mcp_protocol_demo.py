#!/usr/bin/env python3
"""
MCP 协议流程演示
基于架构文章中的初始化流程
"""

import json
import asyncio
import sys
from typing import Dict, Any


class MCPProtocolDemo:
    """MCP 协议演示"""
    
    def __init__(self):
        self.step = 0
    
    def print_step(self, title: str, content: Dict[str, Any]):
        """打印协议步骤"""
        self.step += 1
        print(f"\n{'='*60}")
        print(f"步骤 {self.step}: {title}")
        print('='*60)
        print(json.dumps(content, indent=2, ensure_ascii=False))
        print()
    
    def demo_initialization_flow(self):
        """演示 MCP 初始化流程"""
        print("🚀 MCP 协议初始化流程演示")
        print("基于官方规范: https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle")
        
        # 步骤1: 客户端发送初始化请求
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "ExampleClient",
                    "version": "1.0.0"
                }
            }
        }
        
        self.print_step("客户端发送初始化请求", initialize_request)
        
        # 步骤2: 服务器返回初始化响应
        initialize_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {
                    "logging": {},
                    "prompts": {
                        "listChanged": True
                    },
                    "resources": {
                        "subscribe": True,
                        "listChanged": True
                    },
                    "tools": {
                        "listChanged": True
                    }
                },
                "serverInfo": {
                    "name": "ExampleServer",
                    "version": "1.0.0"
                },
                "instructions": "这是一个示例 MCP 服务器"
            }
        }
        
        self.print_step("服务器返回初始化响应", initialize_response)
        
        # 步骤3: 客户端发送已初始化通知
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        self.print_step("客户端发送已初始化通知", initialized_notification)
        
        print("✅ 初始化流程完成，连接准备就绪！")
    
    def demo_message_exchange(self):
        """演示消息交换"""
        print("\n\n🔄 MCP 消息交换演示")
        
        # 工具列表请求
        tools_list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        self.print_step("请求工具列表", tools_list_request)
        
        # 工具列表响应
        tools_list_response = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "读取文件内容",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "文件路径"
                                }
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "list_directory",
                        "description": "列出目录内容",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "目录路径"
                                }
                            },
                            "required": ["path"]
                        }
                    }
                ]
            }
        }
        
        self.print_step("工具列表响应", tools_list_response)
        
        # 工具调用请求
        tool_call_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "read_file",
                "arguments": {
                    "path": "/example/file.txt"
                }
            }
        }
        
        self.print_step("调用工具", tool_call_request)
        
        # 工具调用响应
        tool_call_response = {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "这是文件的内容示例..."
                    }
                ]
            }
        }
        
        self.print_step("工具调用结果", tool_call_response)
    

    
   

def main():
    """主演示函数"""
    demo = MCPProtocolDemo()
    
    print("🎯 MCP 协议完整演示")
    print("参考资料: Model Context Protocol 官方规范")
    print("版本: 2025-03-26")
    
    # 运行各个演示
    demo.demo_initialization_flow()
    demo.demo_message_exchange()
    
 
if __name__ == "__main__":
    main()

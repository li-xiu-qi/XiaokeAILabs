#!/usr/bin/env python3
"""
MCP 工具服务器 - 提供各种实用工具
"""

import datetime
import os
from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message, PromptMessage, TextContent

# 创建 MCP 应用
mcp = FastMCP("Local Tools Server")

@mcp.tool()
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        else:
            return "错误: 表达式包含不允许的字符"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 基础 prompt，返回字符串（会自动转换为 user 消息）
@mcp.prompt
def ask_about_topic(topic: str) -> str:
    """
    生成一个 user 消息，询问某个主题的解释。
    :param topic: 主题名称
    :return: 格式化的用户消息字符串
    """
    return f"Can you please explain the concept of '{topic}'?"


# prompt，返回特定的 PromptMessage 类型
@mcp.prompt
def generate_code_request(language: str, task_description: str) -> PromptMessage:
    """
    生成一个 user 消息，请求代码生成。
    :param language: 编程语言
    :param task_description: 任务描述
    :return: PromptMessage 类型的用户消息
    """
    content = f"Write a {language} function that performs the following task: {task_description}"
    return PromptMessage(role="user", content=TextContent(type="text", text=content))

# 基础动态资源，返回字符串
@mcp.resource("resource://greeting")
def get_greeting() -> str:
    """
    提供一个简单的问候消息。
    :return: 问候字符串
    """
    return "Hello from FastMCP Resources!"


# 返回 JSON 数据的资源（dict 会自动序列化）
@mcp.resource("data://config")
def get_config() -> dict:
    """
    提供应用配置，返回 JSON 格式。
    :return: 配置字典
    """
    return {
        "theme": "dark",
        "version": "1.2.0",
        "features": ["tools", "resources"],
    }

if __name__ == "__main__":
    # 使用标准输入输出模式启动服务器
    # mcp.run(transport="stdio") 
    
    # 使用 HTTP 模式启动服务器（FastMCP 2.0 推荐的方式）
    mcp.run(transport="http", host="127.0.0.1", port=9000)

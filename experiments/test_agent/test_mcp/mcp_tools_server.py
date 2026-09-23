#!/usr/bin/env python3
"""
MCP 工具服务器 - 提供各种实用工具
"""

import datetime
import os
from fastmcp import FastMCP

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

@mcp.tool()
def get_time() -> str:
    """获取当前时间"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前时间: {current_time}"

@mcp.tool()
def list_files(directory: str = ".") -> str:
    """列出指定目录的文件"""
    try:
        files = os.listdir(directory)
        return f"目录 {directory} 中的文件:\n" + "\n".join(f"- {file}" for file in files)
    except Exception as e:
        return f"读取目录错误: {str(e)}"

@mcp.tool()
def echo(text: str) -> str:
    """回显输入的文本"""
    return f"回显: {text}"

@mcp.tool()
def read_env_var(var_name: str) -> str:
    """读取环境变量"""
    value = os.getenv(var_name)
    if value:
        return f"环境变量 {var_name}: {value}"
    else:
        return f"环境变量 {var_name} 未设置"

@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """写入文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件: {filename}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"

@mcp.tool()
def read_file(filename: str) -> str:
    """读取文件内容"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件 {filename} 内容:\\n{content}"
    except Exception as e:
        return f"读取文件失败: {str(e)}"

if __name__ == "__main__":
    mcp.run()

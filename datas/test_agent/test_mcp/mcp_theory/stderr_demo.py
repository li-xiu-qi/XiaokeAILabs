#!/usr/bin/env python3
"""
简单演示 stderr 和 stdout 的区别
"""

import subprocess

def simple_demo():
    """简单演示"""
    print("=== 简单演示 stderr 和 stdout 的区别 ===")
    
    # 创建一个同时输出到两个流的程序
    code = '''
import sys
print("这是正常输出", file=sys.stdout)
print("这是错误信息", file=sys.stderr)
print("数据处理结果: 42", file=sys.stdout)
print("[DEBUG] 内部状态正常", file=sys.stderr)
'''
    
    process = subprocess.Popen(
        ['python', '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    print("从 stdout 读到:")
    print(f"  '{stdout.strip()}'")
    
    print("\n从 stderr 读到:")
    print(f"  '{stderr.strip()}'")

def mcp_example():
    """MCP 场景示例"""
    print("\n=== MCP 中的实际应用 ===")
    
    server_code = '''
import json
import sys

# 服务器日志到 stderr
print("MCP服务器启动", file=sys.stderr)

# 读取一个请求
request_line = '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# 解析请求 - 日志到 stderr
print(f"收到请求: tools/list", file=sys.stderr)

# 构造响应 - 输出到 stdout
response = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
print(json.dumps(response), file=sys.stdout)

# 完成日志到 stderr
print("请求处理完成", file=sys.stderr)
'''
    
    process = subprocess.Popen(
        ['python', '-c', server_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    
    print("JSON-RPC 响应 (stdout):")
    print(f"  {stdout.strip()}")
    
    print("\n服务器日志 (stderr):")
    for line in stderr.strip().split('\n'):
        if line:
            print(f"  [LOG] {line}")

if __name__ == "__main__":
    simple_demo()
    mcp_example()

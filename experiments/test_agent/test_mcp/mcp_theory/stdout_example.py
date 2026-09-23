#!/usr/bin/env python3
"""
演示 stdout=subprocess.PIPE 的使用
"""

import subprocess
import time
import json
import threading

def example_without_stdout_pipe():
    """不使用stdout管道的例子 - 无法读取子进程输出"""
    print("=== 不使用stdout管道的例子 ===")
    
    # 启动一个会输出数据的子进程，但没有创建stdout管道
    code = '''
import time
for i in range(3):
    print(f"子进程输出第 {i+1} 条消息")
    time.sleep(1)
print("子进程结束")
'''
    
    process = subprocess.Popen(['python', '-c', code])
    # 注意：没有设置 stdout=subprocess.PIPE
    # 子进程的输出会直接显示在终端，但父进程无法程序化地读取
    
    print("父进程无法读取子进程的输出数据，只能等待子进程结束")
    process.wait()
    print("子进程已结束，但我们无法获取它输出了什么内容")

def example_with_stdout_pipe():
    """使用stdout管道的例子 - 可以读取子进程输出"""
    print("\n=== 使用stdout管道的例子 ===")
    
    # 创建一个会输出数据的子进程
    code = '''
import time
import sys
for i in range(3):
    message = f"子进程输出第 {i+1} 条消息"
    print(message)
    sys.stdout.flush()  # 立即刷新输出
    time.sleep(1)
print("子进程结束")
'''
    
    # 启动子进程，创建stdout管道
    process = subprocess.Popen(
        ['python', '-c', code],
        stdout=subprocess.PIPE,   # 创建管道连接到子进程的stdout
        stderr=subprocess.PIPE,   # 也创建stderr管道
        text=True                 # 以文本模式处理
    )
    
    print("父进程开始读取子进程的输出...")
    
    # 方法1：逐行读取输出
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(f"父进程读到: {line.strip()}")
    
    # 等待子进程结束
    process.wait()
    print("成功读取了所有子进程输出！")

def example_realtime_output():
    """实时读取子进程输出的例子"""
    print("\n=== 实时读取子进程输出的例子 ===")
    
    # 创建一个持续输出的子进程
    code = '''
import time
import sys
import random

for i in range(5):
    message = f"实时消息 {i+1}: {random.randint(1, 100)}"
    print(message)
    sys.stdout.flush()
    time.sleep(2)
print("实时输出结束")
'''
    
    process = subprocess.Popen(
        ['python', '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # 无缓冲，确保实时性
    )
    
    print("开始实时读取输出...")
    
    # 使用线程实时读取输出
    def read_output():
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[实时] {line.strip()}")
    
    # 启动读取线程
    reader_thread = threading.Thread(target=read_output)
    reader_thread.daemon = True
    reader_thread.start()
    
    # 等待子进程结束
    process.wait()
    reader_thread.join(timeout=1)
    print("实时读取完成！")

def example_mcp_like_communication():
    """类似MCP的双向通信例子"""
    print("\n=== 类似MCP的双向通信例子 ===")
    
    # 创建一个模拟MCP服务器的子进程
    server_code = '''
import json
import sys

# 向stderr输出调试信息
print("MCP服务器启动", file=sys.stderr)
sys.stderr.flush()

# 从stdin读取请求，向stdout输出响应
for line in sys.stdin:
    try:
        request = json.loads(line.strip())
        method = request.get("method", "")
        
        # 处理不同的方法
        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}, "prompts": {}},
                    "serverInfo": {"name": "TestServer", "version": "1.0.0"}
                }
            }
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "tools": [
                        {"name": "echo", "description": "回显文本"},
                        {"name": "add", "description": "两数相加"}
                    ]
                }
            }
        elif method == "tools/call":
            tool_name = request.get("params", {}).get("name", "")
            args = request.get("params", {}).get("arguments", {})
            
            if tool_name == "echo":
                result = {"output": f"Echo: {args.get('text', '')}"}
            elif tool_name == "add":
                result = {"result": args.get("a", 0) + args.get("b", 0)}
            else:
                result = {"error": "Unknown tool"}
            
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -1, "message": f"Unknown method: {method}"}
            }
        
        # 输出响应到stdout
        print(json.dumps(response))
        sys.stdout.flush()
        
        # 调试信息到stderr
        print(f"处理了方法: {method}", file=sys.stderr)
        sys.stderr.flush()
        
    except Exception as e:
        print(f"处理请求时出错: {e}", file=sys.stderr)
        sys.stderr.flush()
'''
    
    # 启动服务器进程
    process = subprocess.Popen(
        ['python', '-c', server_code],
        stdin=subprocess.PIPE,    # 用于发送请求
        stdout=subprocess.PIPE,   # 用于接收响应 ← 这是关键！
        stderr=subprocess.PIPE,   # 用于接收调试信息
        text=True,
        bufsize=0
    )
    
    # 启动stderr监控线程
    def monitor_stderr():
        for line in iter(process.stderr.readline, ''):
            if line.strip():
                print(f"[服务器调试] {line.strip()}")
    
    stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
    stderr_thread.start()
    
    time.sleep(0.5)  # 等待服务器启动
    
    # 发送请求并读取响应
    requests = [
        {"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {}},
        {"jsonrpc": "2.0", "method": "tools/list", "id": 2},
        {"jsonrpc": "2.0", "method": "tools/call", "id": 3, "params": {"name": "echo", "arguments": {"text": "Hello MCP!"}}},
        {"jsonrpc": "2.0", "method": "tools/call", "id": 4, "params": {"name": "add", "arguments": {"a": 10, "b": 20}}}
    ]
    
    for req in requests:
        # 发送请求（通过stdin管道）
        json_str = json.dumps(req) + '\n'
        print(f"\n客户端发送: {req['method']}")
        process.stdin.write(json_str)
        process.stdin.flush()
        
        # 读取响应（通过stdout管道）← 这就是stdout管道的作用！
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"客户端收到: {response.get('result', response.get('error'))}")
        
        time.sleep(0.5)
    
    # 关闭连接
    process.stdin.close()
    process.terminate()
    process.wait()

def compare_with_without_stdout_pipe():
    """对比有无stdout管道的区别"""
    print("\n=== 对比有无stdout管道的区别 ===")
    
    code = '''
import sys
print("重要数据: 计算结果是 42")
print("状态: 计算完成")
sys.exit(0)
'''
    
    print("1. 没有stdout管道 - 输出直接显示在终端:")
    process1 = subprocess.Popen(['python', '-c', code])
    process1.wait()
    print("   → 我们看到了输出，但无法在代码中获取这些数据\n")
    
    print("2. 有stdout管道 - 可以在代码中获取输出:")
    process2 = subprocess.Popen(
        ['python', '-c', code],
        stdout=subprocess.PIPE,
        text=True
    )
    stdout, _ = process2.communicate()
    print("   → 通过代码获取的输出:")
    for line in stdout.split('\n'):
        if line.strip():
            print(f"     程序读到: {line}")

if __name__ == "__main__":
    example_without_stdout_pipe()
    example_with_stdout_pipe()
    example_realtime_output()
    example_mcp_like_communication()
    compare_with_without_stdout_pipe()

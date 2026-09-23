#!/usr/bin/env python3
"""
演示 stdin=subprocess.PIPE 的使用
"""

import subprocess
import time
import json

def example_without_pipe():
    """不使用管道的例子 - 无法向子进程发送数据"""
    print("=== 不使用管道的例子 ===")
    
    # 启动 Python 解释器，但没有创建管道
    process = subprocess.Popen(['python', '-c', 'print("子进程启动"); input("等待输入:")'])
    
    # 无法向子进程发送数据，子进程会一直等待用户在终端输入
    print("子进程在等待输入，但我们无法通过代码发送数据给它")
    process.wait()

def example_with_pipe():
    """使用管道的例子 - 可以向子进程发送数据"""
    print("\n=== 使用管道的例子 ===")
    
    # 创建一个简单的子进程，从stdin读取数据
    code = '''
import sys
print("子进程启动，等待接收数据...")
for line in sys.stdin:
    print(f"收到数据: {line.strip()}")
    if line.strip() == "exit":
        break
print("子进程结束")
'''
    
    # 启动子进程，创建stdin管道
    process = subprocess.Popen(
        ['python', '-c', code],
        stdin=subprocess.PIPE,    # 创建管道连接到子进程的stdin
        stdout=subprocess.PIPE,   # 也创建stdout管道以接收输出
        stderr=subprocess.PIPE,   # 创建stderr管道以接收错误
        text=True                 # 以文本模式处理
    )
    
    # 现在我们可以通过管道向子进程发送数据
    messages = ["Hello from parent!", "这是第二条消息", "这是第三条消息", "exit"]
    
    for msg in messages:
        print(f"父进程发送: {msg}")
        process.stdin.write(msg + '\n')  # 写入数据到子进程的stdin
        process.stdin.flush()            # 立即发送数据
        time.sleep(0.5)                  # 稍等一下
    
    # 关闭stdin并等待子进程结束
    process.stdin.close()
    stdout, stderr = process.communicate()
    
    print(f"子进程输出:\n{stdout}")
    if stderr:
        print(f"子进程错误:\n{stderr}")

def mcp_like_example():
    """类似MCP的JSON-RPC通信例子"""
    print("\n=== 类似MCP的JSON-RPC通信例子 ===")
    
    # 模拟一个简单的JSON-RPC服务器
    server_code = '''
import json
import sys

print("JSON-RPC服务器启动", file=sys.stderr)

for line in sys.stdin:
    try:
        request = json.loads(line.strip())
        print(f"收到请求: {request}", file=sys.stderr)
        
        # 简单的echo服务
        if request.get("method") == "echo":
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"message": f"Echo: {request.get('params', {}).get('text', '')}"}
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -1, "message": "Unknown method"}
            }
        
        print(json.dumps(response))
        sys.stdout.flush()
        
    except Exception as e:
        print(f"处理请求时出错: {e}", file=sys.stderr)
'''
    
    # 启动服务器进程
    process = subprocess.Popen(
        ['python', '-c', server_code],
        stdin=subprocess.PIPE,    # 用于发送JSON-RPC请求
        stdout=subprocess.PIPE,   # 用于接收JSON-RPC响应
        stderr=subprocess.PIPE,   # 用于接收调试信息
        text=True,
        bufsize=0                 # 无缓冲，实时通信
    )
    
    # 发送JSON-RPC请求
    requests = [
        {"jsonrpc": "2.0", "method": "echo", "params": {"text": "Hello MCP!"}, "id": 1},
        {"jsonrpc": "2.0", "method": "echo", "params": {"text": "这是第二个请求"}, "id": 2},
        {"jsonrpc": "2.0", "method": "unknown", "params": {}, "id": 3}
    ]
    
    for req in requests:
        # 发送请求
        json_str = json.dumps(req) + '\n'
        print(f"客户端发送: {req}")
        process.stdin.write(json_str)
        process.stdin.flush()
        
        # 读取响应
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"客户端收到: {response}")
        
        time.sleep(0.5)
    
    # 关闭连接
    process.stdin.close()
    process.terminate()
    process.wait()

if __name__ == "__main__":
    # example_without_pipe()  # 这个会卡住，所以先注释掉
    example_with_pipe()
    mcp_like_example()

#!/usr/bin/env python3
"""
MCP Stdio 客户端实现
基于架构文章中的代码实现
"""

import subprocess
import json
import threading
import time
from typing import Dict, Any, Optional, List


class StdioMCPClient:
    """MCP Stdio 传输客户端"""

    # 错误信息常量
    _NOT_INITIALIZED_ERROR = "客户端尚未初始化"

    def __init__(self, server_command: List[str]):
        """
        初始化客户端

        Args:
            server_command: 启动服务器的命令，如 ['python', 'simple_mcp_server.py']
        """
        self.server_command = server_command
        self.process = None
        self.request_id = 0
        self.initialized = False

    def start(self):
        """启动服务器进程并建立连接"""
        print(f"启动 MCP 服务器: {' '.join(self.server_command)}")

        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # 无缓冲，实时通信
        )

        # 启动后台线程监控错误输出
        self._start_error_monitor()

        print("服务器进程已启动")

    def _start_error_monitor(self):
        """启动错误监控线程"""
        def monitor_stderr():
            if self.process and self.process.stderr:
                for line in iter(self.process.stderr.readline, ''):
                    if line.strip():
                        print(f"[服务器] {line.strip()}")

        thread = threading.Thread(target=monitor_stderr, daemon=True)
        thread.start()

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """发送请求并等待响应"""
        if not self.process:
            raise RuntimeError("服务器进程未启动")

        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id
        }

        if params is not None:
            request["params"] = params

        # 发送请求
        json_str = json.dumps(request) + '\n'
        print(f"发送请求: {request}")

        self.process.stdin.write(json_str)
        self.process.stdin.flush()

        # 读取响应
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("服务器无响应")

        response = json.loads(response_line.strip())
        print(f"收到响应: {response}")

        # 检查错误
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"服务器错误 {error['code']}: {error['message']}")

        return response.get("result", {})

    def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """发送通知（无需响应）"""
        if not self.process:
            raise RuntimeError("服务器进程未启动")

        notification = {
            "jsonrpc": "2.0",
            "method": method
        }

        if params is not None:
            notification["params"] = params

        # 发送通知
        json_str = json.dumps(notification) + '\n'
        print(f"发送通知: {notification}")

        self.process.stdin.write(json_str)
        self.process.stdin.flush()

    def initialize(self) -> Dict[str, Any]:
        """执行 MCP 初始化流程"""
        print("开始 MCP 初始化流程...")

        # 1. 发送初始化请求
        init_params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "SimpleMCPClient",
                "version": "1.0.0"
            }
        }

        result = self.send_request("initialize", init_params)

        # 2. 发送已初始化通知
        self.send_notification("notifications/initialized")

        self.initialized = True
        print("MCP 初始化完成!")

        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        if not self.initialized:
            raise RuntimeError(self._NOT_INITIALIZED_ERROR)

        result = self.send_request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if not self.initialized:
            raise RuntimeError(self._NOT_INITIALIZED_ERROR)

        params = {
            "name": name,
            "arguments": arguments
        }
        return self.send_request("tools/call", params)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """获取可用提示列表"""
        if not self.initialized:
            raise RuntimeError(self._NOT_INITIALIZED_ERROR)

        result = self.send_request("prompts/list")
        return result.get("prompts", [])

    def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """获取提示"""
        if not self.initialized:
            raise RuntimeError(self._NOT_INITIALIZED_ERROR)

        params = {
            "name": name,
            "arguments": arguments
        }
        return self.send_request("prompts/get", params)

    def close(self):
        """关闭连接"""
        if self.process:
            print("关闭 MCP 连接...")
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("连接已关闭")


def main():
    """客户端测试主函数"""
    client = StdioMCPClient(['python', 'simple_mcp_server.py'])

    try:
        client.start()
        time.sleep(1)  # 等待服务器启动

        server_info = client.initialize()
        print(f"\n服务器信息: {server_info}")

        print("\n=== 测试工具列表 ===")
        tools = client.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")

        print("\n=== 测试工具调用 ===")
        echo_result = client.call_tool("echo", {"text": "Hello, MCP!"})
        print(f"Echo 结果: {echo_result}")

        calc_result = client.call_tool("calculate", {"expression": "10 + 5 * 2"})
        print(f"计算结果: {calc_result}")

        print("\n=== 测试提示列表 ===")
        prompts = client.list_prompts()
        for prompt in prompts:
            print(f"- {prompt['name']}: {prompt['description']}")

        print("\n=== 测试获取提示 ===")
        greeting_prompt = client.get_prompt("greeting", {"name": "筱可"})
        print(f"问候提示: {greeting_prompt}")

    except Exception as e:
        print(f"错误: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    main()

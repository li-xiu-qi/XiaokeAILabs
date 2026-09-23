#!/usr/bin/env python3
"""
JSON-RPC 2.0 处理器
基于架构文章中的代码实现
"""

import json
import sys
import time
from typing import Dict, Any, Optional, Union


class JSONRPCHandler:
    """JSON-RPC 2.0 消息处理器"""
    
    def __init__(self):
        self.request_id = 0
    
    def create_request(self, method: str, params: Optional[Union[Dict[str, Any], list]] = None) -> Dict[str, Any]:
        """创建 JSON-RPC 请求"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self.request_id
        }
        if params is not None:
            request["params"] = params
        return request
    
    def create_response(self, request_id: Union[int, str, None], result: Any = None, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建 JSON-RPC 响应"""
        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        if error:
            response["error"] = error
        else:
            response["result"] = result
        return response
    
    def create_notification(self, method: str, params: Optional[Union[Dict[str, Any], list]] = None) -> Dict[str, Any]:
        """创建 JSON-RPC 通知"""
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params is not None:
            notification["params"] = params
        return notification
    
    def create_error(self, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """创建错误对象"""
        error = {
            "code": code,
            "message": message
        }
        if data is not None:
            error["data"] = data
        return error
    
    def send_message(self, message: Dict[str, Any]):
        """通过 stdout 发送 JSON-RPC 消息"""
        json_str = json.dumps(message, ensure_ascii=False)
        print(json_str, flush=True)  # 立即刷新输出缓冲区
    
    def receive_message(self) -> Optional[Dict[str, Any]]:
        """从 stdin 接收 JSON-RPC 消息"""
        try:
            line = sys.stdin.readline().strip()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError as e:
            # 返回解析错误响应
            error_response = self.create_response(
                None, 
                error=self.create_error(
                    -32700,
                    "Parse error",
                    str(e)
                )
            )
            self.send_message(error_response)
            return None
    
    def validate_request(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """验证 JSON-RPC 请求格式"""
        # 检查基本字段
        if message.get("jsonrpc") != "2.0":
            return self.create_error(-32600, "Invalid Request", "Missing or invalid 'jsonrpc' field")
        
        if "method" not in message:
            return self.create_error(-32600, "Invalid Request", "Missing 'method' field")
        
        if not isinstance(message["method"], str):
            return self.create_error(-32600, "Invalid Request", "'method' must be a string")
        
        # 检查参数类型
        if "params" in message:
            params = message["params"]
            if not isinstance(params, (dict, list)):
                return self.create_error(-32602, "Invalid params", "params must be Object or Array")
        
        return None
    
    def is_notification(self, message: Dict[str, Any]) -> bool:
        """判断消息是否为通知（没有 id 字段）"""
        return "id" not in message


# JSON-RPC 错误代码常量
class JSONRPCError:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # 服务器自定义错误（-32000 到 -32099）
    SERVER_ERROR_BASE = -32000


def demo_jsonrpc_usage():
    """演示 JSON-RPC 处理器的使用"""
    handler = JSONRPCHandler()
    
    print("=== JSON-RPC 2.0 处理器演示 ===\n")
    
    # 1. 创建请求
    print("1. 创建各种类型的消息:")
    
    # 位置参数请求
    request1 = handler.create_request("subtract", [42, 23])
    print(f"位置参数请求: {json.dumps(request1, indent=2)}")
    
    # 命名参数请求
    request2 = handler.create_request("subtract", {"subtrahend": 23, "minuend": 42})
    print(f"\n命名参数请求: {json.dumps(request2, indent=2)}")
    
    # 通知
    notification = handler.create_notification("update", {"progress": 50})
    print(f"\n通知消息: {json.dumps(notification, indent=2)}")
    
    # 2. 创建响应
    print("\n2. 创建响应:")
    
    # 成功响应
    success_response = handler.create_response(1, 19)
    print(f"成功响应: {json.dumps(success_response, indent=2)}")
    
    # 错误响应
    error_response = handler.create_response(
        2, 
        error=handler.create_error(
            JSONRPCError.METHOD_NOT_FOUND,
            "Method not found",
            "The method 'foobar' does not exist"
        )
    )
    print(f"\n错误响应: {json.dumps(error_response, indent=2)}")
    
    # 3. 批处理
    print("\n3. 批处理示例:")
    batch_request = [
        handler.create_request("sum", [1, 2, 4]),
        handler.create_request("get_data"),
        handler.create_notification("notify_hello")
    ]
    print(f"批处理请求: {json.dumps(batch_request, indent=2)}")
    
    batch_response = [
        handler.create_response(1, 7),
        handler.create_response(2, ["hello", "world"])
        # 通知无需响应
    ]
    print(f"\n批处理响应: {json.dumps(batch_response, indent=2)}")
    
    # 4. 消息验证
    print("\n4. 消息验证:")
    
    valid_message = {"jsonrpc": "2.0", "method": "test", "id": 1}
    error = handler.validate_request(valid_message)
    print(f"有效消息验证: {error if error else '通过'}")
    
    invalid_message = {"method": "test", "id": 1}  # 缺少 jsonrpc 字段
    error = handler.validate_request(invalid_message)
    print(f"无效消息验证: {json.dumps(error, indent=2) if error else '通过'}")


if __name__ == "__main__":
    demo_jsonrpc_usage()

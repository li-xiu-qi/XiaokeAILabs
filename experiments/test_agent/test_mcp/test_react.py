#!/usr/bin/env python3
"""
测试 ReAct 智能体
"""
import asyncio
import sys
import os

from react_agent_v4 import ReActAgentV4

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_react_agent():
    """测试 ReAct 智能体"""
    agent = ReActAgentV4()
    
    try:
        print("🔧 正在启动 MCP 连接...")
        await agent.start_mcp_connection()
        
        print("\n🧪 测试开始")
        
        # 测试简单任务
        test_message = "现在几点了？"
        print(f"📝 测试问题: {test_message}")
        
        response = await agent.chat(test_message)
        print(f"✅ 测试结果: {response}")
        
        print("\n🧪 测试复杂任务")
        
        # 测试复杂任务
        complex_message = "帮我计算 10+20*3，然后列出当前目录的文件"
        print(f"📝 复杂测试: {complex_message}")
        
        response = await agent.chat(complex_message)
        print(f"✅ 复杂测试结果: {response}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await agent.close()
        print("\n✅ 测试完成")

if __name__ == "__main__":
    asyncio.run(test_react_agent())

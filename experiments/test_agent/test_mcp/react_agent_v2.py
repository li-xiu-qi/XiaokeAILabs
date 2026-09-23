#!/usr/bin/env python3
"""
ReAct 智能体 - 基于 MCP 工具的反应式推理智能体
实现 Reasoning and Acting 模式：思考 -> 行动 -> 观察 -> 再思考
"""

import os
import json
import requests
import asyncio
import re
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载环境变量
load_dotenv()

class ReActAgent:
    def __init__(self):
        self.api_key = os.getenv("VOLCES_API_KEY")
        self.base_url = os.getenv("VOLCES_BASE_URL")
        self.model = os.getenv("VOLCES_TEXT_MODEL")
        self.mcp_session = None
        self.available_tools = {}
        self.max_iterations = 8  # 增加最大推理循环次数
        
    async def start_mcp_connection(self):
        """启动 MCP 连接"""
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_tools_server.py"],
        )
        
        self.server_connection = stdio_client(server_params)
        self.read, self.write = await self.server_connection.__aenter__()
        self.mcp_session = ClientSession(self.read, self.write)
        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()
        
        # 获取可用工具
        tools = await self.mcp_session.list_tools()
        for tool in tools.tools:
            self.available_tools[tool.name] = tool.description
            
        print(f"✅ MCP 工具服务器已连接，可用工具: {list(self.available_tools.keys())}")
    
    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """调用 MCP 工具"""
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)
            return result.content[0].text
        except Exception as e:
            return f"工具调用失败: {str(e)}"
    
    def call_ai_with_react_prompt(self, user_message: str, conversation_history: list, is_first_turn: bool = False) -> str:
        """使用 ReAct 提示词调用 AI"""
        
        tools_description = "\n".join([f"- {name}: {desc}" for name, desc in self.available_tools.items()])
        
        if is_first_turn:
            # 第一轮强制要求执行工具
            system_prompt = f"""你是一个智能助手，使用 ReAct (Reasoning and Acting) 模式来解决问题。

可用工具：
{tools_description}

重要规则：
1. 这是第一轮对话，你必须先执行工具获取信息，不能直接给出 Final Answer
2. 仔细分析用户需求，确定需要哪些工具
3. 每次只执行一个工具

严格按照以下格式输出：
Thought: [分析用户需求，决定要执行什么工具]
Action: [工具名称] [参数JSON]

例如用户问"计算1+2然后告诉我时间"，你应该：
Thought: 用户需要计算1+2，我先执行计算工具
Action: calculate {{"expression": "1+2"}}

现在处理用户请求：{user_message}"""
        else:
            # 后续轮次可以选择继续执行工具或给出答案
            system_prompt = f"""你是一个智能助手，继续使用 ReAct 模式。

可用工具：
{tools_description}

根据之前的工具执行结果，决定下一步行动：

选项1 - 如果还需要更多信息，继续执行工具：
Thought: [分析还需要什么信息]
Action: [工具名称] [参数JSON]

选项2 - 如果信息已足够，给出最终答案：
Thought: [基于工具结果的分析]
Final Answer: [基于真实工具结果的答案]

记住：答案必须基于真实的工具执行结果，不能编造任何信息。"""

        # 构建对话历史
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史
        for entry in conversation_history:
            messages.append({"role": "user", "content": entry["content"]})
        
        # 如果不是第一轮，添加当前用户消息
        if not is_first_turn:
            messages.append({"role": "user", "content": "基于上述工具执行结果，继续推理"})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API 调用失败: {response.status_code}"
        except Exception as e:
            return f"API 错误: {str(e)}"
    
    def parse_react_response(self, response: str) -> dict:
        """解析 ReAct 响应"""
        result = {
            "thought": "",
            "action": None,
            "action_params": {},
            "final_answer": "",
            "type": "unknown"
        }
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                current_section = "thought"
                result["thought"] = line[8:].strip()
                result["type"] = "thought"
            elif line.startswith("Action:"):
                current_section = "action"
                action_content = line[7:].strip()
                # 解析 Action: tool_name {params}
                parts = action_content.split(' ', 1)
                if len(parts) >= 1:
                    result["action"] = parts[0]
                    if len(parts) > 1:
                        try:
                            result["action_params"] = json.loads(parts[1])
                        except:
                            result["action_params"] = {}
                result["type"] = "action"
            elif line.startswith("Final Answer:"):
                current_section = "final"
                result["final_answer"] = line[13:].strip()
                result["type"] = "final"
            elif current_section and line:
                # 继续添加到当前部分
                if current_section == "thought":
                    result["thought"] += " " + line
                elif current_section == "final":
                    result["final_answer"] += " " + line
        
        return result
    
    async def react_reasoning_loop(self, user_message: str) -> str:
        """执行 ReAct 推理循环"""
        conversation_history = []
        iteration = 0
        
        print(f"\n🤖 开始 ReAct 推理循环处理: {user_message}\n")
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"🔄 第 {iteration} 轮推理")
            
            # 1. 调用 AI 进行推理
            is_first_turn = (iteration == 1)
            ai_response = self.call_ai_with_react_prompt(user_message, conversation_history, is_first_turn)
            
            # 2. 解析响应
            parsed = self.parse_react_response(ai_response)
            
            print(f"💭 Thought: {parsed['thought']}")
            
            if parsed["type"] == "final":
                print(f"✅ Final Answer: {parsed['final_answer']}\n")
                return parsed['final_answer']
            
            elif parsed["type"] == "action":
                action_name = parsed["action"]
                action_params = parsed["action_params"]
                
                print(f"🔧 Action: {action_name} {json.dumps(action_params, ensure_ascii=False)}")
                
                # 3. 执行工具
                if action_name in self.available_tools:
                    tool_result = await self.call_mcp_tool(action_name, action_params)
                    print(f"👁️ Observation: {tool_result}")
                    
                    # 4. 将结果添加到对话历史
                    conversation_history.append({
                        "content": f"Thought: {parsed['thought']}\nAction: {action_name} {json.dumps(action_params)}\nObservation: {tool_result}"
                    })
                else:
                    error_msg = f"工具 {action_name} 不存在，可用工具: {list(self.available_tools.keys())}"
                    print(f"❌ Error: {error_msg}")
                    conversation_history.append({
                        "content": f"Thought: {parsed['thought']}\nAction: {action_name} {json.dumps(action_params)}\nObservation: {error_msg}"
                    })
            else:
                # 如果AI只给出了思考但没有行动，提醒它执行工具
                print("⚠️ AI只给出了思考，提醒执行工具")
                conversation_history.append({
                    "content": f"你刚才只进行了思考：{parsed['thought']}，但没有执行任何工具。请选择一个合适的工具来获取信息。"
                })
            
            print()  # 空行分隔
        
        return f"推理循环达到最大次数 ({self.max_iterations})，未能得到最终答案。"
    
    async def chat(self, user_message: str) -> str:
        """处理用户消息 - ReAct 模式"""
        try:
            return await self.react_reasoning_loop(user_message)
        except Exception as e:
            return f"处理错误: {str(e)}"
    
    async def close(self):
        """关闭 MCP 连接"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if hasattr(self, 'server_connection'):
            await self.server_connection.__aexit__(None, None, None)

async def main():
    agent = ReActAgent()
    
    try:
        # 启动 MCP 连接
        await agent.start_mcp_connection()
        
        print("\n🧠 ReAct 智能体已启动！")
        print("💡 我会使用推理-行动-观察循环来解决问题")
        print("📝 输入 'quit' 退出\n")
        
        while True:
            try:
                user_input = input("👤 您: ")
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input.strip():
                    continue
                
                response = await agent.chat(user_input)
                print(f"🤖 最终回答: {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {str(e)}\n")
    
    finally:
        await agent.close()
        print("\n👋 ReAct 智能体已关闭")

if __name__ == "__main__":
    asyncio.run(main())

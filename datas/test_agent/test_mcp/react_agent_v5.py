#!/usr/bin/env python3
"""
ReAct 智能体 V5 - 智能工具选择策略
基于 MCP 工具的反应式推理智能体，智能选择是否使用工具
实现 Reasoning and Acting 模式：思考 -> 行动 -> 观察 -> 再思考
核心原则：根据问题类型智能选择是否使用工具
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

class ReActAgentV5:
    def __init__(self):
        self.api_key = os.getenv("GUIJI_API_KEY")
        self.base_url = os.getenv("GUIJI_BASE_URL")
        self.model = os.getenv("GUIJI_MODEL")
        self.mcp_session = None
        self.available_tools = {}
        self.max_iterations = 10
        self.tool_capabilities = {
            "calculate": ["计算", "数学", "运算", "算式", "加", "减", "乘", "除", "+", "-", "*", "/", "^", "pow"],
            "get_time": ["时间", "现在", "几点", "当前时间", "日期"],
            "list_files": ["文件", "目录", "列表", "ls", "dir", "文件夹", "查看文件"],
            "read_file": ["读取", "打开", "查看内容", "文件内容"],
            "write_file": ["写入", "创建", "保存", "写文件"],
            "run_command": ["执行", "运行", "命令", "shell", "cmd"],
            "weather": ["天气", "气温", "下雨", "晴天"],
            "search": ["搜索", "查找", "找"],
            "set_memory": ["记住", "存储", "记录", "保存到内存"],
            "get_memory": ["回忆", "获取记忆", "读取内存", "之前保存的"],
        }
        
    def analyze_user_intent(self, user_message: str) -> list:
        """分析用户意图，判断哪些工具可能有用"""
        user_message_lower = user_message.lower()
        applicable_tools = []
        
        for tool_name, keywords in self.tool_capabilities.items():
            if tool_name in self.available_tools:
                for keyword in keywords:
                    if keyword in user_message_lower:
                        applicable_tools.append(tool_name)
                        break
        
        return list(set(applicable_tools))
        
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
        """使用 ReAct 提示词调用 AI - 智能选择策略"""
        tools_description = "\n".join([f"- {name}: {desc}" for name, desc in self.available_tools.items()])
        
        # 分析用户意图，找出可能适用的工具
        applicable_tools = self.analyze_user_intent(user_message)
        
        if is_first_turn:
            # 第一轮：智能选择是否使用工具
            system_prompt = f"""你是一个基于工具的ReAct智能助手，擅长通过工具获取准确信息来解决问题。

🔧 可用工具：
{tools_description}

🎯 用户请求：{user_message}
🎯 建议工具：{applicable_tools if applicable_tools else "需要分析确定"}

📋 ReAct工作流程：
1. Thought: 分析问题，判断是否需要工具
2. Action: 如果需要工具，执行一个工具获取信息
3. Observation: 观察工具返回结果
4. 重复1-3直到有足够信息
5. Final Answer: 基于已有信息给出完整答案

⚠️ 重要规则：
- 如果问题可以直接回答且不需要实时信息，可以直接给出答案
- 如果需要获取实时信息、计算、文件操作等，使用相应工具
- 每次只执行一个工具
- 工具参数必须是有效的JSON格式

输出格式：
方案A（使用工具）：
Thought: [分析为什么需要使用工具]
Action: [工具名] [JSON参数]

方案B（直接回答）：
Thought: [分析为什么可以直接回答]
Final Answer: [完整的回答]

现在开始处理用户请求！"""
        else:
            # 后续轮次：基于已有信息决定下一步
            system_prompt = f"""继续ReAct推理流程。

🔧 可用工具：
{tools_description}

🎯 原始请求：{user_message}

📊 当前状态分析：
回顾上一步的 Observation。根据已执行的工具和获得的信息，现在需要判断：

方案A - 继续使用工具（如果需要更多信息）：
Thought: [分析为什么还需要更多信息，以及需要哪个工具来获取这些信息。]
Action: [工具名] [JSON参数]

方案B - 给出最终答案（如果信息已足够）:
Thought: [分析已有信息是否足够回答用户问题。确认信息已足够。]
Final Answer: [基于已有信息的完整、准确答案。]

⚠️ 决策标准：
- 如果已有信息足够回答用户问题，直接选择方案B
- 如果确实还需要获取新信息，选择方案A
- 答案必须基于真实的工具结果，不能编造

请选择合适的方案并执行："""

        # 构建对话历史
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加对话历史
        for entry in conversation_history:
            messages.append({"role": "user", "content": entry["content"]})
        
        # 如果不是第一轮，添加当前用户消息
        if not is_first_turn:
            messages.append({"role": "user", "content": "基于上述工具执行结果，继续推理。"})
        
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
                if current_section == "thought":
                    result["thought"] += " " + line
                elif current_section == "final":
                    result["final_answer"] += " " + line
        
        return result
    
    async def react_reasoning_loop(self, user_message: str) -> str:
        """执行 ReAct 推理循环 - 智能选择工具版本"""
        conversation_history = []
        iteration = 0
        
        print(f"\n🤖 开始 ReAct 推理循环处理 (智能模式 V5): {user_message}\n")
        
        applicable_tools = self.analyze_user_intent(user_message)
        if applicable_tools:
            print(f"🎯 检测到可能需要的工具: {applicable_tools}")
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"🔄 第 {iteration} 轮推理")
            
            is_first_turn = (iteration == 1)
            ai_response = self.call_ai_with_react_prompt(user_message, conversation_history, is_first_turn)
            parsed = self.parse_react_response(ai_response)
            
            print(f"💭 Thought: {parsed['thought']}")
            
            if parsed["type"] == "final":
                print(f"✅ Final Answer: {parsed['final_answer']}\n")
                return parsed['final_answer']
            
            elif parsed["type"] == "action":
                action_name = parsed["action"]
                action_params = parsed["action_params"]
                
                print(f"🔧 Action: {action_name} {json.dumps(action_params, ensure_ascii=False)}")
                
                if action_name in self.available_tools:
                    tool_result = await self.call_mcp_tool(action_name, action_params)
                    print(f"👁️ Observation: {tool_result}")
                    conversation_history.append({
                        "content": f"Thought: {parsed['thought']}\nAction: {action_name} {json.dumps(action_params)}\nObservation: {tool_result}"
                    })
                else:
                    error_msg = f"工具 {action_name} 不存在或不可用。可用工具: {list(self.available_tools.keys())}"
                    print(f"❌ Error: {error_msg}")
                    conversation_history.append({
                        "content": f"Thought: {parsed['thought']}\nAction: {action_name} {json.dumps(action_params)}\nObservation: {error_msg}"
                    })
            else:
                # AI只给出了思考，提醒它执行工具或给出最终答案
                print("⚠️ AI只给出了思考，要求执行工具或给出最终答案")
                conversation_history.append({
                    "content": f"你刚才只进行了思考：{parsed['thought']}，请执行一个工具来获取信息，或者基于已有信息给出最终答案。"
                })
            
            print()
        
        # 达到最大迭代次数，尝试总结答案
        print(f"⚠️ 推理循环达到最大次数 ({self.max_iterations})，尝试基于已有信息总结答案")
        
        summary_prompt_messages = [{"role": "system", "content": "请根据以下对话历史，总结出一个最终答案来回应用户的原始请求。"}]
        summary_prompt_messages.append({"role": "user", "content": f"原始请求: {user_message}"})
        for entry in conversation_history:
            summary_prompt_messages.append({"role": "assistant", "content": entry["content"]})
        summary_prompt_messages.append({"role": "user", "content": "请总结最终答案。"})

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "messages": summary_prompt_messages, "temperature": 0.1, "max_tokens": 500}
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                summary_answer = response.json()["choices"][0]["message"]["content"]
                print(f"✅ Final Answer (summarized): {summary_answer}\n")
                return summary_answer
            else:
                return f"推理循环达到最大次数，且无法生成总结性答案。API 错误: {response.status_code}"
        except Exception as e:
            return f"推理循环达到最大次数，且总结答案时发生API错误: {str(e)}"

    async def chat(self, user_message: str) -> str:
        """处理用户消息 - ReAct 智能模式 V5"""
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
    agent = ReActAgentV5()
    
    try:
        await agent.start_mcp_connection()
        
        print("\n🧠 ReAct 智能体 V5 已启动！(智能模式)")
        print("💡 我会根据问题类型智能选择是否使用工具")
        print("📝 输入 'q' 退出\n")
        
        while True:
            try:
                user_input = input("👤 您: ")
                if user_input.lower() in ['q', 'quit', 'exit', '退出']:
                    break
                
                if not user_input.strip():
                    continue
                
                response = await agent.chat(user_input)
                print(f"🤖 最终回答 (V5): {response}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {str(e)}\n")
    
    finally:
        await agent.close()
        print("\n👋 ReAct 智能体 V5 已关闭")

if __name__ == "__main__":
    asyncio.run(main())

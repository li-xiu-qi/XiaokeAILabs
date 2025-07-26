import time
import os
from openai import OpenAI



class OpenAICompatibleLLM:
    """
    一个使用OpenAI兼容API的大语言模型类。
    它会从环境变量中读取配置。
    """
    def __init__(self, model="gpt-3.5-turbo"):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        if not api_key or not base_url:
            raise ValueError("错误：请设置 OPENAI_API_KEY 和 OPENAI_API_BASE 环境变量。")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        print(f"🤖 LLM客户端已初始化，将使用模型: {self.model}，接入点: {base_url}")

    def generate(self, context):
        """使用API根据上下文生成回答。"""
        print(f"🤖 [LLM API] 正在根据 {len(context)} 条消息调用API生成回答...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=context,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用API时出错: {e}")
            return "抱歉，我在处理您的请求时遇到了一个问题。"

    def summarize(self, messages_to_summarize):
        """使用API对一段对话历史进行摘要。"""
        num_messages = len(messages_to_summarize)
        print(f"🤖 [LLM API] 正在将 {num_messages} 条旧消息发送到API进行摘要...")
        
        # 创建一个专门用于摘要的上下文
        summary_prompt = [
            {"role": "system", "content": "你是一个高效的对话摘要助手。请根据下面的对话内容，生成一个简洁、准确的摘要，保留关键信息。"},
        ] + messages_to_summarize

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用API进行摘要时出错: {e}")
            return f"对前{num_messages}条消息的摘要生成失败。"

def count_tokens(text):
    """一个简单的函数，通过计算词数来模拟计算token数量。
    注意: 这只是一个粗略的模拟，实际token数应使用tiktoken等库计算。"""
    return len(text.split())

# --- 基础记忆类 (无需修改) ---

class BaseMemory:
    """所有记忆管理策略的基类。"""
    def __init__(self, llm, max_tokens=100):
        self.llm = llm
        self.history = []
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        """添加一条消息到历史记录中。"""
        message = {"role": role, "content": content}
        self.history.append(message)
        # 模拟LLM生成回应
        if role == 'user':
            assistant_response = self.llm.generate(self.get_context())
            self.history.append({"role": "assistant", "content": assistant_response})
        self.enforce_policy() # 应用记忆策略

    def get_context(self):
        """获取当前可用的上下文。"""
        return self.history

    def get_total_tokens(self):
        """计算当前历史记录的总token数。"""
        return sum(count_tokens(msg['content']) for msg in self.history)

    def enforce_policy(self):
        """子类需要实现这个方法来定义自己的记忆管理策略。"""
        raise NotImplementedError

    def print_status(self):
        """打印当前记忆状态。"""
        print(f"  [状态] 消息数量: {len(self.history)}, "
              f"Token数量: {self.get_total_tokens()}/{self.max_tokens}")
        print("  " + "="*50)
        for msg in self.history:
            print(f"    {msg['role']}: {msg['content']}")
        print("  " + "="*50 + "\n")


# --- 策略 1: 滑动窗口 (Sliding Window) (无需修改) ---

class SlidingWindowMemory(BaseMemory):
    """
    使用滑动窗口策略管理记忆。
    当token数量超过限制时，从最前面移除旧的消息。
    """
    def enforce_policy(self):
        print("-> 应用滑动窗口策略...")
        while self.get_total_tokens() > self.max_tokens and len(self.history) > 1:
            removed_message = self.history.pop(0)
            print(f"  [滑动窗口] Token超限，移除最旧的消息: '{removed_message['content'][:30]}...'")

# --- 策略 2: 摘要 (Summarization) (无需修改) ---

class SummarizationMemory(BaseMemory):
    """
    使用摘要策略管理记忆。
    当token数量达到阈值时，将旧消息进行摘要。
    """
    def enforce_policy(self):
        print("-> 应用摘要策略...")
        if self.get_total_tokens() > self.max_tokens and len(self.history) > 3:
            print(f"  [摘要] Token超限，准备对旧消息进行摘要。")
            num_to_summarize = len(self.history) // 2
            messages_to_summarize = self.history[:num_to_summarize]
            summary = self.llm.summarize(messages_to_summarize)
            summary_message = {"role": "system", "content": f"对话早期内容的摘要: {summary}"}
            self.history = [summary_message] + self.history[num_to_summarize:]
            print(f"  [摘要] 创建了新的摘要并替换了 {num_to_summarize} 条旧消息。")

# --- 策略 3: 混合策略 (Hybrid) (无需修改) ---

class HybridMemory(BaseMemory):
    """
    混合策略：保留最近的几条消息，并对更早的消息进行摘要。
    """
    def __init__(self, llm, max_tokens=100, recent_message_limit=4):
        super().__init__(llm, max_tokens)
        self.recent_message_limit = recent_message_limit

    def enforce_policy(self):
        print("-> 应用混合策略...")
        while self.get_total_tokens() > self.max_tokens and len(self.history) > self.recent_message_limit:
            num_to_process = len(self.history) - self.recent_message_limit
            messages_to_process = self.history[:num_to_process]
            if self.history[0]['role'] == 'system' and '摘要' in self.history[0]['content']:
                print(f"  [混合策略] Token超限，将 {len(messages_to_process)-1} 条更早的消息合并到现有摘要中。")
                summary_message = self.history.pop(0)
                messages_to_summarize = messages_to_process + [summary_message]
            else:
                messages_to_summarize = messages_to_process
                print(f"  [混合策略] Token超限，为 {len(messages_to_summarize)} 条最早的消息创建摘要。")
            summary = self.llm.summarize(messages_to_summarize)
            summary_message = {"role": "system", "content": f"对话早期内容的摘要: {summary}"}
            self.history = [summary_message] + self.history[num_to_process:]

# --- 演示 ---

if __name__ == "__main__":
    try:
        # 初始化真实的LLM客户端
        llm = OpenAICompatibleLLM()

        conversation = [
            "你好！我们来聊聊水果吧。我最喜欢苹果。",
            "苹果确实不错，尤其是富士苹果。你喜欢香蕉吗？",
            "香蕉也很好，富含钾元素。我们接下来聊聊旅行计划怎么样？",
            "好主意！我想去云南旅游，听说那里风景很美。",
            "云南的大理和丽江都非常棒。你打算什么时候去？",
            "也许是秋天吧，天气比较凉爽。不过预算是个问题，需要好好规划一下。",
            "是的，规划很重要。我们可以先从机票和住宿开始查起。"
        ]

        print("\n" + "#"*20 + " 1. 滑动窗口策略演示 " + "#"*20)
        sliding_memory = SlidingWindowMemory(llm, max_tokens=150) # 增加了token限制以适应更长的真实回复
        for turn in conversation[:3]: # 仅演示前几轮，避免过多API调用
            sliding_memory.add_message('user', turn)
            sliding_memory.print_status()

        print("\n" + "#"*20 + " 2. 摘要策略演示 " + "#"*20)
        summarization_memory = SummarizationMemory(llm, max_tokens=150)
        for turn in conversation[:4]: # 仅演示前几轮
            summarization_memory.add_message('user', turn)
            summarization_memory.print_status()
            
        print("\n" + "#"*20 + " 3. 混合策略演示 " + "#"*20)
        hybrid_memory = HybridMemory(llm, max_tokens=150, recent_message_limit=4)
        for turn in conversation[:5]: # 仅演示前几轮
            hybrid_memory.add_message('user', turn)
            hybrid_memory.print_status()

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"程序运行中发生未知错误: {e}")
